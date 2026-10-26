//! Camera calculations use metres in f64, then convert view-relative coordinates
//! for drawing. None of these conversions writes back to accepted particle data.

use super::RenderError;
use crate::scene::Particle;
use serde::{Deserialize, Serialize};

pub(crate) type V = [f64; 3];
pub(crate) fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}
pub(crate) fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
pub(crate) fn mul(a: V, s: f64) -> V {
    a.map(|x| x * s)
}
pub(crate) fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn unit(a: V) -> Result<V, RenderError> {
    let n = dot(a, a).sqrt();
    if !n.is_finite() || n <= 0.0 {
        return Err(RenderError(
            "camera direction must be finite and non-zero".into(),
        ));
    }
    Ok(mul(a, 1.0 / n))
}
pub(crate) fn position(p: &Particle) -> V {
    [
        p.position_m.x.into(),
        p.position_m.y.into(),
        p.position_m.z.into(),
    ]
}

/// Perspective camera in right-handed, positive-Y-up world metres.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Camera {
    /// Eye position in metres.
    pub eye_m: V,
    /// Look-at position in metres.
    pub target_m: V,
    /// Reference up direction; must not be parallel to the view.
    pub up: V,
    /// Vertical field of view in degrees, strictly between 1 and 179.
    pub vertical_fov_deg: f64,
    /// Positive near clipping distance in metres.
    pub near_m: f64,
    /// Far clipping distance, greater than near.
    pub far_m: f64,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            eye_m: [1.2, 1.0, 1.2],
            target_m: [0.3, 0.4, 0.3],
            up: [0.0, 1.0, 0.0],
            vertical_fov_deg: 45.0,
            near_m: 0.001,
            far_m: 100.0,
        }
    }
}
impl Camera {
    /// Validates vectors, clipping planes and the representability of projection.
    pub fn validate(&self, aspect: f64) -> Result<(), RenderError> {
        if !self
            .eye_m
            .into_iter()
            .chain(self.target_m)
            .chain(self.up)
            .all(f64::is_finite)
            || !self.vertical_fov_deg.is_finite()
            || !(1.0..179.0).contains(&self.vertical_fov_deg)
            || self.vertical_fov_deg == 1.0
            || !self.near_m.is_finite()
            || !self.far_m.is_finite()
            || self.near_m <= 0.0
            || self.far_m <= self.near_m
            || !aspect.is_finite()
            || aspect <= 0.0
        {
            return Err(RenderError(
                "invalid camera vectors, field of view, aspect or clipping distances".into(),
            ));
        }
        let (right, up, forward, distance) = self.basis()?;
        let _ = (right, up, forward);
        let near = self.near_m / distance;
        let far = self.far_m / distance;
        let fy = 1.0 / (self.vertical_fov_deg.to_radians() / 2.0).tan();
        let values = [
            fy / aspect,
            fy,
            far / (far - near),
            -near * far / (far - near),
        ];
        if !values.into_iter().all(|x| (x as f32).is_finite()) || (values[3] as f32) == 0.0 {
            return Err(RenderError(
                "camera projection cannot be represented by the renderer".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn basis(&self) -> Result<(V, V, V, f64), RenderError> {
        let delta = sub(self.target_m, self.eye_m);
        let distance = dot(delta, delta).sqrt();
        let forward = unit(delta)?;
        let side = cross(forward, unit(self.up)?);
        if dot(side, side) <= 1e-24 {
            return Err(RenderError(
                "camera up must not be parallel to view direction".into(),
            ));
        }
        let right = unit(side)?;
        let up = cross(right, forward);
        Ok((right, up, forward, distance))
    }
    /// Projects a world point to top-left-origin pixels and graphics depth `[0,1]`.
    /// Points outside the frustum are still returned; the caller can inspect depth.
    pub fn project(&self, point: V, width: u32, height: u32) -> Result<V, RenderError> {
        self.validate(f64::from(width) / f64::from(height))?;
        let (r, u, f, _) = self.basis()?;
        let d = sub(point, self.eye_m);
        let z = dot(d, f);
        let fy = 1.0 / (self.vertical_fov_deg.to_radians() / 2.0).tan();
        let x = dot(d, r) * fy / (f64::from(width) / f64::from(height)) / z;
        let y = dot(d, u) * fy / z;
        let depth = self.far_m / (self.far_m - self.near_m) * (1.0 - self.near_m / z);
        let result = [
            (x + 1.0) * f64::from(width) / 2.0,
            (1.0 - y) * f64::from(height) / 2.0,
            depth,
        ];
        if !result.into_iter().all(f64::is_finite) {
            return Err(RenderError("point has no finite projection".into()));
        }
        Ok(result)
    }
    /// Explicitly frames initial bounds expanded by the display particle radius.
    pub fn fit(
        &mut self,
        particles: &[Particle],
        radius: f64,
        aspect: f64,
    ) -> Result<(), RenderError> {
        let (lo, hi) = bounds(particles, radius)?;
        self.fit_bounds(lo, hi, aspect)
    }

    /// Fits explicit finite bounds, retaining the camera's selected vertical field of view.
    ///
    /// # Errors
    /// Rejects invalid bounds or a camera that cannot represent the resulting projection.
    pub fn fit_bounds(&mut self, lo: V, hi: V, aspect: f64) -> Result<(), RenderError> {
        if !lo.into_iter().chain(hi).all(f64::is_finite) || (0..3).any(|i| hi[i] <= lo[i]) {
            return Err(RenderError(
                "camera bounds must be finite and increasing".into(),
            ));
        }
        let target = mul(add(lo, hi), 0.5);
        let diagonal = sub(hi, lo);
        let r = dot(diagonal, diagonal).sqrt() / 2.0;
        let vertical = self.vertical_fov_deg.to_radians() / 2.0;
        let horizontal = (vertical.tan() * aspect).atan();
        let distance = 1.1 * r / vertical.min(horizontal).sin();
        let next = Self {
            target_m: target,
            eye_m: add(target, mul(unit([1.0, 0.7, 1.0])?, distance)),
            up: [0.0, 1.0, 0.0],
            near_m: (r / 1000.0).max(1e-6),
            far_m: distance + 10.0 * r,
            ..self.clone()
        };
        next.validate(aspect)?;
        *self = next;
        Ok(())
    }
    /// Fits the finite declared domain, otherwise the supplied accepted particles.
    /// Empty unbounded scenes retain the built-in fixed fallback.
    ///
    /// # Errors
    /// Rejects an invalid particle radius, aspect or unrepresentable camera.
    pub fn fit_scene(
        &mut self,
        particles: &[Particle],
        planes: &[crate::boundary::Plane],
        radius: f64,
        aspect: f64,
    ) -> Result<(), RenderError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(RenderError(
                "camera radius must be positive and finite".into(),
            ));
        }
        if let Some((mut lo, mut hi)) = super::domain_bounds(planes) {
            for axis in 0..3 {
                lo[axis] -= radius;
                hi[axis] += radius;
            }
            if self.fit_bounds(lo, hi, aspect).is_ok() {
                return Ok(());
            }
        }
        if particles.is_empty() {
            let fallback = Self::default();
            fallback.validate(aspect)?;
            *self = fallback;
            Ok(())
        } else {
            self.fit(particles, radius, aspect)
        }
    }
    /// Orbits around the target. Pitch is clamped away from the up-axis poles.
    pub fn orbit(&mut self, yaw: f64, pitch: f64) -> Result<(), RenderError> {
        let d = sub(self.eye_m, self.target_m);
        let radius = dot(d, d).sqrt();
        let azimuth = d[0].atan2(d[2]) + yaw;
        let elevation = ((d[1] / radius).asin() + pitch).clamp(-1.55, 1.55);
        let next = add(
            self.target_m,
            [
                radius * elevation.cos() * azimuth.sin(),
                radius * elevation.sin(),
                radius * elevation.cos() * azimuth.cos(),
            ],
        );
        let mut camera = self.clone();
        camera.eye_m = next;
        camera.up = [0.0, 1.0, 0.0];
        camera.validate(1.0)?;
        *self = camera;
        Ok(())
    }
    /// Moves eye and target together in the camera plane, in fractions of distance.
    pub fn pan(&mut self, x: f64, y: f64) -> Result<(), RenderError> {
        let (r, u, _, d) = self.basis()?;
        let shift = mul(add(mul(r, x), mul(u, y)), d);
        let mut camera = self.clone();
        camera.eye_m = add(self.eye_m, shift);
        camera.target_m = add(self.target_m, shift);
        camera.validate(1.0)?;
        *self = camera;
        Ok(())
    }
    /// Scales the eye-target distance without changing scene coordinates.
    pub fn zoom(&mut self, factor: f64) -> Result<(), RenderError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(RenderError("invalid zoom factor".into()));
        }
        let mut next = self.clone();
        next.eye_m = add(self.target_m, mul(sub(self.eye_m, self.target_m), factor));
        next.validate(1.0)?;
        *self = next;
        Ok(())
    }
}

pub(crate) fn bounds(particles: &[Particle], radius: f64) -> Result<(V, V), RenderError> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(RenderError(
            "particle radius must be finite and positive".into(),
        ));
    }
    if particles.is_empty() {
        return Ok(([-0.5; 3], [0.5; 3]));
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in particles {
        let v = position(p);
        for i in 0..3 {
            lo[i] = lo[i].min(v[i] - radius);
            hi[i] = hi[i].max(v[i] + radius);
        }
    }
    Ok((lo, hi))
}
