//! Particle representations share accepted centres, colours and world-space
//! lighting. Selection changes graphics geometry only, never simulation state.

use super::RenderError;

/// Opaque particle geometry, independent of numerical mode and scene inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleRenderer {
    /// Established indexed sphere with 16 longitudes and 12 latitude bands.
    #[default]
    Indexed,
    /// An eight-longitude, six-band sphere for comparison at small projected sizes.
    Low,
    /// A perspective-correct analytic sphere bounded by a two-triangle rectangle.
    Impostor,
    /// Frame-wide mesh resolution chosen conservatively from the largest particle.
    Adaptive,
}
impl std::str::FromStr for ParticleRenderer {
    type Err = RenderError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "indexed" => Ok(Self::Indexed),
            "low" => Ok(Self::Low),
            "impostor" => Ok(Self::Impostor),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(RenderError(
                "particle renderer must be indexed, low, impostor or adaptive".into(),
            )),
        }
    }
}

/// Largest projected half-extent controls one mesh for the entire frame. This
/// avoids regrouping particles or changing their identity/draw order.
pub(super) fn adaptive_level(radius_px: f64) -> usize {
    if radius_px <= 3.0 {
        2
    } else if radius_px <= 6.0 {
        1
    } else {
        0
    }
}

/// Half the exact perspective tangent interval, in pixels. The caller uses the
/// indexed fallback when a sphere intersects the near plane or eye plane.
pub(super) fn projected_radius(centre: [f32; 3], radius: f32, fy: f64, height: u32) -> f64 {
    let [x, y, z] = centre.map(f64::from);
    let r = f64::from(radius);
    let denominator = z * z - r * r;
    if denominator <= 0.0 {
        return f64::INFINITY;
    }
    f64::from(height) * 0.5 * fy * r * ((x * x).max(y * y) + z * z - r * r).max(0.0).sqrt()
        / denominator
}

pub(super) fn sphere_mesh(longitudes: u32, latitudes: u32) -> (Vec<f32>, Vec<u16>) {
    let point = |lat: u32, lon: u32| {
        let t = std::f32::consts::PI * lat as f32 / latitudes as f32;
        let p = std::f32::consts::TAU * lon as f32 / longitudes as f32;
        [t.sin() * p.cos(), t.cos(), t.sin() * p.sin()]
    };
    // Do not weld the seam or poles: their almost equal floats belong to the
    // existing mesh's rasterisation and remain distinct in the control.
    let mut vertices = Vec::new();
    for lat in 0..=latitudes {
        for lon in 0..=longitudes {
            vertices.extend(point(lat, lon));
        }
    }
    let mut indices = Vec::new();
    for lat in 0..latitudes {
        for lon in 0..longitudes {
            for (a, b) in [
                (lat, lon),
                (lat + 1, lon),
                (lat + 1, lon + 1),
                (lat, lon),
                (lat + 1, lon + 1),
                (lat, lon + 1),
            ] {
                indices.push((a * (longitudes + 1) + b) as u16);
            }
        }
    }
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    #[test]
    fn lod_keeps_large_particles_and_promotes_at_each_threshold() {
        assert_eq!(super::adaptive_level(2.0), 2);
        assert_eq!(super::adaptive_level(3.0), 2);
        assert_eq!(super::adaptive_level(3.001), 1);
        assert_eq!(super::adaptive_level(6.0), 1);
        assert_eq!(super::adaptive_level(6.001), 0);
        assert_eq!(super::adaptive_level(f64::INFINITY), 0);
    }
    #[test]
    fn perspective_radius_matches_axis_tangency_and_handles_eye_intersection() {
        let value = super::projected_radius([0.0, 0.0, 5.0], 1.0, 1.0, 200);
        assert!((value - 100.0 / 24.0_f64.sqrt()).abs() < 1e-12);
        assert!(super::projected_radius([0.0, 0.0, 0.5], 1.0, 1.0, 200).is_infinite());
    }
}
