//! Infinite physical planes are outlined only where they intersect a frozen
//! display box. The box adds no collision constraints or physical wall extent.

use super::{
    RenderError,
    camera::{V, add, bounds, cross, dot, mul, sub, unit},
};
use crate::{boundary::Plane, scene::Particle};

/// Line segment endpoints in world metres, ordered by source plane and angle.
#[derive(Clone, Debug)]
pub struct BoundaryLines {
    /// Frozen display-box lower corner in metres.
    pub lower_m: V,
    /// Frozen display-box upper corner in metres.
    pub upper_m: V,
    /// Consecutive pairs are independent line segments.
    pub vertices_m: Vec<V>,
}
impl BoundaryLines {
    /// Clips each plane to the initial bounds, using the contract's empty fallback.
    pub fn new(
        particles: &[Particle],
        planes: &[Plane],
        spacing: f64,
    ) -> Result<Self, RenderError> {
        let (mut lo, mut hi) = bounds(particles, spacing / 2.0)?;
        // The display box starts at unexpanded particle bounds; radius expansion
        // belongs to camera fitting, not the clipped-plane rule.
        if !particles.is_empty() {
            for i in 0..3 {
                lo[i] += spacing / 2.0;
                hi[i] -= spacing / 2.0;
            }
        }
        let side = (0..3).map(|i| hi[i] - lo[i]).fold(0.0, f64::max);
        let expand = (2.0 * spacing).max(side / 2.0);
        for i in 0..3 {
            lo[i] -= expand;
            hi[i] += expand;
        }
        let corners: Vec<V> = (0..8)
            .map(|mask| std::array::from_fn(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] }))
            .collect();
        let mut vertices = Vec::new();
        for plane in planes {
            let n = plane.normal();
            let n = [n.x.into(), n.y.into(), n.z.into()];
            let offset = f64::from(plane.offset());
            let mut points: Vec<V> = Vec::new();
            let tolerance = (side + 2.0 * expand) * 1e-10;
            for a in 0..8 {
                for axis in 0..3 {
                    let b = a ^ (1 << axis);
                    if b < a {
                        continue;
                    }
                    let va = corners[a];
                    let vb = corners[b];
                    let da = dot(n, va) - offset;
                    let db = dot(n, vb) - offset;
                    let mut insert = |p: V| {
                        if !points.iter().any(|q| {
                            let d = sub(*q, p);
                            dot(d, d) <= tolerance * tolerance
                        }) {
                            points.push(p);
                        }
                    };
                    if da.abs() <= tolerance {
                        insert(va);
                    }
                    if db.abs() <= tolerance {
                        insert(vb);
                    }
                    if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                        insert(add(va, mul(sub(vb, va), da / (da - db))));
                    }
                }
            }
            if points.len() < 3 {
                continue;
            }
            let centre = mul(
                points.iter().copied().fold([0.0; 3], add),
                1.0 / points.len() as f64,
            );
            let reference = if n[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let u = unit(cross(n, reference))?;
            let v = cross(n, u);
            points.sort_by(|a, b| {
                let a = sub(*a, centre);
                let b = sub(*b, centre);
                dot(a, v)
                    .atan2(dot(a, u))
                    .total_cmp(&dot(b, v).atan2(dot(b, u)))
            });
            for i in 0..points.len() {
                vertices.extend([points[i], points[(i + 1) % points.len()]]);
            }
        }
        Ok(Self {
            lower_m: lo,
            upper_m: hi,
            vertices_m: vertices,
        })
    }
}
