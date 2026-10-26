//! A bounded plane intersection supplies a stable view of the complete domain.
//! Unbounded or numerically unresolved geometry falls back to particle bounds.

use super::camera::{V, add, cross, dot, mul};
use crate::boundary::Plane;

/// Returns finite domain bounds when the plane intersection has no recession ray.
///
/// This is a presentation calculation in f64; it does not change physical planes
/// or the solver's binary32 projection. Empty/unbounded inputs return `None`.
#[must_use]
pub fn domain_bounds(planes: &[Plane]) -> Option<(V, V)> {
    if planes.len() < 4 {
        return None;
    }
    let normals: Vec<V> = planes
        .iter()
        .map(|p| {
            let n = p.normal();
            [f64::from(n.x), f64::from(n.y), f64::from(n.z)]
        })
        .collect();
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let ray = |direction: V| {
        let norm = dot(direction, direction).sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return false;
        }
        let direction = mul(direction, 1.0 / norm);
        [1.0, -1.0].into_iter().any(|sign| {
            normals
                .iter()
                .all(|&n| sign * dot(n, direction) >= -64.0 * f64::EPSILON)
        })
    };
    // A non-zero recession cone has a pairwise boundary ray, or lineality.
    // Axis/normal crosses also cover parallel-plane slabs and lower-rank cases.
    if axes.into_iter().any(ray) {
        return None;
    }
    for (i, &normal) in normals.iter().enumerate() {
        if axes.into_iter().any(|axis| ray(cross(normal, axis))) {
            return None;
        }
        for &other in &normals[i + 1..] {
            if ray(cross(normal, other)) {
                return None;
            }
        }
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for i in 0..planes.len() {
        for j in i + 1..planes.len() {
            for k in j + 1..planes.len() {
                let determinant = dot(normals[i], cross(normals[j], normals[k]));
                if determinant == 0.0 {
                    continue;
                }
                let point = mul(
                    add(
                        add(
                            mul(cross(normals[j], normals[k]), f64::from(planes[i].offset())),
                            mul(cross(normals[k], normals[i]), f64::from(planes[j].offset())),
                        ),
                        mul(cross(normals[i], normals[j]), f64::from(planes[k].offset())),
                    ),
                    1.0 / determinant,
                );
                if !point.into_iter().all(f64::is_finite) {
                    continue;
                }
                if planes.iter().zip(&normals).all(|(plane, &normal)| {
                    let offset = f64::from(plane.offset());
                    let scale = point
                        .iter()
                        .map(|v| v.abs())
                        .sum::<f64>()
                        .max(offset.abs())
                        .max(1.0);
                    dot(normal, point) >= offset - 128.0 * f64::EPSILON * scale
                }) {
                    for axis in 0..3 {
                        lo[axis] = lo[axis].min(point[axis]);
                        hi[axis] = hi[axis].max(point[axis]);
                    }
                }
            }
        }
    }
    lo.into_iter()
        .chain(hi)
        .all(f64::is_finite)
        .then_some((lo, hi))
}
