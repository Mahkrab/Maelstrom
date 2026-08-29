//! TODO Add top level comments here. Pretty please. 


use crate::{boundary::Plane, math::Vec3};

use super::{MAX_PLANES, PLANE_UNIT_LENGTH_SQUARED_TOLERANCE, Particle, PlaneDocument, SceneError};

pub(super) fn resolve_planes(documents: &[PlaneDocument]) -> Result<Vec<Plane>, SceneError> {
    if documents.len() > MAX_PLANES {
        return Err(SceneError::TooManyPlanes {
            maximum: MAX_PLANES,
            actual: documents.len(),
        });
    }
    documents
        .iter()
        .map(|plane| {
            Plane::new(
                Vec3::new(plane.normal[0], plane.normal[1], plane.normal[2]),
                plane.offset_m,
                PLANE_UNIT_LENGTH_SQUARED_TOLERANCE,
            )
            .map_err(SceneError::Numerical)
        })
        .collect()
}

pub(super) fn validate_plane_compatibility(planes: &[Plane]) -> Result<(), SceneError> {
    for first in 0..planes.len() {
        for second in first + 1..planes.len() {
            let normal_sum = planes[first].normal() + planes[second].normal();
            let antiparallel = normal_sum == Vec3::ZERO;
            if antiparallel && planes[first].offset() > -planes[second].offset() {
                return Err(SceneError::ContradictoryPlanes { first, second });
            }
        }
    }
    if feasible_plane_point(planes) {
        Ok(())
    } else {
        Err(SceneError::NoFeasiblePlaneIntersection)
    }
}

pub(super) fn validate_bounds(particles: &[Particle], planes: &[Plane]) -> Result<(), SceneError> {
    for particle in particles {
        for (index, plane) in planes.iter().enumerate() {
            if !plane.contains(particle.position_m)? {
                return Err(SceneError::ParticleOutsidePlane {
                    particle: particle.identity,
                    plane: index,
                });
            }
        }
    }
    Ok(())
}

fn feasible_plane_point(planes: &[Plane]) -> bool {
    if point_satisfies_planes([0.0, 0.0, 0.0], planes) {
        return true;
    }
    for first in 0..planes.len() {
        let (normal, offset) = plane_f64(planes[first]);
        let normal_squared = dot64(normal, normal);
        let candidate = scale64(normal, offset / normal_squared);
        if point_satisfies_planes(candidate, planes) {
            return true;
        }
        for second in first + 1..planes.len() {
            let (other_normal, other_offset) = plane_f64(planes[second]);
            if let Some(candidate) =
                two_plane_minimum_norm(normal, offset, other_normal, other_offset)
                && point_satisfies_planes(candidate, planes)
            {
                return true;
            }
            for third in second + 1..planes.len() {
                let (third_normal, third_offset) = plane_f64(planes[third]);
                if let Some(candidate) = three_plane_intersection(
                    normal,
                    offset,
                    other_normal,
                    other_offset,
                    third_normal,
                    third_offset,
                ) && point_satisfies_planes(candidate, planes)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn two_plane_minimum_norm(
    first: [f64; 3],
    first_offset: f64,
    second: [f64; 3],
    second_offset: f64,
) -> Option<[f64; 3]> {
    let aa = dot64(first, first);
    let ab = dot64(first, second);
    let bb = dot64(second, second);
    let determinant = aa * bb - ab * ab;
    if determinant.abs() <= 1.0e-12 {
        return None;
    }
    let first_scale = (first_offset * bb - second_offset * ab) / determinant;
    let second_scale = (second_offset * aa - first_offset * ab) / determinant;
    Some(add64(
        scale64(first, first_scale),
        scale64(second, second_scale),
    ))
}

fn three_plane_intersection(
    first: [f64; 3],
    first_offset: f64,
    second: [f64; 3],
    second_offset: f64,
    third: [f64; 3],
    third_offset: f64,
) -> Option<[f64; 3]> {
    let second_cross_third = cross64(second, third);
    let determinant = dot64(first, second_cross_third);
    if determinant.abs() <= 1.0e-12 {
        return None;
    }
    let numerator = add64(
        add64(
            scale64(second_cross_third, first_offset),
            scale64(cross64(third, first), second_offset),
        ),
        scale64(cross64(first, second), third_offset),
    );
    Some(scale64(numerator, 1.0 / determinant))
}

fn point_satisfies_planes(point: [f64; 3], planes: &[Plane]) -> bool {
    planes.iter().all(|&plane| {
        let (normal, offset) = plane_f64(plane);
        let value = dot64(normal, point);
        let scale = value.abs().max(offset.abs()).max(f64::MIN_POSITIVE);
        let tolerance = 64.0 * f64::EPSILON * scale;
        value + tolerance >= offset
    })
}

fn plane_f64(plane: Plane) -> ([f64; 3], f64) {
    let normal = plane.normal();
    (
        [
            f64::from(normal.x),
            f64::from(normal.y),
            f64::from(normal.z),
        ],
        f64::from(plane.offset()),
    )
}

fn dot64(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross64(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale64(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn add64(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}
