//! Box authoring expands to the same inward half-spaces as explicit planes.
//! Bounds constrain particle centres, preserving the solver's boundary model.

use super::{ContainerBox, MAX_PLANES, PlaneDocument, SceneDocument, SceneError};
use crate::config::ConfigError;

pub(super) fn expanded_planes(document: &SceneDocument) -> Result<Vec<PlaneDocument>, SceneError> {
    let count = document
        .planes
        .len()
        .checked_add(if document.container.is_some() { 6 } else { 0 })
        .ok_or(SceneError::TooManyPlanes {
            maximum: MAX_PLANES,
            actual: usize::MAX,
        })?;
    if count > MAX_PLANES {
        return Err(SceneError::TooManyPlanes {
            maximum: MAX_PLANES,
            actual: count,
        });
    }
    let mut planes = Vec::with_capacity(count);
    if let Some(container) = &document.container {
        let (min, max) = bounds(container)?;
        for axis in 0..3 {
            let mut normal = [0.0; 3];
            normal[axis] = 1.0;
            planes.push(PlaneDocument {
                normal,
                offset_m: min[axis],
            });
            normal[axis] = -1.0;
            planes.push(PlaneDocument {
                normal,
                offset_m: -max[axis],
            });
        }
    }
    planes.extend_from_slice(&document.planes);
    Ok(planes)
}

fn bounds(container: &ContainerBox) -> Result<([f32; 3], [f32; 3]), SceneError> {
    let (min, max) = match (
        container.min_m,
        container.max_m,
        container.origin_m,
        container.dimensions_m,
    ) {
        (Some(min), Some(max), None, None) => (min, max),
        (None, None, Some(origin), Some(dimensions)) => {
            if dimensions.iter().any(|x| !x.is_finite() || *x <= 0.0) {
                return Err(invalid(
                    "dimensions_m must contain three positive finite extents",
                ));
            }
            (
                origin,
                std::array::from_fn(|axis| origin[axis] + dimensions[axis]),
            )
        }
        _ => {
            return Err(invalid(
                "select min_m and max_m, or origin_m and dimensions_m, without mixing forms",
            ));
        }
    };
    for axis in 0..3 {
        if !min[axis].is_finite() || !max[axis].is_finite() || min[axis] >= max[axis] {
            return Err(invalid(
                "bounds must be finite and strictly increasing on every axis after binary32 calculation",
            ));
        }
    }
    Ok((min, max))
}

fn invalid(reason: &str) -> SceneError {
    ConfigError::InvalidField {
        field: "container",
        reason: reason.into(),
    }
    .into()
}
