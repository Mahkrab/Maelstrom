//! # Smoothed Particle Hydrodynamics kernels
//!
//! Three-dimensional Poly6 and Spiky functions used by the PBF density and
//! position-correction calculations.

use std::f32::consts::PI;

use crate::{NumericalError, math::Vec3, validation};

/// ### Calculates a Poly6 density weight
///
/// The kernel has strict support:
///
/// - `distance < support_radius` returns the Poly6 weight;
/// - `distance >= support_radius` returns exactly `0.0`.
///
/// ### Errors
///
/// Returns [`NumericalError`] when `distance` is negative or non-finite, when
/// `support_radius`  is not positive and finite, or when an intermediate value
/// cannot be represented by `f32`.
pub fn poly6(distance: f32, support_radius: f32) -> Result<f32, NumericalError> {
    validation::non_negative_scalar("kernel distance", distance)?;
    validation::positive_scalar("support radius", support_radius)?;

    if distance >= support_radius {
        return Ok(0.0);
    }
    let support_squared =
        validation::finite_result("squared support radius", support_radius * support_radius)?;
    let distance_squared =
        validation::finite_result("squared kernel distance", distance * distance)?;
    let support_term =
        validation::finite_result("Poly6 support term", support_squared - distance_squared)?;
    let support_ninth =
        validation::finite_result("support radius to the ninth power", support_radius.powi(9))?;
    let denominator = validation::finite_result("Poly6 denominator", 64.0 * PI * support_ninth)?;
    let coefficient = validation::finite_result("Poly6 coefficient", 315.0 / denominator)?;
    let support_term_cubed =
        validation::finite_result("cubed Poly6 support term", support_term.powi(3))?;

    validation::finite_result("Poly6 value", coefficient * support_term_cubed)
}

/// ### Calculates a Spiky correction direction
///
/// A zero displacement has no unique direction, so it returns [`Vec3::ZERO`].
/// Displacements at or beyond `support_radius` also return zero.
///
/// ### Errors
///
/// Returns [`NumericalError`] when an input is non-finite, when
/// `support_radius` is not positive, or when an intermediate value cannot be
/// represented by `f32`.
pub fn spiky_direction(displacement: Vec3, support_radius: f32) -> Result<Vec3, NumericalError> {
    validation::finite_vector("kernel displacement", displacement)?;
    validation::positive_scalar("support radius", support_radius)?;

    let distance = displacement.checked_length("kernel displacement")?;
    if distance == 0.0 || distance >= support_radius {
        return Ok(Vec3::ZERO);
    }

    let support_sixth =
        validation::finite_result("support radius to the sixth power", support_radius.powi(6))?;
    let denominator = validation::finite_result("Spiky denominator", PI * support_sixth)?;
    let coefficient = validation::finite_result("Spiky coefficient", -45.0 / denominator)?;
    let remaining_support =
        validation::finite_result("Spiky support difference", support_radius - distance)?;
    let scale = validation::finite_result(
        "Spiky scale",
        coefficient * (remaining_support * remaining_support) / distance,
    )?;
    validation::finite_vector_result("Spiky direction", displacement * scale)
}
