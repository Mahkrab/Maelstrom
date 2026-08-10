//! # Particle motion
//!
//! Stateless functions for accelerating velocity, predicting position and
//! rebuilding velocity from corrected motion.

use crate::{NumericalError, math::Vec3, validation};

/// ### Applies external acceleration to a velocity
///
/// This returns `accepted_velocity + external_acceleration * timestep`; it does
/// not modify the supplied velocity.
///
/// # Errors
///
/// Returns [`NumericalError`] for a non-positive timestep, non-finite input or
/// non-finite result.
pub fn post_acceleration_velocity(
    accepted_velocity: Vec3,
    external_acceleration: Vec3,
    timestep: f32,
) -> Result<Vec3, NumericalError> {
    validation::finite_vector("accepted velocity", accepted_velocity)?;
    validation::finite_vector("external acceleration", external_acceleration)?;
    validation::positive_scalar("timestep", timestep)?;

    let velocity_delta = validation::finite_vector_result(
        "acceleration velocity delta",
        external_acceleration * timestep,
    )?;
    validation::finite_vector_result(
        "post-acceleration velocity",
        accepted_velocity + velocity_delta,
    )
}

/// ### Predicts a position from the accelerated velocity
///
/// # Errors
///
/// Returns [`NumericalError`] for a non-positive timestep, non-finite input or
/// non-finite result.
pub fn predict_position(
    accepted_position: Vec3,
    post_acceleration_velocity: Vec3,
    timestep: f32,
) -> Result<Vec3, NumericalError> {
    validation::finite_vector("accepted position", accepted_position)?;
    validation::finite_vector("post-acceleration velocity", post_acceleration_velocity)?;
    validation::positive_scalar("timestep", timestep)?;

    let displacement = validation::finite_vector_result(
        "predicted displacement",
        post_acceleration_velocity * timestep,
    )?;
    validation::finite_vector_result("predicted position", accepted_position + displacement)
}

/// ### Rebuilds velocity from the final displacement
///
/// The velocity includes every correction already present in `final_position`.
///
/// # Errors
///
/// Returns [`NumericalError`] for a non-positive timestep, non-finite input or
/// non-finite result.
pub fn reconstruct_velocity(
    accepted_position: Vec3,
    final_position: Vec3,
    timestep: f32,
) -> Result<Vec3, NumericalError> {
    validation::finite_vector("accepted position", accepted_position)?;
    validation::finite_vector("final position", final_position)?;
    validation::positive_scalar("timestep", timestep)?;

    let displacement = validation::finite_vector_result(
        "reconstructed displacement",
        final_position - accepted_position,
    )?;
    let reciprocal_timestep = validation::finite_result("reciprocal timestep", 1.0 / timestep)?;
    validation::finite_vector_result("reconstructed velocity", displacement * reciprocal_timestep)
}
