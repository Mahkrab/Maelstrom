//! # Simulation diagnostics
//!
//! Checks that report useful numerical information without changing the
//! simulation inputs.

use crate::{NumericalError, math::Vec3, validation};

/// ### Courant-Friedrichs-Lewy (CFL) timestep limit
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CflLimit {
    /// All particles are stationary, so there is no finite limit to report.
    Unbounded,
    /// Largest timestep permitted by the velocity based CFL calculation
    Finite(f32),
}

/// ### Result of a velocity-based CFL check
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct CflDiagnostic {
    /// Largest speed found in the supplied velocity slice.
    pub maximum_speed: f32,
    /// Calculated timestep limit, or [`CflLimit::Unbounded`] at zero speed.
    pub limit: CflLimit,
    /// Whether `timestep` is greater than a finite limit.
    pub warning: bool,
}

/// ### Checks a timestep against particle velocities
///
/// Velocities are read in slice order *like pie*. Equality with a finite limit passes. If
/// every speed is zero, the function returns [`CflLimit::Unbounded`] without
/// dividing by zero.
///
/// The result is diagnostic only: this function never changes `timestep`.
///
/// # Errors
///
/// Returns [`NumericalError`] when a scalar parameter is not positive and
/// finite, a velocity is non-finite, or the speed or limit calculation
/// overflows `f32`.
pub fn cfl_diagnostic(
    post_acceleration_velocities: &[Vec3],
    timestep: f32,
    particle_diameter: f32,
    cfl_factor: f32,
) -> Result<CflDiagnostic, NumericalError> {
    validation::positive_scalar("timestep", timestep)?;
    validation::positive_scalar("particle diameter", particle_diameter)?;
    validation::positive_scalar("CFL factor", cfl_factor)?;

    let mut maximum_speed = 0.0_f32;
    for &velocity in post_acceleration_velocities {
        validation::finite_vector("post-acceleration velocity", velocity)?;
        let speed = velocity.checked_length("post-acceleration velocity")?;
        if speed > maximum_speed {
            maximum_speed = speed;
        }
    }

    if maximum_speed == 0.0 {
        return Ok(CflDiagnostic {
            maximum_speed,
            limit: CflLimit::Unbounded,
            warning: false,
        });
    }

    let displacement_ratio = validation::finite_result(
        "CFL diameter-to-speed ratio",
        particle_diameter / maximum_speed,
    )?;
    let limit = validation::finite_result("finite CFL limit", cfl_factor * displacement_ratio)?;

    Ok(CflDiagnostic {
        maximum_speed,
        limit: CflLimit::Finite(limit),
        warning: timestep > limit,
    })
}
