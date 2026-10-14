//! # Simulation diagnostics
//!
//! Checks that report useful numerical information without changing the
//! simulation inputs.

use crate::{NumericalError, math::Vec3, validation};

/// Relative density error over every particle in one accepted snapshot.
/// Both metrics are unavailable (`None`) precisely when `sample_count` is zero.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct DensityAudit {
    /// Number of fluid particles evaluated, including isolated particles.
    pub sample_count: usize,
    /// Arithmetic mean of absolute relative errors; dimensionless.
    pub mean_error: Option<f32>,
    /// Largest absolute relative error; dimensionless.
    pub maximum_error: Option<f32>,
}

/// Audits accepted positions with a fresh strict-support brute-force search.
///
/// Self contributes once, and all particles and sums follow stable identity
/// order. All arithmetic is binary32. Finite error, however large, is diagnostic
/// only. A non-finite distance, density or aggregate returns an error without
/// changing the snapshot. Call this outside the solver timing boundary.
pub fn audit_density(
    snapshot: &crate::simulation::Snapshot,
    support_radius: f32,
    normalised_rest_density: f32,
) -> Result<DensityAudit, NumericalError> {
    validation::positive_scalar("audit rest density", normalised_rest_density)?;
    let positions: Vec<_> = snapshot.particles().iter().map(|p| p.position_m).collect();
    let fresh = crate::neighbours::brute_force(&positions, support_radius)?;
    let count = positions.len();
    if count == 0 {
        return Ok(DensityAudit {
            sample_count: 0,
            mean_error: None,
            maximum_error: None,
        });
    }
    let mut sum = 0.0_f32;
    let mut maximum = 0.0_f32;
    for particle in 0..count {
        let density =
            crate::fluid::normalised_density(particle, &positions, &fresh, support_radius)?;
        let error = crate::fluid::density_constraint(density, normalised_rest_density)?.abs();
        sum = validation::finite_result("ordered density-error sum", sum + error)?;
        maximum = maximum.max(error);
    }
    let mean = validation::finite_result("mean density error", sum / count as f32)?;
    Ok(DensityAudit {
        sample_count: count,
        mean_error: Some(mean),
        maximum_error: Some(maximum),
    })
}

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
/// ### Errors
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
