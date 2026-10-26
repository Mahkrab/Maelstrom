//! Velocity effects read one complete reconstructed snapshot and produce another.
//! Their output remains tentative until the timestep's normal acceptance checks.

pub(crate) mod vorticity;
pub use vorticity::{VorticityParameters, vorticity_confinement};

use crate::{
    NumericalError, execution::Execution, fluid::InteractionSets, kernel, math::Vec3, validation,
};

/// Applies equal-mass XSPH with dimensionless Poly6/rest-density weights.
/// Rows are fixed for the timestep; current distances decide strict support.
/// The coefficient is applied once per call, without an additional timestep factor.
///
/// # Errors
/// Rejects mismatched inputs, invalid scalars, non-finite state and intermediate overflow.
pub fn xsph_viscosity(
    positions: &[Vec3],
    velocities: &[Vec3],
    interactions: &InteractionSets,
    radius_m: f32,
    rest_density_per_m3: f32,
    coefficient: f32,
) -> Result<Vec<Vec3>, NumericalError> {
    xsph_with_execution(
        positions,
        velocities,
        interactions,
        radius_m,
        rest_density_per_m3,
        coefficient,
        &Execution::default(),
    )
}

pub(crate) fn xsph_with_execution(
    positions: &[Vec3],
    velocities: &[Vec3],
    interactions: &InteractionSets,
    radius_m: f32,
    rest_density_per_m3: f32,
    coefficient: f32,
    execution: &Execution,
) -> Result<Vec<Vec3>, NumericalError> {
    validation::positive_scalar("XSPH support radius", radius_m)?;
    validation::positive_scalar("XSPH normalised rest density", rest_density_per_m3)?;
    if !coefficient.is_finite() || !(0.0..=1.0).contains(&coefficient) {
        return Err(NumericalError::InvalidSmoothingCoefficient { value: coefficient });
    }
    if positions.len() != velocities.len() {
        return Err(NumericalError::LengthMismatch {
            quantity: "XSPH velocities",
            expected: positions.len(),
            actual: velocities.len(),
        });
    }
    if interactions.particle_count() != positions.len() {
        return Err(NumericalError::InteractionSetCount {
            expected: positions.len(),
            actual: interactions.particle_count(),
        });
    }
    for (&position, &velocity) in positions.iter().zip(velocities) {
        validation::finite_vector("XSPH position", position)?;
        validation::finite_vector("XSPH velocity", velocity)?;
    }
    if coefficient == 0.0 {
        return Ok(velocities.to_vec());
    }
    execution.map(positions.len(), |i| {
        let row = interactions
            .for_particle(i)
            .ok_or(NumericalError::ParticleIndexOutOfBounds {
                particle: i,
                particle_count: interactions.particle_count(),
            })?;
        let mut sum = Vec3::ZERO;
        for &j in row {
            let distance = validation::finite_vector_result(
                "particle displacement",
                positions[i] - positions[j],
            )?
            .checked_length("particle displacement")?;
            if distance >= radius_m {
                continue;
            }
            let weight = validation::finite_result(
                "XSPH kernel weight",
                kernel::poly6(distance, radius_m)? / rest_density_per_m3,
            )?;
            let difference = validation::finite_vector_result(
                "XSPH velocity difference",
                velocities[j] - velocities[i],
            )?;
            let contribution =
                validation::finite_vector_result("XSPH pair contribution", difference * weight)?;
            sum = validation::finite_vector_result("XSPH ordered sum", sum + contribution)?;
        }
        let delta =
            validation::finite_vector_result("XSPH velocity correction", sum * coefficient)?;
        validation::finite_vector_result("XSPH smoothed velocity", velocities[i] + delta)
    })
}
