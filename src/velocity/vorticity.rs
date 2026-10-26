//! Curl and its magnitude gradient use separate complete snapshots. Confinement
//! restores rotational motion before any subsequent velocity smoothing.

use crate::{
    NumericalError, execution::Execution, fluid::InteractionSets, kernel, math::Vec3, validation,
};

/// Rest-volume confinement controls in metre/second units.
#[derive(Clone, Copy, Debug)]
pub struct VorticityParameters {
    /// Strict kernel support radius in metres.
    pub radius_m: f32,
    /// Equal-mass normalised rest density in inverse cubic metres.
    pub rest_density_per_m3: f32,
    /// Non-negative confinement strength in metres per second.
    pub strength_m_s: f32,
    /// Integration interval in seconds.
    pub timestep_s: f32,
}

/// Applies vorticity confinement, reading reconstructed velocities and final
/// projected positions through fixed identity-ordered neighbour rows.
/// Zero strength returns the validated input unchanged. No neighbour search runs.
///
/// # Errors
/// Rejects invalid controls, mismatched inputs, non-finite state and overflow.
pub fn vorticity_confinement(
    positions: &[Vec3],
    velocities: &[Vec3],
    interactions: &InteractionSets,
    parameters: VorticityParameters,
) -> Result<Vec<Vec3>, NumericalError> {
    with_execution(
        positions,
        velocities,
        interactions,
        parameters,
        &Execution::default(),
        &mut || Ok(()),
    )
}

pub(crate) fn with_execution<E: From<NumericalError>>(
    positions: &[Vec3],
    velocities: &[Vec3],
    interactions: &InteractionSets,
    p: VorticityParameters,
    execution: &Execution,
    between: &mut impl FnMut() -> Result<(), E>,
) -> Result<Vec<Vec3>, E> {
    validation::positive_scalar("vorticity support radius", p.radius_m)?;
    validation::positive_scalar("vorticity normalised rest density", p.rest_density_per_m3)?;
    validation::non_negative_scalar("vorticity strength", p.strength_m_s)?;
    validation::positive_scalar("vorticity timestep", p.timestep_s)?;
    if positions.len() != velocities.len() {
        return Err(NumericalError::LengthMismatch {
            quantity: "vorticity velocities",
            expected: positions.len(),
            actual: velocities.len(),
        }
        .into());
    }
    if interactions.particle_count() != positions.len() {
        return Err(NumericalError::InteractionSetCount {
            expected: positions.len(),
            actual: interactions.particle_count(),
        }
        .into());
    }
    for (&position, &velocity) in positions.iter().zip(velocities) {
        validation::finite_vector("vorticity position", position)?;
        validation::finite_vector("vorticity velocity", velocity)?;
    }
    if p.strength_m_s == 0.0 {
        return Ok(velocities.to_vec());
    }
    let curl: Vec<Vec3> = execution.map(positions.len(), |i| -> Result<Vec3, NumericalError> {
        let mut sum = Vec3::ZERO;
        for &j in interactions
            .for_particle(i)
            .ok_or(NumericalError::ParticleIndexOutOfBounds {
                particle: i,
                particle_count: positions.len(),
            })?
        {
            let displacement = validation::finite_vector_result(
                "particle displacement",
                positions[i] - positions[j],
            )?;
            if displacement.checked_length("particle displacement")? >= p.radius_m {
                continue;
            }
            let gradient = gradient(displacement, p)?;
            let difference = validation::finite_vector_result(
                "vorticity velocity difference",
                velocities[j] - velocities[i],
            )?;
            let contribution = validation::finite_vector_result(
                "vorticity curl contribution",
                gradient.cross(difference),
            )?;
            sum = validation::finite_vector_result("vorticity ordered curl", sum + contribution)?;
        }
        Ok(sum)
    })?;
    let magnitudes: Vec<f32> =
        execution.map(curl.len(), |i| curl[i].checked_length("vorticity curl"))?;
    between()?;
    Ok(
        execution.map(positions.len(), |i| -> Result<Vec3, NumericalError> {
            let mut eta = Vec3::ZERO;
            for &j in
                interactions
                    .for_particle(i)
                    .ok_or(NumericalError::ParticleIndexOutOfBounds {
                        particle: i,
                        particle_count: positions.len(),
                    })?
            {
                let displacement = validation::finite_vector_result(
                    "particle displacement",
                    positions[i] - positions[j],
                )?;
                if displacement.checked_length("particle displacement")? >= p.radius_m {
                    continue;
                }
                let gradient = gradient(displacement, p)?;
                let difference = validation::finite_result(
                    "vorticity magnitude difference",
                    magnitudes[j] - magnitudes[i],
                )?;
                let contribution = validation::finite_vector_result(
                    "vorticity gradient contribution",
                    gradient * difference,
                )?;
                eta = validation::finite_vector_result(
                    "vorticity magnitude gradient",
                    eta + contribution,
                )?;
            }
            let magnitude = eta.checked_length("vorticity magnitude gradient")?;
            if magnitude == 0.0 {
                return Ok(velocities[i]);
            }
            let normal = validation::finite_vector_result(
                "vorticity direction",
                Vec3::new(eta.x / magnitude, eta.y / magnitude, eta.z / magnitude),
            )?;
            let rotation =
                validation::finite_vector_result("vorticity rotation", normal.cross(curl[i]))?;
            let acceleration = validation::finite_vector_result(
                "vorticity acceleration",
                rotation * p.strength_m_s,
            )?;
            let delta = validation::finite_vector_result(
                "vorticity velocity correction",
                acceleration * p.timestep_s,
            )?;
            validation::finite_vector_result("confined velocity", velocities[i] + delta)
        })?,
    )
}

fn gradient(displacement: Vec3, p: VorticityParameters) -> Result<Vec3, NumericalError> {
    let g = kernel::spiky_direction(displacement, p.radius_m)?;
    validation::finite_vector_result(
        "vorticity kernel gradient",
        Vec3::new(
            g.x / p.rest_density_per_m3,
            g.y / p.rest_density_per_m3,
            g.z / p.rest_density_per_m3,
        ),
    )
}
