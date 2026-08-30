//! TODO: Addd top level documntation.

use crate::{config::ResolvedConfiguration, math::Vec3, validation};

use super::{
    LatticeEndpointRule, LatticeOrder, MAX_EXPLICIT_PARTICLES, MAX_LATTICE_PARTICLES,
    PARTICLE_SEPARATION_RELATIVE_TOLERANCE, Particle, ParticleSource, SceneError,
};

pub(super) fn resolve_particles(
    source: &ParticleSource,
    configuration: &ResolvedConfiguration,
    allow_coincident: bool,
) -> Result<Vec<Particle>, SceneError> {
    match source {
        ParticleSource::Explicit { items } => {
            let requested = items.len() as u64;
            if items.len() > MAX_EXPLICIT_PARTICLES {
                return Err(SceneError::ParticleCountTooLarge {
                    construction: "explicit",
                    maximum: MAX_EXPLICIT_PARTICLES,
                    requested,
                });
            }
            let mut particles = Vec::with_capacity(items.len());
            let mut previous = None;
            for item in items {
                if let Some(previous) = previous
                    && previous >= item.identity
                {
                    return Err(SceneError::ParticleIdentityOrder {
                        previous,
                        next: item.identity,
                    });
                }
                let position = array_to_vec3("initial position", item.position_m)?;
                let velocity = array_to_vec3("initial velocity", item.velocity_m_s)?;
                particles.push(Particle {
                    identity: item.identity,
                    position_m: position,
                    velocity_m_s: velocity,
                });
                previous = Some(item.identity);
            }
            validate_explicit_separation(
                &particles,
                configuration.derived.particle_diameter_m,
                allow_coincident,
            )?;
            Ok(particles)
        }
        ParticleSource::Lattice {
            first_identity,
            origin_m,
            counts,
            spacing_m,
            endpoint_rule: LatticeEndpointRule::CountedHalfOpen,
            order: LatticeOrder::XThenYThenZ,
            initial_velocity_m_s,
        } => resolve_lattice(
            *first_identity,
            *origin_m,
            *counts,
            *spacing_m,
            *initial_velocity_m_s,
            configuration,
        ),
    }
}

fn resolve_lattice(
    first_identity: u64,
    origin_m: [f32; 3],
    counts: [u32; 3],
    spacing_m: f32,
    initial_velocity_m_s: [f32; 3],
    configuration: &ResolvedConfiguration,
) -> Result<Vec<Particle>, SceneError> {
    for (axis, count) in ["x", "y", "z"].into_iter().zip(counts) {
        if count == 0 {
            return Err(SceneError::EmptyLatticeAxis { axis });
        }
    }
    validation::positive_scalar("scene lattice spacing", spacing_m)?;
    if spacing_m.to_bits() != configuration.numerical.particle_spacing_m.to_bits() {
        return Err(SceneError::ParticleSpacingMismatch {
            scene: spacing_m,
            configuration: configuration.numerical.particle_spacing_m,
        });
    }
    let requested = u64::from(counts[0])
        .checked_mul(u64::from(counts[1]))
        .and_then(|count| count.checked_mul(u64::from(counts[2])))
        .ok_or(SceneError::ParticleCountTooLarge {
            construction: "lattice",
            maximum: MAX_LATTICE_PARTICLES,
            requested: u64::MAX,
        })?;
    if requested > MAX_LATTICE_PARTICLES as u64 {
        return Err(SceneError::ParticleCountTooLarge {
            construction: "lattice",
            maximum: MAX_LATTICE_PARTICLES,
            requested,
        });
    }
    if requested > 0 {
        first_identity
            .checked_add(requested - 1)
            .ok_or(SceneError::ParticleIdentityOverflow)?;
    }
    let origin = array_to_vec3("lattice origin", origin_m)?;
    let velocity = array_to_vec3("initial velocity", initial_velocity_m_s)?;
    validate_lattice_axes(origin, counts, spacing_m)?;
    let mut particles = Vec::with_capacity(requested as usize);
    for x in 0..counts[0] {
        for y in 0..counts[1] {
            for z in 0..counts[2] {
                let offset = Vec3::new(
                    spacing_m * x as f32,
                    spacing_m * y as f32,
                    spacing_m * z as f32,
                );
                let position =
                    validation::finite_vector_result("lattice particle position", origin + offset)?;
                let identity = first_identity
                    .checked_add(particles.len() as u64)
                    .ok_or(SceneError::ParticleIdentityOverflow)?;
                particles.push(Particle {
                    identity,
                    position_m: position,
                    velocity_m_s: velocity,
                });
            }
        }
    }
    Ok(particles)
}

fn validate_explicit_separation(
    particles: &[Particle],
    diameter: f32,
    allow_coincident: bool,
) -> Result<(), SceneError> {
    for first in 0..particles.len() {
        for second in first + 1..particles.len() {
            let separation = (particles[first].position_m - particles[second].position_m)
                .checked_length("initial particle separation")?;
            let tolerance = diameter * PARTICLE_SEPARATION_RELATIVE_TOLERANCE;
            let definitely_overlapping = separation + tolerance < diameter;
            if definitely_overlapping && !(allow_coincident && separation == 0.0) {
                return Err(SceneError::OverlappingParticles {
                    first: particles[first].identity,
                    second: particles[second].identity,
                });
            }
        }
    }
    Ok(())
}

fn validate_lattice_axes(origin: Vec3, counts: [u32; 3], spacing: f32) -> Result<(), SceneError> {
    for (axis, quantity, first, count) in [
        ("x", "lattice x coordinate", origin.x, counts[0]),
        ("y", "lattice y coordinate", origin.y, counts[1]),
        ("z", "lattice z coordinate", origin.z, counts[2]),
    ] {
        let mut previous = first;
        for index in 1..count {
            let coordinate = validation::finite_result(quantity, first + spacing * index as f32)?;
            if coordinate <= previous {
                return Err(SceneError::LatticeCoordinateCollapse {
                    axis,
                    previous_index: index - 1,
                    next_index: index,
                });
            }
            previous = coordinate;
        }
    }
    Ok(())
}

fn array_to_vec3(quantity: &'static str, values: [f32; 3]) -> Result<Vec3, SceneError> {
    let vector = Vec3::new(values[0], values[1], values[2]);
    validation::finite_vector(quantity, vector).map_err(SceneError::Numerical)
}
