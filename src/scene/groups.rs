//! Named sources become one ordered particle vector before solver setup.
//! Names aid authoring; they do not introduce separate materials or solver work.

use super::{
    MAX_SCENE_PARTICLES, Particle, ParticleGroup, ParticleSource, SceneError,
    particles::resolve_source, separation::validate_separation,
};
use crate::config::{ConfigError, ResolvedConfiguration};
use std::collections::BTreeSet;

/// Maximum named initial sources accepted in one scene.
pub const MAX_PARTICLE_GROUPS: usize = 256;

pub(super) fn resolve(
    groups: &[ParticleGroup],
    configuration: &ResolvedConfiguration,
    allow_coincident: bool,
) -> Result<Vec<Particle>, SceneError> {
    if groups.len() > MAX_PARTICLE_GROUPS {
        return Err(invalid("at most 256 named groups are supported"));
    }
    let mut names = BTreeSet::new();
    let mut total = 0usize;
    for group in groups {
        if group.name.trim().is_empty() || group.name.chars().any(char::is_control) {
            return Err(invalid(
                "group names must contain non-whitespace text without control characters",
            ));
        }
        if !names.insert(group.name.as_str()) {
            return Err(invalid(&format!("duplicate group name {:?}", group.name)));
        }
        let count = source_count(&group.source).map_err(|error| context(group, error))?;
        total = total.checked_add(count).ok_or_else(|| too_many(u64::MAX))?;
        if total > MAX_SCENE_PARTICLES {
            return Err(too_many(total as u64));
        }
    }
    let mut particles: Vec<Particle> = Vec::with_capacity(total);
    let mut lattice_ranges = Vec::new();
    for group in groups {
        let automatic = if matches!(
            group.source,
            ParticleSource::Lattice {
                first_identity: None,
                ..
            }
        ) {
            particles
                .last()
                .map_or(Ok(0), |p| {
                    p.identity
                        .checked_add(1)
                        .ok_or(SceneError::ParticleIdentityOverflow)
                })
                .map_err(|error| context(group, error))?
        } else {
            0
        };
        let next = resolve_source(
            &group.source,
            configuration,
            allow_coincident,
            automatic,
            false,
        )
        .map_err(|error| context(group, error))?;
        if let (Some(previous), Some(first)) = (particles.last(), next.first())
            && previous.identity >= first.identity
        {
            return Err(context(
                group,
                SceneError::ParticleIdentityOrder {
                    previous: previous.identity,
                    next: first.identity,
                },
            ));
        }
        if matches!(group.source, ParticleSource::Lattice { .. }) {
            lattice_ranges.push(particles.len()..particles.len() + next.len());
        }
        particles.extend(next);
    }
    validate_separation(
        &particles,
        configuration.derived.particle_diameter_m,
        allow_coincident,
        &lattice_ranges,
    )?;
    Ok(particles)
}

fn source_count(source: &ParticleSource) -> Result<usize, SceneError> {
    match source {
        ParticleSource::Explicit { items } => {
            if items.len() > MAX_SCENE_PARTICLES {
                return Err(SceneError::ParticleCountTooLarge {
                    construction: "explicit",
                    maximum: MAX_SCENE_PARTICLES,
                    requested: items.len() as u64,
                });
            }
            Ok(items.len())
        }
        ParticleSource::Lattice { counts, .. } => {
            let count = counts
                .iter()
                .try_fold(1u64, |n, &axis| n.checked_mul(u64::from(axis)))
                .ok_or_else(|| too_many(u64::MAX))?;
            if count > MAX_SCENE_PARTICLES as u64 {
                return Err(too_many(count));
            }
            usize::try_from(count).map_err(|_| too_many(count))
        }
        ParticleSource::Groups { .. } => Err(invalid("nested groups are not supported")),
    }
}

fn too_many(requested: u64) -> SceneError {
    SceneError::ParticleCountTooLarge {
        construction: "groups",
        maximum: MAX_SCENE_PARTICLES,
        requested,
    }
}
fn invalid(reason: &str) -> SceneError {
    ConfigError::InvalidField {
        field: "particles.groups",
        reason: reason.into(),
    }
    .into()
}
fn context(group: &ParticleGroup, error: SceneError) -> SceneError {
    SceneError::ParticleGroup {
        name: group.name.clone(),
        error: Box::new(error),
    }
}
