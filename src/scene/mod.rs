//! # Versioned deterministic scenes
//!
//! A scene fixes the initial particles, static half-spaces and numerical
//! configuration for one simulation input. This module parses and resolves
//! that input deterministically *fancy*, leaving timestep execution to the simulator.

mod container;
mod document;
mod geometry;
mod groups;
mod inspection;
mod particles;
mod resolved;
mod separation;

pub use document::{
    ConfigurationSelection, ConfigurationSource, ContainerBox, LatticeEndpointRule, LatticeOrder,
    ParticleDocument, ParticleGroup, ParticleSource, PlaneDocument, RandomnessPolicy, RunDefaults,
    SceneDocument, parse_scene,
};
pub use resolved::{
    MAX_EXPLICIT_PARTICLES, MAX_LATTICE_PARTICLES, MAX_PLANES, MAX_SCENE_PARTICLES,
    PARTICLE_SEPARATION_RELATIVE_TOLERANCE, PLANE_UNIT_LENGTH_SQUARED_TOLERANCE, Particle,
    ResolvedScene, SceneError, load_resolved_scene, resolve_scene,
    resolve_scene_with_configuration,
};

pub use groups::MAX_PARTICLE_GROUPS;
