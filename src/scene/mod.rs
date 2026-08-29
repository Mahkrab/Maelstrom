//! # Versioned deterministic scenes
//!
//! A scene fixes the initial particles, static half-spaces and numerical
//! configuration for one simulation input. This module parses and resolves
//! that input deterministically *fancy*, leaving timestep execution to the simulator.

mod document;
mod geometry;
mod inspection;
mod particles;
mod resolved;

pub use document::{
    ConfigurationSelection, ConfigurationSource, LatticeEndpointRule, LatticeOrder,
    ParticleDocument, ParticleSource, PlaneDocument, RandomnessPolicy, SceneDocument, parse_scene,
};
pub use resolved::{
    MAX_EXPLICIT_PARTICLES, MAX_LATTICE_PARTICLES, MAX_PLANES,
    PARTICLE_SEPARATION_RELATIVE_TOLERANCE, PLANE_UNIT_LENGTH_SQUARED_TOLERANCE, Particle,
    ResolvedScene, SceneError, load_resolved_scene, resolve_scene,
};
