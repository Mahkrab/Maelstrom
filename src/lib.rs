//! Headless Position-Based Fluids simulation with ordered binary32 arithmetic.
//!
//! [`simulation::Simulation`] owns accepted state and CPU resources, combining neighbour
//! discovery with the shared numerical foundations. Configuration and scenes
//! provide inputs; callers can inspect snapshots between transactional timesteps.

pub mod boundary;
pub mod config;
pub mod diagnostics;
pub mod error;
pub mod execution;
pub mod fluid;
pub mod integration;
pub mod kernel;
pub mod math;
pub mod neighbours;
pub mod reference;
pub mod scene;
pub mod simulation;
pub mod workflow;

mod validation;

pub use error::NumericalError;
