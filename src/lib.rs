//! Headless Position-Based Fluids simulation with deterministic serial arithmetic.
//!
//! [`simulation::SerialSimulation`] owns accepted state and combines neighbour
//! discovery with the shared numerical foundations. Configuration and scenes
//! provide inputs; callers can inspect snapshots between transactional timesteps.

pub mod boundary;
pub mod config;
pub mod diagnostics;
pub mod error;
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
