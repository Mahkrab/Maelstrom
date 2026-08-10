//! # Maelstrom numerical library
//!
//! Reusable maths for Maelstrom's Position-Based Fluids (PBF) simulator.
//! The library provides:
//!
//! - three-dimensional `f32` vectors and smoothing kernels
//! - density constraints and position corrections
//! - plane projection, motion integration and CFL diagnostics
//! - the reference particle-lattice calculation
//!
//! Neighbour discovery and simulation-state management are kept separate from
//! these calculations so they can share the same numerical code.

pub mod boundary;
pub mod diagnostics;
pub mod error;
pub mod fluid;
pub mod integration;
pub mod kernel;
pub mod math;
pub mod reference;

mod validation;

pub use error::NumericalError;
