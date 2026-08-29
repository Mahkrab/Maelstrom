//! # Numerical errors
//!
//! Errors returned when an input is outside its valid range, or a calculation
//! produces a value that the simulator cannot use safely.

use std::fmt;

/// ### Error returned by a checked numerical operation
///
/// Input errors name the value that should be corrected. Result errors identify
/// the calculation that overflowed or otherwise became non-finite.
#[derive(Clone, Debug, PartialEq)]
pub enum NumericalError {
    /// A scalar input is NaN or infinite.
    NonFiniteScalar { quantity: &'static str, value: f32 },
    /// At least one component of a vector input is NaN or infinite.
    NonFiniteVector { quantity: &'static str },
    /// A value that must be greatder than zero, is zero, or negative.
    NonPositiveScalar { quantity: &'static str, value: f32 },
    /// A value that may be zero is negative.
    NegativeScalar { quantity: &'static str, value: f32 },
    /// A calculation produced NaN or infinity from otherwise finite inputs.
    NonFiniteResult { quantity: &'static str },
    /// The artificial-pressure sample point is not strictly inside kernel support.
    InvalidReferenceSeparation {
        separation: f32,
        support_radius: f32,
    },
    /// A plane normal has no direction.
    ZeroPlaneNormal,
    /// A plane normal is outside the permitted distance from unit length.
    NonUnitPlaneNormal { length_squared: f32, tolerance: f32 },
    /// The tolerance used to check a plane normal is negative or non-finite.
    InvalidUnitTolerance { tolerance: f32 },
    /// There is not exactly onee interaction row for each particle.
    InteractionSetCount { expected: usize, actual: usize },
    /// An interaction row refers to a particle that does not exist.
    InteractionIndexOutOfBounds {
        particle: usize,
        neighbour: usize,
        particle_count: usize,
    },
    /// A particle appears in its own directional-interaction row.
    SelfInteraction { particle: usize },
    /// Interaction indices are duplicated or are not in ascending order.
    InteractionOrder {
        particle: usize,
        previous: usize,
        next: usize,
    },
    /// A requested particle does not exist in the supplied snapshot.
    ParticleIndexOutOfBounds {
        particle: usize,
        particle_count: usize,
    },
    /// Parallel slices that describe the same particles have different lengths.
    LengthMismatch {
        quantity: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A lattice support ratio would require an impractical enumeration.
    SupportRatioTooLarge { ratio: f32, maximum: f32 },
}

impl fmt::Display for NumericalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteScalar { quantity, value } => {
                write!(formatter, "{quantity} must be finite, got {value}")
            }
            Self::NonFiniteVector { quantity } => {
                write!(formatter, "{quantity} must contain only finite components")
            }
            Self::NonPositiveScalar { quantity, value } => {
                write!(formatter, "{quantity} must be positive, got {value}")
            }
            Self::NegativeScalar { quantity, value } => {
                write!(formatter, "{quantity} must be positive, got {value}")
            }
            Self::NonFiniteResult { quantity } => {
                write!(formatter, "calculation produced a non-finite {quantity}")
            }
            Self::InvalidReferenceSeparation {
                separation,
                support_radius,
            } => write!(
                formatter,
                "artificial-pressure reference separation must be in (0, {support_radius}), got {separation}"
            ),
            Self::ZeroPlaneNormal => write!(formatter, "plane normal must not be zero"),
            Self::NonUnitPlaneNormal {
                length_squared,
                tolerance,
            } => write!(
                formatter,
                "plane normal length squared must be within {tolerance} of 1, got {length_squared}"
            ),
            Self::InvalidUnitTolerance { tolerance } => write!(
                formatter,
                "plane unit-normal tolerance must be finite and positive, got {tolerance}"
            ),
            Self::InteractionSetCount { expected, actual } => write!(
                formatter,
                "interaction-set count must match particle count: expected {expected}, got {actual}"
            ),
            Self::InteractionIndexOutOfBounds {
                particle,
                neighbour,
                particle_count,
            } => write!(
                formatter,
                "particle {particle} references neighbour {neighbour}, but particle count is {particle_count}"
            ),
            Self::SelfInteraction { particle } => {
                write!(
                    formatter,
                    "particle {particle} must not appear in its interaction set"
                )
            }
            Self::InteractionOrder {
                particle,
                previous,
                next,
            } => write!(
                formatter,
                "particle {particle} interaction indices must be unique and ascending, got {previous} then {next}"
            ),
            Self::ParticleIndexOutOfBounds {
                particle,
                particle_count,
            } => write!(
                formatter,
                "particle index {particle} is invalid for particle count {particle_count}"
            ),
            Self::LengthMismatch {
                quantity,
                expected,
                actual,
            } => write!(
                formatter,
                "{quantity} length must be {expected}, got {actual}"
            ),
            Self::SupportRatioTooLarge { ratio, maximum } => write!(
                formatter,
                "lattice support ratio must be at most {maximum}, got {ratio}"
            ),
        }
    }
}

impl std::error::Error for NumericalError {}
