//! Failure types for serial simulation construction and execution.
//!
//! Failed or cancelled timesteps leave the accepted snapshot unchanged. A run
//! failure keeps the outcomes that were accepted before the failed step.

use std::fmt;

use crate::{NumericalError, config::ConfigError};

use super::StepOutcome;

/// A failure while constructing the solver or advancing one timestep.
///
/// Returning this error never commits the attempted timestep; earlier accepted
/// snapshots remain available.
#[derive(Debug)]
pub enum StepError {
    /// The supplied numerical configuration is invalid or unsupported.
    Configuration(ConfigError),
    /// A checked input or intermediate calculation failed.
    Numerical(NumericalError),
    /// Stable identities are duplicated or out of ascending order.
    IdentityOrder {
        /// Identity immediately preceding the invalid entry.
        previous: u64,
        /// Identity of the invalid entry.
        next: u64,
    },
    /// A final position fails a configured halfspace after all projection passes.
    OutsidePlane {
        /// Stable identity of the particle outside the halfspace.
        identity: u64,
        /// Zero-based index in the configured plane order.
        plane: usize,
    },
    /// The cancellation predicate requested that working state be discarded.
    Cancelled,
    /// Another accepted step would overflow the completed-step counter.
    StepCountOverflow,
}

impl From<NumericalError> for StepError {
    fn from(error: NumericalError) -> Self {
        Self::Numerical(error)
    }
}

impl From<ConfigError> for StepError {
    fn from(error: ConfigError) -> Self {
        Self::Configuration(error)
    }
}

impl fmt::Display for StepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => {
                write!(formatter, "invalid solver configuration: {error}")
            }
            Self::Numerical(error) => write!(formatter, "simulation calculation failed: {error}"),
            Self::IdentityOrder { previous, next } => write!(
                formatter,
                "particle identities must be unique and ascending, got {previous} then {next}"
            ),
            Self::OutsidePlane { identity, plane } => {
                write!(formatter, "particle {identity} is outside plane {plane}")
            }
            Self::Cancelled => write!(formatter, "timestep cancelled; no state accepted"),
            Self::StepCountOverflow => write!(formatter, "completed-step counter would overflow"),
        }
    }
}

impl std::error::Error for StepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::Numerical(error) => Some(error),
            _ => None,
        }
    }
}

/// An incomplete multi-step request, retaining outcomes only for accepted steps.
#[derive(Debug)]
pub struct RunFailure {
    /// Outcomes from this request before the first failed or cancelled step.
    pub completed: Vec<StepOutcome>,
    /// Reason the next attempted step did not commit.
    pub error: StepError,
}

impl fmt::Display for RunFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "run stopped after {} accepted steps: {}",
            self.completed.len(),
            self.error
        )
    }
}

impl std::error::Error for RunFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
