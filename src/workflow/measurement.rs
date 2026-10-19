//! Measurement records stay separate from the reproducible numerical identity.
//! Only accepted steps contribute to successful-work timings; failed attempts
//! retain their own elapsed time alongside the ordinary failure record.

use serde::Serialize;

use crate::simulation::StepMeasurement;

/// Measurement overhead selected explicitly by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementMode {
    /// Two outer monotonic clock reads per successful step.
    Total,
    /// Total plus internal intervals, for a separately labelled stage profile.
    Stages,
}

/// Optional additive result data. Ordinary unmeasured result JSON is unchanged.
#[derive(Debug, Serialize)]
pub struct RunMeasurements {
    /// Version of the measurement fields and interval definitions.
    pub schema_version: u32,
    /// Whether internal profiling overhead was enabled.
    pub mode: MeasurementMode,
    /// One entry per accepted step, in the same order as the CFL diagnostics.
    pub steps: Vec<StepMeasurement>,
    /// Solver construction, including private workers or CUDA context/module setup.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_construction_s: Option<f64>,
    /// Consuming solver destruction, outside completed-step totals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_teardown_s: Option<f64>,
    /// Elapsed seconds of an unsuccessful solver attempt, excluded from totals.
    pub failed_attempt_s: Option<f64>,
}

impl RunMeasurements {
    pub(super) fn new(mode: MeasurementMode) -> Self {
        Self {
            schema_version: 1,
            mode,
            steps: Vec::new(),
            failed_attempt_s: None,
            solver_construction_s: None,
            solver_teardown_s: None,
        }
    }
}
