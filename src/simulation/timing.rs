//! Optional monotonic stage clocks observe the existing serial execution path.
//! Disabled clocks compile to no operations; successful step totals also include
//! local destruction and clock overhead outside the contiguous stage intervals.

use std::time::Instant;

use serde::Serialize;

/// Contiguous internal intervals, summed across all configured Jacobi iterations.
/// Values are elapsed wall-clock seconds, including descheduling and validation.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StageTimes {
    /// Initial cancellation/state checks, acceleration and mandatory CFL diagnostic.
    pub validation_and_acceleration_s: f64,
    /// Position prediction and its cancellation poll.
    pub prediction_s: f64,
    /// Complete brute-force discovery, validation and its cancellation poll.
    pub neighbour_search_s: f64,
    /// Densities, multipliers, corrections, physical-density validation and poll.
    pub jacobi_s: f64,
    /// Candidate construction, ordered plane projection, position replacement and poll.
    pub projection_s: f64,
    /// Velocity reconstruction, final state/plane checks, poll and atomic acceptance.
    pub acceptance_s: f64,
}

/// One accepted step's measurement. A failed attempt never supplies one.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct StepMeasurement {
    /// Around the complete solver call; excludes workflow observations and recording.
    pub total_s: f64,
    /// Present only for an explicitly instrumented profile, not primary samples.
    pub stages: Option<StageTimes>,
}

pub(super) enum Stage {
    ValidationAndAcceleration,
    Prediction,
    NeighbourSearch,
    Jacobi,
    Projection,
    Acceptance,
}

pub(super) struct StageClock<const ENABLED: bool> {
    previous: Option<Instant>,
    times: StageTimes,
}

impl<const ENABLED: bool> StageClock<ENABLED> {
    pub(super) fn new() -> Self {
        Self {
            previous: if ENABLED { Some(Instant::now()) } else { None },
            times: StageTimes::default(),
        }
    }

    pub(super) fn mark(&mut self, stage: Stage) {
        if ENABLED && let Some(previous) = self.previous {
            let now = Instant::now();
            let seconds = now.duration_since(previous).as_secs_f64();
            let interval = match stage {
                Stage::ValidationAndAcceleration => &mut self.times.validation_and_acceleration_s,
                Stage::Prediction => &mut self.times.prediction_s,
                Stage::NeighbourSearch => &mut self.times.neighbour_search_s,
                Stage::Jacobi => &mut self.times.jacobi_s,
                Stage::Projection => &mut self.times.projection_s,
                Stage::Acceptance => &mut self.times.acceptance_s,
            };
            *interval += seconds;
            self.previous = Some(now);
        }
    }

    pub(super) fn finish(self) -> StageTimes {
        self.times
    }
}
