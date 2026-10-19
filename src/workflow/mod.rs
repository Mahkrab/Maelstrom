//! Headless execution joins resolved inputs, transactional stepping and declared
//! observations. Only the solver's step call advances state. Audits, copies and
//! output happen after it returns, outside the primary solver timing boundary.
//!
//! Optional measurements wrap solver calls only. The workflow retains every accepted CFL
//! outcome and each requested checkpoint in memory; callers choose checkpoint
//! frequency according to the size of their input.

mod measurement;
mod output;
mod record;

pub use measurement::{MeasurementMode, RunMeasurements};
pub use output::{OutputError, OutputFormat};
pub use record::{ParticleRecord, RecordError, RecordedInput, SCHEMA_VERSION, SnapshotRecord};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    diagnostics::{CflDiagnostic, CflLimit, DensityAudit, audit_density},
    execution::ExecutionMode,
    neighbours::SearchBackend,
    scene::ResolvedScene,
    simulation::{SerialSimulation, StepError},
};

/// Historical four-iteration brute-force identity, retained for API compatibility.
pub const BACKEND_ID: &str = "serial-brute-force-pbf-binary32-four-jacobi-v2";

fn backend_id(backend: SearchBackend, iterations: u32, execution: ExecutionMode) -> String {
    let search = match backend {
        SearchBackend::BruteForce => "brute-force",
        SearchBackend::UniformGrid => "uniform-grid",
    };
    let count = if iterations == 4 {
        "four".to_owned()
    } else {
        iterations.to_string()
    };
    let mode = match execution {
        ExecutionMode::Serial => "serial",
        ExecutionMode::Threaded { .. } => "threaded",
        ExecutionMode::Cuda { .. } => "cuda",
    };
    let version = if matches!(execution, ExecutionMode::Cuda { .. }) {
        1
    } else {
        2
    };
    format!("{mode}-{search}-pbf-binary32-{count}-jacobi-v{version}")
}

/// Execution choices that do not change resolved physical input identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionOptions {
    /// Execution resources; serial by default. Parallel modes require grid discovery.
    pub execution: ExecutionMode,
    /// Discovery implementation; defaults to the historical brute-force path.
    pub backend: SearchBackend,
    /// Optional elapsed-time instrumentation, excluded from replay identity.
    pub measurement: Option<MeasurementMode>,
}

/// Work requested from the initial state. Checkpoint zero observes that initial
/// state; all other checkpoints follow acceptance of the named step. Checkpoints
/// must be unique, ascending and no greater than `steps`. An empty list disables
/// auditing, while the last accepted snapshot is always retained.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    /// Number of timesteps requested; zero is a valid completed request.
    #[serde(with = "record::decimal")]
    pub steps: u64,
    /// Accepted step boundaries at which to copy state and audit density.
    #[serde(with = "record::decimal_vec")]
    pub checkpoints: Vec<u64>,
}

impl RunRequest {
    fn validate(&self) -> Result<(), RecordError> {
        if self.checkpoints.iter().any(|&step| step > self.steps)
            || self.checkpoints.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(RecordError::Invalid(
                "checkpoints must be unique, ascending and within the requested steps".into(),
            ));
        }
        Ok(())
    }
}

/// Final disposition of the whole request, including requested observations and
/// any attempted publication. Accepted step count alone does not imply success.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    /// All requested steps and checkpoints completed without CFL warnings.
    Completed,
    /// All requested work completed, with at least one accepted CFL warning.
    CompletedWithWarnings,
    /// Input or request validation failed before timestep execution.
    InputValidationFailure,
    /// A timestep failed; the last accepted snapshot is retained.
    SimulationFailure,
    /// A cancellation request prevented completion of the requested work.
    Cancelled,
    /// A requested audit failed after observing, but never changing, accepted state.
    AuditFailure,
    /// Encoding or publication failed; prior failures and accepted work are retained.
    OutputFailure,
}

impl CompletionStatus {
    /// Whether all requested simulation and observation work completed and no
    /// attempted publication failed. CFL warnings remain successful completion.
    #[must_use]
    pub fn is_complete(self) -> bool {
        matches!(self, Self::Completed | Self::CompletedWithWarnings)
    }
}

/// Context for one failure. Multiple failures are retained if publishing an
/// already incomplete run also fails.
#[derive(Clone, Debug, Serialize)]
pub struct Failure {
    /// Classification of this failure, independent of subsequent failures.
    pub status: CompletionStatus,
    /// Last fully accepted step when the failure was reported.
    #[serde(with = "record::decimal")]
    pub completed_steps: u64,
    /// Context from the checked operation that failed.
    pub message: String,
}

/// Portable CFL limit, with unboundedness represented by a named variant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", content = "seconds", rename_all = "snake_case")]
pub enum RecordedCflLimit {
    /// Zero maximum speed supplies no finite velocity-based limit.
    Unbounded,
    /// Finite velocity-based limit in seconds.
    Finite(f32),
}

/// Post-acceleration diagnostics for one accepted timestep.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StepDiagnostic {
    /// Accepted step number, starting at one.
    #[serde(with = "record::decimal")]
    pub completed_steps: u64,
    /// Maximum post-acceleration particle speed in metres per second.
    pub maximum_speed_m_s: f32,
    /// Limit used for this step's warning-only CFL check.
    pub cfl_limit: RecordedCflLimit,
    /// Whether the fixed timestep exceeded the finite CFL limit.
    pub warning: bool,
}

impl StepDiagnostic {
    fn new(completed_steps: u64, cfl: CflDiagnostic) -> Self {
        Self {
            completed_steps,
            maximum_speed_m_s: cfl.maximum_speed,
            cfl_limit: match cfl.limit {
                CflLimit::Unbounded => RecordedCflLimit::Unbounded,
                CflLimit::Finite(seconds) => RecordedCflLimit::Finite(seconds),
            },
            warning: cfl.warning,
        }
    }
}

/// An explicitly requested observation. `density` is absent only if that audit
/// failed; empty successful audits have zero samples and unavailable metrics.
#[derive(Clone, Debug, Serialize)]
pub struct Checkpoint {
    snapshot: SnapshotRecord,
    density: Option<DensityAudit>,
}

impl Checkpoint {
    /// Accepted state at this observation boundary.
    #[must_use]
    pub fn snapshot(&self) -> &SnapshotRecord {
        &self.snapshot
    }
    /// Density audit, or `None` if the result records an audit failure here.
    #[must_use]
    pub fn density(&self) -> Option<DensityAudit> {
        self.density
    }
}

/// Stable execution identity and build context. This is a repeatable request
/// identity, not a unique invocation identifier or a certification of correctness.
#[derive(Clone, Debug, Serialize)]
pub struct RunMetadata {
    /// Runtime device identity for CUDA; absent in historical and CPU records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<crate::cuda::DeviceInfo>,
    /// SHA-256 of backend, application version, input key, steps and worker settings.
    /// Checkpoint selection, labels and provenance do not affect this identity.
    pub run_id: String,
    /// SHA-256 of the exact existing scene comparison key.
    pub input_id: String,
    /// SHA-256 of the exact existing configuration comparison key.
    pub configuration_id: String,
    /// Versioned backend arithmetic contract.
    pub backend: String,
    /// Package version compiled into this application.
    pub application_version: String,
    /// Optional source/build identifier supplied through `MAELSTROM_BUILD_REVISION`
    /// when compiling. Absence is explicit and never treated as a known revision.
    pub build_revision: Option<String>,
    /// Target operating system of the compiled application.
    pub target_os: String,
    /// Target architecture of the compiled application.
    pub target_arch: String,
    /// Whether this build enables Rust debug assertions.
    pub debug_assertions: bool,
}

impl RunMetadata {
    fn new(
        input: &RecordedInput,
        steps: u64,
        search: SearchBackend,
        execution: ExecutionMode,
    ) -> Self {
        let input_id = digest(input.comparison_key());
        let configuration_id = digest(input.configuration_key());
        let application_version = env!("CARGO_PKG_VERSION").to_owned();
        let backend = backend_id(search, input.solver_iterations(), execution);
        let resources = match execution {
            ExecutionMode::Serial => String::new(),
            ExecutionMode::Threaded { workers } => format!("|workers={workers}"),
            ExecutionMode::Cuda { device, threads } => format!(
                "|device={device}|threads={threads}|target={}",
                option_env!("MAELSTROM_CUDA_ARCH").unwrap_or("unbuilt")
            ),
        };
        let run_id = digest(&format!(
            "maelstrom-run-v1|{backend}|{application_version}|{input_id}|{steps}{resources}"
        ));
        Self {
            device: None,
            run_id,
            input_id,
            configuration_id,
            backend,
            application_version,
            build_revision: option_env!("MAELSTROM_BUILD_REVISION").map(str::to_owned),
            target_os: std::env::consts::OS.to_owned(),
            target_arch: std::env::consts::ARCH.to_owned(),
            debug_assertions: cfg!(debug_assertions),
        }
    }
}

fn digest(value: &str) -> String {
    let hex: String = Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("sha256:{hex}")
}

/// Immutable observations and a coherent completion record for one request.
/// Invalid inputs have no reproducible input record, identity or accepted state;
/// runs admitted for execution retain their initial or last accepted snapshot.
#[derive(Debug, Serialize)]
pub struct RunResult {
    schema_version: u32,
    application_version: String,
    backend: String,
    #[serde(skip_serializing_if = "ExecutionMode::is_serial")]
    execution: ExecutionMode,
    request: RunRequest,
    #[serde(with = "record::decimal")]
    completed_steps: u64,
    metadata: Option<RunMetadata>,
    input: Option<RecordedInput>,
    steps: Vec<StepDiagnostic>,
    checkpoints: Vec<Checkpoint>,
    final_snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    measurements: Option<RunMeasurements>,
    // Keep disposition last: a truncated stream is not a complete JSON document.
    failures: Vec<Failure>,
    status: CompletionStatus,
}

impl RunResult {
    /// Requested execution mode; admitted threaded runs use exactly this pool size.
    /// Invalid-input results may retain a rejected setting and have no worker pool.
    #[must_use]
    pub fn execution(&self) -> ExecutionMode {
        self.execution
    }

    /// Whole-request completion, including any failed publication attempts.
    #[must_use]
    pub fn status(&self) -> CompletionStatus {
        self.status
    }
    /// Original work and observation request.
    #[must_use]
    pub fn request(&self) -> &RunRequest {
        &self.request
    }
    /// Reproducibility and comparison metadata, present for validated inputs.
    #[must_use]
    pub fn metadata(&self) -> Option<&RunMetadata> {
        self.metadata.as_ref()
    }
    /// Complete resolved inputs used by the solver, if validation succeeded.
    #[must_use]
    pub fn input(&self) -> Option<&RecordedInput> {
        self.input.as_ref()
    }
    /// All accepted-step diagnostics, including warnings preceding a failure.
    #[must_use]
    pub fn steps(&self) -> &[StepDiagnostic] {
        &self.steps
    }
    /// Requested observations reached before the run stopped.
    #[must_use]
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }
    /// Last accepted state, including initial state when no step was accepted.
    #[must_use]
    pub fn final_snapshot(&self) -> Option<&SnapshotRecord> {
        self.final_snapshot.as_ref()
    }
    /// Optional solver measurements; entries follow the accepted-step diagnostics.
    /// A failed attempt is timed separately and never enters successful totals.
    #[must_use]
    pub fn measurements(&self) -> Option<&RunMeasurements> {
        self.measurements.as_ref()
    }
    /// Number of accepted steps, independent of the overall completion status.
    #[must_use]
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }
    /// Number of accepted steps that exceeded the velocity-based CFL limit.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.steps.iter().filter(|s| s.warning).count()
    }
    /// Failures in occurrence order; publication cannot erase an earlier failure.
    #[must_use]
    pub fn failures(&self) -> &[Failure] {
        &self.failures
    }

    /// Encodes the same result used by the human summary. All numerical values
    /// originate from validated inputs or successful checked calculations.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub(super) fn fail(&mut self, status: CompletionStatus, message: String, completed_steps: u64) {
        self.status = status;
        self.failures.push(Failure {
            status,
            message,
            completed_steps,
        });
    }

    fn observe(&mut self, solver: &SerialSimulation) -> bool {
        let step = solver.snapshot().completed_steps();
        if self.request.checkpoints.binary_search(&step).is_err() {
            return true;
        }
        let config = solver.configuration().derived;
        let audit = audit_density(
            solver.snapshot(),
            config.support_radius_m,
            config.normalised_rest_density_per_m3,
        );
        let snapshot = SnapshotRecord::capture(solver.snapshot());
        match audit {
            Ok(density) => {
                self.checkpoints.push(Checkpoint {
                    snapshot,
                    density: Some(density),
                });
                true
            }
            Err(error) => {
                self.checkpoints.push(Checkpoint {
                    snapshot,
                    density: None,
                });
                self.fail(CompletionStatus::AuditFailure, error.to_string(), step);
                false
            }
        }
    }
}

/// Runs resolved inputs through the default serial solver with requested observations.
/// Input validation failures are explicit results with zero accepted work.
#[must_use]
pub fn run(scene: &ResolvedScene, request: RunRequest) -> RunResult {
    run_with_cancellation(scene, request, || false)
}

/// Runs with the solver's cooperative cancellation semantics. The predicate is
/// only called by timestep execution, so checkpoint selection does not change
/// polling order, numerical state or CFL warnings. A zero-step request has no
/// cancellable timestep, but still performs any declared initial-state audit.
#[must_use]
pub fn run_with_cancellation(
    scene: &ResolvedScene,
    request: RunRequest,
    cancelled: impl FnMut() -> bool,
) -> RunResult {
    run_with_options(scene, request, ExecutionOptions::default(), cancelled)
}

/// Runs with explicitly enabled elapsed wall-clock measurements. Total-only mode
/// avoids internal clock reads; stage mode is a separate instrumented profile.
/// Audits, snapshots, input setup, record allocation and output remain outside
/// every primary timer. Numerical execution and cancellation polls are shared
/// with the unmeasured workflow. Timing never participates in replay identity.
#[must_use]
pub fn run_measured_with_cancellation(
    scene: &ResolvedScene,
    request: RunRequest,
    mode: MeasurementMode,
    cancelled: impl FnMut() -> bool,
) -> RunResult {
    run_with_options(
        scene,
        request,
        ExecutionOptions {
            measurement: Some(mode),
            ..Default::default()
        },
        cancelled,
    )
}

/// Runs selected discovery and execution resources through one observation path.
/// Grid construction, enumeration, row sorting, dispatch and barriers stay inside
/// solver timing. Private worker setup and teardown remain outside that boundary.
/// Each invocation owns its solver; cancellation and failure retain only its own
/// last accepted state. Invalid inputs are explicit results with no accepted work.
#[must_use]
pub fn run_with_options(
    scene: &ResolvedScene,
    request: RunRequest,
    options: ExecutionOptions,
    mut cancelled: impl FnMut() -> bool,
) -> RunResult {
    let mut result = RunResult {
        schema_version: SCHEMA_VERSION,
        application_version: env!("CARGO_PKG_VERSION").to_owned(),
        backend: backend_id(
            options.backend,
            scene.configuration.numerical.solver_iterations,
            options.execution,
        ),
        execution: options.execution,
        request,
        completed_steps: 0,
        metadata: None,
        input: None,
        steps: Vec::new(),
        checkpoints: Vec::new(),
        final_snapshot: None,
        measurements: options.measurement.map(RunMeasurements::new),
        failures: Vec::new(),
        status: CompletionStatus::Completed,
    };
    let setup = (|| {
        result.request.validate()?;
        record::scene_version(scene.schema_version)?;
        let solver =
            SerialSimulation::from_scene_with_execution(scene, options.backend, options.execution)?;
        let input = RecordedInput::capture(scene, &solver);
        // Also validate the portable record, so every admitted input can replay.
        input.resolve()?;
        Ok::<_, RecordError>((solver, input))
    })();
    let (mut solver, input) = match setup {
        Ok(value) => value,
        Err(error) => {
            result.fail(
                CompletionStatus::InputValidationFailure,
                error.to_string(),
                0,
            );
            return result;
        }
    };
    result.metadata = Some(RunMetadata::new(
        &input,
        result.request.steps,
        options.backend,
        solver.execution(),
    ));
    if let Some(metadata) = &mut result.metadata {
        metadata.device = solver.device().cloned();
    }
    result.input = Some(input);
    if result.observe(&solver) {
        for _ in 0..result.request.steps {
            // This call is the primary solver boundary. No audit, snapshot copy,
            // record conversion, encoding or presentation occurs inside it.
            let attempt = if let Some(measurements) = &mut result.measurements {
                let start = std::time::Instant::now();
                match solver.step_measured_with_cancellation(
                    measurements.mode == MeasurementMode::Stages,
                    &mut cancelled,
                ) {
                    Ok((outcome, measurement)) => {
                        measurements.steps.push(measurement);
                        Ok(outcome)
                    }
                    Err(error) => {
                        measurements.failed_attempt_s = Some(start.elapsed().as_secs_f64());
                        Err(error)
                    }
                }
            } else {
                solver.step_with_cancellation(&mut cancelled)
            };
            match attempt {
                Ok(outcome) => result
                    .steps
                    .push(StepDiagnostic::new(outcome.completed_steps, outcome.cfl)),
                Err(error) => {
                    let status = if matches!(error, StepError::Cancelled) {
                        CompletionStatus::Cancelled
                    } else {
                        CompletionStatus::SimulationFailure
                    };
                    result.fail(
                        status,
                        error.to_string(),
                        solver.snapshot().completed_steps(),
                    );
                    break;
                }
            }
            if !result.observe(&solver) {
                break;
            }
        }
    }
    result.completed_steps = solver.snapshot().completed_steps();
    result.final_snapshot = Some(SnapshotRecord::capture(solver.snapshot()));
    if result.status == CompletionStatus::Completed && result.warning_count() > 0 {
        result.status = CompletionStatus::CompletedWithWarnings;
    }
    result
}

/// Extracts and validates reproducible inputs and the original request from a
/// result document. This does not certify the document's reported outcomes or
/// promise that cancellation or an external output failure will recur on replay.
/// This compatibility helper accepts brute-force records only. Use
/// [`replay_execution`] to retain the discovery selection of either backend.
pub fn replay_inputs(json: &str) -> Result<(ResolvedScene, RunRequest), RecordError> {
    let (scene, request, backend) = replay_execution(json)?;
    if backend != SearchBackend::BruteForce {
        return Err(RecordError::Invalid(
            "grid replay requires replay_execution to retain its backend".into(),
        ));
    }
    Ok((scene, request))
}

/// Restores physical inputs, request and recorded discovery selection.
/// Historical brute-force IDs retain their exact meaning. Unknown arithmetic or
/// discovery contracts fail rather than silently selecting another implementation.
/// Measurements are deliberately not restored, and outcomes are not certified.
pub fn replay_execution(
    json: &str,
) -> Result<(ResolvedScene, RunRequest, SearchBackend), RecordError> {
    let (scene, request, options) = replay_options(json)?;
    if !options.execution.is_serial() {
        return Err(RecordError::Invalid(
            "parallel replay requires replay_options to retain execution settings".into(),
        ));
    }
    Ok((scene, request, options.backend))
}

/// Restores discovery and exact worker settings alongside physical inputs.
/// Historical records without execution settings remain serial. Threaded records
/// require valid explicit settings and their matching backend contract. Replay
/// never reselects workers from the current machine or restores measurements.
pub fn replay_options(
    json: &str,
) -> Result<(ResolvedScene, RunRequest, ExecutionOptions), RecordError> {
    #[derive(Deserialize)]
    struct Replay {
        schema_version: u32,
        backend: String,
        #[serde(default)]
        execution: ExecutionMode,
        request: RunRequest,
        input: Option<RecordedInput>,
    }
    let replay: Replay = serde_json::from_str(json)?;
    record::version(replay.schema_version)?;
    let iterations = replay
        .input
        .as_ref()
        .map_or(4, RecordedInput::solver_iterations);
    replay.execution.validate().map_err(StepError::from)?;
    let backend = [SearchBackend::BruteForce, SearchBackend::UniformGrid].into_iter()
        .find(|&candidate| replay.backend == backend_id(candidate, iterations, replay.execution))
        .ok_or_else(|| RecordError::Invalid(format!(
            "record backend {:?} is incompatible with {:?} or {:?}; use the original binary or rerun the scene with a current backend",
            replay.backend, backend_id(SearchBackend::BruteForce, iterations, replay.execution), backend_id(SearchBackend::UniformGrid, iterations, replay.execution)
        )))?;
    if !replay.execution.is_serial() && backend != SearchBackend::UniformGrid {
        return Err(RecordError::Invalid(
            "parallel execution requires uniform-grid discovery".into(),
        ));
    }
    replay.request.validate()?;
    let input = replay
        .input
        .ok_or_else(|| RecordError::Invalid("result has no validated input to replay".into()))?;
    Ok((
        input.resolve()?,
        replay.request,
        ExecutionOptions {
            backend,
            execution: replay.execution,
            measurement: None,
        },
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn identity_digest_matches_the_standard_sha256_abc_vector() {
        assert_eq!(
            super::digest("abc"),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
