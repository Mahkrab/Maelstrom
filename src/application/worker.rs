//! The worker owns all solver lifetimes. Publishing a snapshot follows acceptance
//! and never borrows solver storage across a thread boundary.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, TryRecvError},
};
use std::time::Instant;

use crate::simulation::{Simulation, StepError};

use super::{
    Action, Command, ControlError, Failure, FailureKind, RunInfo, RunSpec, RunStatus, SharedRun,
};

struct ActiveRun {
    // Drop the solver on this thread before releasing the observation owner.
    solver: Option<Simulation>,
    shared: Arc<SharedRun>,
    status: RunStatus,
    target: u64,
    generation: u64,
    acceptance_trace: Option<AcceptanceTrace>,
}

// Capacity qualification needs every accepted boundary, rather than coalesced
// display observations. Buffer opt-in timestamps and write only during teardown.
struct AcceptanceTrace {
    file: std::fs::File,
    origin: Instant,
    rows: Vec<(u64, u128)>,
    truncated: bool,
}
impl AcceptanceTrace {
    fn open() -> Option<Self> {
        let path = std::env::var_os("MAELSTROM_ACCEPTANCE_PROFILE")?;
        match std::fs::File::create_new(path) {
            Ok(file) => Some(Self {
                file,
                origin: Instant::now(),
                rows: vec![(0, 0)],
                truncated: false,
            }),
            Err(error) => {
                eprintln!("maelstrom acceptance profile unavailable: {error}");
                None
            }
        }
    }
    fn record(&mut self, step: u64, at: Instant) {
        if self.rows.len() < 1_000_000 {
            self.rows
                .push((step, at.duration_since(self.origin).as_nanos()));
        } else if !self.truncated {
            self.truncated = true;
            eprintln!(
                "maelstrom acceptance profile reached its 1000000-row memory budget; the trace is incomplete; simulation continues"
            );
        }
    }
}
impl Drop for AcceptanceTrace {
    fn drop(&mut self) {
        use std::io::Write;
        let mut writer = std::io::BufWriter::new(&self.file);
        let result = (|| -> std::io::Result<()> {
            writeln!(writer, "step,wall_ns")?;
            for (step, time) in &self.rows {
                writeln!(writer, "{step},{time}")?;
            }
            writer.flush()
        })();
        if let Err(error) = result {
            eprintln!("maelstrom acceptance profile write: {error}");
        }
    }
}

impl ActiveRun {
    fn set_status(&mut self, status: RunStatus) {
        self.status = status;
        self.shared.update(|o| o.status = status);
        if status.is_terminal() {
            self.solver.take();
        }
    }

    fn cancelled(&self, stopping: &AtomicBool) -> bool {
        stopping.load(Ordering::Acquire) || self.shared.cancelled.load(Ordering::Acquire)
    }

    fn fail(&mut self, kind: FailureKind, message: String) {
        self.status = RunStatus::Failed;
        self.shared.update(|o| {
            o.status = RunStatus::Failed;
            o.failure = Some(Failure { kind, message });
        });
        self.solver.take();
    }

    fn step(&mut self, stopping: &AtomicBool) {
        let Some(solver) = &mut self.solver else {
            return;
        };
        let shared = &self.shared;
        let start = Instant::now();
        match solver.step_with_cancellation(|| {
            stopping.load(Ordering::Acquire) || shared.cancelled.load(Ordering::Acquire)
        }) {
            Ok(outcome) => {
                let accepted_at = self.acceptance_trace.as_ref().map(|_| Instant::now());
                let solver_s = start.elapsed().as_secs_f64();
                let copy_start = Instant::now();
                let snapshot = Arc::new(solver.snapshot().clone());
                let snapshot_s = copy_start.elapsed().as_secs_f64();
                if outcome.completed_steps == self.target {
                    self.status = RunStatus::Completed;
                }
                self.shared.update(|o| {
                    o.status = self.status;
                    o.snapshot = Some(snapshot);
                    o.last_cfl = Some(outcome.cfl);
                    o.warning_count += u64::from(outcome.cfl.warning);
                    o.timings.solver_s += solver_s;
                    o.timings.snapshot_s += snapshot_s;
                });
                if let Some((trace, at)) = self.acceptance_trace.as_mut().zip(accepted_at) {
                    trace.record(outcome.completed_steps, at);
                }
                if self.status.is_terminal() {
                    self.solver.take();
                }
            }
            Err(StepError::Cancelled) => self.set_status(RunStatus::Cancelled),
            Err(error) => self.fail(FailureKind::Simulation, error.to_string()),
        }
    }

    fn control(
        &mut self,
        action: Action,
        stopping: &AtomicBool,
    ) -> Result<RunStatus, ControlError> {
        match (self.status, action) {
            (RunStatus::Running, Action::Pause) => self.set_status(RunStatus::Paused),
            (RunStatus::Paused, Action::Resume) => self.set_status(RunStatus::Running),
            (RunStatus::Paused, Action::SingleStep) => self.step(stopping),
            (status, action) => return Err(ControlError::InvalidTransition { status, action }),
        }
        Ok(self.status)
    }
}

impl Drop for ActiveRun {
    fn drop(&mut self) {
        if std::thread::panicking() && !self.status.is_terminal() {
            self.shared.update(|o| {
                o.status = RunStatus::Failed;
                o.failure = Some(Failure {
                    kind: FailureKind::WorkerPanic,
                    message: "simulation worker panicked".into(),
                });
            });
        }
    }
}

fn construct(spec: RunSpec, shared: Arc<SharedRun>, stopping: &AtomicBool) -> ActiveRun {
    let generation = shared
        .observation
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .generation;
    let mut active = ActiveRun {
        solver: None,
        shared,
        status: RunStatus::Constructing,
        target: spec.steps,
        generation,
        acceptance_trace: None,
    };
    if active.cancelled(stopping) {
        active.set_status(RunStatus::Cancelled);
        return active;
    }
    active.set_status(RunStatus::Constructing);
    let solver = if !matches!(spec.scene.schema_version, 1 | 2) {
        Err(crate::config::ConfigError::UnsupportedSchema {
            document: "scene",
            supported: crate::config::CURRENT_INPUT_SCHEMA_VERSION,
            found: spec.scene.schema_version,
        }
        .into())
    } else {
        Simulation::from_scene_with_execution(&spec.scene, spec.backend, spec.execution)
    };
    match solver {
        Ok(solver) => {
            let info = Arc::new(RunInfo {
                backend: solver.backend(),
                execution: solver.execution(),
                device: solver.device().cloned(),
                fixed_timestep_s: solver.configuration().numerical.fixed_timestep_s,
                requested_steps: spec.steps,
            });
            let snapshot = Arc::new(solver.snapshot().clone());
            active.shared.update(|o| {
                o.info = Some(info);
                o.snapshot = Some(snapshot);
            });
            active.solver = Some(solver);
            active.acceptance_trace = AcceptanceTrace::open();
            active.set_status(if active.cancelled(stopping) {
                RunStatus::Cancelled
            } else if spec.steps == 0 {
                RunStatus::Completed
            } else if spec.initially_paused {
                RunStatus::Paused
            } else {
                RunStatus::Running
            });
        }
        Err(error) => active.fail(FailureKind::Construction, error.to_string()),
    }
    active
}

pub(super) fn run(
    receiver: Receiver<Command>,
    stopping: Arc<AtomicBool>,
    admission: Arc<Mutex<()>>,
) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_loop(&receiver, &stopping)
    }));
    {
        let _admission = admission.lock().unwrap_or_else(|e| e.into_inner());
        stopping.store(true, Ordering::Release);
    }
    for command in receiver.try_iter() {
        match command {
            Command::Start { shared, reply, .. } => {
                shared.update(|o| {
                    o.status = if outcome.is_err() {
                        RunStatus::Failed
                    } else {
                        RunStatus::Cancelled
                    };
                    if outcome.is_err() {
                        o.failure = Some(Failure {
                            kind: FailureKind::WorkerPanic,
                            message: "simulation worker panicked".into(),
                        });
                    }
                });
                let _ = reply.try_send(Ok(if outcome.is_err() {
                    RunStatus::Failed
                } else {
                    RunStatus::Cancelled
                }));
            }
            Command::Control { reply, .. } => {
                let _ = reply.try_send(Err(ControlError::Stopped));
            }
            Command::Wake => {}
            #[cfg(test)]
            Command::Panic => {}
        }
    }
    if let Err(payload) = outcome {
        std::panic::resume_unwind(payload);
    }
}

fn run_loop(receiver: &Receiver<Command>, stopping: &AtomicBool) {
    let mut active: Option<ActiveRun> = None;
    loop {
        if stopping.load(Ordering::Acquire) {
            break;
        }
        if let Some(run) = &mut active
            && !run.status.is_terminal()
            && run.cancelled(stopping)
        {
            run.set_status(RunStatus::Cancelled);
        }
        let command = if active
            .as_ref()
            .is_some_and(|run| run.status == RunStatus::Running)
        {
            match receiver.try_recv() {
                Ok(command) => Some(command),
                Err(TryRecvError::Empty) => {
                    if let Some(run) = &mut active {
                        run.step(stopping);
                    }
                    None
                }
                Err(TryRecvError::Disconnected) => break,
            }
        } else {
            match receiver.recv() {
                Ok(command) => Some(command),
                Err(_) => break,
            }
        };
        match command {
            Some(Command::Start {
                spec,
                shared,
                reply,
            }) => {
                if let Some(run) = &mut active
                    && !run.status.is_terminal()
                {
                    run.set_status(RunStatus::Cancelled);
                }
                let next = construct(*spec, shared, stopping);
                let _ = reply.try_send(Ok(next.status));
                active = Some(next);
            }
            Some(Command::Control {
                generation,
                action,
                reply,
            }) => {
                let result = match &mut active {
                    Some(run) if run.generation == generation => run.control(action, stopping),
                    _ => Err(ControlError::StaleGeneration),
                };
                let _ = reply.try_send(result);
            }
            Some(Command::Wake) | None => {}
            #[cfg(test)]
            Some(Command::Panic) => panic!("injected worker panic"),
        }
    }
    if let Some(run) = &mut active
        && !run.status.is_terminal()
    {
        run.set_status(RunStatus::Cancelled);
    }
}
