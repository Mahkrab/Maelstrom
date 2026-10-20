//! A presentation layer controls a worker through bounded commands and reads
//! owned accepted observations. The solver, including any CUDA context, is
//! constructed, stepped and destroyed on that worker alone.
//!
//! Running is unpaced: every fixed timestep is executed, independently of display
//! clocks. Observations coalesce in one slot per run; this is a display boundary,
//! not a lossless checkpoint stream for offline export.

#[cfg(test)]
mod tests;
mod worker;

use std::{
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::{
    cuda::DeviceInfo, diagnostics::CflDiagnostic, execution::ExecutionMode,
    neighbours::SearchBackend, scene::ResolvedScene, simulation::Snapshot,
};

/// Maximum queued requests. A full queue rejects a request without changing a run.
pub const COMMAND_CAPACITY: usize = 8;

/// Inputs fixed for the lifetime of a run. Changing these means starting again
/// from step zero, never resuming a saved final observation.
#[derive(Clone, Debug)]
pub struct RunSpec {
    /// Resolved physical input; worker construction validates it again.
    pub scene: ResolvedScene,
    /// Discovery implementation, independent of presentation settings.
    pub backend: SearchBackend,
    /// Explicit resources, with no fallback to another execution mode.
    pub execution: ExecutionMode,
    /// Authoritative number of accepted timesteps requested from step zero.
    pub steps: u64,
    /// Start at a paused initial snapshot instead of immediately advancing.
    pub initially_paused: bool,
}

/// Worker lifecycle at a completed synchronous boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStatus {
    /// Admitted to the bounded queue, but not yet constructed.
    Queued,
    /// Validating inputs and acquiring the selected solver resources.
    Constructing,
    /// No numerical work until a resume or single-step command.
    Paused,
    /// Advancing fixed timesteps as fast as the solver permits.
    Running,
    /// All requested timesteps were accepted, including a zero-step request.
    Completed,
    /// Cancellation retained the last accepted state, if construction succeeded.
    Cancelled,
    /// Construction, stepping or the worker failed; inspect the retained failure.
    Failed,
}

impl RunStatus {
    /// Whether further numerical work requires a new run.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

/// Controls applied in queue order between timesteps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Stop before the next attempt. An already synchronous step may finish.
    Pause,
    /// Continue a paused run with its original fixed request.
    Resume,
    /// Attempt exactly one step while paused; remain paused unless terminal.
    SingleStep,
}

/// Responsibility that failed, separate from the diagnostic text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// Input validation or resource acquisition failed.
    Construction,
    /// An attempted timestep failed without accepting its working state.
    Simulation,
    /// The worker unwound unexpectedly; no further commands can be processed.
    WorkerPanic,
}

/// A retained terminal failure. It cannot be cleared by a control command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Which responsibility failed.
    pub kind: FailureKind,
    /// Context supplied by the owning boundary, without input file paths.
    pub message: String,
}

/// Immutable effective settings after successful solver construction.
#[derive(Clone, Debug)]
pub struct RunInfo {
    /// Effective discovery choice.
    pub backend: SearchBackend,
    /// Effective worker/device settings.
    pub execution: ExecutionMode,
    /// Runtime CUDA device properties, absent for CPU execution.
    pub device: Option<DeviceInfo>,
    /// Unchanged binary32 fixed timestep in seconds.
    pub fixed_timestep_s: f32,
    /// Authoritative requested step count.
    pub requested_steps: u64,
}

/// A complete view of one run, never solver scratch. Retaining this value keeps
/// its accepted snapshot alive even as newer observations replace the slot.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Controller-local, monotonically increasing run identity.
    pub generation: u64,
    /// State at the last published boundary.
    pub status: RunStatus,
    /// Absent until successful construction; always the last accepted state.
    pub snapshot: Option<Arc<Snapshot>>,
    /// Effective settings, absent on failed construction.
    pub info: Option<Arc<RunInfo>>,
    /// CFL diagnostic of the most recently accepted step, absent at step zero.
    pub last_cfl: Option<CflDiagnostic>,
    /// Total warnings over accepted steps, even if display observations were lost.
    pub warning_count: u64,
    /// Sticky terminal failure, if any.
    pub failure: Option<Failure>,
}

impl Observation {
    /// Simulated seconds derived from the accepted index, never from a clock.
    #[must_use]
    pub fn simulated_time_s(&self) -> Option<f64> {
        Some(
            self.snapshot.as_ref()?.completed_steps() as f64
                * f64::from(self.info.as_ref()?.fixed_timestep_s),
        )
    }
}

struct SharedRun {
    observation: Mutex<Observation>,
    changed: Condvar,
    cancelled: AtomicBool,
}

impl SharedRun {
    fn update(&self, change: impl FnOnce(&mut Observation)) {
        // A panic must still permit retention and teardown. No solver storage is
        // protected by this lock, and publication never calls presentation code.
        let mut observation = self.observation.lock().unwrap_or_else(|e| e.into_inner());
        change(&mut observation);
        self.changed.notify_all();
    }
}

/// An independently retained run outcome. Old handles remain readable after
/// restart; they have separate observation slots and cannot overwrite a new run.
#[derive(Clone)]
pub struct RunHandle {
    shared: Arc<SharedRun>,
    wake: SyncSender<Command>,
}

impl RunHandle {
    /// Copies the latest observation; particle storage is shared immutably.
    #[must_use]
    pub fn observe(&self) -> Observation {
        self.shared
            .observation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Latches cancellation without needing queue capacity. Synchronous stages
    /// may finish before observing it. Terminal results are not reclassified.
    pub fn cancel(&self) {
        self.shared.cancelled.store(true, Ordering::Release);
        // A full queue already gives the worker something to wake for.
        let _ = self.wake.try_send(Command::Wake);
    }

    /// Waits for a terminal outcome, or returns `None` at the observation timeout.
    /// A timeout neither cancels nor restarts work and is not a shutdown deadline.
    #[must_use]
    pub fn wait_for_terminal(&self, timeout: Duration) -> Option<Observation> {
        let observation = self
            .shared
            .observation
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (observation, _) = self
            .shared
            .changed
            .wait_timeout_while(observation, timeout, |observation| {
                !observation.status.is_terminal()
            })
            .unwrap_or_else(|e| e.into_inner());
        observation
            .status
            .is_terminal()
            .then(|| observation.clone())
    }
}

/// A request rejected before admission or by the authoritative worker state.
#[derive(Debug)]
pub enum ControlError {
    /// The fixed command queue is full; the caller may retry.
    QueueFull,
    /// Shutdown or worker loss prevents processing another request.
    Stopped,
    /// A control targets a superseded run.
    StaleGeneration,
    /// Another generation cannot be represented.
    GenerationExhausted,
    /// The worker's state does not permit this action.
    InvalidTransition {
        /// State when the worker inspected the command.
        status: RunStatus,
        /// Rejected control.
        action: Action,
    },
    /// Thread creation failed before any request was admitted.
    Spawn(std::io::Error),
    /// Joining detected an unexpected worker panic.
    WorkerPanic,
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => write!(f, "application command queue is full"),
            Self::Stopped => write!(f, "application worker is stopped"),
            Self::StaleGeneration => write!(f, "control belongs to a superseded run"),
            Self::GenerationExhausted => write!(f, "application run generation exhausted"),
            Self::InvalidTransition { status, action } => {
                write!(f, "cannot {action:?} a {status:?} run")
            }
            Self::Spawn(error) => write!(f, "could not create application worker: {error}"),
            Self::WorkerPanic => write!(f, "application worker panicked"),
        }
    }
}

impl std::error::Error for ControlError {}

type Reply = SyncSender<Result<RunStatus, ControlError>>;

/// One lossless acknowledgement per admitted command, separate from coalesced
/// display observations. Dropping a receipt never blocks the worker.
pub struct Receipt {
    receiver: Receiver<Result<RunStatus, ControlError>>,
}

impl Receipt {
    /// Waits for construction or control completion; errors do not erase a run.
    pub fn wait(self) -> Result<RunStatus, ControlError> {
        self.receiver.recv().unwrap_or(Err(ControlError::Stopped))
    }

    /// Polls once for acknowledgement. Consume a successful result only once.
    pub fn try_result(&self) -> Option<Result<RunStatus, ControlError>> {
        match self.receiver.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(ControlError::Stopped)),
        }
    }
}

enum Command {
    Start {
        spec: Box<RunSpec>,
        shared: Arc<SharedRun>,
        reply: Reply,
    },
    Control {
        generation: u64,
        action: Action,
        reply: Reply,
    },
    Wake,
    #[cfg(test)]
    Panic,
}

/// Presentation-side owner of a single worker. It contains no solver or device
/// context. Use non-blocking shutdown and poll `is_finished` from an event loop;
/// `join` and the fallback destructor wait for synchronous solver work to finish.
pub struct Controller {
    sender: Option<SyncSender<Command>>,
    thread: Option<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    admission: Arc<Mutex<()>>,
    current: Option<RunHandle>,
    generation: u64,
}

impl Controller {
    /// Creates an idle worker without constructing a solver or graphics resource.
    pub fn new() -> Result<Self, ControlError> {
        let (sender, receiver) = mpsc::sync_channel(COMMAND_CAPACITY);
        let stopping = Arc::new(AtomicBool::new(false));
        let admission = Arc::new(Mutex::new(()));
        let stop = stopping.clone();
        let worker_admission = admission.clone();
        let thread = thread::Builder::new()
            .name("maelstrom-simulation".into())
            .spawn(move || worker::run(receiver, stop, worker_admission))
            .map_err(ControlError::Spawn)?;
        Ok(Self {
            sender: Some(sender),
            thread: Some(thread),
            stopping,
            admission,
            current: None,
            generation: 0,
        })
    }

    fn send(&self, command: Command) -> Result<(), ControlError> {
        // Serialise admission with worker exit so no request can enter after
        // the worker has drained its queue, including during panic recovery.
        let _admission = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        if self.stopping.load(Ordering::Acquire) {
            return Err(ControlError::Stopped);
        }
        let sender = self.sender.as_ref().ok_or(ControlError::Stopped)?;
        sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) => ControlError::QueueFull,
            TrySendError::Disconnected(_) => ControlError::Stopped,
        })
    }

    /// Starts or restarts from initial inputs. Queue admission is atomic: failure
    /// leaves the old run selected and uncancelled. Successful admission cancels
    /// the old generation; its outcome remains accessible through its handle.
    /// Setup failures belong to the new handle and never hide the old outcome.
    pub fn start(&mut self, spec: RunSpec) -> Result<(RunHandle, Receipt), ControlError> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(ControlError::GenerationExhausted)?;
        let shared = Arc::new(SharedRun {
            observation: Mutex::new(Observation {
                generation,
                status: RunStatus::Queued,
                snapshot: None,
                info: None,
                last_cfl: None,
                warning_count: 0,
                failure: None,
            }),
            changed: Condvar::new(),
            cancelled: AtomicBool::new(false),
        });
        let (reply, receiver) = mpsc::sync_channel(1);
        self.send(Command::Start {
            spec: Box::new(spec),
            shared: shared.clone(),
            reply,
        })?;
        if let Some(old) = &self.current {
            old.cancel();
        }
        let wake = self.sender.as_ref().ok_or(ControlError::Stopped)?.clone();
        let handle = RunHandle { shared, wake };
        self.current = Some(handle.clone());
        self.generation = generation;
        Ok((handle, Receipt { receiver }))
    }

    /// Queues a control for this controller's current run. The receipt reports
    /// whether the worker accepted the transition at its next step boundary.
    pub fn control(&self, handle: &RunHandle, action: Action) -> Result<Receipt, ControlError> {
        if !self
            .current
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(&current.shared, &handle.shared))
        {
            return Err(ControlError::StaleGeneration);
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        self.send(Command::Control {
            generation: self.generation,
            action,
            reply,
        })?;
        Ok(Receipt { receiver })
    }

    /// Begins cooperative shutdown without waiting or requiring queue capacity.
    pub fn request_shutdown(&mut self) {
        self.stopping.store(true, Ordering::Release);
        if let Some(current) = &self.current {
            current.cancel();
        }
        if let Some(sender) = &self.sender {
            let _ = sender.try_send(Command::Wake);
        }
        self.sender.take();
    }

    /// Whether the worker has returned and all its solver resources were dropped.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Requests shutdown and joins the worker. There is no deadline guarantee.
    pub fn join(mut self) -> Result<(), ControlError> {
        self.request_shutdown();
        self.join_worker()
    }

    fn join_worker(&mut self) -> Result<(), ControlError> {
        if self
            .thread
            .take()
            .is_some_and(|thread| thread.join().is_err())
        {
            return Err(ControlError::WorkerPanic);
        }
        Ok(())
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.request_shutdown();
        let _ = self.join_worker();
    }
}
