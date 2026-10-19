//! Execution settings belong to each solver, separately from its physical inputs.
//! A private pool lives across timesteps. Indexed results are joined before
//! errors are inspected, preserving serial error precedence and stage barriers.

use std::fmt;

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use serde::{Deserialize, Serialize};

/// Resource bound for an explicit pool, independent of machine CPU capacity.
/// Oversubscription is allowed deliberately, but unbounded thread creation is not.
pub const MAX_WORKERS: usize = 256;

/// Execution resources. A CPU worker is a pool thread, not a physical core.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionMode {
    /// The historical calling-thread execution path, with no worker pool.
    #[default]
    Serial,
    /// Exactly this many private workers; valid counts are `1..=MAX_WORKERS`.
    Threaded {
        /// Requested and effective pool size. Construction never silently clamps it.
        workers: usize,
    },
    /// One explicit CUDA device, with a reproducible block size.
    Cuda {
        /// Process-visible ordinal; unavailable devices fail without fallback.
        device: u32,
        /// Lanes per block: 32, 64, 128 or 256.
        threads: u32,
    },
}

impl ExecutionMode {
    /// Selects the process's available parallelism, capped at [`MAX_WORKERS`].
    /// Failure to query capacity is explicit; Rayon environment settings are ignored.
    pub fn threaded_default() -> Result<Self, ExecutionError> {
        let workers = std::thread::available_parallelism()
            .map_err(ExecutionError::UnavailableCapacity)?
            .get()
            .min(MAX_WORKERS);
        Ok(Self::Threaded { workers })
    }

    /// Checks the resource domain without creating any workers.
    pub fn validate(self) -> Result<(), ExecutionError> {
        if let Self::Cuda { device, threads } = self
            && (device > i32::MAX as u32 || ![32, 64, 128, 256].contains(&threads))
        {
            return Err(ExecutionError::InvalidCudaSettings);
        }
        if let Self::Threaded { workers } = self
            && !(1..=MAX_WORKERS).contains(&workers)
        {
            return Err(ExecutionError::InvalidWorkers { requested: workers });
        }
        Ok(())
    }

    pub(crate) fn is_serial(&self) -> bool {
        matches!(self, Self::Serial)
    }
}

/// An invalid execution request or an unavailable worker resource.
#[derive(Debug)]
pub enum ExecutionError {
    /// CUDA device or block size is outside the supported request domain.
    InvalidCudaSettings,
    /// CPU particle scheduling cannot execute a device request.
    RequiresCpu,
    /// Worker count lies outside the supported resource bound.
    InvalidWorkers {
        /// Count supplied by the caller.
        requested: usize,
    },
    /// Threaded simulation requires uniform-grid discovery.
    RequiresUniformGrid,
    /// The default CPU capacity could not be queried.
    UnavailableCapacity(std::io::Error),
    /// The operating system or scheduler could not construct the private pool.
    PoolBuild(rayon::ThreadPoolBuildError),
    /// The scheduler could not honour the exact requested size.
    WorkerCount {
        /// Requested pool size.
        requested: usize,
        /// Size supplied by the scheduler.
        effective: usize,
    },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCudaSettings => write!(
                f,
                "CUDA requires a device ordinal in 0..=2147483647 and 32, 64, 128 or 256 lanes per block"
            ),
            Self::RequiresCpu => write!(f, "CUDA execution requires the device solver resource"),
            Self::InvalidWorkers { requested } => {
                write!(f, "worker count {requested} must be in 1..={MAX_WORKERS}")
            }
            Self::RequiresUniformGrid => {
                write!(f, "parallel execution requires uniform-grid discovery")
            }
            Self::UnavailableCapacity(error) => write!(
                f,
                "cannot determine available parallelism: {error}; select an explicit worker count"
            ),
            Self::PoolBuild(error) => write!(f, "cannot create worker pool: {error}"),
            Self::WorkerCount {
                requested,
                effective,
            } => write!(
                f,
                "requested {requested} workers but scheduler supplied {effective}"
            ),
        }
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnavailableCapacity(error) => Some(error),
            Self::PoolBuild(error) => Some(error),
            _ => None,
        }
    }
}

/// Reusable CPU resource for independent particle work. No global pool is used.
/// Each call completes all dispatched work before returning, including on error.
#[derive(Debug, Default)]
pub struct Execution {
    pool: Option<ThreadPool>,
}

impl Execution {
    /// Creates a private fixed-size pool, or no workers for serial execution.
    /// Pool creation is setup work and is outside timestep measurements.
    pub fn new(mode: ExecutionMode) -> Result<Self, ExecutionError> {
        mode.validate()?;
        match mode {
            ExecutionMode::Serial => Ok(Self::default()),
            ExecutionMode::Cuda { .. } => Err(ExecutionError::RequiresCpu),
            ExecutionMode::Threaded { workers } => Self::from_pool(
                workers,
                ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .thread_name(|index| format!("maelstrom-{index}"))
                    .build(),
            ),
        }
    }

    fn from_pool(
        workers: usize,
        pool: Result<ThreadPool, rayon::ThreadPoolBuildError>,
    ) -> Result<Self, ExecutionError> {
        let pool = pool.map_err(ExecutionError::PoolBuild)?;
        let effective = pool.current_num_threads();
        if effective != workers {
            return Err(ExecutionError::WorkerCount {
                requested: workers,
                effective,
            });
        }
        Ok(Self { pool: Some(pool) })
    }

    /// Effective mode. A successful threaded construction has the requested size.
    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.pool
            .as_ref()
            .map_or(ExecutionMode::Serial, |pool| ExecutionMode::Threaded {
                workers: pool.current_num_threads(),
            })
    }

    pub(crate) fn map<T: Send, E: Send>(
        &self,
        count: usize,
        operation: impl Fn(usize) -> Result<T, E> + Sync + Send,
    ) -> Result<Vec<T>, E> {
        match &self.pool {
            None => (0..count).map(operation).collect(),
            Some(pool) => {
                // Collect Result values in index order before inspecting them.
                // A parallel Result collection could expose whichever error wins
                // a race, instead of the first failing particle in serial order.
                let results: Vec<Result<T, E>> =
                    pool.install(|| (0..count).into_par_iter().map(operation).collect());
                results.into_iter().collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn joined_work_uses_distinct_workers_and_reports_lowest_index_failure() {
        let execution = Execution::new(ExecutionMode::Threaded { workers: 2 }).unwrap();
        for _ in 0..8 {
            let barrier = Barrier::new(2);
            let finished = AtomicUsize::new(0);
            let identities = Mutex::new(Vec::new());
            let result: Result<Vec<()>, usize> = execution.map(2, |i| {
                identities.lock().unwrap().push(std::thread::current().id());
                barrier.wait();
                if i == 0 {
                    std::thread::yield_now();
                }
                finished.fetch_add(1, Ordering::SeqCst);
                Err(i)
            });
            assert_eq!(result, Err(0));
            assert_eq!(finished.load(Ordering::SeqCst), 2);
            let ids = identities.into_inner().unwrap();
            assert_ne!(ids[0], ids[1]);
        }
    }

    #[test]
    fn workers_are_reused_and_terminate_after_the_resource_is_dropped() {
        let handles = Mutex::new(Vec::new());
        let starts = Arc::new(AtomicUsize::new(0));
        let exits = Arc::new(AtomicUsize::new(0));
        let pool = ThreadPoolBuilder::new()
            .num_threads(3)
            .spawn_handler(|thread| {
                let starts = starts.clone();
                let exits = exits.clone();
                let handle = std::thread::Builder::new().spawn(move || {
                    starts.fetch_add(1, Ordering::SeqCst);
                    thread.run();
                    exits.fetch_add(1, Ordering::SeqCst);
                })?;
                handles.lock().unwrap().push(handle);
                Ok(())
            })
            .build();
        let execution = Execution::from_pool(3, pool).unwrap();
        for _ in 0..4 {
            let barrier = Barrier::new(3);
            let result = execution
                .map(3, |i| {
                    barrier.wait();
                    Ok::<_, ()>(i)
                })
                .unwrap();
            assert_eq!(result, vec![0, 1, 2]);
            assert_eq!(starts.load(Ordering::SeqCst), 3);
            assert_eq!(exits.load(Ordering::SeqCst), 0);
        }
        drop(execution);
        for handle in handles.into_inner().unwrap() {
            handle.join().unwrap();
        }
        assert_eq!(exits.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn unavailable_workers_are_a_typed_setup_failure() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(2)
            .spawn_handler(|_| Err(std::io::Error::other("injected spawn failure")))
            .build();
        let error = Execution::from_pool(2, pool).unwrap_err();
        assert!(matches!(error, ExecutionError::PoolBuild(_)));
        assert!(error.to_string().contains("injected spawn failure"));
    }
}
