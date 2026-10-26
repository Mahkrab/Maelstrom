//! Automatic execution is a declared resource policy, never a timing claim.
//! Explicit resources cannot disappear into a silent fallback.
use super::{
    CliError,
    arguments::{ExecutionKind, Selection},
    storage::Preferences,
};

pub(super) fn provenance(args: &Selection, preferences: &Preferences) -> serde_json::Value {
    let explicit = args.backend.is_some() || args.execution.is_some();
    let origin = |argument: bool, preference: bool| {
        if argument {
            "command_line"
        } else if preference {
            "application_preferences"
        } else {
            "automatic"
        }
    };
    serde_json::json!({
        "backend":origin(explicit,preferences.backend.is_some()),
        "threads":origin(args.threads.is_some(),!explicit && preferences.threads.is_some()),
        "device":origin(args.device.is_some(),!explicit && preferences.device.is_some()),
        "block_size":origin(args.block_size.is_some(),!explicit && preferences.block_size.is_some()),
    })
}
use maelstrom::{execution::ExecutionMode, neighbours::SearchBackend, workflow::ExecutionOptions};

pub(super) fn resolve(
    args: &Selection,
    preferences: &Preferences,
) -> Result<ExecutionOptions, CliError> {
    let explicit = args.backend.is_some() || args.execution.is_some();
    let backend = args
        .backend
        .as_deref()
        .or(preferences.backend.as_deref())
        .unwrap_or("auto");
    let threads = args
        .threads
        .or(if explicit { None } else { preferences.threads });
    let device = args
        .device
        .or(if explicit { None } else { preferences.device });
    let block = args.block_size.or(if explicit {
        None
    } else {
        preferences.block_size
    });
    let legacy = args.execution;
    if legacy.is_some() && !["uniform-grid", "brute-force", "auto"].contains(&backend) {
        return Err(CliError::usage(
            "--execution is a legacy option; use a single --backend selection".into(),
        ));
    }
    let mode = if let Some(execution) = legacy {
        match (backend, execution) {
            ("brute-force", ExecutionKind::Serial) => "brute-force",
            ("brute-force", _) => {
                return Err(CliError::usage(
                    "parallel execution requires uniform-grid".into(),
                ));
            }
            (_, ExecutionKind::Serial) => "serial-grid",
            (_, ExecutionKind::Threaded) => "threaded-grid",
            (_, ExecutionKind::Cuda) => "cuda",
        }
    } else {
        backend
    };
    let requested_cuda = device.is_some() || block.is_some();
    if threads.is_some() && requested_cuda {
        return Err(CliError::usage(
            "--threads cannot be combined with CUDA resources".into(),
        ));
    }
    let mode = match mode {
        "auto" if requested_cuda => "cuda",
        "auto" if threads.is_some() => "threaded-grid",
        "auto" => {
            // Probe only before execution. Once selected, a CUDA failure is final.
            if maelstrom::cuda::CudaContext::new(0).is_ok() {
                "cuda"
            } else {
                "serial-grid"
            }
        }
        "uniform-grid" => "serial-grid",
        "cuda-grid" => "cuda",
        mode => mode,
    };
    if threads.is_some() && mode != "threaded-grid" {
        return Err(CliError::usage("--threads requires threaded-grid".into()));
    }
    if requested_cuda && mode != "cuda" {
        return Err(CliError::usage("--device/--block-size require cuda".into()));
    }
    let (backend, execution) = match mode {
        "brute-force" => (SearchBackend::BruteForce, ExecutionMode::Serial),
        "serial-grid" => (SearchBackend::UniformGrid, ExecutionMode::Serial),
        "threaded-grid" => (
            SearchBackend::UniformGrid,
            match threads {
                Some(workers) => ExecutionMode::Threaded { workers },
                None => {
                    ExecutionMode::threaded_default().map_err(|e| CliError::input(e.to_string()))?
                }
            },
        ),
        "cuda" => (
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda {
                device: device.unwrap_or(0),
                threads: block.unwrap_or(128),
            },
        ),
        _ => {
            return Err(CliError::usage(
                "backend must be auto, brute-force, serial-grid, threaded-grid or cuda".into(),
            ));
        }
    };
    execution
        .validate()
        .map_err(|e| CliError::usage(e.to_string()))?;
    Ok(ExecutionOptions {
        backend,
        execution,
        measurement: None,
    })
}
