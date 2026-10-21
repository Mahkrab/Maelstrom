//! Export source selection is explicit: saved history never silently becomes a
//! new simulation. Physical resolution reuses the ordinary headless CLI boundary.

use super::{
    CliError,
    arguments::{Export, ExportSource},
    inputs,
};
use maelstrom::{
    execution::ExecutionMode,
    export::{ExportRequest, Source},
    neighbours::SearchBackend,
    render::{GraphicsSelection, Settings},
    workflow::ExecutionOptions,
};
use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub(super) fn launch(args: &Export) -> Result<u8, CliError> {
    let destination = args
        .output
        .as_deref()
        .ok_or_else(|| CliError::usage("export requires --output DIR".into()))?;
    let mut settings = match &args.render_config {
        Some(path) => Settings::parse(
            &fs::read_to_string(path)
                .map_err(|e| CliError::input(format!("selected render file: {}", e.kind())))?,
        )
        .map_err(|e| CliError::input(e.to_string()))?,
        None => Settings::default(),
    };
    if let Some(v) = args.width {
        settings.width = v;
    }
    if let Some(v) = args.height {
        settings.height = v;
    }
    if let Some(v) = args.step_stride {
        settings.step_stride = v;
    }
    settings
        .validate()
        .map_err(|e| CliError::input(e.to_string()))?;
    let content = match &args.source {
        ExportSource::Saved { file, .. } | ExportSource::Rerun { file } => Some(
            fs::read_to_string(file)
                .map_err(|e| CliError::input(format!("selected result file: {}", e.kind())))?,
        ),
        _ => None,
    };
    let resolved = match &args.source {
        ExportSource::Direct {
            inputs: selection, ..
        } => Some(inputs::resolve(selection)?),
        _ => None,
    };
    let source = match &args.source {
        ExportSource::Saved { last_step, .. } => Source::Saved {
            json: content
                .as_deref()
                .ok_or_else(|| CliError::input("missing selected result".into()))?,
            last_step: *last_step,
        },
        ExportSource::Rerun { .. } => Source::Rerun {
            json: content
                .as_deref()
                .ok_or_else(|| CliError::input("missing selected result".into()))?,
        },
        ExportSource::Direct {
            steps,
            mode,
            workers,
            device,
            block_size,
            ..
        } => {
            if mode != "threaded-grid" && workers.is_some() {
                return Err(CliError::usage("--workers requires threaded-grid".into()));
            }
            if mode != "cuda-grid" && (device.is_some() || block_size.is_some()) {
                return Err(CliError::usage(
                    "--device/--block-size require cuda-grid".into(),
                ));
            }
            let (backend, execution) = match mode.as_str() {
                "brute-force" => (SearchBackend::BruteForce, ExecutionMode::Serial),
                "serial-grid" => (SearchBackend::UniformGrid, ExecutionMode::Serial),
                "threaded-grid" => (
                    SearchBackend::UniformGrid,
                    ExecutionMode::Threaded {
                        workers: workers.unwrap_or(4),
                    },
                ),
                "cuda-grid" => (
                    SearchBackend::UniformGrid,
                    ExecutionMode::Cuda {
                        device: device.unwrap_or(0),
                        threads: block_size.unwrap_or(128),
                    },
                ),
                _ => return Err(CliError::usage("unsupported export execution mode".into())),
            };
            execution
                .validate()
                .map_err(|e| CliError::usage(e.to_string()))?;
            let scene = resolved
                .as_ref()
                .and_then(|r| r.scene.as_ref())
                .ok_or_else(|| CliError::usage("export direct requires SCENE".into()))?;
            Source::Direct {
                scene,
                steps: *steps,
                options: ExecutionOptions {
                    backend,
                    execution,
                    measurement: None,
                },
            }
        }
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancel);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|e| CliError::input(format!("cancellation handler: {e}")))?;
    maelstrom::export::export(
        ExportRequest {
            source,
            settings,
            graphics: GraphicsSelection {
                index: args.graphics_adapter,
                software: args.software,
            },
            fit_camera: args.fit_camera,
            destination,
        },
        || cancel.load(Ordering::Relaxed),
    )
    .map_err(|e| CliError::output(e.to_string()))?;
    Ok(0)
}
