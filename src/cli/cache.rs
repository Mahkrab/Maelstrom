//! Cache commands distinguish fresh simulation from rendering immutable history.
//! The same numerical selection and presentation resolvers serve existing commands.
use super::{
    CliError,
    arguments::{CacheCommand, CacheRender, Cli},
    storage::Storage,
};
use maelstrom::cache::{BakeRequest, Cache};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn cancellation() -> Result<Arc<AtomicBool>, CliError> {
    let flag = Arc::new(AtomicBool::new(false));
    let signal = flag.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|e| CliError::input(e.to_string()))?;
    Ok(flag)
}
fn failure(error: impl std::fmt::Display, cancel: &AtomicBool) -> CliError {
    CliError {
        code: if cancel.load(Ordering::Relaxed) {
            130
        } else {
            5
        },
        message: error.to_string(),
    }
}

pub(super) fn manage(cli: &Cli, storage: &Storage, command: &CacheCommand) -> Result<u8, CliError> {
    let cancel = cancellation()?;
    match command {
        CacheCommand::Bake(args) => {
            let resolved = super::inputs::resolve_in(&args.inputs, storage)?;
            let steps = resolved.steps(&args.limit)?;
            let scene = resolved
                .scene
                .as_ref()
                .ok_or_else(|| CliError::usage("cache bake requires SCENE".into()))?;
            let options = super::selection::resolve(&args.selection, &storage.preferences()?)?;
            let mut bar = Progress::new();
            let result = maelstrom::cache::bake(
                BakeRequest {
                    scene,
                    steps,
                    stride: args.frame_stride,
                    options,
                    destination: &args.output,
                },
                || cancel.load(Ordering::Relaxed),
                |n, total| bar.update(n, total),
            );
            bar.finish();
            let value = result.map_err(|e| failure(e, &cancel))?;
            super::print_document(
                cli.format,
                format!(
                    "Cached {} observations to {}; no images generated.",
                    value["frames"],
                    args.output.display()
                ),
                value,
            )?;
        }
        CacheCommand::Inspect {
            cache,
            allow_partial,
        }
        | CacheCommand::Validate {
            cache,
            allow_partial,
            ..
        } => {
            let opened = Cache::open_with_cancellation(cache, *allow_partial, || {
                cancel.load(Ordering::Relaxed)
            })
            .map_err(|e| {
                if cancel.load(Ordering::Relaxed) {
                    failure(e, &cancel)
                } else {
                    CliError::input(e.to_string())
                }
            })?;
            if let CacheCommand::Validate { full, .. } = command {
                if *full {
                    opened
                        .validate_full(|| cancel.load(Ordering::Relaxed))
                        .map_err(|e| failure(e, &cancel))?;
                } else {
                    for index in 0..opened.frame_count() {
                        if cancel.load(Ordering::Relaxed) {
                            return Err(failure("cache validation cancelled", &cancel));
                        }
                        opened
                            .frame(index)
                            .map_err(|e| CliError::input(e.to_string()))?;
                    }
                }
            }
            super::print_document(
                cli.format,
                format!(
                    "Particle cache: {} committed frames; complete={}; no simulation performed.",
                    opened.frame_count(),
                    opened.is_complete()
                ),
                opened.info(),
            )?;
        }
    }
    Ok(0)
}

#[cfg(not(feature = "render"))]
pub(super) fn render(_: &Cli, _: &CacheRender) -> Result<u8, CliError> {
    Err(CliError::input(
        "cache rendering requires render support; reinstall with --features render or viewer"
            .into(),
    ))
}

#[cfg(feature = "render")]
pub(super) fn render(cli: &Cli, args: &CacheRender) -> Result<u8, CliError> {
    use maelstrom::{
        export::{CacheSelection, CachedRenderRequest, render_cache_with_summary},
        presentation::CameraMode,
        render::GraphicsSelection,
    };
    if args.presentation.frame_stride.is_some() {
        return Err(CliError::usage(
            "cached rendering uses --every in cache-frame indices, not --frame-stride".into(),
        ));
    }
    if args.allow_partial && args.frame.is_none() && args.last_frame.is_none() {
        return Err(CliError::usage(
            "--allow-partial requires an explicit --frame or --last-frame".into(),
        ));
    }
    let cache =
        Cache::open(&args.cache, args.allow_partial).map_err(|e| CliError::input(e.to_string()))?;
    let last = cache
        .frame_count()
        .checked_sub(1)
        .ok_or_else(|| CliError::input("cache has no committed frames".into()))?;
    let selection = CacheSelection {
        first: args.frame.or(args.first_frame).unwrap_or(0),
        last: args.frame.or(args.last_frame).unwrap_or(last),
        every: args.every.unwrap_or(1),
    };
    let selected = super::presentation::resolve(None, &args.presentation)?;
    let cancel = cancellation()?;
    let mut bar = Progress::new();
    let result = render_cache_with_summary(
        CachedRenderRequest {
            cache: &cache,
            settings: selected.effective.settings,
            graphics: GraphicsSelection {
                index: args.graphics_adapter,
                software: args.software,
            },
            fit_camera: selected.effective.camera_mode == CameraMode::Fit,
            selection,
            destination: &args.output,
        },
        || cancel.load(Ordering::Relaxed),
        |n, total| bar.update(n as u64, total as u64),
    );
    bar.finish();
    let value = result.map_err(|e| failure(e, &cancel))?;
    super::print_document(
        cli.format,
        format!(
            "Rendered {} cached frames to {}; no simulation performed.",
            value["frame_count"],
            args.output.display()
        ),
        serde_json::json!({"schema_version":2,"complete":true,"frame_count":value["frame_count"],"path":args.output,"source_cache":cache.identity(),"timing_profile":value["timing_profile"]}),
    )?;
    Ok(0)
}

struct Progress {
    terminal: bool,
    last: std::time::Instant,
}
impl Progress {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self {
            terminal: std::io::stderr().is_terminal(),
            last: std::time::Instant::now() - std::time::Duration::from_secs(1),
        }
    }
    fn update(&mut self, n: u64, total: u64) {
        use std::io::Write;
        if self.terminal && self.last.elapsed() >= std::time::Duration::from_millis(100) {
            eprint!("\r\x1b[2K{n}/{total} frames");
            let _ = std::io::stderr().flush();
            self.last = std::time::Instant::now();
        }
    }
    fn finish(&self) {
        if self.terminal {
            eprintln!();
        }
    }
}
