//! Offline sequences sample accepted step indices. A synchronous frame boundary
//! applies backpressure without giving graphics any control over numerical work.

mod offscreen;
mod source;
mod timing;

use crate::{
    render::{BoundaryLines, Graphics, GraphicsSelection, Settings},
    scene::ResolvedScene,
    simulation::Simulation,
    workflow::{ExecutionOptions, RecordedInput, RunMetadata, SnapshotRecord, publish_bytes},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use source::Recorded;
use std::{fmt, fs, path::Path, time::Instant};

/// A rejected input or incomplete export. Published files are retained.
#[derive(Debug)]
pub struct ExportError(pub String);
impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ExportError {}
fn fail(error: impl fmt::Display) -> ExportError {
    ExportError(error.to_string())
}
fn hash(bytes: &[u8]) -> String {
    format!(
        "sha256:{}",
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}

/// Ordered, unique accepted boundaries, including initial and requested final state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FrameSchedule {
    /// Zero-based frame order; integers are exact u64 step indices.
    pub steps: Vec<u64>,
}
impl FrameSchedule {
    /// Rejects overflow and schedules exceeding one million frames before allocation.
    pub fn new(last_step: u64, stride: u64) -> Result<Self, ExportError> {
        if stride == 0 {
            return Err(fail("step stride must be positive"));
        }
        let count = (last_step / stride)
            .checked_add(1)
            .and_then(|n| n.checked_add(u64::from(!last_step.is_multiple_of(stride))))
            .ok_or_else(|| fail("frame count overflow"))?;
        if count > 1_000_000 {
            return Err(fail("frame count exceeds 1000000"));
        }
        let mut steps: Vec<_> = (0..=last_step / stride).map(|i| i * stride).collect();
        if steps.last() != Some(&last_step) {
            steps.push(last_step);
        }
        Ok(Self { steps })
    }
}

/// Numerical source selection is explicit; no missing frame triggers a rerun.
pub enum Source<'a> {
    /// Start a fresh scene at zero using the selected execution resources.
    Direct {
        scene: &'a ResolvedScene,
        steps: u64,
        options: ExecutionOptions,
    },
    /// Draw stored accepted states. Partial results require an explicit last step.
    Saved {
        json: &'a str,
        last_step: Option<u64>,
    },
    /// Rerun the entire recorded request and compare every stored checkpoint.
    Rerun { json: &'a str },
}

/// Resolved presentation choices and the required new destination directory.
pub struct ExportRequest<'a> {
    /// Saved observations, recorded-input rerun, or direct execution.
    pub source: Source<'a>,
    /// Strict presentation schema, independent of physical inputs.
    pub settings: Settings,
    /// Graphics selection, independent of simulation execution.
    pub graphics: GraphicsSelection,
    /// Resolve a fitted camera once from initial geometry.
    pub fit_camera: bool,
    /// Must not already exist; its parent must exist.
    pub destination: &'a Path,
}

/// One closed, published and hashed frame.
#[derive(Debug, Serialize, Deserialize)]
pub struct Frame {
    /// Contiguous zero-based frame index.
    pub index: usize,
    /// Accepted numerical step index, serialised as a decimal string.
    pub step: String,
    /// Simulated time, never an implicit playback FPS.
    pub simulated_seconds: f64,
    /// Fixed basename within the destination directory.
    pub file: String,
    /// SHA-256 of encoded PNG bytes.
    pub png_sha256: String,
    /// SHA-256 of decoded top-left opaque RGBA8 pixels.
    pub rgba_sha256: String,
    /// SHA-256 of the portable accepted observation, including velocities.
    pub snapshot_sha256: String,
}

/// Exports a sequence without a window or coalesced controller observations.
/// On failure, `progress.json` remains explicitly incomplete; no manifest is
/// published. Cancellation is cooperative between solver stages and whole frames.
pub fn export(
    request: ExportRequest<'_>,
    cancelled: impl FnMut() -> bool,
) -> Result<Value, ExportError> {
    export_using(request, cancelled, offscreen::FrameOperations::default())
}

fn export_using(
    request: ExportRequest<'_>,
    mut cancelled: impl FnMut() -> bool,
    operations: offscreen::FrameOperations,
) -> Result<Value, ExportError> {
    let started = Instant::now();
    let mut timings = timing::ExportTimings::default();
    let mut settings = request.settings;
    settings.validate().map_err(fail)?;
    let recorded = match &request.source {
        Source::Saved { json, .. } | Source::Rerun { json } => Some(Recorded::parse(json)?),
        _ => None,
    };
    let (scene, last_step, options, mode) = match &request.source {
        Source::Direct {
            scene,
            steps,
            options,
        } => (*scene, *steps, *options, "direct"),
        Source::Saved { last_step, .. } => {
            let r = recorded
                .as_ref()
                .ok_or_else(|| fail("missing saved source"))?;
            let last = match last_step {
                Some(s) if *s <= r.completed => *s,
                Some(_) => return Err(fail("saved last step exceeds accepted count")),
                None if r.complete => r.steps,
                None => {
                    return Err(fail(
                        "partial saved result requires an explicit --last-step",
                    ));
                }
            };
            (&r.scene, last, r.options, "saved")
        }
        Source::Rerun { .. } => {
            let r = recorded
                .as_ref()
                .ok_or_else(|| fail("missing recorded source"))?;
            (&r.scene, r.steps, r.options, "rerun")
        }
    };
    options.execution.validate().map_err(fail)?;
    if options.execution != crate::execution::ExecutionMode::Serial
        && options.backend != crate::neighbours::SearchBackend::UniformGrid
    {
        return Err(fail(
            "parallel export execution requires uniform-grid discovery",
        ));
    }
    let schedule = FrameSchedule::new(last_step, settings.step_stride)?;
    let dt = f64::from(scene.configuration.numerical.fixed_timestep_s);
    if schedule.steps.iter().any(|s| !(*s as f64 * dt).is_finite()) {
        return Err(fail("simulated timestamp overflow"));
    }
    if mode == "saved" {
        recorded
            .as_ref()
            .ok_or_else(|| fail("missing saved source"))?
            .require_schedule(&schedule)?;
    }
    // Validate solver inputs without acquiring execution resources. Saved mode
    // never constructs the recorded CUDA or threaded backend.
    let initial = Simulation::from_scene(scene).map_err(fail)?;
    let input = RecordedInput::capture(scene, &initial);
    input.resolve().map_err(fail)?;
    if let Some(expected) = recorded.as_ref().and_then(|r| r.snapshots.get(&0)) {
        source::compare(
            expected,
            &SnapshotRecord::capture(initial.snapshot()),
            crate::execution::ExecutionMode::Serial,
        )?;
    }
    drop(initial);
    let spacing = f64::from(scene.configuration.numerical.particle_spacing_m);
    let boundaries = BoundaryLines::new(&scene.particles, &scene.planes, spacing).map_err(fail)?;
    if request.fit_camera {
        settings
            .camera
            .fit_scene(
                &scene.particles,
                &scene.planes,
                spacing / 2.0,
                f64::from(settings.width) / f64::from(settings.height),
            )
            .map_err(fail)?;
    }
    match fs::symlink_metadata(request.destination) {
        Ok(_) => return Err(fail("output destination already exists")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(fail(format!(
                "inspect output destination: {}",
                error.kind()
            )));
        }
    }
    let instance = Graphics::instance();
    let graphics = pollster::block_on(Graphics::new(
        &instance,
        request.graphics,
        None,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
    .map_err(fail)?;
    let _ = graphics
        .depth(settings.width, settings.height)
        .map_err(fail)?;
    // Exclusive creation rejects directories, files and symlinks equally.
    fs::create_dir(request.destination)
        .map_err(|e| fail(format!("create new output directory: {}", e.kind())))?;
    timings.setup = started.elapsed();
    let start = Instant::now();
    let mut progress = json!({"schema_version":1,"complete":false,"source_mode":mode,"expected_frames":schedule.steps.len(),"frames":[],"failures":[],"original":recorded.as_ref().map(|r|&r.provenance)});
    write_progress(request.destination, &progress)?;
    timings.progress += start.elapsed();
    let result = (|| {
        let mut frames = Vec::with_capacity(schedule.steps.len());
        let mut writer = offscreen::FrameWriter {
            operations,
            graphics: &graphics,
            settings: &settings,
            boundaries: &boundaries,
            spacing,
            mass_kg: scene.configuration.derived.particle_mass_kg,
            dt,
            destination: request.destination,
            timings: timing::FrameTimings::default(),
        };
        let mut diagnostics = Vec::new();
        let start = Instant::now();
        let mut solver = if mode == "saved" {
            None
        } else {
            Some(
                Simulation::from_scene_with_execution(scene, options.backend, options.execution)
                    .map_err(fail)?,
            )
        };
        let metadata = solver.as_ref().map(|s| {
            let mut m = RunMetadata::new(&input, last_step, options.backend, options.execution);
            m.device = s.device().cloned();
            m
        });
        timings.setup += start.elapsed();
        let mut next = 0;
        for step in 0..=last_step {
            // Saved schedules can be sparse and enormous. Iterate scheduled
            // observations directly below rather than walking unstored work.
            if mode == "saved" {
                break;
            }
            if cancelled() {
                return Err(fail("export cancelled"));
            }
            let s = solver
                .as_mut()
                .ok_or_else(|| fail("missing rerun solver"))?;
            if step > 0 {
                let start = Instant::now();
                let outcome = s.step_with_cancellation(&mut cancelled).map_err(fail)?;
                timings.solver_steps += start.elapsed();
                diagnostics.push(crate::workflow::StepDiagnostic::new(step, outcome.cfl));
                progress["accepted_steps"] = json!(step.to_string());
            }
            let start = Instant::now();
            let snapshot = SnapshotRecord::capture(s.snapshot());
            timings.snapshot_capture += start.elapsed();
            let start = Instant::now();
            if let Some(expected) = recorded.as_ref().and_then(|r| r.snapshots.get(&step))
                && let Err(error) = source::compare(expected, &snapshot, options.execution)
            {
                let report = json!({"schema_version":1,"policy":"reference-comparison-v1; serial exact bits including signed zero","error":error.to_string(),"expected":expected,"actual":snapshot,"metadata":metadata,"source":recorded.as_ref().map(|r|&r.provenance)});
                if let Err(output) = publish_bytes(
                    &request.destination.join("comparison-failure.json"),
                    &serde_json::to_vec_pretty(&report).map_err(fail)?,
                ) {
                    return Err(fail(format!(
                        "{error}; comparison evidence publication failed: {}",
                        output.kind()
                    )));
                }
                return Err(error);
            }
            timings.comparison += start.elapsed();
            if schedule.steps.get(next) == Some(&step) {
                let frame = writer.write(&snapshot, next)?;
                frames.push(frame);
                next += 1;
                let start = Instant::now();
                progress["frames"] = serde_json::to_value(&frames).map_err(fail)?;
                progress["accepted_steps"] = json!(step.to_string());
                write_progress(request.destination, &progress)?;
                timings.progress += start.elapsed();
            }
        }
        if mode == "saved" {
            let r = recorded
                .as_ref()
                .ok_or_else(|| fail("missing saved source"))?;
            for (index, step) in schedule.steps.iter().enumerate() {
                if cancelled() {
                    return Err(fail("export cancelled"));
                }
                let snapshot = r
                    .snapshots
                    .get(step)
                    .ok_or_else(|| fail("missing saved frame"))?;
                frames.push(writer.write(snapshot, index)?);
                let start = Instant::now();
                progress["frames"] = serde_json::to_value(&frames).map_err(fail)?;
                progress["accepted_steps"] = json!(step.to_string());
                write_progress(request.destination, &progress)?;
                timings.progress += start.elapsed();
            }
        }
        if frames.len() != schedule.steps.len() {
            return Err(fail("incomplete frame schedule"));
        }
        if cancelled() {
            return Err(fail("export cancelled before completion publication"));
        }
        graphics.check().map_err(fail)?;
        timings.frames = writer.timings;
        let timing_profile = timings.profile(started.elapsed())?;
        let manifest = json!({"schema_version":1,"complete":true,"source_mode":mode,"application_version":env!("CARGO_PKG_VERSION"),
            "build":{"commit":option_env!("MAELSTROM_EXPORT_COMMIT"),"dirty":option_env!("MAELSTROM_EXPORT_DIRTY"),"source_sha256":option_env!("MAELSTROM_EXPORT_SOURCE"),"lock_sha256":option_env!("MAELSTROM_EXPORT_LOCK"),"target":option_env!("MAELSTROM_EXPORT_TARGET"),"rustc":option_env!("MAELSTROM_EXPORT_RUSTC"),"debug_assertions":cfg!(debug_assertions),"profile":option_env!("MAELSTROM_EXPORT_PROFILE"),"opt_level":option_env!("MAELSTROM_EXPORT_OPT_LEVEL"),"rustflags_sha256":option_env!("MAELSTROM_EXPORT_CARGO_ENCODED_RUSTFLAGS"),"features":{"render":true,"viewer":cfg!(feature="viewer"),"cuda":cfg!(feature="cuda")}},
            "backend":RunMetadata::new(&input,last_step,options.backend,options.execution).backend,"execution":options.execution,"request":{"steps":last_step.to_string(),"checkpoints":schedule.steps.iter().map(u64::to_string).collect::<Vec<_>>()},"input":input,"effective_execution":options.execution,"discovery":format!("{:?}",options.backend),"run_metadata":metadata,"original":recorded.as_ref().map(|r|&r.provenance),
            "accepted_steps":last_step.to_string(),"status":if mode=="saved" {"saved_observations_rendered"} else if diagnostics.iter().any(|d|d.warning) {"completed_with_warnings"} else {"completed"},"frame_count":frames.len(),"step_stride":settings.step_stride.to_string(),"frames":frames,"render":settings,
            "boundary_display_box":{"lower_m":boundaries.lower_m,"upper_m":boundaries.upper_m},
            "renderer":{"revision":"opaque-spheres-lines-v2","shader_sha256":hash(include_bytes!("../render/particles.wgsl")),"format":"Rgba8UnormSrgb","samples":1,"overlays":false},
            "graphics":{"backend":format!("{:?}",graphics.adapter_info.backend),"name":graphics.adapter_info.name,"device_type":format!("{:?}",graphics.adapter_info.device_type),"vendor":graphics.adapter_info.vendor,"device":graphics.adapter_info.device,"driver":graphics.adapter_info.driver,"driver_info":graphics.adapter_info.driver_info,"selection_index":request.graphics.index,"software_requested":request.graphics.software},
            "encoder":{"crate":"png","version":"0.18.1","colour":"RGBA8 sRGB","compression":"Balanced","filter":"Adaptive"},
            "timing_seconds":{"solver_steps":timings.solver_steps.as_secs_f64(),"graphics_readback":timings.frames.graphics_readback.as_secs_f64(),"encoding":timings.frames.png_encoding.as_secs_f64()},"timing_profile":timing_profile,"accepted_diagnostics":diagnostics,
            "destination_label":request.destination.file_name().map(|s|s.to_string_lossy())});
        publish_bytes(
            &request.destination.join("manifest.json"),
            &serde_json::to_vec_pretty(&manifest).map_err(fail)?,
        )
        .map_err(|e| fail(format!("manifest publication: {}", e.kind())))?;
        Ok(manifest)
    })();
    if let Err(error) = &result {
        progress["failures"]
            .as_array_mut()
            .ok_or_else(|| fail("invalid progress"))?
            .push(json!(error.to_string()));
        if let Err(output) = write_progress(request.destination, &progress) {
            return Err(fail(format!(
                "{error}; recording incomplete outcome also failed: {output}"
            )));
        }
    }
    result
}

fn write_progress(directory: &Path, progress: &Value) -> Result<(), ExportError> {
    // Only our own progress file is replaced. Frame and manifest publication
    // always uses exclusive staging and hard links, retaining the existing policy.
    let temp = directory.join(".progress.tmp");
    publish_bytes(&temp, &serde_json::to_vec_pretty(progress).map_err(fail)?)
        .map_err(|e| fail(format!("progress staging: {}", e.kind())))?;
    fs::rename(temp, directory.join("progress.json"))
        .map_err(|e| fail(format!("progress update: {}", e.kind())))
}

#[cfg(test)]
mod tests;
