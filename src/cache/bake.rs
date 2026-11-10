//! Cache production observes the unchanged solver at scheduled accepted indices.
//! Diagnostics stream to disk; only bounded immutable particle copies are queued.
use super::*;
use crate::{
    scene::ResolvedScene,
    simulation::Simulation,
    workflow::{ExecutionOptions, RecordedInput, RunMetadata, SnapshotRecord, StepDiagnostic},
};
use std::{
    io::{BufWriter, Write},
    time::Instant,
};

/// A fresh simulation and its required new particle-cache directory.
pub struct BakeRequest<'a> {
    /// Fully resolved numerical inputs; no presentation values enter the solver.
    pub scene: &'a ResolvedScene,
    /// Requested accepted timesteps, starting from zero.
    pub steps: u64,
    /// Output frequency in accepted steps, always including both endpoints.
    pub stride: u64,
    /// Explicit simulation resources and discovery selection.
    pub options: ExecutionOptions,
    /// Exclusive new destination; its parent must already exist.
    pub destination: &'a Path,
}

/// Simulates once, asynchronously prepares frames, and transactionally closes the cache.
/// On error, committed receipts identify the retained prefix. No PNG work occurs.
pub fn bake(
    request: BakeRequest<'_>,
    mut cancelled: impl FnMut() -> bool,
    mut progress: impl FnMut(u64, u64),
) -> Result<serde_json::Value, CacheError> {
    let started = Instant::now();
    let schedule = Schedule::new(request.steps, request.stride)?;
    let dt = f64::from(request.scene.configuration.numerical.fixed_timestep_s);
    if !((request.steps as f64) * dt).is_finite()
        || request.scene.particles.len() > crate::scene::MAX_SCENE_PARTICLES
    {
        return Err(invalid("cache input exceeds supported limits"));
    }
    if cancelled() {
        return Err(CacheError::Cancelled);
    }
    match fs::symlink_metadata(request.destination) {
        Ok(_) => return Err(invalid("cache destination already exists")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let mut solver = Simulation::from_scene_with_execution(
        request.scene,
        request.options.backend,
        request.options.execution,
    )
    .map_err(invalid)?;
    let input = RecordedInput::capture(request.scene, &solver);
    let input_bytes = serde_json::to_vec_pretty(&input)?;
    let identities: Vec<_> = solver
        .snapshot()
        .particles()
        .iter()
        .flat_map(|p| p.identity.to_le_bytes())
        .collect();
    let mut metadata = RunMetadata::new(
        &input,
        request.steps,
        request.options.backend,
        request.options.execution,
    );
    metadata.device = solver.device().cloned();
    let specification = Request {
        schema_version: 1,
        kind: "maelstrom-particle-cache".into(),
        steps: request.steps.to_string(),
        stride: request.stride.to_string(),
        particle_count: solver.snapshot().particles().len(),
        timestep_bits: request
            .scene
            .configuration
            .numerical
            .fixed_timestep_s
            .to_bits(),
        input_sha256: digest(&input_bytes),
        identities_sha256: digest(&identities),
        execution: request.options.execution,
        backend: metadata.backend.clone(),
        metadata: serde_json::to_value(metadata)?,
        build: build_identity(),
    };
    let bytes = serde_json::to_vec_pretty(&specification)?;
    let request_hash = digest(&bytes);
    fs::create_dir(request.destination)?;
    fs::create_dir(request.destination.join("frames"))?;
    fs::create_dir(request.destination.join("receipts"))?;
    publish(&request.destination.join("input.json"), &input_bytes)?;
    publish(&request.destination.join("identities.bin"), &identities)?;
    publish(&request.destination.join("request.json"), &bytes)?;
    sync_directory(request.destination)?;
    drop(input_bytes);
    drop(identities);
    drop(input);
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(request.destination.join("diagnostics.jsonl"))?;
    let mut diagnostics_file = BufWriter::with_capacity(64 * 1024, file);
    let mut diagnostics_hash = Sha256::new();
    let mut writer = writer::Writer::new(request.destination, request_hash.clone())?;
    let mut solver_s = 0.0;
    let mut capture_s = 0.0;
    let mut diagnostics_s = 0.0;
    let mut warnings = 0;
    let mut accepted = 0;
    let setup_s = started.elapsed().as_secs_f64();
    progress(0, schedule.len());
    let result = (|| {
        let mut next = 0;
        for step in 0..=request.steps {
            if cancelled() {
                return Err(CacheError::Cancelled);
            }
            if step > 0 {
                let start = Instant::now();
                let outcome = solver
                    .step_with_cancellation(&mut cancelled)
                    .map_err(invalid)?;
                solver_s += start.elapsed().as_secs_f64();
                accepted = step;
                let start = Instant::now();
                let diagnostic = StepDiagnostic::new(step, outcome.cfl);
                warnings += u64::from(diagnostic.warning);
                let mut bytes = serde_json::to_vec(&diagnostic)?;
                bytes.push(b'\n');
                diagnostics_file.write_all(&bytes)?;
                diagnostics_hash.update(&bytes);
                diagnostics_s += start.elapsed().as_secs_f64();
            }
            if schedule.step(next) == Some(step) {
                writer.collect(&mut cancelled)?;
                progress(writer.count, schedule.len());
                let start = Instant::now();
                let snapshot = SnapshotRecord::capture(solver.snapshot());
                capture_s += start.elapsed().as_secs_f64();
                writer.submit(snapshot, next)?;
                next += 1;
                if step == 0 {
                    writer.collect(&mut cancelled)?;
                    progress(writer.count, schedule.len());
                }
            }
        }
        writer.collect(&mut cancelled)?;
        progress(writer.count, schedule.len());
        if cancelled() {
            return Err(CacheError::Cancelled);
        }
        if writer.count != schedule.len() {
            return Err(invalid("incomplete cache frame schedule"));
        }
        diagnostics_file.flush()?;
        diagnostics_file.get_ref().sync_all()?;
        let receipt_hash = writer.chain();
        let diagnostic_hash = hex(diagnostics_hash.clone().finalize().as_slice());
        let timings = serde_json::json!({"schema_version":1,"boundary":"bake_entry_to_before_completion_publication","setup_s":setup_s,"solver_s":solver_s,"capture_s":capture_s,"diagnostics_s":diagnostics_s,"worker":writer.timings,"worker_overlaps_simulation":true,"elapsed_s":started.elapsed().as_secs_f64(),"max_pending_particle_frames":1});
        let completion = Completion {
            schema_version: 1,
            complete: true,
            request_sha256: request_hash.clone(),
            frames: writer.count,
            accepted_steps: accepted.to_string(),
            warnings,
            cache_id: result_id(&request_hash, &receipt_hash, &diagnostic_hash),
            receipts_sha256: receipt_hash,
            diagnostics_sha256: diagnostic_hash,
            timings,
        };
        publish(
            &request.destination.join("cache.json"),
            &serde_json::to_vec_pretty(&completion)?,
        )?;
        sync_directory(request.destination)?;
        serde_json::to_value(completion).map_err(Into::into)
    })();
    if let Err(error) = &result {
        // Drop joins the writer before any function return. A prepared frame
        // without its receipt is an orphan, never part of the committed prefix.
        let _ = diagnostics_file.flush();
        let failure = serde_json::json!({"schema_version":1,"complete":false,"accepted_steps":accepted.to_string(),"committed_frames":writer.count,"error":error.to_string()});
        if let Err(output) = publish(
            &request.destination.join("failure.json"),
            &serde_json::to_vec_pretty(&failure)?,
        ) {
            return Err(invalid(format!(
                "{error}; recording cache failure also failed: {output}"
            )));
        }
    }
    result
}
