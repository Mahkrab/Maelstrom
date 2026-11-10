//! Cached rendering owns graphics and image output only. Typed, checked frames
//! supply canonical provenance without serialising the same particles again.
use super::*;
use crate::cache::Cache;

/// Inclusive selection in cache-frame indices, independent of solver timesteps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheSelection {
    /// First cached frame to render.
    pub first: u64,
    /// Last cached frame to render (inclusive).
    pub last: u64,
    /// Positive increment in cache-frame indices. A non-matching last index is not appended.
    pub every: u64,
}
impl CacheSelection {
    pub(crate) fn count(self, available: u64) -> Result<u64, ExportError> {
        if self.every == 0 || self.first > self.last || self.last >= available {
            return Err(fail("invalid cached frame range"));
        }
        Ok((self.last - self.first) / self.every + 1)
    }
}

/// Presentation request over already validated cache metadata.
pub struct CachedRenderRequest<'a> {
    /// Complete cache, or an explicitly opened partial prefix.
    pub cache: &'a Cache,
    /// Presentation only; its step_stride does not resample the cache.
    pub settings: Settings,
    /// Independent graphics selection, never a CUDA device request.
    pub graphics: GraphicsSelection,
    /// Fit once against the frozen initial geometry.
    pub fit_camera: bool,
    /// Inclusive cached-frame range.
    pub selection: CacheSelection,
    /// Required new image sequence directory.
    pub destination: &'a Path,
}

/// Renders cached states without constructing a simulation or running fluid steps.
/// Schema-2 image manifests describe sparse schedules explicitly; schema 1 is unchanged.
pub fn render_cache(
    request: CachedRenderRequest<'_>,
    mut cancelled: impl FnMut() -> bool,
    mut notify: impl FnMut(usize, usize),
) -> Result<Value, ExportError> {
    let started = Instant::now();
    let scene = request.cache.scene();
    let count = request.selection.count(request.cache.frame_count())?;
    let last_step = request
        .cache
        .step(request.selection.last)
        .ok_or_else(|| fail("cached frame is unavailable"))?;
    if last_step as f64 * f64::from(scene.configuration.numerical.fixed_timestep_s) > 1e12 {
        return Err(fail("cached timestamp exceeds image playback limit"));
    }
    let mut settings = request.settings;
    settings.validate().map_err(fail)?;
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
    if cancelled() {
        return Err(fail("cached rendering cancelled"));
    }
    match fs::symlink_metadata(request.destination) {
        Ok(_) => return Err(fail("output destination already exists")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(fail(e)),
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
    fs::create_dir(request.destination).map_err(fail)?;
    let mut progress = json!({"schema_version":2,"complete":false,"source_mode":"cache","source_cache":request.cache.info(),"frames":[],"failures":[]});
    write_progress(request.destination, &progress)?;
    let mut timings = timing::ExportTimings {
        setup: started.elapsed(),
        ..Default::default()
    };
    let mut read_validate = std::time::Duration::ZERO;
    let mut cache_reading = 0.0;
    let mut cache_validation = 0.0;
    let mut cache_decoding = 0.0;
    let result = (|| {
        let mut writer = offscreen::FrameWriter {
            operations: offscreen::FrameOperations::default(),
            graphics: &graphics,
            settings: &settings,
            boundaries: &boundaries,
            spacing,
            mass_kg: scene.configuration.derived.particle_mass_kg,
            dt: f64::from(scene.configuration.numerical.fixed_timestep_s),
            destination: request.destination,
            timings: Default::default(),
            worker: encoding::Worker::new()?,
        };
        let mut frames = Vec::with_capacity(count as usize);
        notify(0, count as usize);
        for index in 0..count {
            if cancelled() {
                return Err(fail("cached rendering cancelled"));
            }
            let source_index = request.selection.first + index * request.selection.every;
            let start = Instant::now();
            let frame = request.cache.frame(source_index).map_err(fail)?;
            read_validate += start.elapsed();
            let read = frame.read_timings();
            cache_reading += read.reading_s;
            cache_validation += read.binary_validation_s;
            cache_decoding += read.decoding_s;
            let prepared = writer.prepare_cached(frame, index as usize);
            if cancelled() {
                return Err(fail("cached rendering cancelled"));
            }
            collect_frame_cancellable(
                &mut writer,
                &mut frames,
                &mut progress,
                &mut timings.progress,
                &mut cancelled,
            )?;
            writer.submit(prepared?)?;
            if index == 0 {
                collect_frame_cancellable(
                    &mut writer,
                    &mut frames,
                    &mut progress,
                    &mut timings.progress,
                    &mut cancelled,
                )?;
            }
            notify(frames.len(), count as usize);
        }
        if cancelled() {
            return Err(fail("cached rendering cancelled"));
        }
        collect_frame_cancellable(
            &mut writer,
            &mut frames,
            &mut progress,
            &mut timings.progress,
            &mut cancelled,
        )?;
        if cancelled() {
            return Err(fail("cached rendering cancelled before publication"));
        }
        graphics.check().map_err(fail)?;
        notify(frames.len(), count as usize);
        timings.frames = writer.timings;
        let cache = request.cache.info();
        let manifest = json!({"schema_version":2,"kind":"maelstrom-cache-render","complete":true,"source_mode":"cache","status":"cached_observations_rendered","application_version":env!("CARGO_PKG_VERSION"),"build":crate::cache::build_identity(),"source_cache":cache,
            "cache_selection":request.selection,"timestep_bits":scene.configuration.numerical.fixed_timestep_s.to_bits(),"frame_count":frames.len(),"frames":frames,"render":settings,
            "input":request.cache.input,"backend":cache["metadata"]["backend"],"execution":cache["execution"],
            "request":{"steps":cache["steps"],"checkpoints":[]},"accepted_steps":request.cache.step(request.cache.frame_count()-1).map(|s|s.to_string()),"step_stride":cache["stride"],
            "renderer":{"revision":"opaque-spheres-lines-v2","shader_sha256":hash(include_bytes!("../render/particles.wgsl")),"format":"Rgba8UnormSrgb","samples":1,"overlays":false},
            "graphics":{"backend":format!("{:?}",graphics.adapter_info.backend),"name":graphics.adapter_info.name,"driver":graphics.adapter_info.driver,"driver_info":graphics.adapter_info.driver_info,"device_type":format!("{:?}",graphics.adapter_info.device_type),"vendor":graphics.adapter_info.vendor,"device":graphics.adapter_info.device,"software_requested":request.graphics.software,"selection_index":request.graphics.index},
            "encoder":{"crate":"png","version":"0.18.1","colour":"RGBA8 sRGB","compression":"Balanced","filter":"Adaptive"},
            "boundary_display_box":{"lower_m":boundaries.lower_m,"upper_m":boundaries.upper_m},
            "timing_seconds":{"solver_steps":0.0,"cache_read_validate":read_validate.as_secs_f64(),"cache_reading":cache_reading,"cache_binary_validation":cache_validation,"cache_decoding":cache_decoding,"cache_canonical_rehashing":0.0,"graphics_readback":timings.frames.graphics_readback.as_secs_f64(),"encoding":timings.frames.png_encoding.as_secs_f64()},"timing_profile":timings.profile(started.elapsed())?});
        publish_bytes(
            &request.destination.join("manifest.json"),
            &serde_json::to_vec_pretty(&manifest).map_err(fail)?,
        )
        .map_err(fail)?;
        Ok(manifest)
    })();
    if let Err(error) = &result {
        progress["failures"] = json!([error.to_string()]);
        if let Err(output) = write_progress(request.destination, &progress) {
            return Err(fail(format!(
                "{error}; recording incomplete outcome also failed: {output}"
            )));
        }
    }
    result
}
