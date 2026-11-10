//! Cached windows share the live surface and camera controls, but own only a
//! bounded disk reader. No controller or numerical backend is constructed.
use super::*;
use crate::{
    cache::Cache,
    render::{BoundaryLines, ParticleRenderer},
    scene::Particle,
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Condvar, Mutex},
    thread,
};

type Frame = Arc<Vec<Particle>>;

// Opt-in diagnostics describe client submissions and CPU work, never scan-out.
fn trace(event: impl FnOnce() -> serde_json::Value) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("MAELSTROM_CACHE_TRACE").is_some()) {
        eprintln!("cache-trace {}", event());
    }
}
struct Reads {
    wanted: u64,
    frames: BTreeMap<u64, Frame>,
    error: Option<String>,
    stop: bool,
    latest: Option<(u64, Frame)>,
}
struct Reader {
    shared: Arc<(Mutex<Reads>, Condvar)>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Reader {
    fn new(cache: Arc<Cache>, budget_mb: usize) -> Result<Self, RenderError> {
        // Conservative reservation covers decoded records, particles and binary
        // workspace, including one superseded read and the displayed frame.
        let per_frame = cache
            .scene()
            .particles
            .len()
            .max(1)
            .checked_mul(256)
            .ok_or_else(|| RenderError("cache memory estimate overflow".into()))?;
        let budget = budget_mb
            .checked_mul(1024 * 1024)
            .ok_or_else(|| RenderError("cache memory budget overflow".into()))?;
        let slots = (budget / per_frame).saturating_sub(2).min(8);
        if slots < 2 {
            return Err(RenderError("cache memory budget must hold two read-ahead frames plus decode/display workspace; increase --cache-memory-mb".into()));
        }
        let shared = Arc::new((
            Mutex::new(Reads {
                wanted: 0,
                frames: BTreeMap::new(),
                error: None,
                stop: false,
                latest: None,
            }),
            Condvar::new(),
        ));
        let state = shared.clone();
        let worker = thread::Builder::new()
            .name("cache-playback-reader".into())
            .spawn(move || {
                loop {
                    let (lock, wake) = &*state;
                    let mut s = match lock.lock() {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    let next = loop {
                        if s.stop {
                            return;
                        }
                        let first = s.wanted;
                        let end = first.saturating_add(slots as u64).min(cache.frame_count());
                        s.frames.retain(|i, _| *i >= first && *i < end);
                        if let Some(i) = (first..end).find(|i| !s.frames.contains_key(i)) {
                            break i;
                        }
                        s = match wake.wait(s) {
                            Ok(s) => s,
                            Err(_) => return,
                        };
                    };
                    drop(s);
                    let started = Instant::now();
                    let loaded = cache.frame(next).map(|f| {
                        trace(|| serde_json::json!({"event":"read", "index":next, "timings":f.read_timings()}));
                        Arc::new(
                            f.snapshot()
                                .particles()
                                .iter()
                                .map(|p| p.particle())
                                .collect::<Vec<_>>(),
                        )
                    });
                    trace(|| serde_json::json!({"event":"loaded", "index":next, "seconds":started.elapsed().as_secs_f64()}));
                    let mut s = match lock.lock() {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    match loaded {
                        Ok(frame) => {
                            // Keep one completed read even if wall time has moved
                            // beyond it. Otherwise slow reads could starve playback
                            // by always completing after their target was superseded.
                            s.latest = Some((next, frame.clone()));
                            if next >= s.wanted && next < s.wanted.saturating_add(slots as u64) {
                                s.frames.insert(next, frame);
                            }
                        }
                        Err(e) => {
                            s.error = Some(e.to_string());
                            return;
                        }
                    }
                }
            })
            .map_err(|e| RenderError(format!("cache reader: {e}")))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }
    fn get(&self, index: u64) -> Result<Option<Frame>, RenderError> {
        Ok(self.request(index, index)?.map(|(_, frame)| frame))
    }
    fn request(&self, index: u64, minimum: u64) -> Result<Option<(u64, Frame)>, RenderError> {
        let (lock, wake) = &*self.shared;
        let mut s = lock
            .lock()
            .map_err(|_| RenderError("cache reader lock failed".into()))?;
        if let Some(e) = &s.error {
            return Err(RenderError(e.clone()));
        }
        s.wanted = index;
        let buffered = s
            .frames
            .range(minimum..=index)
            .next_back()
            .map(|(i, f)| (*i, f.clone()));
        let latest = s
            .latest
            .as_ref()
            .filter(|(i, _)| *i >= minimum && *i <= index)
            .cloned();
        let result = match (buffered, latest) {
            (Some(a), Some(b)) => Some(if a.0 >= b.0 { a } else { b }),
            (a, b) => a.or(b),
        };
        wake.notify_one();
        Ok(result)
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        if let Ok(mut s) = lock.lock() {
            s.stop = true;
            wake.notify_one();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Opens exact cached observations in the shared viewer surface. Disk reads
/// run on the background reader. Real-time playback may skip presentations;
/// frame-accurate playback waits for each exact observation.
#[allow(clippy::too_many_arguments)]
pub fn run_cache(
    cache: Cache,
    mut settings: Settings,
    selection: GraphicsSelection,
    fit: bool,
    paused: bool,
    budget_mb: usize,
    save: Option<PathBuf>,
    renderer: ParticleRenderer,
    policy: CachePlayback,
) -> Result<(), RenderError> {
    settings.validate()?;
    if cache.frame_count() == 0 {
        return Err(RenderError("cache contains no frames".into()));
    }
    #[cfg(target_os = "linux")]
    check_keyboard_libraries()?;
    let scene = cache.scene();
    let spacing = f64::from(scene.configuration.numerical.particle_spacing_m);
    let boundaries = BoundaryLines::new(&scene.particles, &scene.planes, spacing)?;
    if fit {
        settings.camera.fit_scene(
            &scene.particles,
            &scene.planes,
            spacing / 2.0,
            settings.width as f64 / settings.height as f64,
        )?;
    }
    let cache = Arc::new(cache);
    let reader = Reader::new(cache.clone(), budget_mb)?;
    let mut app = CachedWindow {
        cache,
        reader,
        settings,
        selection,
        boundaries,
        renderer,
        save,
        window: None,
        display: None,
        current: None,
        index: 0,
        wanted: 0,
        paused,
        looping: false,
        speed: 1.0,
        last: Instant::now(),
        remaining: 0.0,
        policy,
        position: 0.0,
        seeking: true,
        started: Instant::now(),
        cursor: [0.0; 2],
        drag: None,
        message: String::new(),
        seek: None,
        error: None,
    };
    create_event_loop()?
        .run_app(&mut app)
        .map_err(|e| window_error("cache viewer", e))?;
    app.error.map_or(Ok(()), Err)
}
struct CachedWindow {
    cache: Arc<Cache>,
    reader: Reader,
    settings: Settings,
    selection: GraphicsSelection,
    boundaries: BoundaryLines,
    renderer: ParticleRenderer,
    save: Option<PathBuf>,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    current: Option<Frame>,
    index: u64,
    wanted: u64,
    paused: bool,
    looping: bool,
    speed: f64,
    last: Instant,
    remaining: f64,
    policy: CachePlayback,
    position: f64,
    seeking: bool,
    started: Instant,
    cursor: [f64; 2],
    drag: Option<MouseButton>,
    message: String,
    seek: Option<String>,
    error: Option<RenderError>,
}
impl CachedWindow {
    fn time(&self, i: u64) -> f64 {
        self.cache.step(i).unwrap_or(0) as f64
            * f64::from(self.cache.scene().configuration.numerical.fixed_timestep_s)
    }
    fn tick_realtime(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        if self.paused || self.seeking || self.current.is_none() {
            return;
        }
        let last = self.cache.frame_count() - 1;
        let terminal = self.time(last);
        let hold = if last > 0 {
            terminal - self.time(last - 1)
        } else {
            1.0 / 60.0
        };
        self.paused = super::cache_clock::advance(
            &mut self.position,
            elapsed,
            self.speed,
            self.looping,
            terminal,
            hold,
        );
        self.wanted =
            super::cache_clock::target(self.cache.frame_count(), self.position, |i| self.time(i));
    }
    fn seek(&mut self, i: u64) {
        self.wanted = i.min(self.cache.frame_count() - 1);
        self.remaining = 0.0;
        self.position = self.time(self.wanted);
        self.seeking = true;
        self.last = Instant::now();
    }
    fn save(&mut self) -> Result<(), RenderError> {
        let path = self.save.as_ref().ok_or_else(|| {
            RenderError("select --camera-output FILE to save presentation".into())
        })?;
        self.settings.validate()?;
        let presentation = crate::presentation::Presentation {
            schema_version: 2,
            settings: self.settings.clone(),
            camera_mode: crate::presentation::CameraMode::Fixed,
            supplied_fields: vec![],
        };
        let text = presentation
            .to_toml()
            .map_err(|e| RenderError(e.to_string()))?;
        crate::presentation::Presentation::parse(&text)?;
        crate::workflow::publish_bytes(path, text.as_bytes())
            .map_err(|e| RenderError(e.to_string()))?;
        self.message = format!("Saved {} (new files only)", path.display());
        Ok(())
    }
    fn key(&mut self, key: &Key, event_loop: &ActiveEventLoop) -> Result<(), RenderError> {
        trace(
            || serde_json::json!({"event":"key", "wall_s":self.started.elapsed().as_secs_f64(), "key":format!("{key:?}")}),
        );
        // Settle elapsed time at the old speed/pause state before controls change it.
        if self.policy == CachePlayback::RealTime {
            self.tick_realtime(Instant::now());
        }
        if let Some(text) = &mut self.seek {
            match key {
                Key::Named(NamedKey::Escape) => self.seek = None,
                Key::Named(NamedKey::Backspace) => {
                    text.pop();
                }
                Key::Named(NamedKey::Enter) => {
                    let i = text
                        .parse::<u64>()
                        .map_err(|_| RenderError("enter a cache-frame index".into()))?;
                    if i >= self.cache.frame_count() {
                        return Err(RenderError("frame outside cache".into()));
                    }
                    self.seek = None;
                    self.paused = true;
                    self.seek(i);
                }
                Key::Character(s) if s.chars().all(|c| c.is_ascii_digit()) && text.len() < 20 => {
                    text.push_str(s)
                }
                _ => {}
            }
            return Ok(());
        }
        match key {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::Space) => {
                self.paused = !self.paused;
                self.last = Instant::now();
            }
            Key::Named(NamedKey::Home) => self.seek(0),
            Key::Named(NamedKey::End) => {
                self.paused = true;
                self.seek(self.cache.frame_count() - 1);
            }
            Key::Named(NamedKey::ArrowLeft) => self.settings.camera.orbit(-0.1, 0.0)?,
            Key::Named(NamedKey::ArrowRight) => self.settings.camera.orbit(0.1, 0.0)?,
            Key::Named(NamedKey::ArrowUp) => self.settings.camera.orbit(0.0, 0.1)?,
            Key::Named(NamedKey::ArrowDown) => self.settings.camera.orbit(0.0, -0.1)?,
            Key::Character(s) => match s.to_lowercase().as_str() {
                "n" | "." => {
                    self.paused = true;
                    self.seek(self.wanted.saturating_add(1));
                }
                "b" | "," => {
                    self.paused = true;
                    self.seek(self.wanted.saturating_sub(1));
                }
                "r" => self.seek(0),
                "l" => self.looping = !self.looping,
                "j" => self.seek = Some(String::new()),
                "t" => {
                    self.policy = if self.policy == CachePlayback::RealTime {
                        CachePlayback::FrameAccurate
                    } else {
                        CachePlayback::RealTime
                    };
                    self.seek(self.index);
                }
                "s" => self.save()?,
                "[" => self.speed = (self.speed / 2.0).max(0.0625),
                "]" => self.speed = (self.speed * 2.0).min(16.0),
                "+" | "=" => self.settings.camera.zoom(0.9)?,
                "-" => self.settings.camera.zoom(1.1)?,
                "f" => {
                    let scene = self.cache.scene();
                    self.settings.camera.fit_scene(
                        &scene.particles,
                        &scene.planes,
                        f64::from(scene.configuration.numerical.particle_spacing_m) / 2.0,
                        self.settings.width as f64 / self.settings.height as f64,
                    )?;
                }
                "p" => {
                    self.renderer = if self.renderer == ParticleRenderer::Indexed {
                        ParticleRenderer::Impostor
                    } else {
                        ParticleRenderer::Indexed
                    };
                }
                _ => {}
            },
            _ => {}
        }
        Ok(())
    }
    fn update(&mut self, now: Instant) -> Result<(), RenderError> {
        let elapsed = now.duration_since(self.last).as_secs_f64();
        if self.policy == CachePlayback::RealTime {
            self.tick_realtime(now);
        } else {
            self.last = now;
        }
        let was_loading = self.current.is_none() || self.index != self.wanted;
        let had_frame = self.current.is_some();
        let minimum = if self.policy == CachePlayback::RealTime && !self.seeking {
            if self.wanted >= self.index {
                self.index
            } else {
                0
            }
        } else {
            self.wanted
        };
        let loaded = if minimum == self.wanted {
            self.reader
                .get(self.wanted)?
                .map(|frame| (self.wanted, frame))
        } else {
            self.reader.request(self.wanted, minimum)?
        };
        let ready = loaded.is_some();
        if let Some((index, frame)) = loaded
            && (was_loading || index != self.index || self.seeking)
        {
            self.current = Some(frame);
            self.index = index;
            if self.seeking {
                self.seeking = false;
                self.last = now;
            }
            self.remaining += if self.index + 1 < self.cache.frame_count() {
                self.time(self.index + 1) - self.time(self.index)
            } else if self.index > 0 {
                self.time(self.index) - self.time(self.index - 1)
            } else {
                1.0 / 60.0
            };
        }
        if self.policy == CachePlayback::FrameAccurate && !self.paused && ready && had_frame {
            self.remaining -= elapsed * self.speed;
            if self.remaining <= 0.0 {
                if self.index + 1 < self.cache.frame_count() {
                    self.wanted = self.index + 1;
                } else if self.looping {
                    self.wanted = 0;
                } else {
                    self.paused = true;
                }
            }
        }
        Ok(())
    }
    fn redraw(&mut self) -> Result<(), RenderError> {
        self.update(Instant::now())?;
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| RenderError("window unavailable".into()))?;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.settings.width = size.width;
        self.settings.height = size.height;
        let loading = self.current.is_none() || self.wanted != self.index;
        let title = format!(
            "Maelstrom cache | frame {}/{} | {:.6} simulation s | {} {}x | {} | {:?} | {:?}",
            self.index,
            self.cache.frame_count() - 1,
            self.time(self.index),
            if self.paused { "paused" } else { "playing" },
            self.speed,
            if loading { "loading" } else { "ready" },
            self.renderer,
            self.policy
        );
        window.set_title(&title);
        let dt = f64::from(self.cache.scene().configuration.numerical.fixed_timestep_s);
        let lines = vec![
            title,
            "Space play/pause | N/B step | J frame index | R/Home restart | End last | L loop"
                .into(),
            "Drag orbit | right-drag pan | wheel zoom | S save | P geometry | T timing | [ ] speed"
                .into(),
            format!(
                "Recorded solver {:.3} Hz | cache uses recorded step gaps | display follows surface | loop {}",
                1.0 / dt,
                self.looping
            ),
            format!(
                "Jump: {} | {}",
                self.seek.as_deref().unwrap_or("press J"),
                self.message
            ),
        ];
        let mut overlay = overlay::panel(&lines, size.width, size.height, 1.5);
        super::overlay::scrubber(
            &mut overlay,
            size.width,
            size.height,
            self.wanted as f64 / (self.cache.frame_count() - 1).max(1) as f64,
        );
        let display = self
            .display
            .as_mut()
            .ok_or_else(|| RenderError("display unavailable".into()))?;
        display.graphics.set_particle_renderer(self.renderer);
        let frame = match display.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                display.resize(size.width, size.height)?;
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RenderError("cache surface validation failed".into()));
            }
        };
        let scene = self.cache.scene();
        let draw = display.graphics.draw_timed(
            &frame.texture.create_view(&Default::default()),
            &display.depth.create_view(&Default::default()),
            &DrawScene {
                particles: self.current.as_deref().map_or(&[], |p| p.as_slice()),
                boundaries: &self.boundaries,
                radius_m: f64::from(scene.configuration.numerical.particle_spacing_m) / 2.0,
                camera: &self.settings.camera,
                colours: &self.settings.particles,
                particle_mass_kg: scene.configuration.derived.particle_mass_kg,
                width: size.width,
                height: size.height,
            },
            &overlay,
        )?;
        window.pre_present_notify();
        display.graphics.queue.present(frame);
        trace(
            || serde_json::json!({"event":"present", "wall_s":self.started.elapsed().as_secs_f64(), "index":self.index, "wanted":self.wanted, "simulation_s":self.time(self.index), "timeline_s":self.position, "paused":self.paused, "speed":self.speed, "loading":loading, "camera":self.settings.camera, "width":size.width, "height":size.height, "policy":format!("{:?}",self.policy), "prepare_s":draw.prepare_s, "upload_s":draw.upload_s, "submit_s":draw.submit_s}),
        );
        Ok(())
    }
}
impl ApplicationHandler for CachedWindow {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = (|| {
            if self.window.is_none() {
                self.window = Some(Arc::new(
                    event_loop
                        .create_window(
                            Window::default_attributes()
                                .with_title("Maelstrom cache")
                                .with_inner_size(winit::dpi::PhysicalSize::new(
                                    self.settings.width,
                                    self.settings.height,
                                )),
                        )
                        .map_err(|e| window_error("cache window", e))?,
                ));
            }
            if self.display.is_none() {
                self.display = Some(Display::new(
                    self.window
                        .as_ref()
                        .ok_or_else(|| RenderError("window unavailable".into()))?
                        .clone(),
                    self.selection,
                    event_loop.owned_display_handle(),
                )?);
                trace(
                    || serde_json::json!({"event":"adapter", "features":format!("{:?}",self.display.as_ref().map(|d| d.graphics.adapter.features()))}),
                );
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.error = Some(e);
            event_loop.exit();
        }
    }
    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.display = None;
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let result = match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                Ok(())
            }
            WindowEvent::Resized(s) => self
                .display
                .as_mut()
                .map_or(Ok(()), |d| d.resize(s.width, s.height)),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.key(&event.logical_key, event_loop)
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.drag = if state == ElementState::Pressed {
                    Some(button)
                } else {
                    None
                };
                Ok(())
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.cursor[0];
                let dy = position.y - self.cursor[1];
                self.cursor = [position.x, position.y];
                // The bottom strip is a continuous scrub control over cached indices.
                if self.drag == Some(MouseButton::Left)
                    && self
                        .window
                        .as_ref()
                        .is_some_and(|w| position.y >= w.inner_size().height as f64 - 30.0)
                {
                    if let Some(w) = &self.window {
                        let f = (position.x / w.inner_size().width.max(1) as f64).clamp(0.0, 1.0);
                        self.paused = true;
                        self.seek((f * (self.cache.frame_count() - 1) as f64).round() as u64);
                    }
                    Ok(())
                } else {
                    match self.drag {
                        Some(MouseButton::Left) => {
                            self.settings.camera.orbit(-dx * 0.006, dy * 0.006)
                        }
                        Some(MouseButton::Right) => {
                            self.settings.camera.pan(-dx * 0.001, dy * 0.001)
                        }
                        _ => Ok(()),
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 100.0,
                };
                self.settings.camera.zoom((-y * 0.1).exp())
            }
            WindowEvent::RedrawRequested => {
                if self.display.is_some()
                    && let Err(e) = self.redraw()
                {
                    self.error = Some(e);
                    event_loop.exit();
                }
                Ok(())
            }
            _ => Ok(()),
        };
        if let Err(e) = result {
            self.message = e.to_string();
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(16),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn inspection_window() -> (CachedWindow, PathBuf) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "maelstrom-cache-clock-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let scene = crate::scene::load_resolved_scene(
            std::path::Path::new("reference/small-lattice.toml"),
            None,
        )
        .unwrap();
        crate::cache::bake(
            crate::cache::BakeRequest {
                scene: &scene,
                steps: 6,
                stride: 2,
                options: Default::default(),
                destination: &root,
            },
            || false,
            |_, _| {},
        )
        .unwrap();
        let cache = Arc::new(Cache::open(&root, false).unwrap());
        let mut app = CachedWindow {
            reader: Reader::new(cache.clone(), 1).unwrap(),
            boundaries: BoundaryLines::new(&scene.particles, &scene.planes, 0.1).unwrap(),
            cache,
            settings: Settings::default(),
            selection: GraphicsSelection::default(),
            renderer: ParticleRenderer::Indexed,
            save: None,
            window: None,
            display: None,
            current: None,
            index: 0,
            wanted: 0,
            paused: true,
            looping: false,
            speed: 1.,
            last: Instant::now(),
            remaining: 0.,
            policy: CachePlayback::RealTime,
            position: 0.,
            seeking: true,
            started: Instant::now(),
            cursor: [0.; 2],
            drag: None,
            message: String::new(),
            seek: None,
            error: None,
        };
        wait_for_seek(&mut app);
        (app, root)
    }
    fn wait_for_seek(app: &mut CachedWindow) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.seeking {
            app.update(Instant::now()).unwrap();
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn same_frame_seek_reanchors_and_paused_inspection_stays_exact() {
        let (mut app, root) = inspection_window();
        for i in [0, 3, 2, 0, 0] {
            app.seek(i);
            wait_for_seek(&mut app);
            assert_eq!(app.index, i);
            assert_eq!(app.position, app.time(i));
            let before = app.position;
            app.tick_realtime(app.last + Duration::from_secs(100));
            assert_eq!(app.position, before);
        }
        drop(app);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn frame_accurate_requests_only_the_next_state_after_a_slow_redraw() {
        let (mut app, root) = inspection_window();
        app.policy = CachePlayback::FrameAccurate;
        app.paused = false;
        app.speed = 16.;
        app.update(app.last + Duration::from_secs(1)).unwrap();
        assert_eq!(app.index, 0);
        assert_eq!(app.wanted, 1);
        drop(app);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn realtime_pause_resume_and_speed_change_keep_fractional_position() {
        let (mut app, root) = inspection_window();
        app.paused = false;
        let start = app.last;
        app.tick_realtime(start + Duration::from_millis(1));
        assert!((app.position - 0.001).abs() < 1e-12);
        app.speed = 2.;
        app.tick_realtime(start + Duration::from_millis(2));
        assert!((app.position - 0.003).abs() < 1e-12);
        app.paused = true;
        app.tick_realtime(start + Duration::from_secs(1));
        assert!((app.position - 0.003).abs() < 1e-12);
        app.paused = false;
        app.tick_realtime(start + Duration::from_millis(1001));
        assert!((app.position - 0.005).abs() < 1e-12);
        drop(app);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn superseded_read_can_progress_realtime_but_never_satisfies_an_exact_seek() {
        let particles = Arc::new(Vec::new());
        let reader = Reader {
            shared: Arc::new((
                Mutex::new(Reads {
                    wanted: 0,
                    frames: BTreeMap::new(),
                    error: None,
                    stop: false,
                    latest: Some((3, particles)),
                }),
                Condvar::new(),
            )),
            worker: None,
        };
        assert_eq!(reader.request(10, 0).unwrap().unwrap().0, 3);
        assert!(reader.get(10).unwrap().is_none());
        assert!(reader.request(10, 4).unwrap().is_none());
        assert!(reader.request(2, 0).unwrap().is_none());
    }
    #[test]
    fn bounded_reader_seeks_exact_states_and_reports_corruption() {
        use crate::cache::{BakeRequest, bake};
        let root =
            std::env::temp_dir().join(format!("maelstrom-cached-viewer-{}", std::process::id()));
        let scene = crate::scene::load_resolved_scene(
            std::path::Path::new("reference/small-lattice.toml"),
            None,
        )
        .unwrap();
        bake(
            BakeRequest {
                scene: &scene,
                steps: 6,
                stride: 2,
                options: Default::default(),
                destination: &root,
            },
            || false,
            |_, _| {},
        )
        .unwrap();
        let cache = Arc::new(Cache::open(&root, false).unwrap());
        assert!(Reader::new(cache.clone(), 0).is_err());
        let reader = Reader::new(cache.clone(), 1).unwrap();
        for index in [0, 3, 1, 2, 0] {
            let deadline = Instant::now() + Duration::from_secs(5);
            let loaded = loop {
                if let Some(f) = reader.get(index).unwrap() {
                    break f;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            };
            let expected: Vec<_> = cache
                .frame(index)
                .unwrap()
                .snapshot()
                .particles()
                .iter()
                .map(|p| p.particle())
                .collect();
            assert_eq!(*loaded, expected);
            assert!(reader.shared.0.lock().unwrap().frames.len() <= 8);
        }
        drop(reader);
        std::fs::write(root.join("frames/frame-00000003.mpc"), b"corrupt").unwrap();
        let reader = Reader::new(cache.clone(), 1).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match reader.get(3) {
                Err(_) => break,
                _ => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
        drop(reader);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn portable_presentation_preserves_interactive_camera_and_colours_exactly() {
        use crate::presentation::{CameraMode, ParticleColours, Presentation};
        let mut settings = Settings {
            schema_version: 2,
            ..Settings::default()
        };
        settings.camera.orbit(0.37, -0.12).unwrap();
        settings.camera.pan(0.1, -0.05).unwrap();
        settings.camera.zoom(0.8).unwrap();
        settings.particles = ParticleColours::KineticEnergy {
            min_energy_j: 0.001,
            max_energy_j: 0.123,
        };
        let original = serde_json::to_value(&settings).unwrap();
        let p = Presentation {
            schema_version: 2,
            settings,
            camera_mode: CameraMode::Fixed,
            supplied_fields: vec![],
        };
        let loaded = Presentation::parse(&p.to_toml().unwrap()).unwrap();
        assert_eq!(loaded.camera_mode, CameraMode::Fixed);
        assert_eq!(serde_json::to_value(loaded.settings).unwrap(), original);
    }
}
