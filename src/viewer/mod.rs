//! The main-thread window adapts events to a non-blocking presentation model.
//! Solver resources stay on the controller worker; graphics failures request
//! cooperative shutdown and keep pumping events until that worker has exited.

mod model;
mod overlay;
mod windowing;
pub(crate) use windowing::create as create_event_loop;
#[cfg(test)]
mod tests;
use crate::{
    application::Action,
    render::{DrawScene, Graphics, GraphicsSelection, RenderError, Settings},
};
use model::Model;
pub use model::{Draft, PresentationResolver, Resolver, ViewPresentation};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

/// Launches a viewer on the calling main thread. The resolver reuses the
/// owning application's physical-input loading and validation rules.
pub fn run(
    draft: Draft,
    settings: Settings,
    graphics: GraphicsSelection,
    fit_camera: bool,
    resolver: Resolver,
) -> Result<(), RenderError> {
    run_internal(draft, settings, graphics, fit_camera, resolver, None)
}

/// Launches with scene-owned presentation and camera-save destinations resolved
/// on each successful apply. Numerical loading remains owned by `resolver`.
pub fn run_with_presentation(
    draft: Draft,
    settings: Settings,
    graphics: GraphicsSelection,
    fit_camera: bool,
    resolver: Resolver,
    presentation: PresentationResolver,
) -> Result<(), RenderError> {
    run_internal(
        draft,
        settings,
        graphics,
        fit_camera,
        resolver,
        Some(presentation),
    )
}

fn run_internal(
    draft: Draft,
    settings: Settings,
    graphics: GraphicsSelection,
    fit_camera: bool,
    resolver: Resolver,
    presentation: Option<PresentationResolver>,
) -> Result<(), RenderError> {
    settings.validate()?;
    #[cfg(target_os = "linux")]
    check_keyboard_libraries()?;
    let event_loop = create_event_loop()?;
    let mut model = Model::new(draft, settings.camera.clone(), resolver)?;
    model.presentation_resolver = presentation;
    if let Err(e) = model.apply() {
        model.message = e.to_string();
    }
    if fit_camera && model.presentation_resolver.is_none() {
        model.fit(f64::from(settings.width) / f64::from(settings.height))?;
    }
    let mut app = Viewer::new(model, settings, graphics);
    let event_result = event_loop
        .run_app(&mut app)
        .map_err(|e| window_error("window event loop", e));
    // Event-loop failure still requests and joins worker teardown.
    app.model.request_shutdown();
    let joined = app.model.join();
    event_result?;
    joined?;
    if let Some(error) = app.error {
        return Err(error);
    }
    Ok(())
}
pub(crate) fn window_error(context: &str, error: impl std::fmt::Display) -> RenderError {
    let message = error.to_string();
    let message = message
        .strip_prefix("os error at ")
        .and_then(|s| s.split_once(": ").map(|(_, reason)| reason))
        .unwrap_or(&message);
    RenderError(format!("{context}: {message}"))
}
#[cfg(target_os = "linux")]
pub(crate) fn check_keyboard_libraries() -> Result<(), RenderError> {
    // Probe through winit's own safe loader before it calls an infallible handle.
    // This reports missing native packages without an upstream loader panic.
    if xkbcommon_dl::xkbcommon_option().is_none()
        || xkbcommon_dl::xkbcommon_compose_option().is_none()
    {
        return Err(RenderError(
            "window keyboard support requires libxkbcommon0".into(),
        ));
    }
    let nonempty = |name| std::env::var(name).is_ok_and(|s| !s.is_empty());
    let backend = windowing::Backend::read()?;
    let x11 = backend == windowing::Backend::X11
        || (backend == windowing::Backend::Auto
            && !nonempty("WAYLAND_DISPLAY")
            && !nonempty("WAYLAND_SOCKET")
            && nonempty("DISPLAY"));
    if x11 && xkbcommon_dl::x11::xkbcommon_x11_option().is_none() {
        return Err(RenderError("X11 keyboard support requires libxkbcommon-x11-0 and libxcb-xkb1; use Wayland when available".into()));
    }
    Ok(())
}
fn drawable(width: u32, height: u32, occluded: bool) -> bool {
    width != 0 && height != 0 && !occluded
}
pub(crate) struct Display {
    instance: wgpu::Instance,
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) graphics: Graphics,
    pub(crate) config: wgpu::SurfaceConfiguration,
    depth: wgpu::Texture,
}
impl Display {
    pub(crate) fn new(
        window: Arc<Window>,
        selection: GraphicsSelection,
        display_handle: winit::event_loop::OwnedDisplayHandle,
    ) -> Result<Self, RenderError> {
        // EGL presentation must use the window's compositor connection rather
        // than the separate surfaceless connection used by offline rendering.
        let instance = Graphics::instance_with_descriptor(
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display_handle)),
        );
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| RenderError(format!("window surface: {e}")))?;
        // Select before inspecting formats; the actual pipeline uses the selected
        // surface's sRGB format, not a guessed channel order.
        let adapter = pollster::block_on(Graphics::select_adapter(
            &instance,
            selection,
            Some(&surface),
        ))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .ok_or_else(|| RenderError("selected surface has no supported sRGB format".into()))?;
        let graphics = pollster::block_on(Graphics::from_adapter(adapter, Some(&surface), format))?;
        eprintln!(
            "maelstrom graphics: {:?} {:?}; {}; driver {} {}; surface {:?}",
            graphics.adapter_info.backend,
            graphics.adapter_info.device_type,
            graphics.adapter_info.name,
            graphics.adapter_info.driver,
            graphics.adapter_info.driver_info,
            format
        );
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .ok_or_else(|| RenderError("surface has no alpha mode".into()))?,
            view_formats: vec![],
        };
        let depth = graphics.depth(config.width, config.height)?;
        surface.configure(&graphics.device, &config);
        graphics.check()?;
        Ok(Self {
            instance,
            surface,
            graphics,
            config,
            depth,
        })
    }
    pub(crate) fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        if !drawable(width, height, false) {
            return Ok(());
        }
        self.depth = self.graphics.depth(width, height)?;
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.graphics.device, &self.config);
        self.graphics.check()
    }
    pub(crate) fn recreate(&mut self, window: Arc<Window>) -> Result<(), RenderError> {
        self.surface = self
            .instance
            .create_surface(window)
            .map_err(|e| RenderError(format!("surface recreation: {e}")))?;
        if !self
            .surface
            .get_capabilities(&self.graphics.adapter)
            .formats
            .contains(&self.config.format)
        {
            return Err(RenderError(
                "recreated surface lost its selected sRGB format".into(),
            ));
        }
        self.surface.configure(&self.graphics.device, &self.config);
        self.graphics.check()
    }
}
struct Viewer {
    model: Model,
    settings: Settings,
    selection: GraphicsSelection,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    closing: bool,
    error: Option<RenderError>,
    edit: Option<String>,
    show_panel: bool,
    refresh_surface: bool,
    field: usize,
    drag: Option<MouseButton>,
    cursor: [f64; 2],
    occluded: bool,
    frames: u64,
    fps: f64,
    fps_start: Instant,
    next_redraw: Instant,
}
impl Viewer {
    fn new(model: Model, settings: Settings, selection: GraphicsSelection) -> Self {
        Self {
            model,
            settings,
            selection,
            window: None,
            display: None,
            closing: false,
            error: None,
            edit: None,
            show_panel: true,
            refresh_surface: false,
            field: 0,
            drag: None,
            cursor: [0.0; 2],
            occluded: false,
            frames: 0,
            fps: 0.0,
            fps_start: Instant::now(),
            next_redraw: Instant::now(),
        }
    }

    fn close(&mut self, error: Option<RenderError>) {
        if let Some(e) = &error {
            eprintln!("maelstrom viewer: {e}");
        }
        if self.error.is_none() {
            self.error = error;
        }
        self.closing = true;
        self.model.request_shutdown();
        if let Some(w) = &self.window {
            w.set_title("Maelstrom - stopping worker; retaining accepted state");
        }
    }
    fn feedback(&mut self, result: Result<(), RenderError>) {
        if let Err(e) = result {
            self.model.message = e.to_string();
        }
    }
    fn key(&mut self, key: &Key) {
        if let Some(text) = &mut self.edit {
            match key {
                Key::Named(NamedKey::Escape) => {
                    self.edit = None;
                }
                Key::Named(NamedKey::Enter) => {
                    *self.model.draft.field_mut(self.field) = text.clone();
                    self.edit = None;
                }
                Key::Named(NamedKey::Delete) => {
                    text.clear();
                }
                Key::Named(NamedKey::Backspace) => {
                    text.pop();
                }
                Key::Character(s)
                    if text.len() + s.len() <= 4096 && !s.chars().any(char::is_control) =>
                {
                    text.push_str(s);
                }
                _ => {}
            }
            return;
        }
        let result = match key {
            Key::Named(NamedKey::Escape) => {
                self.close(None);
                Ok(())
            }
            Key::Named(NamedKey::Tab) => {
                self.field = (self.field + 1) % self.model.draft.fields().len();
                Ok(())
            }
            Key::Named(NamedKey::Enter) => {
                self.edit = Some(self.model.draft.fields()[self.field].1.into());
                Ok(())
            }
            Key::Named(NamedKey::Space) => self.model.pause_resume(),
            Key::Named(NamedKey::ArrowLeft) => self.model.camera.orbit(-0.1, 0.0),
            Key::Named(NamedKey::ArrowRight) => self.model.camera.orbit(0.1, 0.0),
            Key::Named(NamedKey::ArrowUp) => self.model.camera.orbit(0.0, 0.1),
            Key::Named(NamedKey::ArrowDown) => self.model.camera.orbit(0.0, -0.1),
            Key::Named(NamedKey::Home) => self.model.restore_camera(),
            Key::Character(s) => match s.to_lowercase().as_str() {
                "a" => self.model.apply(),
                "b" => {
                    let modes = ["brute-force", "serial-grid", "threaded-grid", "cuda-grid"];
                    let i = modes
                        .iter()
                        .position(|mode| *mode == self.model.draft.mode)
                        .unwrap_or(0);
                    self.model.draft.mode = modes[(i + 1) % 4].into();
                    Ok(())
                }
                "1" | "2" | "3" | "4" | "5" | "6" | "7" => {
                    let names = ["04", "06", "08", "10", "12", "16", "20"];
                    let i = s.parse::<usize>().unwrap_or(1) - 1;
                    self.model.draft.scene = format!("falling-block-{}", names[i]);
                    Ok(())
                }
                "n" => self.model.action(Action::SingleStep),
                "d" => self.model.audit(),
                "h" => {
                    self.show_panel = !self.show_panel;
                    Ok(())
                }
                "g" => {
                    self.refresh_surface = true;
                    Ok(())
                }
                "r" => self.model.reset(),
                "c" => {
                    self.model.cancel();
                    Ok(())
                }
                "f" => {
                    let size = self
                        .window
                        .as_ref()
                        .map(|w| w.inner_size())
                        .unwrap_or(winit::dpi::PhysicalSize::new(1, 1));
                    self.model
                        .fit(f64::from(size.width.max(1)) / f64::from(size.height.max(1)))
                }
                "s" => {
                    let size = self.window.as_ref().map(|w| w.inner_size()).unwrap_or(
                        winit::dpi::PhysicalSize::new(self.settings.width, self.settings.height),
                    );
                    self.model
                        .save_camera(f64::from(size.width.max(1)) / f64::from(size.height.max(1)))
                }
                "o" => {
                    self.model.show_retained = !self.model.show_retained;
                    Ok(())
                }
                "+" | "=" => self.model.camera.zoom(0.9),
                "-" => self.model.camera.zoom(1.1),
                _ => Ok(()),
            },
            _ => Ok(()),
        };
        self.feedback(result);
    }
    fn lines(&self) -> Vec<String> {
        let mut lines = vec![
            "MAELSTROM | accepted particle observations".into(),
            "A apply draft | Space run/pause | N step | R reset | C cancel".into(),
            "Tab field / Enter edit+save / Delete clear / Esc edit or close".into(),
            "Drag orbit | right-drag pan | wheel zoom | F fit | Home restore | S save camera"
                .into(),
            "B draft mode | 1-7 scenes | D audit | H panel | G surface".into(),
        ];
        for (i, (label, value)) in self.model.draft.fields().iter().enumerate() {
            let shown = if let Some(text) = self.edit.as_ref().filter(|_| i == self.field) {
                format!("{text}_")
            } else if i <= 1 {
                std::path::Path::new(value)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(value)
                    .to_owned()
            } else {
                value.to_string()
            };
            lines.push(format!(
                "{} {label}: {shown}",
                if i == self.field { ">" } else { " " }
            ));
        }
        lines.push(format!("Feedback: {}", self.model.message));
        if let Some(s) = self.model.displayed() {
            let o = s.handle.observe();
            let steps = o.snapshot.as_ref().map_or(0, |s| s.completed_steps());
            lines.push(format!(
                "{}generation {} | {:?} | {steps}/{} steps | {} simulated s",
                if self.model.show_retained {
                    "PRIOR "
                } else {
                    "SELECTED "
                },
                o.generation,
                o.status,
                s.spec.steps,
                o.simulated_time_s()
                    .map_or_else(|| "unavailable".into(), |s| format!("{s:.6}"))
            ));
            lines.push(format!(
                "Particles: {} | effective: {}",
                o.snapshot.as_ref().map_or(0, |s| s.particles().len()),
                o.info.as_ref().map_or_else(
                    || if o.status.is_terminal() {
                        "not acquired".into()
                    } else {
                        "pending construction".into()
                    },
                    |i| format!("{:?} / {:?}", i.backend, i.execution)
                )
            ));
            let numerical = &s.spec.scene.configuration.numerical;
            lines.push(format!(
                "Numerics: {:.1} Hz | Jacobi {} | correction {} | relaxation {}",
                1.0 / numerical.fixed_timestep_s,
                numerical.solver_iterations,
                numerical.correction_scale,
                numerical.relaxation_scale
            ));
            if o.snapshot.is_none() {
                lines.push(format!(
                    "Accepted snapshot unavailable; {} requested particles",
                    s.spec.scene.particles.len()
                ));
            }
            if let Some(device) = o.info.as_ref().and_then(|i| i.device.as_ref()) {
                lines.push(format!(
                    "CUDA: {} (ordinal {})",
                    device.name, device.ordinal
                ));
            }
            lines.push(format!(
                "CFL warnings: {} | latest: {:?}",
                o.warning_count, o.last_cfl
            ));
            if let Some(failure) = o.failure {
                lines.push(format!("FAILURE {:?}: {}", failure.kind, failure.message));
            }
            if !self.model.show_retained
                && let Some(old) = &self.model.retained
            {
                let o = old.handle.observe();
                lines.push(format!(
                    "Prior generation {}: {:?}, accepted {}; O to inspect",
                    o.generation,
                    o.status,
                    o.snapshot.as_ref().map_or(0, |s| s.completed_steps())
                ));
            }
        }
        lines.push("Plane outlines are clipped depictions; metres, Y up".into());
        lines.push(self.model.audit_message.clone());
        lines.push(format!(
            "Redraw FPS: {:.1} | solver timing and throughput not measured",
            self.fps
        ));
        lines
    }
    fn redraw(&mut self) -> Result<(), RenderError> {
        self.model.poll();
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| RenderError("window unavailable".into()))?
            .clone();
        let size = window.inner_size();
        if !drawable(size.width, size.height, self.occluded) {
            return Ok(());
        }
        if let Some(selected) = self.model.displayed() {
            let observation = selected.handle.observe();
            window.set_title(&format!(
                "Maelstrom | {}gen {} {:?} | {}/{} steps | warnings {}",
                if self.model.show_retained {
                    "PRIOR "
                } else {
                    ""
                },
                observation.generation,
                observation.status,
                observation
                    .snapshot
                    .as_ref()
                    .map_or(0, |s| s.completed_steps()),
                selected.spec.steps,
                observation.warning_count
            ));
        }
        let lines = self.lines();
        let overlay = if self.show_panel {
            overlay::panel(
                &lines,
                size.width,
                size.height,
                (1.5 * window.scale_factor() as f32).clamp(1.5, 3.0),
            )
        } else {
            Vec::new()
        };
        let display = self
            .display
            .as_mut()
            .ok_or_else(|| RenderError("graphics unavailable".into()))?;
        display.graphics.check()?;
        if self.refresh_surface {
            display.recreate(window.clone())?;
            self.refresh_surface = false;
        }
        let mut suboptimal = false;
        let frame = match display.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Present this acquired image, then refresh configuration on the next redraw.
                suboptimal = true;
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                display.resize(size.width, size.height)?;
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                display.recreate(window)?;
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RenderError("surface acquisition validation failed".into()));
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let depth = display.depth.create_view(&Default::default());
        if let Some(s) = self.model.displayed() {
            let observation = s.handle.observe();
            let particles = observation
                .snapshot
                .as_ref()
                .map_or(&[][..], |s| s.particles());
            display.graphics.draw(
                &view,
                &depth,
                &DrawScene {
                    particles,
                    boundaries: &s.boundaries,
                    radius_m: f64::from(s.spec.scene.configuration.numerical.particle_spacing_m)
                        / 2.0,
                    camera: &self.model.camera,
                    colours: &s
                        .presentation
                        .as_ref()
                        .map_or(&self.settings, |p| &p.presentation.settings)
                        .particles,
                    particle_mass_kg: s.spec.scene.configuration.derived.particle_mass_kg,
                    width: size.width,
                    height: size.height,
                },
                &overlay,
            )?;
        } else {
            let empty = crate::render::BoundaryLines::new(&[], &[], 0.05)?;
            display.graphics.draw(
                &view,
                &depth,
                &DrawScene {
                    particles: &[],
                    boundaries: &empty,
                    radius_m: 0.025,
                    camera: &self.model.camera,
                    colours: &self.settings.particles,
                    particle_mass_kg: 1.0,
                    width: size.width,
                    height: size.height,
                },
                &overlay,
            )?;
        }
        window.pre_present_notify();
        display.graphics.queue.present(frame);
        if suboptimal {
            display.resize(size.width, size.height)?;
        }
        self.frames += 1;
        let elapsed = self.fps_start.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.fps = self.frames as f64 / elapsed;
            self.frames = 0;
            self.fps_start = Instant::now();
        }
        Ok(())
    }
}
impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.closing {
            return;
        }
        self.occluded = false;
        if self.window.is_none() {
            match event_loop.create_window(
                Window::default_attributes()
                    .with_active(false)
                    .with_title("Maelstrom - particle viewer")
                    .with_inner_size(winit::dpi::PhysicalSize::new(
                        self.settings.width,
                        self.settings.height,
                    )),
            ) {
                Ok(window) => self.window = Some(Arc::new(window)),
                Err(e) => {
                    self.close(Some(window_error("window creation", e)));
                    return;
                }
            }
        }
        if self.display.is_none()
            && let Some(window) = &self.window
        {
            match Display::new(
                window.clone(),
                self.selection,
                event_loop.owned_display_handle(),
            ) {
                Ok(display) => self.display = Some(display),
                Err(e) => self.close(Some(e)),
            }
        }
    }
    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.display = None;
        self.occluded = true;
    }
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if self.closing {
            return;
        }
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => self.close(None),
            WindowEvent::Resized(size) => {
                if let Some(d) = &mut self.display {
                    let result = d.resize(size.width, size.height);
                    if let Err(e) = result {
                        self.close(Some(e));
                    }
                }
            }
            WindowEvent::Occluded(value) => self.occluded = value,
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                self.key(&event.logical_key)
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.drag = if state == ElementState::Pressed {
                    Some(button)
                } else {
                    None
                };
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.cursor[0];
                let dy = position.y - self.cursor[1];
                self.cursor = [position.x, position.y];
                let result = match self.drag {
                    Some(MouseButton::Left) => self.model.camera.orbit(-dx * 0.006, dy * 0.006),
                    Some(MouseButton::Right) => self.model.camera.pan(-dx * 0.001, dy * 0.001),
                    _ => Ok(()),
                };
                self.feedback(result);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / 100.0,
                };
                let result = self.model.camera.zoom((-delta * 0.1).exp());
                self.feedback(result);
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.redraw() {
                    self.close(Some(e));
                }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.model.poll();
        if self.closing {
            if self.model.is_finished() {
                event_loop.exit();
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + Duration::from_millis(16),
                ));
            }
            return;
        }
        let now = Instant::now();
        if now >= self.next_redraw {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            self.next_redraw = now + Duration::from_millis(16);
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw));
    }
}
