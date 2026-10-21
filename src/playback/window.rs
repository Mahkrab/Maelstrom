//! The playback window owns an image texture and a presentation clock only.
//! Closing it never needs solver cancellation or a simulation worker join.

use super::{Sequence, clock::Clock, image::Image};
use crate::{
    render::{GraphicsSelection, RenderError},
    viewer::{Display, window_error},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

/// Plays existing images on the main thread at recorded simulated time, scaled
/// by `speed` (0.01..=100). No particle rendering or numerical execution occurs.
/// The final frame remains visible. Repeating holds it for the preceding interval.
/// Missing/corrupt frames and graphics failures stop playback without changing files.
pub fn run(
    sequence: Sequence,
    speed: f64,
    paused: bool,
    repeat: bool,
    selection: GraphicsSelection,
) -> Result<(), RenderError> {
    let clock = Clock::new(
        sequence
            .frames
            .iter()
            .map(|f| f.simulated_seconds)
            .collect(),
        speed,
        paused,
        repeat,
    )?;
    // Validate the first image before opening a window, without retaining history.
    let first = sequence.decode(0)?;
    #[cfg(target_os = "linux")]
    crate::viewer::check_keyboard_libraries()?;
    let event_loop =
        EventLoop::new().map_err(|e| window_error("playback event loop unavailable", e))?;
    eprintln!(
        "maelstrom playback: saved images only; no simulation. Space play/pause; Left/Right seek; Home/End endpoints; R restart; +/- speed; Escape close"
    );
    let mut app = Player {
        sequence,
        clock,
        selection,
        window: None,
        display: None,
        image: None,
        loaded: None,
        first: Some(first),
        last_tick: Instant::now(),
        next_redraw: Instant::now(),
        error: None,
        occluded: false,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| window_error("playback event loop", e))?;
    if let Some(e) = app.error {
        return Err(e);
    }
    Ok(())
}
struct Player {
    sequence: Sequence,
    clock: Clock,
    selection: GraphicsSelection,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    image: Option<Image>,
    loaded: Option<usize>,
    first: Option<Vec<u8>>,
    last_tick: Instant,
    next_redraw: Instant,
    error: Option<RenderError>,
    occluded: bool,
}
impl Player {
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: RenderError) {
        self.error = Some(error);
        event_loop.exit();
    }
    fn tick(&mut self) {
        let now = Instant::now();
        self.clock
            .advance(now.duration_since(self.last_tick).as_secs_f64());
        self.last_tick = now;
    }
    fn redraw(&mut self) -> Result<(), RenderError> {
        self.next_redraw = Instant::now() + Duration::from_millis(16);
        let (Some(window), Some(display), Some(image)) =
            (&self.window, &mut self.display, &self.image)
        else {
            return Ok(());
        };
        let size = window.inner_size();
        if self.occluded || size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let index = self.clock.frame();
        if self.loaded != Some(index) {
            let rgba = self
                .first
                .take()
                .filter(|_| index == 0)
                .map(Ok)
                .unwrap_or_else(|| self.sequence.decode(index))?;
            image.upload(&display.graphics, &rgba)?;
            self.loaded = Some(index);
        }
        let f = &self.sequence.frames[index];
        window.set_title(&format!(
            "Maelstrom playback | {} | frame {}/{} | step {} | {:.6} s | {:.2}x",
            if self.clock.playing {
                "playing"
            } else {
                "paused"
            },
            index + 1,
            self.sequence.frame_count(),
            f.step,
            f.simulated_seconds,
            self.clock.speed
        ));
        let mut refresh = false;
        let frame = match display.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                refresh = true;
                f
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                display.resize(size.width, size.height)?;
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                display.recreate(window.clone())?;
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RenderError(
                    "playback surface acquisition validation failed".into(),
                ));
            }
        };
        image.draw(
            &display.graphics,
            &frame.texture.create_view(&Default::default()),
            size.width,
            size.height,
        )?;
        window.pre_present_notify();
        display.graphics.queue.present(frame);
        if refresh {
            display.resize(size.width, size.height)?;
        }
        Ok(())
    }
}
impl ApplicationHandler for Player {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let result = (|| {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Maelstrom playback")
                            .with_inner_size(winit::dpi::PhysicalSize::new(
                                self.sequence.width.min(1280),
                                self.sequence.height.min(720),
                            )),
                    )
                    .map_err(|e| window_error("playback window unavailable", e))?,
            );
            let display = Display::new(window.clone(), self.selection)?;
            let image = Image::new(
                &display.graphics,
                display.config.format,
                self.sequence.width,
                self.sequence.height,
            )?;
            self.window = Some(window);
            self.display = Some(display);
            self.image = Some(image);
            self.last_tick = Instant::now();
            self.redraw()
        })();
        if let Err(e) = result {
            self.fail(event_loop, e);
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|w| w.id() != id) {
            return;
        }
        self.tick();
        let request_redraw = !matches!(event, WindowEvent::RedrawRequested);
        let result = match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                Ok(())
            }
            WindowEvent::Resized(size) => self
                .display
                .as_mut()
                .map_or(Ok(()), |d| d.resize(size.width, size.height)),
            WindowEvent::Occluded(value) => {
                self.occluded = value;
                Ok(())
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) if !event.repeat => self.clock.toggle(),
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.clock.seek(self.clock.frame().saturating_sub(1))
                    }
                    Key::Named(NamedKey::ArrowRight) => self.clock.seek(self.clock.frame() + 1),
                    Key::Named(NamedKey::Home) => self.clock.seek(0),
                    Key::Named(NamedKey::End) => self.clock.seek(self.sequence.frame_count() - 1),
                    Key::Character(key) => match key.as_str() {
                        "r" | "R" => self.clock.restart(),
                        "+" | "=" => self.clock.speed = (self.clock.speed * 2.0).min(100.0),
                        "-" | "_" => self.clock.speed = (self.clock.speed / 2.0).max(0.01),
                        _ => {}
                    },
                    _ => {}
                }
                Ok(())
            }
            _ => return,
        };
        if let Err(e) = result {
            self.fail(event_loop, e);
        } else if request_redraw {
            // Controls, resize and exposure must redraw even while paused.
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.clock.playing && !self.occluded {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw));
            if Instant::now() >= self.next_redraw
                && let Some(window) = &self.window
            {
                window.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}
