//! Draft edits are separate from the selected generation. Only validated requests
//! reach the worker; retained outcomes are explicitly labelled when inspected.

use crate::application::{Action, Controller, Receipt, RunHandle, RunSpec, RunStatus};
use crate::presentation::{CameraMode, CameraSaveTarget, Presentation};
use crate::render::{BoundaryLines, Camera, RenderError};

/// Editable fields for the next run. Applying them always starts from step zero.
#[derive(Clone, Debug)]
pub struct Draft {
    /// Bundled name or external scene path, resolved by the existing input loader.
    pub scene: String,
    /// Explicit base configuration path; empty preserves the scene selection.
    pub config: String,
    /// brute-force, serial-grid, threaded-grid or cuda-grid.
    pub mode: String,
    /// reference, fast, experimental, scene, or blank to use the application default/preference.
    pub numerical_mode: String,
    /// Threaded CPU pool size.
    pub workers: String,
    /// CUDA ordinal.
    pub device: String,
    /// CUDA lanes per block.
    pub block_size: String,
    /// Finite accepted-step limit.
    pub steps: String,
}
impl Draft {
    pub(crate) fn fields(&self) -> [(&'static str, &str); 8] {
        [
            ("Scene", &self.scene),
            ("Config (blank = scene)", &self.config),
            ("Backend", &self.mode),
            ("Workers", &self.workers),
            ("CUDA device", &self.device),
            ("CUDA block", &self.block_size),
            ("Steps", &self.steps),
            (
                "Numerics (default / reference / fast / experimental / scene)",
                &self.numerical_mode,
            ),
        ]
    }
    pub(crate) fn field_mut(&mut self, index: usize) -> &mut String {
        match index {
            0 => &mut self.scene,
            1 => &mut self.config,
            2 => &mut self.mode,
            3 => &mut self.workers,
            4 => &mut self.device,
            5 => &mut self.block_size,
            6 => &mut self.steps,
            _ => &mut self.numerical_mode,
        }
    }
}

pub(crate) struct Selected {
    pub handle: RunHandle,
    pub spec: RunSpec,
    pub boundaries: BoundaryLines,
    pub presentation: Option<ViewPresentation>,
    pub initial_camera: Camera,
}
/// Resolves a draft through the application's existing physical-input precedence.
pub type Resolver = Box<dyn Fn(&Draft) -> Result<RunSpec, String>>;

/// Scene-owned visual settings and the explicit camera-save destination.
#[derive(Clone, Debug)]
pub struct ViewPresentation {
    /// Validated settings for the selected scene, including temporary CLI overrides.
    pub presentation: Presentation,
    /// Destination captured for this selection; bundled scenes have no writable target.
    pub camera_save: Option<CameraSaveTarget>,
}
/// Resolves visual settings independently of the physical request and worker resources.
pub type PresentationResolver = Box<dyn Fn(&Draft) -> Result<ViewPresentation, String>>;

struct SaveJob {
    generation: u64,
    camera: Camera,
    thread: std::thread::JoinHandle<Result<(), crate::presentation::CameraSaveError>>,
}

/// Non-blocking presentation state; window events only admit commands and poll receipts.
pub(crate) struct Model {
    /// Draft inputs for the next apply, independent of the selected physical request.
    pub draft: Draft,
    /// Presentation-only camera; controls never change accepted particles.
    pub camera: Camera,
    /// Latest input or control feedback, retained until the next operation.
    pub message: String,
    pub(crate) controller: Controller,
    pub(crate) selected: Option<Selected>,
    pub(crate) retained: Option<Selected>,
    pub(crate) show_retained: bool,
    pub(super) receipts: Vec<Receipt>,
    pub(super) resolver: Resolver,
    pub(super) presentation_resolver: Option<PresentationResolver>,
    save: Option<SaveJob>,
    audit: Option<std::thread::JoinHandle<String>>,
    /// Latest requested density audit, labelled by generation and accepted step.
    pub audit_message: String,
}
impl Model {
    /// Starts an idle worker. Draft resolution happens only on apply.
    pub fn new(draft: Draft, camera: Camera, resolver: Resolver) -> Result<Self, RenderError> {
        let controller = Controller::new().map_err(|e| RenderError(e.to_string()))?;
        Ok(Self {
            draft,
            camera,
            message: "Ready: apply a draft to start paused".into(),
            controller,
            selected: None,
            retained: None,
            show_retained: false,
            receipts: Vec::new(),
            resolver,
            presentation_resolver: None,
            save: None,
            audit: None,
            audit_message: "Density audit: D to request (outside solver work)".into(),
        })
    }
    /// Validates a complete draft before admitting a replacement generation.
    pub fn apply(&mut self) -> Result<(), RenderError> {
        let spec = (self.resolver)(&self.draft).map_err(RenderError)?;
        let presentation = self
            .presentation_resolver
            .as_ref()
            .map(|resolve| resolve(&self.draft))
            .transpose()
            .map_err(RenderError)?;
        let mut camera = self.camera.clone();
        if let Some(view) = &presentation {
            let settings = &view.presentation.settings;
            settings.validate()?;
            camera = settings.camera.clone();
            if view.presentation.camera_mode == CameraMode::Fit {
                camera.fit_scene(
                    &spec.scene.particles,
                    &spec.scene.planes,
                    f64::from(spec.scene.configuration.numerical.particle_spacing_m) / 2.0,
                    f64::from(settings.width) / f64::from(settings.height),
                )?;
            }
        }
        self.start(spec, presentation, camera.clone())?;
        self.camera = camera;
        Ok(())
    }
    fn start(
        &mut self,
        spec: RunSpec,
        presentation: Option<ViewPresentation>,
        initial_camera: Camera,
    ) -> Result<(), RenderError> {
        self.poll();
        let spacing = f64::from(spec.scene.configuration.numerical.particle_spacing_m);
        let boundaries = BoundaryLines::new(&spec.scene.particles, &spec.scene.planes, spacing)?;
        let (handle, receipt) = self
            .controller
            .start(spec.clone())
            .map_err(|e| RenderError(e.to_string()))?;
        if let Some(old) = self.selected.take() {
            // Keep the last accepted outcome even after repeated setup failures.
            if old.handle.observe().snapshot.is_some() || self.retained.is_none() {
                self.retained = Some(old);
            }
        }
        self.selected = Some(Selected {
            handle,
            spec,
            boundaries,
            presentation,
            initial_camera,
        });
        self.receipts.push(receipt);
        self.show_retained = false;
        self.message = "New run admitted; constructing on worker".into();
        Ok(())
    }
    /// Restarts the selected original request, ignoring unapplied draft edits.
    pub fn reset(&mut self) -> Result<(), RenderError> {
        let mut spec = self
            .selected
            .as_ref()
            .ok_or_else(|| RenderError("no selected run".into()))?
            .spec
            .clone();
        spec.initially_paused = true;
        let selected = self
            .selected
            .as_ref()
            .ok_or_else(|| RenderError("no selected run".into()))?;
        self.start(
            spec,
            selected.presentation.clone(),
            selected.initial_camera.clone(),
        )
    }
    /// Admits an action without waiting for a synchronous numerical stage.
    pub fn action(&mut self, action: Action) -> Result<(), RenderError> {
        self.poll();
        let selected = self
            .selected
            .as_ref()
            .ok_or_else(|| RenderError("no selected run".into()))?;
        let receipt = self
            .controller
            .control(&selected.handle, action)
            .map_err(|e| RenderError(e.to_string()))?;
        self.receipts.push(receipt);
        self.message = format!("Queued {action:?}; waiting for worker acknowledgement");
        Ok(())
    }
    /// Toggles running/paused state using the latest observation.
    pub fn pause_resume(&mut self) -> Result<(), RenderError> {
        let status = self
            .selected
            .as_ref()
            .ok_or_else(|| RenderError("no selected run".into()))?
            .handle
            .observe()
            .status;
        match status {
            RunStatus::Paused => self.action(Action::Resume),
            RunStatus::Running => self.action(Action::Pause),
            _ => Err(RenderError(
                "pause/resume requires a constructed active run".into(),
            )),
        }
    }
    /// Latches cancellation independently of queue capacity.
    pub fn cancel(&mut self) {
        if let Some(s) = &self.selected {
            s.handle.cancel();
            self.message = "Cancellation requested; retaining last accepted state".into();
        }
    }
    /// Polls all admitted receipts without blocking the event loop.
    pub fn poll(&mut self) {
        if self
            .audit
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
            && let Some(thread) = self.audit.take()
        {
            self.audit_message = thread
                .join()
                .unwrap_or_else(|_| "Density audit worker failed".into());
        }

        self.receipts.retain(|r| match r.try_result() {
            None => true,
            Some(result) => {
                self.message = match result {
                    Ok(status) => format!("Worker acknowledged {status:?}"),
                    Err(e) => e.to_string(),
                };
                false
            }
        });
        if self
            .save
            .as_ref()
            .is_some_and(|job| job.thread.is_finished())
            && let Some(job) = self.save.take()
        {
            self.message = match job.thread.join() {
                Ok(Ok(())) => {
                    for selected in [&mut self.selected, &mut self.retained]
                        .into_iter()
                        .flatten()
                    {
                        if selected.handle.observe().generation == job.generation {
                            selected.initial_camera = job.camera.clone();
                        }
                    }
                    format!("Saved camera for generation {}", job.generation)
                }
                Ok(Err(error)) => error.to_string(),
                Err(_) => "Camera-save worker failed".into(),
            };
        }
    }
    /// Audits one owned accepted snapshot on a separate thread. Only one audit
    /// may be in flight, and its result keeps the original generation and step.
    pub fn audit(&mut self) -> Result<(), RenderError> {
        if self.audit.is_some() {
            return Err(RenderError("density audit already in progress".into()));
        }
        let selected = self
            .displayed()
            .ok_or_else(|| RenderError("no displayed run".into()))?;
        let observation = selected.handle.observe();
        let snapshot = observation
            .snapshot
            .ok_or_else(|| RenderError("no accepted snapshot to audit".into()))?;
        let generation = observation.generation;
        let step = snapshot.completed_steps();
        let derived = &selected.spec.scene.configuration.derived;
        let support = derived.support_radius_m;
        let rest = derived.normalised_rest_density_per_m3;
        self.audit = Some(
            std::thread::Builder::new()
                .name("density observation".into())
                .spawn(
                    move || match crate::diagnostics::audit_density(&snapshot, support, rest) {
                        Ok(a) => format!(
                            "Density gen {generation} step {step}: n={} mean={:?} max={:?}",
                            a.sample_count, a.mean_error, a.maximum_error
                        ),
                        Err(e) => format!("Density gen {generation} step {step}: {e}"),
                    },
                )
                .map_err(|e| RenderError(format!("density audit thread: {e}")))?,
        );
        self.audit_message = format!("Density gen {generation} step {step}: audit in progress");
        Ok(())
    }
    pub(crate) fn displayed(&self) -> Option<&Selected> {
        if self.show_retained {
            self.retained.as_ref()
        } else {
            self.selected.as_ref()
        }
    }
    /// Fits the displayed domain, or its current accepted particles when unbounded.
    pub fn fit(&mut self, aspect: f64) -> Result<(), RenderError> {
        if let Some(s) = self.displayed() {
            let mut camera = self.camera.clone();
            let observation = s.handle.observe();
            let particles = observation
                .snapshot
                .as_ref()
                .map_or(s.spec.scene.particles.as_slice(), |p| p.particles());
            camera.fit_scene(
                particles,
                &s.spec.scene.planes,
                f64::from(s.spec.scene.configuration.numerical.particle_spacing_m) / 2.0,
                aspect,
            )?;
            self.camera = camera;
        }
        Ok(())
    }
    /// Restores the displayed generation's last loaded or successfully saved camera.
    pub fn restore_camera(&mut self) -> Result<(), RenderError> {
        self.camera = self
            .displayed()
            .ok_or_else(|| RenderError("no displayed run".into()))?
            .initial_camera
            .clone();
        Ok(())
    }
    /// Queues a camera-only save without blocking the window or solver worker.
    pub fn save_camera(&mut self, aspect: f64) -> Result<(), RenderError> {
        if self.save.is_some() {
            return Err(RenderError("camera save already in progress".into()));
        }
        self.camera.validate(aspect)?;
        let selected = self
            .displayed()
            .ok_or_else(|| RenderError("no displayed run".into()))?;
        let target = selected
            .presentation
            .as_ref()
            .and_then(|p| p.camera_save.clone())
            .ok_or_else(|| {
                RenderError(
                    "camera saving requires an editable scene; use scenes create first".into(),
                )
            })?;
        let generation = selected.handle.observe().generation;
        let camera = self.camera.clone();
        let saved = camera.clone();
        let thread = std::thread::Builder::new()
            .name("camera save".into())
            .spawn(move || target.save(&saved, aspect))
            .map_err(|e| RenderError(format!("camera-save worker: {}", e.kind())))?;
        self.save = Some(SaveJob {
            generation,
            camera,
            thread,
        });
        self.message = format!("Saving camera for generation {generation}");
        Ok(())
    }
    /// Requests teardown; the event loop must keep polling until the worker finishes.
    pub fn request_shutdown(&mut self) {
        self.controller.request_shutdown();
    }
    /// Whether joining can proceed without waiting for numerical work.
    pub fn is_finished(&self) -> bool {
        self.controller.is_finished()
            && self
                .audit
                .as_ref()
                .is_none_or(std::thread::JoinHandle::is_finished)
            && self
                .save
                .as_ref()
                .is_none_or(|job| job.thread.is_finished())
    }
    /// Joins the worker after it has stopped, preserving retained observations.
    pub fn join(mut self) -> Result<(), RenderError> {
        self.controller.request_shutdown();
        let worker = self
            .controller
            .join()
            .map_err(|e| RenderError(e.to_string()));
        let audit = self.audit.take().map(|audit| audit.join());
        let save = self.save.take().map(|job| job.thread.join());
        worker?;
        if audit.is_some_and(|result| result.is_err()) {
            return Err(RenderError(
                "density audit worker failed during teardown".into(),
            ));
        }
        if let Some(save) = save {
            save.map_err(|_| RenderError("camera-save worker failed".into()))?
                .map_err(|e| RenderError(e.to_string()))?;
        }
        Ok(())
    }
}
