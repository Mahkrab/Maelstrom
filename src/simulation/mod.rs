//! Transactional Position-Based Fluids timesteps with selectable execution.
//!
//! The solver constructs each timestep in working state, then replaces its
//! snapshot only after the numerical and boundary checks succeed.

mod error;
mod state;
pub(crate) mod timing;

pub use error::{RunFailure, StepError};
pub use state::Snapshot;
pub use timing::{StageTimes, StepMeasurement};

use std::time::Instant;
use timing::{Stage, StageClock};

use crate::{
    boundary::{Plane, project_in_order},
    config::ResolvedConfiguration,
    diagnostics::{CflDiagnostic, cfl_diagnostic},
    execution::{Execution, ExecutionError, ExecutionMode},
    fluid::{PbfParameters, compute_jacobi_stage_with_execution},
    integration::{post_acceleration_velocity, predict_position, reconstruct_velocity},
    neighbours::SearchBackend,
    scene::{PLANE_UNIT_LENGTH_SQUARED_TOLERANCE, Particle, ResolvedScene},
    validation,
};
use state::validate_particles;

/// Diagnostic outcome recorded after an accepted timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct StepOutcome {
    /// Completed-step counter after this step was accepted.
    pub completed_steps: u64,
    /// Post-acceleration CFL check. A warning does not prevent acceptance.
    pub cfl: CflDiagnostic,
}

/// PBF solver with private state, selectable discovery and execution resources.
/// Threaded mode distributes Jacobi and enabled velocity-effect rows.
/// CUDA runs discovery and numerical stages on its private device stream.
#[derive(Debug)]
pub struct Simulation {
    execution: Execution,
    mode: ExecutionMode,
    #[cfg(feature = "cuda")]
    cuda: Option<crate::cuda::solver::CudaSolver>,
    backend: SearchBackend,
    configuration: ResolvedConfiguration,
    parameters: PbfParameters,
    planes: Vec<Plane>,
    accepted: Snapshot,
}

/// Historical solver name. Existing constructors still select serial execution.
pub type SerialSimulation = Simulation;

impl Simulation {
    /// Copies a scene's simulation inputs into an independent solver.
    ///
    /// Scene metadata does not affect execution. All numerical inputs are checked
    /// again by [`Self::new`], since resolved scene records are caller-mutable.
    pub fn from_scene(scene: &ResolvedScene) -> Result<Self, StepError> {
        Self::from_scene_with_backend(scene, SearchBackend::BruteForce)
    }

    /// Copies a scene into an independent solver using the selected discovery.
    /// Validation and accepted-state semantics are shared by both backends.
    pub fn from_scene_with_backend(
        scene: &ResolvedScene,
        backend: SearchBackend,
    ) -> Result<Self, StepError> {
        Self::from_scene_with_execution(scene, backend, ExecutionMode::Serial)
    }

    /// Copies a scene into an independent solver with explicit execution resources.
    /// Parallel execution requires uniform-grid discovery. Invalid settings and
    /// unavailable resources fail construction before any step is attempted.
    pub fn from_scene_with_execution(
        scene: &ResolvedScene,
        backend: SearchBackend,
        execution: ExecutionMode,
    ) -> Result<Self, StepError> {
        Self::new_with_execution(
            &scene.configuration,
            scene.particles.clone(),
            scene.planes.clone(),
            backend,
            execution,
        )
    }

    /// Creates a solver and sorts its input particles by stable identity once.
    ///
    /// Sparse identities, empty inputs and coincident particles are valid.
    /// Finite initial positions may lie outside planes because each step applies
    /// projection; scene construction may impose stricter input rules.
    ///
    /// Returns an error for invalid configuration, inconsistent derived values,
    /// an unsupported solver iteration count, invalid plane normals, duplicate
    /// identities, or non-finite particle positions or velocities.
    pub fn new(
        configuration: &ResolvedConfiguration,
        particles: Vec<Particle>,
        planes: Vec<Plane>,
    ) -> Result<Self, StepError> {
        Self::new_with_backend(configuration, particles, planes, SearchBackend::BruteForce)
    }

    /// Creates a solver with explicit discovery. Inputs are validated and sorted
    /// as in [`Self::new`]; selection does not change the physical configuration.
    pub fn new_with_backend(
        configuration: &ResolvedConfiguration,
        particles: Vec<Particle>,
        planes: Vec<Plane>,
        backend: SearchBackend,
    ) -> Result<Self, StepError> {
        Self::new_with_execution(
            configuration,
            particles,
            planes,
            backend,
            ExecutionMode::Serial,
        )
    }

    /// Creates independently owned state and a reusable execution resource.
    /// Numerical validation is identical to [`Self::new`]. Parallel modes accept
    /// only grid discovery, and never silently substitutes a serial fallback.
    pub fn new_with_execution(
        configuration: &ResolvedConfiguration,
        mut particles: Vec<Particle>,
        planes: Vec<Plane>,
        backend: SearchBackend,
        execution: ExecutionMode,
    ) -> Result<Self, StepError> {
        execution.validate()?;
        if !execution.is_serial() && backend != SearchBackend::UniformGrid {
            return Err(ExecutionError::RequiresUniformGrid.into());
        }
        configuration.validate()?;
        for plane in &planes {
            Plane::new(
                plane.normal(),
                plane.offset(),
                PLANE_UNIT_LENGTH_SQUARED_TOLERANCE,
            )?;
        }
        particles.sort_unstable_by_key(|particle| particle.identity);
        validate_particles(&particles)?;
        #[cfg(feature = "cuda")]
        let cuda = match execution {
            ExecutionMode::Cuda { device, threads } => {
                Some(crate::cuda::solver::CudaSolver::new(device, threads)?)
            }
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        if let ExecutionMode::Cuda { device, .. } = execution {
            crate::cuda::CudaContext::new(device)?;
        }
        Ok(Self {
            execution: Execution::new(if matches!(execution, ExecutionMode::Cuda { .. }) {
                ExecutionMode::Serial
            } else {
                execution
            })?,
            mode: execution,
            #[cfg(feature = "cuda")]
            cuda,
            backend,
            configuration: configuration.clone(),
            parameters: configuration.pbf_parameters()?,
            planes,
            accepted: Snapshot {
                completed_steps: 0,
                particles,
            },
        })
    }

    /// Borrows the last fully accepted state. Clone it to retain a checkpoint
    /// while subsequent steps advance the solver.
    #[must_use]
    pub fn snapshot(&self) -> &Snapshot {
        &self.accepted
    }

    /// Borrows the validated configuration used by every step.
    #[must_use]
    pub fn configuration(&self) -> &ResolvedConfiguration {
        &self.configuration
    }

    /// Discovery selected at construction, fixed for this solver's lifetime.
    #[must_use]
    pub fn backend(&self) -> SearchBackend {
        self.backend
    }

    /// Effective execution settings, fixed for this solver's lifetime.
    #[must_use]
    pub fn execution(&self) -> ExecutionMode {
        self.mode
    }

    /// Selected device properties for CUDA, absent for CPU execution.
    pub fn device(&self) -> Option<&crate::cuda::DeviceInfo> {
        #[cfg(feature = "cuda")]
        {
            self.cuda.as_ref().map(|cuda| cuda.context.device())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Attempts one complete step. Every error leaves accepted state unchanged.
    pub fn step(&mut self) -> Result<StepOutcome, StepError> {
        self.step_with_cancellation(|| false)
    }

    /// Attempts one step with cooperative cancellation between numerical stages.
    ///
    /// A `true` result returns [`StepError::Cancelled`] and discards
    /// all working states. Each search or Jacobi calculation finishes before the
    /// next poll; callers should keep a cancellation request latched until seen.
    /// A predicate panic also occurs before acceptance and cannot partially commit.
    pub fn step_with_cancellation(
        &mut self,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<StepOutcome, StepError> {
        self.step_inner(&mut cancelled, &mut StageClock::<false>::new())
    }

    /// Times one successful step using a monotonic clock. With `stages` enabled,
    /// internal stage clocks add profiling overhead; use total-only samples for
    /// primary comparisons. Errors retain transactional state and return no
    /// successful measurement. The caller may time failed attempts separately.
    pub fn step_measured_with_cancellation(
        &mut self,
        stages: bool,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<(StepOutcome, StepMeasurement), StepError> {
        let start = Instant::now();
        let (outcome, stage_times) = if stages {
            let mut clock = StageClock::<true>::new();
            let outcome = self.step_inner(&mut cancelled, &mut clock)?;
            (outcome, Some(clock.finish()))
        } else {
            let outcome = self.step_inner(&mut cancelled, &mut StageClock::<false>::new())?;
            (outcome, None)
        };
        let total_s = start.elapsed().as_secs_f64();
        Ok((
            outcome,
            StepMeasurement {
                total_s,
                stages: stage_times,
            },
        ))
    }

    fn step_inner<const PROFILE: bool>(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
        clock: &mut StageClock<PROFILE>,
    ) -> Result<StepOutcome, StepError> {
        check_cancellation(cancelled)?;
        validate_particles(&self.accepted.particles)?;
        let completed_steps = self
            .accepted
            .completed_steps
            .checked_add(1)
            .ok_or(StepError::StepCountOverflow)?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            let (particles, cfl) = cuda.step(
                &self.configuration,
                &self.accepted.particles,
                &self.planes,
                cancelled,
                clock,
            )?;
            validate_particles(&particles)?;
            for particle in &particles {
                for (index, plane) in self.planes.iter().enumerate() {
                    if !plane.contains(particle.position_m)? {
                        return Err(StepError::OutsidePlane {
                            identity: particle.identity,
                            plane: index,
                        });
                    }
                }
            }
            check_cancellation(cancelled)?;
            self.accepted = Snapshot {
                completed_steps,
                particles,
            };
            clock.mark(Stage::Acceptance);
            return Ok(StepOutcome {
                completed_steps,
                cfl,
            });
        }
        let model = &self.configuration.numerical;
        let derived = &self.configuration.derived;
        let mut velocities = Vec::with_capacity(self.accepted.particles.len());
        for particle in &self.accepted.particles {
            velocities.push(post_acceleration_velocity(
                particle.velocity_m_s,
                model.external_acceleration_m_s2,
                model.fixed_timestep_s,
            )?);
        }
        let cfl = cfl_diagnostic(
            &velocities,
            model.fixed_timestep_s,
            derived.particle_diameter_m,
            model.cfl_factor,
        )?;
        check_cancellation(cancelled)?;

        clock.mark(Stage::ValidationAndAcceleration);
        let mut positions = Vec::with_capacity(velocities.len());
        for (particle, velocity) in self.accepted.particles.iter().zip(velocities) {
            positions.push(predict_position(
                particle.position_m,
                velocity,
                model.fixed_timestep_s,
            )?);
        }
        check_cancellation(cancelled)?;
        clock.mark(Stage::Prediction);
        let fixed = self
            .backend
            .discover(&positions, self.parameters.support_radius())?;
        check_cancellation(cancelled)?;

        clock.mark(Stage::NeighbourSearch);
        for _ in 0..model.solver_iterations {
            let stage = compute_jacobi_stage_with_execution(
                &positions,
                &fixed,
                self.parameters,
                &self.execution,
            )?;
            for density in stage.normalised_densities {
                validation::finite_result("physical density", density * derived.particle_mass_kg)?;
            }
            check_cancellation(cancelled)?;
            clock.mark(Stage::Jacobi);
            let mut candidates = Vec::with_capacity(positions.len());
            for (position, correction) in positions.iter().zip(stage.corrections) {
                candidates.push(validation::finite_vector_result(
                    "candidate position",
                    *position + correction,
                )?);
            }
            for candidate in &mut candidates {
                *candidate = project_in_order(*candidate, &self.planes)?;
            }
            positions = candidates;
            check_cancellation(cancelled)?;
            clock.mark(Stage::Projection);
        }

        let mut particles = Vec::with_capacity(positions.len());
        for (old, &position_m) in self.accepted.particles.iter().zip(&positions) {
            particles.push(Particle {
                identity: old.identity,
                position_m,
                velocity_m_s: reconstruct_velocity(
                    old.position_m,
                    position_m,
                    model.fixed_timestep_s,
                )?,
            });
        }
        if model.vorticity_confinement_enabled && model.vorticity_confinement_strength_m_s != 0.0 {
            check_cancellation(cancelled)?;
            let velocities: Vec<_> = particles
                .iter()
                .map(|particle| particle.velocity_m_s)
                .collect();
            let confined = crate::velocity::vorticity::with_execution(
                &positions,
                &velocities,
                &fixed,
                crate::velocity::VorticityParameters {
                    radius_m: derived.support_radius_m,
                    rest_density_per_m3: derived.normalised_rest_density_per_m3,
                    strength_m_s: model.vorticity_confinement_strength_m_s,
                    timestep_s: model.fixed_timestep_s,
                },
                &self.execution,
                &mut || check_cancellation(cancelled),
            )?;
            check_cancellation(cancelled)?;
            for (particle, velocity) in particles.iter_mut().zip(confined) {
                particle.velocity_m_s = velocity;
            }
        }
        if model.xsph_viscosity_enabled && model.xsph_viscosity_coefficient != 0.0 {
            check_cancellation(cancelled)?;
            let velocities: Vec<_> = particles
                .iter()
                .map(|particle| particle.velocity_m_s)
                .collect();
            let smoothed = crate::velocity::xsph_with_execution(
                &positions,
                &velocities,
                &fixed,
                derived.support_radius_m,
                derived.normalised_rest_density_per_m3,
                model.xsph_viscosity_coefficient,
                &self.execution,
            )?;
            check_cancellation(cancelled)?;
            for (particle, velocity) in particles.iter_mut().zip(smoothed) {
                particle.velocity_m_s = velocity;
            }
        }
        validate_particles(&particles)?;
        // Fixed rows are immutable and validated at construction. Their indices
        // still refer to the same identities because no stage reorders particles.
        for particle in &particles {
            for (index, plane) in self.planes.iter().enumerate() {
                if !plane.contains(particle.position_m)? {
                    return Err(StepError::OutsidePlane {
                        identity: particle.identity,
                        plane: index,
                    });
                }
            }
        }
        check_cancellation(cancelled)?;
        self.accepted = Snapshot {
            completed_steps,
            particles,
        };
        clock.mark(Stage::Acceptance);
        Ok(StepOutcome {
            completed_steps,
            cfl,
        })
    }

    /// Executes `steps` complete timesteps, retaining every accepted CFL outcome.
    /// A zero-step request succeeds without changing state. On failure, the error
    /// carries only completed outcomes and the solver retains its last checkpoint.
    pub fn run_steps(&mut self, steps: u64) -> Result<Vec<StepOutcome>, RunFailure> {
        self.run_steps_with_cancellation(steps, || false)
    }

    /// Executes steps with the cancellation semantics of [`Self::step_with_cancellation`].
    /// Call individual steps and read [`Self::snapshot`] between them when custom
    /// checkpoint intervals are needed; observations are outside numerical work.
    pub fn run_steps_with_cancellation(
        &mut self,
        steps: u64,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<StepOutcome>, RunFailure> {
        let mut completed = Vec::new();
        for _ in 0..steps {
            match self.step_with_cancellation(&mut cancelled) {
                Ok(outcome) => completed.push(outcome),
                Err(error) => return Err(RunFailure { completed, error }),
            }
        }
        Ok(completed)
    }
}

pub(crate) fn check_cancellation(cancelled: &mut impl FnMut() -> bool) -> Result<(), StepError> {
    if cancelled() {
        Err(StepError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolve_effective_configuration;

    #[test]
    fn exhausted_step_counter_cannot_wrap_or_accept_another_step() {
        let configuration = resolve_effective_configuration(None, None, None).unwrap();
        let mut solver = SerialSimulation::new(&configuration, vec![], vec![]).unwrap();
        solver.accepted.completed_steps = u64::MAX - 1;
        assert_eq!(solver.step().unwrap().completed_steps, u64::MAX);
        let before = solver.snapshot().clone();
        assert!(matches!(solver.step(), Err(StepError::StepCountOverflow)));
        assert_eq!(solver.snapshot(), &before);
    }
}
