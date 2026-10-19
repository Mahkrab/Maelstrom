//! Device scratch persists across timesteps; accepted state belongs to Simulation.
//! Each attempt reloads that state, so cancellation or numerical failure can retry
//! without exposing or trusting a previous attempt's partial device buffers.

use super::{CudaContext, CudaError};
use crate::{
    NumericalError,
    boundary::Plane,
    config::ResolvedConfiguration,
    diagnostics::{CflDiagnostic, cfl_diagnostic},
    math::Vec3,
    scene::Particle,
    simulation::{
        StepError, check_cancellation,
        timing::{Stage, StageClock},
    },
};

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(super) struct Settings {
    count: u32,
    padded: u32,
    words: u32,
    planes: u32,
    timestep: f32,
    radius: f32,
    rest_density: f32,
    relaxation: f32,
    strength: f32,
    reference_weight: f32,
    exponent: f32,
    correction_scale: f32,
    mass: f32,
    acceleration: [f32; 3],
    width: f64,
    stride: u32,
    span: u32,
    validate_pairs: u32,
    reserved: u32,
}
const _: () = {
    assert!(size_of::<Settings>() == 88 && align_of::<Settings>() == 8);
    assert!(std::mem::offset_of!(Settings, width) == 64);
};

#[derive(Debug)]
pub(crate) struct CudaSolver {
    pub(crate) context: CudaContext,
    threads: u32,
}

fn array(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}
fn vector(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

fn settings(
    configuration: &ResolvedConfiguration,
    count: usize,
    planes: usize,
) -> Result<Settings, StepError> {
    let count = u32::try_from(count).map_err(|_| CudaError::Capacity)?;
    let padded = count
        .max(1)
        .checked_next_power_of_two()
        .ok_or(CudaError::Capacity)?;
    let planes = u32::try_from(planes).map_err(|_| CudaError::Capacity)?;
    let model = &configuration.numerical;
    let derived = &configuration.derived;
    let parameters = configuration.pbf_parameters()?;
    Ok(Settings {
        count,
        padded,
        words: count.div_ceil(32),
        planes,
        timestep: model.fixed_timestep_s,
        radius: derived.support_radius_m,
        rest_density: derived.normalised_rest_density_per_m3,
        relaxation: parameters.relaxation(),
        strength: parameters.artificial_pressure().strength(),
        reference_weight: crate::kernel::poly6(
            parameters.artificial_pressure().reference_separation(),
            derived.support_radius_m,
        )?,
        exponent: parameters.artificial_pressure().exponent(),
        correction_scale: model.correction_scale,
        mass: derived.particle_mass_kg,
        acceleration: array(model.external_acceleration_m_s2),
        ..Default::default()
    })
}

fn storage_sizes(s: Settings) -> Result<[usize; 13], CudaError> {
    let n = u128::from(s.count);
    let sizes = [
        n * 12,
        n * 12,
        n * 12,
        n * 12,
        n * 4,
        n * 4,
        n * 4,
        n * 12,
        n * 12,
        u128::from(s.planes) * 16,
        u128::from(s.padded) * 32,
        n * u128::from(s.words) * 4,
        n * 4,
    ];
    let mut output = [0; 13];
    for (out, size) in output.iter_mut().zip(sizes) {
        *out = usize::try_from(size)
            .ok()
            .filter(|&size| size <= isize::MAX as usize)
            .ok_or(CudaError::Capacity)?;
    }
    Ok(output)
}

fn host_values<T>(values: impl ExactSizeIterator<Item = T>) -> Result<Vec<T>, CudaError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| CudaError::HostAllocation)?;
    output.extend(values);
    Ok(output)
}

impl CudaSolver {
    pub(crate) fn new(device: u32, threads: u32) -> Result<Self, CudaError> {
        Ok(Self {
            context: CudaContext::new(device)?,
            threads,
        })
    }

    fn prepare(
        &mut self,
        s: Settings,
        particles: &[Particle],
        planes: &[Plane],
    ) -> Result<(), CudaError> {
        for (slot, size) in storage_sizes(s)?.into_iter().enumerate() {
            self.context.inner.reserve(slot as u32, size)?;
        }
        let positions: Vec<_> = host_values(particles.iter().map(|p| array(p.position_m)))?;
        let velocities: Vec<_> = host_values(particles.iter().map(|p| array(p.velocity_m_s)))?;
        let planes = host_values(
            planes
                .iter()
                .map(|p| [p.normal().x, p.normal().y, p.normal().z, p.offset()]),
        )?;
        self.context.inner.upload(0, &positions)?;
        self.context.inner.upload(1, &velocities)?;
        self.context.inner.upload(9, &planes)
    }

    fn operation(&mut self, s: Settings, operation: u32) -> Result<(), StepError> {
        self.context.inner.launch(s, operation, self.threads)?;
        if operation != 2 && operation != 3 {
            let errors: Vec<u32> = self.context.inner.download(12, s.count as usize)?;
            if let Some(code) = errors.into_iter().find(|&code| code != 0) {
                let quantity = QUANTITIES
                    .get(code as usize)
                    .ok_or(CudaError::Verification {
                        index: code as usize,
                    })?;
                return Err(NumericalError::NonFiniteResult { quantity }.into());
            }
        }
        Ok(())
    }

    fn discover(&mut self, mut s: Settings, positions: &[Vec3]) -> Result<(), StepError> {
        let mut largest = 0.0_f64;
        let mut lower = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut upper = Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX);
        for p in positions {
            largest = largest.max(f64::from(p.x.abs().max(p.y.abs()).max(p.z.abs())));
            lower = Vec3::new(lower.x.min(p.x), lower.y.min(p.y), lower.z.min(p.z));
            upper = Vec3::new(upper.x.max(p.x), upper.y.max(p.y), upper.z.max(p.z));
        }
        s.validate_pairs =
            u32::from(positions.len() > 1 && !(upper - lower).length_squared().is_finite());
        let minimum = f64::from(s.radius)
            .max(2.0_f64.powi(-72))
            .max(largest * 2.0_f64.powi(-50));
        s.width = 2.0_f64.powi(-72);
        while s.width < minimum {
            s.width *= 2.0;
        }
        self.operation(s, 2)?;
        let mut span = 2_u64;
        while span <= u64::from(s.padded) {
            s.span = span as u32;
            s.stride = s.span / 2;
            while s.stride != 0 {
                self.operation(s, 3)?;
                s.stride /= 2;
            }
            span *= 2;
        }
        self.operation(s, 4)
    }

    pub(crate) fn step<const PROFILE: bool>(
        &mut self,
        configuration: &ResolvedConfiguration,
        particles: &[Particle],
        planes: &[Plane],
        cancelled: &mut impl FnMut() -> bool,
        clock: &mut StageClock<PROFILE>,
    ) -> Result<(Vec<Particle>, CflDiagnostic), StepError> {
        let s = settings(configuration, particles.len(), planes.len())?;
        self.prepare(s, particles, planes)?;
        self.operation(s, 0)?;
        let velocities = host_values(
            self.context
                .inner
                .download(3, particles.len())?
                .into_iter()
                .map(vector),
        )?;
        let cfl = cfl_diagnostic(
            &velocities,
            s.timestep,
            configuration.derived.particle_diameter_m,
            configuration.numerical.cfl_factor,
        )?;
        check_cancellation(cancelled)?;
        clock.mark(Stage::ValidationAndAcceleration);
        self.operation(s, 1)?;
        check_cancellation(cancelled)?;
        clock.mark(Stage::Prediction);
        let positions = host_values(
            self.context
                .inner
                .download(2, particles.len())?
                .into_iter()
                .map(vector),
        )?;
        self.discover(s, &positions)?;
        check_cancellation(cancelled)?;
        clock.mark(Stage::NeighbourSearch);
        for _ in 0..configuration.numerical.solver_iterations {
            for op in [5, 6, 7, 8] {
                self.operation(s, op)?;
            }
            check_cancellation(cancelled)?;
            clock.mark(Stage::Jacobi);
            self.operation(s, 9)?;
            self.operation(s, 10)?;
            check_cancellation(cancelled)?;
            clock.mark(Stage::Projection);
        }
        self.operation(s, 11)?;
        let positions: Vec<[f32; 3]> = self.context.inner.download(2, particles.len())?;
        let velocities: Vec<[f32; 3]> = self.context.inner.download(3, particles.len())?;
        let output = host_values(particles.iter().zip(positions).zip(velocities).map(
            |((old, p), v)| Particle {
                identity: old.identity,
                position_m: vector(p),
                velocity_m_s: vector(v),
            },
        ))?;
        Ok((output, cfl))
    }
}

const QUANTITIES: [&str; 52] = [
    "unused",
    "neighbour search displacement",
    "squared vector length",
    "vector length",
    "squared support radius",
    "squared kernel distance",
    "Poly6 support term",
    "support radius to the ninth power",
    "Poly6 denominator",
    "Poly6 coefficient",
    "cubed Poly6 support term",
    "Poly6 value",
    "support radius to the sixth power",
    "Spiky denominator",
    "Spiky coefficient",
    "Spiky support difference",
    "Spiky scale",
    "Spiky direction",
    "particle displacement",
    "acceleration velocity delta",
    "post-acceleration velocity",
    "predicted displacement",
    "predicted position",
    "ordered normalised density",
    "normalised density ratio",
    "density constraint",
    "reciprocal normalised rest density",
    "ordered owner direction sum",
    "owner substituted direction",
    "neighbour substituted direction",
    "squared substituted direction",
    "ordered squared-direction sum",
    "relaxed multiplier denominator",
    "relaxed multiplier",
    "artificial-pressure kernel ratio",
    "artificial-pressure shaped ratio",
    "artificial-pressure value",
    "pair multiplier sum",
    "correction coefficient",
    "pair correction contribution",
    "ordered correction sum",
    "particle position correction",
    "scaled position correction",
    "physical density",
    "candidate position",
    "plane dot product",
    "plane constraint",
    "plane projection correction",
    "plane-projected position",
    "reconstructed displacement",
    "reciprocal timestep",
    "reconstructed velocity",
];

#[cfg(test)]
#[path = "solver_tests.rs"]
mod tests;
