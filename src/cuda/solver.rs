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
    planes: u32,
    validate_pairs: u32,
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
    entries: u64,
}
const _: () = {
    assert!(size_of::<Settings>() == 88 && align_of::<Settings>() == 8);
    assert!(std::mem::offset_of!(Settings, width) == 64);
};

#[derive(Debug)]
pub(crate) struct CudaSolver {
    pub(crate) context: CudaContext,
    threads: u32,
    neighbour_capacity: usize,
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

fn storage_sizes(s: Settings) -> Result<[usize; 14], CudaError> {
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
        u128::from(s.entries) * 4,
        n * 4,
        (n + 1) * 8,
    ];
    let mut output = [0; 14];
    for (out, size) in output.iter_mut().zip(sizes) {
        *out = usize::try_from(size)
            .ok()
            .filter(|&size| size <= isize::MAX as usize)
            .ok_or(CudaError::Capacity)?;
    }
    Ok(output)
}

/// Replace independently counted row lengths with checked exclusive offsets.
fn prefix_offsets(counts: &mut [u64], particles: u32) -> Result<u64, CudaError> {
    if counts.len() != particles as usize + 1 || counts.first() != Some(&0) {
        return Err(CudaError::NeighbourStorage { particle: 0 });
    }
    let mut total = 0_u64;
    for (index, value) in counts.iter_mut().enumerate().skip(1) {
        if *value > u64::from(particles.saturating_sub(1)) {
            return Err(CudaError::NeighbourStorage {
                particle: index - 1,
            });
        }
        total = total.checked_add(*value).ok_or(CudaError::Capacity)?;
        *value = total;
    }
    usize::try_from(total)
        .ok()
        .and_then(|entries| entries.checked_mul(size_of::<u32>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(CudaError::Capacity)?;
    Ok(total)
}

fn host_values<T>(values: impl ExactSizeIterator<Item = T>) -> Result<Vec<T>, CudaError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| CudaError::HostAllocation)?;
    output.extend(values);
    Ok(output)
}

fn diagnostic_error(record: u64, particles: u32) -> Result<(), StepError> {
    if record == u64::MAX {
        return Ok(());
    }
    let particle = (record >> 32) as usize;
    let quantity = record as u32;
    if particle >= particles as usize || quantity == 0 || quantity > 52 {
        return Err(CudaError::Diagnostic { particle, quantity }.into());
    }
    if quantity == 52 {
        return Err(CudaError::NeighbourStorage { particle }.into());
    }
    Err(NumericalError::NonFiniteResult {
        quantity: QUANTITIES[quantity as usize],
    }
    .into())
}

impl CudaSolver {
    pub(crate) fn new(device: u32, threads: u32) -> Result<Self, CudaError> {
        Ok(Self {
            context: CudaContext::new(device)?,
            threads,
            neighbour_capacity: 0,
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
        self.context.inner.upload(9, &planes)?;
        self.context.inner.begin_attempt()
    }

    fn operation(&mut self, s: Settings, operation: u32) -> Result<(), StepError> {
        if operation == 2 || operation == 3 {
            return self
                .context
                .inner
                .launch(s, operation, self.threads)
                .map_err(Into::into);
        }
        let record = self
            .context
            .inner
            .checked_launch(s, operation, self.threads)?;
        diagnostic_error(record, s.count)
    }

    fn group(&mut self, s: Settings, first_operation: u32) -> Result<(), StepError> {
        let record = self
            .context
            .inner
            .checked_group(s, first_operation, self.threads)?;
        diagnostic_error(record, s.count)
    }

    fn discover(&mut self, s: &mut Settings, positions: &[Vec3]) -> Result<(), StepError> {
        s.entries = 0;
        if s.count == 0 {
            self.context.inner.upload(13, &[0_u64])?;
            return Ok(());
        }
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
        self.operation(*s, 2)?;
        if s.padded > 1 {
            self.context.inner.launch(*s, 13, self.threads)?;
        }
        self.operation(*s, 4)?;
        let mut offsets: Vec<u64> = self.context.inner.download(13, s.count as usize + 1)?;
        s.entries = prefix_offsets(&mut offsets, s.count)?;
        let required = storage_sizes(*s)?[11];
        // Membership changes as the block moves. Geometric growth avoids a
        // device reallocation for every small increase, without capping a row.
        let capacity = if required > self.neighbour_capacity {
            self.neighbour_capacity
                .checked_add(self.neighbour_capacity / 2)
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .unwrap_or(required)
                .max(required)
        } else {
            self.neighbour_capacity
        };
        self.context.inner.reserve(11, capacity)?;
        self.neighbour_capacity = capacity;
        self.context.inner.upload(13, &offsets)?;
        self.operation(*s, 12)
    }

    pub(crate) fn step<const PROFILE: bool>(
        &mut self,
        configuration: &ResolvedConfiguration,
        particles: &[Particle],
        planes: &[Plane],
        cancelled: &mut impl FnMut() -> bool,
        clock: &mut StageClock<PROFILE>,
    ) -> Result<(Vec<Particle>, CflDiagnostic), StepError> {
        let mut s = settings(configuration, particles.len(), planes.len())?;
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
        self.discover(&mut s, &positions)?;
        check_cancellation(cancelled)?;
        clock.mark(Stage::NeighbourSearch);
        for _ in 0..configuration.numerical.solver_iterations {
            self.group(s, 5)?;
            check_cancellation(cancelled)?;
            clock.mark(Stage::Jacobi);
            self.group(s, 9)?;
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
