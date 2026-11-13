//! Accepted device vectors survive timesteps alongside Simulation's host mirror.
//! Attempts write separate working vectors. Only final host validation permits
//! promotion, so cancellation and retries never depend on partial device output.

use super::{CudaContext, CudaError};
use crate::{
    NumericalError,
    boundary::Plane,
    config::ResolvedConfiguration,
    diagnostics::{CflDiagnostic, cfl_from_maximum_speed},
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
    xsph: f32,
    vorticity: f32,
    neighbour_layout: u32,
    grid_min: [i64; 3],
    grid_dims: [u32; 3],
    grid_cells: u32,
    tile_base: u64,
}
const _: () = {
    assert!(size_of::<Settings>() == 152 && align_of::<Settings>() == 8);
    assert!(std::mem::offset_of!(Settings, width) == 64);
    assert!(std::mem::offset_of!(Settings, grid_min) == 104);
    assert!(std::mem::offset_of!(Settings, tile_base) == 144);
};

#[derive(Debug)]
pub(crate) struct CudaSolver {
    pub(crate) context: CudaContext,
    threads: u32,
    neighbour_capacity: usize,
    index_capacity: usize,
    resident: bool,
    force_sorted: bool,
    host_positions: Vec<[f32; 3]>,
    host_velocities: Vec<u32>,
    host_particles: Vec<Particle>,
}

pub(crate) struct CudaStepOutput {
    pub(crate) particles: Vec<Particle>,
    pub(crate) cfl: CflDiagnostic,
    pub(crate) planes_checked: bool,
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
        xsph: if model.xsph_viscosity_enabled {
            model.xsph_viscosity_coefficient
        } else {
            0.0
        },
        vorticity: if model.vorticity_confinement_enabled {
            model.vorticity_confinement_strength_m_s
        } else {
            0.0
        },
        acceleration: array(model.external_acceleration_m_s2),
        ..Default::default()
    })
}

// Dense-grid admission is bounded both absolutely and relative to occupied
// particles. Sparse volumes retain the existing sorted GPU path. Width is the
// shared exact power of two >= radius, keeping the adjacent-cell radius at one.
const MAX_GRID_CELLS: u64 = 16 * 1024 * 1024;
fn tile_base(s: Settings) -> usize {
    if s.tile_base == 0 {
        s.padded as usize * 4
    } else {
        s.tile_base as usize
    }
}
fn configure_grid(s: &mut Settings, lower: Vec3, upper: Vec3) {
    if s.count == 0 || !s.width.is_finite() || s.width < f64::from(s.radius) || s.width <= 0.0 {
        return;
    }
    let mut cells = 1_u64;
    for (axis, (low, high)) in array(lower).into_iter().zip(array(upper)).enumerate() {
        let lo = (f64::from(low) / s.width).floor();
        let hi = (f64::from(high) / s.width).floor();
        // The width policy bounds live coordinates by 2^50. Check before any
        // integer cast so this helper also rejects malformed/extreme settings.
        if !lo.is_finite()
            || !hi.is_finite()
            || lo < -(1_i64 << 51) as f64
            || hi > (1_i64 << 51) as f64
            || lo > hi
        {
            return;
        }
        let dimension = hi as i64 - lo as i64 + 1;
        let Ok(dimension) = u32::try_from(dimension) else {
            return;
        };
        let Some(product) = cells.checked_mul(u64::from(dimension)) else {
            return;
        };
        cells = product;
        if cells > MAX_GRID_CELLS || cells > (u64::from(s.count) * 8).max(4096) {
            return;
        }
        s.grid_min[axis] = lo as i64;
        s.grid_dims[axis] = dimension;
    }
    s.grid_cells = cells as u32;
    let words = 3 * cells + 1 + 2 * u64::from(s.count) + (cells + 1).div_ceil(256);
    s.tile_base = words.div_ceil(2);
}

fn max_index_bytes(s: Settings) -> Result<usize, CudaError> {
    let n = u128::from(s.count);
    let cells = u128::from(MAX_GRID_CELLS).min((n * 8).max(4096));
    let words = 3 * cells + 1 + 2 * n + (cells + 1).div_ceil(256);
    let grid = (words.div_ceil(2) + n.div_ceil(32) + 1) * 8;
    let limit = grid.max(u128::from(s.padded) * 64);
    usize::try_from(limit)
        .ok()
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or(CudaError::Capacity)
}

fn storage_sizes(s: Settings) -> Result<[usize; 14], CudaError> {
    let n = u128::from(s.count);
    let velocity = if n == 0 { 0 } else { n * 12 + 16 };
    let sizes = [
        n * 12,
        velocity,
        n * 12,
        velocity,
        n * 4,
        n * 4,
        n * 4,
        (n * 12).max(32),
        // Candidate scratch first holds three Jacobi scalars and a deferred
        // error word; position/velocity candidates later use its vector prefix.
        n * 16,
        u128::from(s.planes) * 16,
        if s.grid_cells == 0 {
            u128::from(s.padded) * 64
        } else {
            (u128::from(s.tile_base) + u128::from(s.count).div_ceil(32) + 1) * 8
        },
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
#[cfg(test)]
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
    if particle >= particles as usize || quantity == 0 || quantity > 70 {
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
            index_capacity: 0,
            resident: false,
            host_positions: Vec::new(),
            host_velocities: Vec::new(),
            host_particles: Vec::new(),
            force_sorted: std::env::var_os("MAELSTROM_CUDA_SPATIAL_INDEX").as_deref()
                == Some(std::ffi::OsStr::new("sort")),
        })
    }

    fn prepare(
        &mut self,
        s: Settings,
        particles: &[Particle],
        planes: &[Plane],
    ) -> Result<(), CudaError> {
        self.resident = false;
        for (slot, size) in storage_sizes(s)?.into_iter().enumerate() {
            // The prediction bounds select and reserve the index later. Avoid
            // a full legacy cell allocation before a compact-grid first step.
            if slot != 10 {
                self.context.inner.reserve(slot as u32, size)?;
            }
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

    /// Promote only after Simulation's final checks; host installation follows
    /// immediately and cannot fail. Inputs and planes are immutable for its life.
    pub(crate) fn accept(&mut self) -> Result<(), CudaError> {
        self.context.inner.accept()
    }

    /// Called only after device promotion and host snapshot replacement succeed.
    pub(crate) fn recycle_host_particles(&mut self, particles: Vec<Particle>) {
        self.host_particles = particles;
    }

    fn operation(&mut self, s: Settings, operation: u32) -> Result<(), StepError> {
        if operation == 2 || operation == 3 {
            self.reserve_index(s)?;
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

    #[cfg(test)]
    fn discover(&mut self, s: &mut Settings, positions: &[Vec3]) -> Result<(), StepError> {
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
        self.select_grid(s, lower, upper);
        self.discover_with_width(s)
    }

    fn select_grid(&self, s: &mut Settings, lower: Vec3, upper: Vec3) {
        s.grid_cells = 0;
        s.tile_base = 0;
        if !self.force_sorted {
            configure_grid(s, lower, upper);
        }
    }

    fn reserve_index(&mut self, s: Settings) -> Result<(), CudaError> {
        let required = storage_sizes(s)?[10];
        if required > self.index_capacity {
            // Moving bounds can add a few cells every step. Amortise those
            // changes without exceeding either representation's validated cap.
            let limit = max_index_bytes(s)?;
            let capacity = self
                .index_capacity
                .checked_add(self.index_capacity / 2)
                .unwrap_or(limit)
                .max(required)
                .min(limit);
            self.context.inner.reserve(10, capacity)?;
            self.index_capacity = capacity;
        }
        Ok(())
    }

    fn discover_with_width(&mut self, s: &mut Settings) -> Result<(), StepError> {
        s.entries = 0;
        if s.count == 0 {
            self.context.inner.upload(13, &[0_u64])?;
            return Ok(());
        }
        self.reserve_index(*s)?;
        // Cell generation, sorting, counting and checked prefix formation share
        // the stream. The host needs only the total before allocating row storage.
        if s.grid_cells != 0 {
            self.context.inner.launch(*s, 26, self.threads)?;
        } else {
            self.operation(*s, 2)?;
            if s.padded > 1 {
                self.context.inner.launch(*s, 13, self.threads)?;
            }
        }
        let record = self.context.inner.prefix(*s, self.threads, true)?;
        diagnostic_error(record, s.count)?;
        s.entries = if s.neighbour_layout == 1 {
            self.context.inner.download_at::<u64>(
                10,
                tile_base(*s) + (s.count as usize).div_ceil(32),
                1,
            )?[0]
        } else {
            self.context
                .inner
                .download_at::<u64>(13, s.count as usize, 1)?[0]
        };
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
        self.operation(*s, 12)
    }

    pub(crate) fn step<const PROFILE: bool>(
        &mut self,
        configuration: &ResolvedConfiguration,
        particles: &[Particle],
        planes: &[Plane],
        cancelled: &mut impl FnMut() -> bool,
        clock: &mut StageClock<PROFILE>,
    ) -> Result<CudaStepOutput, StepError> {
        let mut s = settings(configuration, particles.len(), planes.len())?;
        if !self.resident {
            self.prepare(s, particles, planes)?;
            self.resident = true;
        } else {
            // Count, identities, configuration and planes belong to one fixed
            // Simulation. Accepted vectors were promoted at its last commit;
            // a failed attempt can overwrite scratch but never these inputs.
            self.context.inner.begin_attempt()?;
        }
        let records = self.context.inner.observe(s, 0, self.threads)?;
        diagnostic_error(records[0], s.count)?;
        let blocks = particles.len().div_ceil(self.threads as usize);
        let summaries: Vec<[f32; 8]> = self.context.inner.download(7, blocks)?;
        // Preserve the old ordering: acceleration diagnostics, completed host
        // observation, then CFL errors, then the first cancellation poll.
        diagnostic_error(records[1], s.count)?;
        let maximum_speed = summaries.iter().map(|s| s[0]).fold(0.0_f32, f32::max);
        let cfl = cfl_from_maximum_speed(
            maximum_speed,
            s.timestep,
            configuration.derived.particle_diameter_m,
            configuration.numerical.cfl_factor,
        )?;
        check_cancellation(cancelled)?;
        clock.mark(Stage::ValidationAndAcceleration);
        let records = self.context.inner.observe(s, 1, self.threads)?;
        diagnostic_error(records[0], s.count)?;
        check_cancellation(cancelled)?;
        clock.mark(Stage::Prediction);
        let summaries: Vec<[f32; 8]> = self.context.inner.download(7, blocks)?;
        let mut largest = 0.0_f64;
        let mut lower = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut upper = Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX);
        for summary in summaries {
            largest = largest.max(f64::from(summary[0]));
            lower = Vec3::new(
                lower.x.min(summary[1]),
                lower.y.min(summary[2]),
                lower.z.min(summary[3]),
            );
            upper = Vec3::new(
                upper.x.max(summary[4]),
                upper.y.max(summary[5]),
                upper.z.max(summary[6]),
            );
        }
        s.validate_pairs =
            u32::from(particles.len() > 1 && !(upper - lower).length_squared().is_finite());
        let minimum = f64::from(s.radius)
            .max(2.0_f64.powi(-72))
            .max(largest * 2.0_f64.powi(-50));
        s.width = 2.0_f64.powi(-72);
        while s.width < minimum {
            s.width *= 2.0;
        }
        // The PC comparison favours compact rows for small inputs. Larger
        // inputs amortise tiled construction across repeated numerical passes.
        s.neighbour_layout = u32::from(particles.len() >= 32_000);
        self.select_grid(&mut s, lower, upper);
        self.discover_with_width(&mut s)?;
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
        self.operation(s, 24)?;
        if s.vorticity != 0.0 {
            check_cancellation(cancelled)?;
            self.operation(s, 16)?;
            check_cancellation(cancelled)?;
            self.operation(s, 17)?;
            check_cancellation(cancelled)?;
            self.operation(s, 15)?;
        }
        if s.xsph != 0.0 {
            check_cancellation(cancelled)?;
            self.operation(s, 14)?;
            check_cancellation(cancelled)?;
            self.operation(s, 15)?;
        }
        self.context
            .inner
            .download_into(2, particles.len(), &mut self.host_positions)?;
        // The trailer is initialized by projection, survives velocity effects,
        // and returns in the same completed copy as the numerical velocities.
        let words = particles
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_add(if particles.is_empty() { 0 } else { 4 }))
            .ok_or(CudaError::Capacity)?;
        self.context
            .inner
            .download_into(3, words, &mut self.host_velocities)?;
        let velocities = &self.host_velocities;
        let planes_checked = if particles.is_empty() {
            true
        } else {
            let index = 3 * particles.len() + (particles.len() & 1);
            let record = u64::from(velocities[index]) | (u64::from(velocities[index + 1]) << 32);
            record == u64::MAX
        };
        let velocities = velocities.chunks_exact(3).take(particles.len()).map(|v| {
            [
                f32::from_bits(v[0]),
                f32::from_bits(v[1]),
                f32::from_bits(v[2]),
            ]
        });
        // The spare is never publicly observable. Failed allocation leaves it
        // intact; failed attempts cannot modify the accepted snapshot.
        let additional = particles.len().saturating_sub(self.host_particles.len());
        super::native::reserve_host(&mut self.host_particles, additional)?;
        self.host_particles.clear();
        self.host_particles.extend(
            particles
                .iter()
                .zip(&self.host_positions)
                .zip(velocities)
                .map(|((old, &p), v)| Particle {
                    identity: old.identity,
                    position_m: vector(p),
                    velocity_m_s: vector(v),
                }),
        );
        Ok(CudaStepOutput {
            particles: std::mem::take(&mut self.host_particles),
            cfl,
            planes_checked,
        })
    }
}

const QUANTITIES: [&str; 71] = [
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
    "neighbour storage",
    "XSPH velocity difference",
    "XSPH kernel weight",
    "XSPH pair contribution",
    "XSPH ordered sum",
    "XSPH velocity correction",
    "XSPH smoothed velocity",
    "vorticity kernel gradient",
    "vorticity velocity difference",
    "vorticity curl contribution",
    "vorticity ordered curl",
    "vorticity magnitude difference",
    "vorticity gradient contribution",
    "vorticity magnitude gradient",
    "vorticity direction",
    "vorticity rotation",
    "vorticity acceleration",
    "vorticity velocity correction",
    "confined velocity",
];

#[cfg(test)]
#[path = "../../tests/unit/cuda/solver_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/residency_tests.rs"]
mod residency_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/correction_tests.rs"]
mod correction_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/search_tests.rs"]
mod search_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/vorticity_tests.rs"]
mod vorticity_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/spatial_sort_tests.rs"]
mod spatial_sort_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/observation_tests.rs"]
mod observation_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/active_tests.rs"]
mod active_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/density_tests.rs"]
mod density_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/plane_tests.rs"]
mod plane_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/mirror_tests.rs"]
mod mirror_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/tiled_tests.rs"]
mod tiled_tests;

#[cfg(test)]
#[path = "../../tests/unit/cuda/compact_grid_tests.rs"]
mod compact_grid_tests;
