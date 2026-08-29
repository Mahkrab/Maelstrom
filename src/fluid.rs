//! # Fluid density and position correction
//!
//! PBF calculations that turn a position snapshot and fixed interaction lists
//! into densities, constraint multipliers and position corrections.
//!
//! [`compute_jacobi_stage`] uses two passes: it finishes every multiplier
//! before calculating any correction. This keeps all particles on the same
//! Jacobi snapshot.

use crate::{NumericalError, kernel, math::Vec3, validation};

/// ### Fixed interaction lists for a particle snapshot
///
/// `rows[i]` contains the particles that may interact with particle `i`.
/// Indices in each row are unique, exclude `i`, and appear in ascending order.
/// The type validates supplied lists but does not search for neighbours.
#[derive(Clone, Debug, PartialEq)]
pub struct InteractionSets {
    rows: Vec<Vec<usize>>,
}

impl InteractionSets {
    /// ### Validates and stores interaction rows
    ///
    /// ### Errors
    ///
    /// Returns [`NumericalError`] when a row:
    ///
    /// - refers to a particle outside `rows`;
    /// - contains its own particle;
    /// - contains duplicate or descending indices.
    pub fn new(rows: Vec<Vec<usize>>) -> Result<Self, NumericalError> {
        let particle_count = rows.len();
        for (particle, neighbours) in rows.iter().enumerate() {
            let mut previous = None;
            for &neighbour in neighbours {
                if neighbour >= particle_count {
                    return Err(NumericalError::InteractionIndexOutOfBounds {
                        particle,
                        neighbour,
                        particle_count,
                    });
                }
                if neighbour == particle {
                    return Err(NumericalError::SelfInteraction { particle });
                }
                if let Some(previous) = previous
                    && previous >= neighbour
                {
                    return Err(NumericalError::InteractionOrder {
                        particle,
                        previous,
                        next: neighbour,
                    });
                }
                previous = Some(neighbour);
            }
        }

        Ok(Self { rows })
    }

    /// Returns the number of particle rows.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns one particle's interaction row, or `None` if it does not exist.
    #[must_use]
    pub fn for_particle(&self, particle: usize) -> Option<&[usize]> {
        self.rows.get(particle).map(Vec::as_slice)
    }
}

/// ### Normalised and physical density from the same support sum
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct DensityEstimate {
    /// Mass-divided density in inverse cubic metres.
    pub normalised: f32,
    /// Physical density in kilograms per cubic metre.
    pub physical: f32,
}

/// ### Spiky directions used by one particle's density constraint
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct SubstitutedDirections {
    owner: usize,
    owner_direction: Vec3,
    neighbour_directions: Vec<(usize, Vec3)>,
}

impl SubstitutedDirections {
    /// Returns the particle whose constraint produced these directions.
    #[must_use]
    pub const fn owner(&self) -> usize {
        self.owner
    }

    /// Returns the direction applied to the owning particle.
    #[must_use]
    pub const fn owner_direction(&self) -> Vec3 {
        self.owner_direction
    }

    /// Returns `(particle index, direction)` pairs in ascending index order.
    #[must_use]
    pub fn neighbour_directions(&self) -> &[(usize, Vec3)] {
        &self.neighbour_directions
    }
}

/// ### Artificial-pressure settings for one support radius
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtificialPressure {
    strength: f32,
    reference_separation: f32,
    exponent: f32,
    support_radius: f32,
    reference_weight: f32,
}

impl ArtificialPressure {
    /// ### Creates validated artificial-pressure settings
    ///
    /// `reference_separation` must be strictly between zero and
    /// `support_radius` so its Poly6 weight can be used a a denominator.
    ///
    /// ### Errors
    ///
    /// Returns [`NumericalError`] when an input is non-finite, when `strength`
    /// is negative, when `exponent` or `support_radius` is not positive, or
    /// when the reference sepration lies outside `(0, support_radius)`.
    pub fn new(
        strength: f32,
        reference_separation: f32,
        exponent: f32,
        support_radius: f32,
    ) -> Result<Self, NumericalError> {
        validation::non_negative_scalar("artificial-pressure strength", strength)?;
        validation::positive_scalar("support radius", support_radius)?;
        validation::finite_scalar(
            "artificial-pressure reference separation",
            reference_separation,
        )?;
        if reference_separation <= 0.0 || reference_separation >= support_radius {
            return Err(NumericalError::InvalidReferenceSeparation {
                separation: reference_separation,
                support_radius,
            });
        }
        validation::positive_scalar("artificial-pressure exponent", exponent)?;

        let reference_weight = kernel::poly6(reference_separation, support_radius)?;
        validation::positive_scalar("artificial-pressure reference weight", reference_weight)?;

        Ok(Self {
            strength,
            reference_separation,
            exponent,
            support_radius,
            reference_weight,
        })
    }

    /// Returns the artificial-pressure strength.
    #[must_use]
    pub const fn strength(self) -> f32 {
        self.strength
    }

    /// Returns the distance used for the reference Poly6 weight.
    #[must_use]
    pub const fn reference_separation(self) -> f32 {
        self.reference_separation
    }

    /// Returns the exponent applied to the Poly6 weight ratio.
    #[must_use]
    pub const fn exponent(self) -> f32 {
        self.exponent
    }

    /// ### Calculates artificial pressure at a particle separation
    ///
    /// Separations at or beyond kernel support return exactly `0.0`.
    ///
    /// ### Errors
    ///
    /// Returns [`NumericalError`] when `distance` is negative or non-finite, or
    /// when the kernel-ratio calculation becomes non-finite.
    pub fn evaluate(self, distance: f32) -> Result<f32, NumericalError> {
        let weight = kernel::poly6(distance, self.support_radius)?;
        if weight == 0.0 || self.strength == 0.0 {
            return Ok(0.0);
        }

        let ratio = validation::finite_result(
            "artificial-pressure kernel ratio",
            weight / self.reference_weight,
        )?;
        let shaped_ratio = validation::finite_result(
            "artificial-pressure shaped ratio",
            ratio.powf(self.exponent),
        )?;
        validation::finite_result("artificial-pressure value", -self.strength * shaped_ratio)
    }
}

/// ### Settings shared by every particle in a Jacobi calculation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbfParameters {
    support_radius: f32,
    normalised_rest_density: f32,
    relaxation: f32,
    artificial_pressure: ArtificialPressure,
}

impl PbfParameters {
    /// ### Creates validated PBF settings/config
    ///
    /// ### Errors
    ///
    /// Returns [`NumericalError`] when the support radius, rest density or
    /// relaxation is not positive and finite, or when the artificial-pressure
    /// settings are invalid.
    pub fn new(
        support_radius: f32,
        normalised_rest_density: f32,
        relaxation: f32,
        artificial_pressure_strength: f32,
        artificial_pressure_reference_separation: f32,
        artificial_pressure_exponent: f32,
    ) -> Result<Self, NumericalError> {
        validation::positive_scalar("support radius", support_radius)?;
        validation::positive_scalar("normalised rest density", normalised_rest_density)?;
        validation::positive_scalar("relaxation", relaxation)?;
        let artificial_pressure = ArtificialPressure::new(
            artificial_pressure_strength,
            artificial_pressure_reference_separation,
            artificial_pressure_exponent,
            support_radius,
        )?;

        Ok(Self {
            support_radius,
            normalised_rest_density,
            relaxation,
            artificial_pressure,
        })
    }

    /// Returns the kernel support radius in metres.
    #[must_use]
    pub const fn support_radius(self) -> f32 {
        self.support_radius
    }

    /// Returns the mass-divided rest density in inverse cubic metres
    #[must_use]
    pub const fn normalised_rest_density(self) -> f32 {
        self.normalised_rest_density
    }

    /// Returns the multiplier relaxation term in inverse square metres.
    #[must_use]
    pub const fn relaxation(self) -> f32 {
        self.relaxation
    }

    /// Returns the validated artificial-pressure settings.
    #[must_use]
    pub const fn artificial_pressure(self) -> ArtificialPressure {
        self.artificial_pressure
    }
}

/// ### Results calculated from one position snapshot
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct JacobiStage {
    /// Mass-divided density for each particle.
    pub normalised_densities: Vec<f32>,
    /// Signed density constraint for each particle.
    pub constraints: Vec<f32>,
    /// Relaxed constraint multiplier for each particle.
    pub multipliers: Vec<f32>,
    /// Position correction for each particle.
    pub corrections: Vec<Vec3>,
}

/// ### Filters interactions using current particle positions
///
/// Only fixed interactions whose current separation satisfies
/// `distance < support_radius` are returned. The original ascending order is
/// preserved.
///
/// ### Errors
///
/// Returns [`NumericalError`] for invalid intraction data, a missing particle,
/// a non-positive support radius, a non-finite position or a non-finite distance.
pub fn active_interactions(
    particle: usize,
    positions: &[Vec3],
    fixed_interactions: &InteractionSets,
    support_radius: f32,
) -> Result<Vec<usize>, NumericalError> {
    validate_snapshot(positions, fixed_interactions)?;
    validation::positive_scalar("support radius", support_radius)?;
    let fixed = interaction_row(particle, fixed_interactions)?;
    active_from_validated_snapshot(particle, positions, fixed, support_radius)
}

/// ### Calculates normalised and physical particle density
///
/// The support sum includes the particle itself exactly once. Fixed
/// interactions outside current kernel support do not contribute.
///
/// ### Errors
///
/// Returns [`NumericalError`] under the same conditions as
/// [`normalised_density`], when `particle_mass` is not positive and finite, or
/// when physical density becomes non-finite.
pub fn estimate_density(
    particle: usize,
    positions: &[Vec3],
    fixed_interactions: &InteractionSets,
    support_radius: f32,
    particle_mass: f32,
) -> Result<DensityEstimate, NumericalError> {
    validation::positive_scalar("particle mass", particle_mass)?;
    let normalised = normalised_density(particle, positions, fixed_interactions, support_radius)?;
    let physical = validation::finite_result("physical density", particle_mass * normalised)?;
    Ok(DensityEstimate {
        normalised,
        physical,
    })
}

/// ### Calculates massdivided particle density
///
/// Terms are added in ascending particle-index order, including the particle's
/// own Poly6 weight at its ordered position.
///
/// ### Errors
///
/// Returns [`NumericalError`] for invalid interaction data, a missing particle,
/// a non-positive support radius, a non-finite position or a non-finite result.
pub fn normalised_density(
    particle: usize,
    positions: &[Vec3],
    fixed_interactions: &InteractionSets,
    support_radius: f32,
) -> Result<f32, NumericalError> {
    validate_snapshot(positions, fixed_interactions)?;
    validation::positive_scalar("support radius", support_radius)?;
    let fixed = interaction_row(particle, fixed_interactions)?;
    let active = active_from_validated_snapshot(particle, positions, fixed, support_radius)?;
    normalised_density_from_active(particle, positions, &active, support_radius)
}

/// ### Calculates the signed density constraint
///
/// The result is `normalised_density / normalised_rest_density - 1` and is not
/// clamped. A negative result therefore remains meaningful.
///
/// ### Errors
///
/// Returns [`NumericalError`] when density is negative or non-finite, rest
/// density is not positive and finite, or the calculation becomes non-finite.
pub fn density_constraint(
    normalised_density: f32,
    normalised_rest_density: f32,
) -> Result<f32, NumericalError> {
    validation::non_negative_scalar("normalised density", normalised_density)?;
    validation::positive_scalar("normalised rest density", normalised_rest_density)?;
    let ratio = validation::finite_result(
        "normalised density ratio",
        normalised_density / normalised_rest_density,
    )?;
    validation::finite_result("density constraint", ratio - 1.0)
}

/// ### Calculates Spiky directions for one density constraint
///
/// The returned owner and neighbour directions use the active interactions at
/// the current positions. Coincident particles receive zero directions from
/// [`crate::kernel::spiky_direction`].
///
/// ### Errors
///
/// Returns [`NumericalError`] for invalid interaction data, a missing particle,
/// invalid radii or rest density, non-finite positions, or non-finite results.
pub fn substituted_directions(
    particle: usize,
    positions: &[Vec3],
    fixed_interactions: &InteractionSets,
    support_radius: f32,
    normalised_rest_density: f32,
) -> Result<SubstitutedDirections, NumericalError> {
    validate_snapshot(positions, fixed_interactions)?;
    validation::positive_scalar("support radius", support_radius)?;
    validation::positive_scalar("normalised rest density", normalised_rest_density)?;
    let fixed = interaction_row(particle, fixed_interactions)?;
    let active = active_from_validated_snapshot(particle, positions, fixed, support_radius)?;
    directions_from_active(
        particle,
        positions,
        &active,
        support_radius,
        normalised_rest_density,
    )
}

/// ### Calculates a relaxed constraint multiplier
///
/// Squared directions are summed in ascending particle-index order, including
/// the owner at its ordered position.
///
/// ### Errors
///
/// Returns [`NumericalError`] when `constraint` is non-finite, `relaxation` is
/// not positive and finite, or the calculation becomes non-finite.
pub fn relaxed_multiplier(
    constraint: f32,
    directions: &SubstitutedDirections,
    relaxation: f32,
) -> Result<f32, NumericalError> {
    validation::finite_scalar("density constraint", constraint)?;
    validation::positive_scalar("relaxation", relaxation)?;

    let mut direction_sum = 0.0_f32;
    let mut owner_added = false;
    for &(neighbour, direction) in &directions.neighbour_directions {
        if !owner_added && directions.owner < neighbour {
            direction_sum = add_squared_direction(direction_sum, directions.owner_direction)?;
            owner_added = true;
        }
        direction_sum = add_squared_direction(direction_sum, direction)?;
    }
    if !owner_added {
        direction_sum = add_squared_direction(direction_sum, directions.owner_direction)?;
    }

    let denominator =
        validation::finite_result("relaxed multiplier denominator", direction_sum + relaxation)?;
    validation::finite_result("relaxed multiplier", -constraint / denominator)
}

/// ### Calculates one complete Jacobi stage
///
/// The function reads `positions` without changing it. It calculates all
/// densities, constraints and multipliers first, then calculates every
/// correction from the same positions and the completed multiplier list.
///
/// ### Errors
///
/// Returns [`NumericalError`] when the position and interaction counts differ,
/// supplied interaction data is invalid, any position is non-finite, or any
/// density, constraint, multiplier, pressure or correction calculation fails.
pub fn compute_jacobi_stage(
    positions: &[Vec3],
    fixed_interactions: &InteractionSets,
    parameters: PbfParameters,
) -> Result<JacobiStage, NumericalError> {
    validate_snapshot(positions, fixed_interactions)?;
    let particle_count = positions.len();
    let mut active_sets = Vec::with_capacity(particle_count);
    for particle in 0..particle_count {
        let fixed = interaction_row(particle, fixed_interactions)?;
        active_sets.push(active_from_validated_snapshot(
            particle,
            positions,
            fixed,
            parameters.support_radius,
        )?);
    }

    let mut normalised_densities = Vec::with_capacity(particle_count);
    let mut constraints = Vec::with_capacity(particle_count);
    let mut multipliers = Vec::with_capacity(particle_count);

    for (particle, active) in active_sets.iter().enumerate() {
        let density =
            normalised_density_from_active(particle, positions, active, parameters.support_radius)?;
        let constraint = density_constraint(density, parameters.normalised_rest_density)?;
        let directions = directions_from_active(
            particle,
            positions,
            active,
            parameters.support_radius,
            parameters.normalised_rest_density,
        )?;
        let multiplier = relaxed_multiplier(constraint, &directions, parameters.relaxation)?;

        normalised_densities.push(density);
        constraints.push(constraint);
        multipliers.push(multiplier);
    }

    // Corrections need both particless multipliers, so finish the full list first.
    let mut corrections = Vec::with_capacity(particle_count);
    for (particle, active) in active_sets.iter().enumerate() {
        let mut correction_sum = Vec3::ZERO;
        for &neighbour in active {
            let (displacement, distance) =
                displacement_and_distance(particle, neighbour, positions)?;
            let spiky = kernel::spiky_direction(displacement, parameters.support_radius)?;
            let pressure = parameters.artificial_pressure.evaluate(distance)?;
            let multiplier_pair = validation::finite_result(
                "pair multiplier sum",
                multipliers[particle] + multipliers[neighbour],
            )?;
            let coefficient =
                validation::finite_result("correction coefficient", multiplier_pair + pressure)?;
            let pair_correction = validation::finite_vector_result(
                "pair correction contribution",
                spiky * coefficient,
            )?;
            correction_sum = validation::finite_vector_result(
                "ordered correction sum",
                correction_sum + pair_correction,
            )?;
        }
        let reciprocal_rest_density = validation::finite_result(
            "reciprocal normalised rest density",
            1.0 / parameters.normalised_rest_density,
        )?;
        corrections.push(validation::finite_vector_result(
            "particle position correction",
            correction_sum * reciprocal_rest_density,
        )?);
    }

    Ok(JacobiStage {
        normalised_densities,
        constraints,
        multipliers,
        corrections,
    })
}

fn validate_snapshot(
    positions: &[Vec3],
    fixed_interactions: &InteractionSets,
) -> Result<(), NumericalError> {
    if fixed_interactions.particle_count() != positions.len() {
        return Err(NumericalError::InteractionSetCount {
            expected: positions.len(),
            actual: fixed_interactions.particle_count(),
        });
    }
    for &position in positions {
        validation::finite_vector("Jacobi snapshot position", position)?;
    }
    Ok(())
}

fn interaction_row(
    particle: usize,
    fixed_interactions: &InteractionSets,
) -> Result<&[usize], NumericalError> {
    fixed_interactions
        .for_particle(particle)
        .ok_or(NumericalError::ParticleIndexOutOfBounds {
            particle,
            particle_count: fixed_interactions.particle_count(),
        })
}

fn active_from_validated_snapshot(
    particle: usize,
    positions: &[Vec3],
    fixed: &[usize],
    support_radius: f32,
) -> Result<Vec<usize>, NumericalError> {
    let mut active = Vec::with_capacity(fixed.len());
    for &neighbour in fixed {
        let (_, distance) = displacement_and_distance(particle, neighbour, positions)?;
        if distance < support_radius {
            active.push(neighbour);
        }
    }
    Ok(active)
}

fn normalised_density_from_active(
    particle: usize,
    positions: &[Vec3],
    active: &[usize],
    support_radius: f32,
) -> Result<f32, NumericalError> {
    let self_weight = kernel::poly6(0.0, support_radius)?;
    let mut density = 0.0_f32;
    let mut self_added = false;

    for &neighbour in active {
        if !self_added && particle < neighbour {
            density =
                validation::finite_result("ordered normalised density", density + self_weight)?;
            self_added = true;
        }
        let (_, distance) = displacement_and_distance(particle, neighbour, positions)?;
        let weight = kernel::poly6(distance, support_radius)?;
        density = validation::finite_result("ordered normalised density", density + weight)?;
    }
    if !self_added {
        density = validation::finite_result("ordered normalised density", density + self_weight)?;
    }

    Ok(density)
}

fn directions_from_active(
    particle: usize,
    positions: &[Vec3],
    active: &[usize],
    support_radius: f32,
    normalised_rest_density: f32,
) -> Result<SubstitutedDirections, NumericalError> {
    let reciprocal_rest_density = validation::finite_result(
        "reciprocal normalised rest density",
        1.0 / normalised_rest_density,
    )?;
    let mut owner_sum = Vec3::ZERO;
    let mut neighbour_directions = Vec::with_capacity(active.len());

    for &neighbour in active {
        let (displacement, _) = displacement_and_distance(particle, neighbour, positions)?;
        let spiky = kernel::spiky_direction(displacement, support_radius)?;
        owner_sum =
            validation::finite_vector_result("ordered owner direction sum", owner_sum + spiky)?;
        neighbour_directions.push((
            neighbour,
            validation::finite_vector_result(
                "neighbour substituted direction",
                (-spiky) * reciprocal_rest_density,
            )?,
        ));
    }

    let owner_direction = validation::finite_vector_result(
        "owner substituted direction",
        owner_sum * reciprocal_rest_density,
    )?;
    Ok(SubstitutedDirections {
        owner: particle,
        owner_direction,
        neighbour_directions,
    })
}

fn add_squared_direction(sum: f32, direction: Vec3) -> Result<f32, NumericalError> {
    let squared =
        validation::finite_result("squared substituted direction", direction.length_squared())?;
    validation::finite_result("ordered squared-direction sum", sum + squared)
}

fn displacement_and_distance(
    particle: usize,
    neighbour: usize,
    positions: &[Vec3],
) -> Result<(Vec3, f32), NumericalError> {
    let displacement = validation::finite_vector_result(
        "particle displacement",
        positions[particle] - positions[neighbour],
    )?;
    let distance = displacement.checked_length("particle displacement")?;
    Ok((displacement, distance))
}
