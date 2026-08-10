//! # Reference particle lattice
//!
//! Calculates rest density and particle mass from the regular cubic lattice
//! used by Maelstrom's reference setup.

use crate::{NumericalError, kernel::poly6, math::Vec3, validation};

/// Number of offsets inside strict support when `support_radius = 2 * spacing`.
pub const REFERENCE_LATTICE_OFFSET_COUNT: usize = 27;

/// ### Values calculated from the reference lattice
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct ReferenceLattice {
    /// Kernel support radius in metres.
    pub support_radius: f32,
    /// Mass-divided rest density in inverse cubic metres.
    pub normalised_rest_density: f32,
    /// Particle mass in kilograms.
    pub particle_mass: f32,
    /// Number of lattice offsets included in the density sum.
    pub offset_count: usize,
}

/// ### Calculates density and mass from the reference lattice
///
/// The function sets `support_radius = 2 * particle_spacing`, visits offsets
/// `(a, b, c)` in ascending order, and sums the 27 offsets inside strict
/// support. Particle mass is then calculated from `physical_rest_density`.
///
/// # Errors
///
/// Returns [`NumericalError`] when either input is not positive and finite, or
/// when an intermediate value cannot be represented by `f32`.
pub fn derive_reference_lattice(
    particle_spacing: f32,
    physical_rest_density: f32,
) -> Result<ReferenceLattice, NumericalError> {
    validation::positive_scalar("particle spacing", particle_spacing)?;
    validation::positive_scalar("physical rest density", physical_rest_density)?;

    let support_radius =
        validation::finite_result("reference support radius", 2.0 * particle_spacing)?;
    let mut normalised_rest_density = 0.0_f32;
    let mut offset_count = 0_usize;

    for a in -1_i32..=1 {
        for b in -1_i32..=1 {
            for c in -1_i32..=1 {
                let offset = Vec3::new(
                    particle_spacing * a as f32,
                    particle_spacing * b as f32,
                    particle_spacing * c as f32,
                );
                validation::finite_vector_result("reference lattice offset", offset)?;
                let distance = offset.checked_length("reference lattice offset")?;
                if distance < support_radius {
                    let weight = poly6(distance, support_radius)?;
                    normalised_rest_density = validation::finite_result(
                        "reference normalised rest density",
                        normalised_rest_density + weight,
                    )?;
                    offset_count += 1;
                }
            }
        }
    }

    let particle_mass = validation::finite_result(
        "reference particle mass",
        physical_rest_density / normalised_rest_density,
    )?;

    Ok(ReferenceLattice {
        support_radius,
        normalised_rest_density,
        particle_mass,
        offset_count,
    })
}
