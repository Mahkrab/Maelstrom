//! # Particle-lattice equilibrium
//!
//! Calculates rest density and particle mass from a regular cubic lattice. The
//! fixed-ratio helper retains the selected serial-reference calculation, while
//! configuration resolution can use the general equilibrium derivation.

use crate::{NumericalError, kernel::poly6, math::Vec3, validation};

/// Number of offsets inside strict support when `support_radius = 2 * spacing`.
pub const REFERENCE_LATTICE_OFFSET_COUNT: usize = 27;

/// Values calculated from a regular lattice equilibrium.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct LatticeEquilibrium {
    /// Kernel support radius in metres.
    pub support_radius: f32,
    /// Mass-divided rest density in inverse cubic metres.
    pub normalised_rest_density: f32,
    /// Particle mass in kilograms.
    pub particle_mass: f32,
    /// Number of lattice offsets included in the density sum.
    pub offset_count: usize,
}

/// Result type returned by the fixed-ratio reference-lattice helper.
pub type ReferenceLattice = LatticeEquilibrium;

/// ### Calculates density and mass from the reference lattice
///
/// The function sets `support_radius = 2 * particle_spacing`, visits offsets
/// `(a, b, c)` in ascending order, and sums the 27 offsets inside strict
/// support. Particle mass is then calculated from `physical_rest_density`.
///
/// ### Errors
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
    derive_lattice_equilibrium(particle_spacing, support_radius, physical_rest_density, 2.0)
}

/// Derives density and mass for a strict-support cubic lattice.
///
/// `maximum_support_ratio` bounds the finite enumeration performed during
/// input resolution. Offsets are visited in ascending `(a, b, c)` order.
///
/// ### Errors
/// TODO: Change to list. <sub>I should really standardise this...</sub>
/// Returns [`NumericalError`] when an input is invalid, the support ratio is
/// greater than `maximum_support_ratio`, an index range is unrepresentable, or
/// an intermediate calculation is non-finite.
pub fn derive_lattice_equilibrium(
    particle_spacing: f32,
    support_radius: f32,
    physical_rest_density: f32,
    maximum_support_ratio: f32,
) -> Result<LatticeEquilibrium, NumericalError> {
    validation::positive_scalar("particle spacing", particle_spacing)?;
    validation::positive_scalar("support radius", support_radius)?;
    validation::positive_scalar("physical rest density", physical_rest_density)?;
    validation::positive_scalar("maximum support ratio", maximum_support_ratio)?;

    let support_ratio =
        validation::finite_result("lattice support ratio", support_radius / particle_spacing)?;
    if support_ratio > maximum_support_ratio {
        return Err(NumericalError::SupportRatioTooLarge {
            ratio: support_ratio,
            maximum: maximum_support_ratio,
        });
    }
    let extent = support_ratio.ceil() as i32;
    let mut normalised_rest_density = 0.0_f32;
    let mut offset_count = 0_usize;

    for a in -extent..=extent {
        for b in -extent..=extent {
            for c in -extent..=extent {
                let offset = Vec3::new(
                    particle_spacing * a as f32,
                    particle_spacing * b as f32,
                    particle_spacing * c as f32,
                );
                validation::finite_vector_result("lattice equilibrium offset", offset)?;
                let distance = offset.checked_length("lattice equilibrium offset")?;
                if distance < support_radius {
                    let weight = poly6(distance, support_radius)?;
                    normalised_rest_density = validation::finite_result(
                        "lattice equilibrium normalised rest density",
                        normalised_rest_density + weight,
                    )?;
                    offset_count += 1;
                }
            }
        }
    }

    let particle_mass = validation::finite_result(
        "lattice equilibrium particle mass",
        physical_rest_density / normalised_rest_density,
    )?;

    Ok(LatticeEquilibrium {
        support_radius,
        normalised_rest_density,
        particle_mass,
        offset_count,
    })
}
