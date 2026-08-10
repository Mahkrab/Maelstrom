//! Tests for the reference lattice density and particle-mass calculation.
//!
//! The expected values come from a separate `f64` scan with wider loop bounds,
//! so the production search limits are not reused as the test oracle.

mod support;

use maelstrom::{
    NumericalError,
    reference::{REFERENCE_LATTICE_OFFSET_COUNT, derive_reference_lattice},
};
use support::{assert_absolute, poly6_f64};

fn independently_derive_lattice(spacing: f64, physical_rest_density: f64) -> (usize, f64, f64) {
    let support_radius = 2.0 * spacing;
    let mut count = 0_usize;
    let mut normalised_rest_density = 0.0_f64;

    // Scan a larger cube so the distance check, rather than the same loop bounds
    // as the implementation, decides which offsets belong to the lattice.
    for a in -2_i32..=2 {
        for b in -2_i32..=2 {
            for c in -2_i32..=2 {
                let x = spacing * f64::from(a);
                let y = spacing * f64::from(b);
                let z = spacing * f64::from(c);
                let distance = (x * x + y * y + z * z).sqrt();
                // Strict support excludes offsets exactly two cells away.
                if distance < support_radius {
                    normalised_rest_density += poly6_f64(distance, support_radius);
                    count += 1;
                }
            }
        }
    }

    (
        count,
        normalised_rest_density,
        physical_rest_density / normalised_rest_density,
    )
}

#[test]
fn documented_lattice_reproduces_27_offsets_density_and_mass() {
    let actual = derive_reference_lattice(0.05, 1000.0).unwrap();
    let (expected_count, expected_density, expected_mass) =
        independently_derive_lattice(0.05, 1000.0);

    // Check the independently counted membership before comparing sums.
    assert_eq!(expected_count, REFERENCE_LATTICE_OFFSET_COUNT);
    assert_eq!(actual.offset_count, REFERENCE_LATTICE_OFFSET_COUNT);
    assert_eq!(actual.support_radius, 0.1);

    // `0.01 m^-3` is about 1.2 ppm. It allows `f32` rounding but catches a
    // missing offset or a changed kernel constant.
    assert_absolute(actual.normalised_rest_density, expected_density, 0.01);
    // `2e-7 kg` is about 1.6 ppm. It catches code that divides by the rounded
    // density shown in the documentation instead of the calculated `f32` sum.
    assert_absolute(actual.particle_mass, expected_mass, 2.0e-7);

    // These fixed anchors detect accidental changes to the documented setup as
    // well as agreement between the two calculations above.
    assert_absolute(actual.normalised_rest_density, 8_078.201_335_157_481, 0.01);
    assert_absolute(actual.particle_mass, 0.123_789_932_748, 2.0e-7);
}

#[test]
fn reference_derivation_is_exactly_repeatable_in_the_same_environment() {
    let first = derive_reference_lattice(0.05, 1000.0).unwrap();
    let second = derive_reference_lattice(0.05, 1000.0).unwrap();

    // Bit comparisons make floating-point repeatability explicit.
    assert_eq!(first, second);
    assert_eq!(
        first.normalised_rest_density.to_bits(),
        second.normalised_rest_density.to_bits()
    );
    assert_eq!(
        first.particle_mass.to_bits(),
        second.particle_mass.to_bits()
    );
}

#[test]
fn reference_derivation_rejects_invalid_domains_and_non_finite_intermediates() {
    // Spacing and physical rest density must be positive.
    for result in [
        derive_reference_lattice(0.0, 1000.0),
        derive_reference_lattice(-0.05, 1000.0),
        derive_reference_lattice(0.05, 0.0),
        derive_reference_lattice(0.05, -1000.0),
    ] {
        assert!(matches!(
            result,
            Err(NumericalError::NonPositiveScalar { .. })
        ));
    }
    // Non-finite inputs are reported separately from invalid signs.
    for result in [
        derive_reference_lattice(f32::NAN, 1000.0),
        derive_reference_lattice(0.05, f32::INFINITY),
    ] {
        assert!(matches!(
            result,
            Err(NumericalError::NonFiniteScalar { .. })
        ));
    }
    // A positive finite spacing can still make the kernel coefficient overflow.
    assert!(matches!(
        derive_reference_lattice(f32::MIN_POSITIVE, 1000.0),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}
