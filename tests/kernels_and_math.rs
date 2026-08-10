//! Tests for vector maths and the Poly6 and Spiky kernels.
//!
//! Exact binary examples use direct equality. Kernel values use independent
//! `f64` equations with small tolerances for the production `f32` results.

mod support;

use maelstrom::{
    NumericalError,
    kernel::{poly6, spiky_direction},
    math::Vec3,
};
use support::{assert_absolute, assert_vec_absolute, poly6_f64, spiky_magnitude_f64};

#[test]
fn vector_operations_preserve_simple_binary32_expectations_exactly() {
    let left = Vec3::new(1.0, -2.0, 4.0);
    let right = Vec3::new(0.5, 2.0, -1.0);

    // These values are exactly representable in binary, so any rounding
    // tolerance would hide an error in the vector operations.
    assert_eq!(left + right, Vec3::new(1.5, 0.0, 3.0));
    assert_eq!(left - right, Vec3::new(0.5, -4.0, 5.0));
    assert_eq!(-right, Vec3::new(-0.5, -2.0, 1.0));
    assert_eq!(left * 0.5, Vec3::new(0.5, -1.0, 2.0));
    assert_eq!(left.dot(right), -7.5);
    assert_eq!(Vec3::new(3.0, 4.0, 0.0).checked_length("test"), Ok(5.0));
}

#[test]
fn vector_length_rejects_non_finite_input_and_overflow() {
    // Invalid input and overflow are separate failure paths.
    assert!(matches!(
        Vec3::new(f32::NAN, 0.0, 0.0).checked_length("test vector"),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        Vec3::new(f32::MAX, f32::MAX, 0.0).checked_length("test vector"),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}

#[test]
fn poly6_matches_an_independent_analytical_value_below_support() {
    let actual = poly6(0.5, 1.0).unwrap();
    let expected = poly6_f64(0.5, 1.0);

    // `2e-6` allows normal `f32` rounding but is small enough to catch a wrong
    // coefficient, power or support term.
    assert_absolute(actual, expected, 2.0e-6);
    assert_absolute(poly6(0.0, 1.0).unwrap(), poly6_f64(0.0, 1.0), 2.0e-6);
}

#[test]
fn poly6_has_exact_zero_at_and_above_strict_support() {
    // The support boundary is excluded: only distances below `h` contribute.
    assert_eq!(poly6(1.0, 1.0), Ok(0.0));
    assert_eq!(poly6(1.25, 1.0), Ok(0.0));
}

#[test]
fn poly6_rejects_every_invalid_argument_domain() {
    // Distance must be finite and non-negative.
    assert!(matches!(
        poly6(-0.25, 1.0),
        Err(NumericalError::NegativeScalar { .. })
    ));
    assert!(matches!(
        poly6(f32::NAN, 1.0),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    // Support radius must be finite, positive and large enough for the kernel
    // coefficient to remain finite.
    assert!(matches!(
        poly6(0.0, 0.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        poly6(0.0, f32::INFINITY),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        poly6(0.0, f32::MIN_POSITIVE),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}

#[test]
fn spiky_direction_matches_magnitude_direction_and_symmetry() {
    let positive = spiky_direction(Vec3::new(0.5, 0.0, 0.0), 1.0).unwrap();
    let negative = spiky_direction(Vec3::new(-0.5, 0.0, 0.0), 1.0).unwrap();
    let magnitude = spiky_magnitude_f64(0.5, 1.0);

    // The gradient points back towards the particle at the origin. Mirroring
    // the displacement must therefore reverse the result exactly.
    // `4e-6` allows the rounding from the coefficient and direction calculation.
    assert_vec_absolute(positive, [-magnitude, 0.0, 0.0], 4.0e-6);
    assert_vec_absolute(negative, [magnitude, 0.0, 0.0], 4.0e-6);
    assert_eq!(negative, -positive);
    assert_absolute(
        positive.checked_length("Spiky result").unwrap(),
        magnitude,
        4.0e-6,
    );
}

#[test]
fn spiky_zero_distance_and_support_boundaries_are_exact() {
    // Coincident particles have no defined direction, while points at or
    // outside support have no kernel contribution.
    assert_eq!(spiky_direction(Vec3::ZERO, 1.0), Ok(Vec3::ZERO));
    assert_eq!(
        spiky_direction(Vec3::new(1.0, 0.0, 0.0), 1.0),
        Ok(Vec3::ZERO)
    );
    assert_eq!(
        spiky_direction(Vec3::new(1.5, 0.0, 0.0), 1.0),
        Ok(Vec3::ZERO)
    );
}

#[test]
fn spiky_rejects_non_finite_vectors_and_invalid_support() {
    // The final case starts with finite components but overflows while finding
    // the displacement length.
    assert!(matches!(
        spiky_direction(Vec3::new(f32::INFINITY, 0.0, 0.0), 1.0),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        spiky_direction(Vec3::ZERO, -1.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        spiky_direction(Vec3::new(f32::MAX, f32::MAX, 0.0), 1.0),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}
