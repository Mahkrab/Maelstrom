//! Independent numerical helpers shared by the integration tests.
//!
//! Each integration test compiles this module separately, so some test crates
//! use only part of it.

#![allow(dead_code)]

use std::f64::consts::PI;

pub fn poly6_f64(distance: f64, support_radius: f64) -> f64 {
    // Keep this equation independent of the production `f32` implementation.
    if distance >= support_radius {
        return 0.0;
    }
    315.0 / (64.0 * PI * support_radius.powi(9))
        * (support_radius * support_radius - distance * distance).powi(3)
}

pub fn spiky_magnitude_f64(distance: f64, support_radius: f64) -> f64 {
    // Callers select a distance inside support, so no boundary branch is needed.
    45.0 / (PI * support_radius.powi(6)) * (support_radius - distance).powi(2)
}

pub fn assert_absolute(actual: f32, expected: f64, tolerance: f32) {
    // Convert the production value to `f64` before measuring its absolute error.
    let error = (f64::from(actual) - expected).abs();
    assert!(
        error <= f64::from(tolerance),
        "actual {actual:?}, expected {expected:?}, absolute error {error:?}, tolerance {tolerance:?}"
    );
}

pub fn assert_vec_absolute(actual: maelstrom::math::Vec3, expected: [f64; 3], tolerance: f32) {
    // Report the failing component through the scalar assertion above.
    assert_absolute(actual.x, expected[0], tolerance);
    assert_absolute(actual.y, expected[1], tolerance);
    assert_absolute(actual.z, expected[2], tolerance);
}
