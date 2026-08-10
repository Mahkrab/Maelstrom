//! # Three-dimensional vector maths
//!
//! A small `f32` vector type containing only the operations used by the fluid
//! calculations.

use std::ops::{Add, AddAssign, Mul, Neg, Sub};

use crate::{NumericalError, validation};

/// ### Three-dimensional `f32` vector
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// Horizontal `x` component.
    pub x: f32,
    /// Vertical `y` component.
    pub y: f32,
    /// Depth `z` component.
    pub z: f32,
}

impl Vec3 {
    /// Vector with all components set to zero (zero vector).
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// ### Creates a vector from its `x`, `y` and `z` components
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns `true` when every component is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Calculates the dot product with `other`.
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculates the squared vector length without taking a square root.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// ### Calculates the vector length
    ///
    /// `quantity` is included in an input error so callers can identify which
    /// vector failed validation.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError`] when a component is non-finite or when the
    /// squared length overflows `f32`.
    pub fn checked_length(self, quantity: &'static str) -> Result<f32, NumericalError> {
        validation::finite_vector(quantity, self)?;
        let length_squared =
            validation::finite_result("squared vector length", self.length_squared())?;
        validation::finite_result("vector length", length_squared.sqrt())
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, vector: Vec3) -> Self::Output {
        vector * self
    }
}
