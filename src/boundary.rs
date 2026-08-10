//! # Static plane boundaries
//!
//! Validation and position projection for planes that keep particles inside a
//! permitted halfspace.

use crate::{NumericalError, math::Vec3, validation};

/// ### Plane whose normal points inato the permitted halfspace
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    normal: Vec3,
    offset: f32,
}

impl Plane {
    /// ### Creates a validated plane
    ///
    /// `unit_length_squared_tolerance` is the accepted absolute difference
    /// between `normal.length_squared()` and `1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError`] when:
    ///
    /// - the normal or offset is non-finite;
    /// - the normal is zero or outside the unit-length tolerance;
    /// - the tolerance is negative or non-finite.
    pub fn new(
        normal: Vec3,
        offset: f32,
        unit_length_squared_tolerance: f32,
    ) -> Result<Self, NumericalError> {
        validation::finite_vector("plane normal", normal)?;
        validation::finite_scalar("plane offset", offset)?;

        if !unit_length_squared_tolerance.is_finite() || unit_length_squared_tolerance < 0.0 {
            return Err(NumericalError::InvalidUnitTolerance {
                tolerance: unit_length_squared_tolerance,
            });
        }

        let length_squared =
            validation::finite_result("plane normal squared length", normal.length_squared())?;

        if length_squared == 0.0 {
            return Err(NumericalError::ZeroPlaneNormal);
        }
        if (length_squared - 1.0).abs() > unit_length_squared_tolerance {
            return Err(NumericalError::NonUnitPlaneNormal {
                length_squared,
                tolerance: unit_length_squared_tolerance,
            });
        }

        Ok(Self { normal, offset })
    }

    /// Returns the plane normal exactly as supplied to [`Plane::new`].
    #[must_use]
    pub const fn normal(self) -> Vec3 {
        self.normal
    }

    /// Returns the plane offset in metres as supplied to [`Plane::new`].
    #[must_use]
    pub const fn offset(self) -> f32 {
        self.offset
    }

    /// ### Calculates the signed plain constraint
    ///
    /// A negative value means that `point` is outside the permitted halfspace.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError`] when `point` is non-finite or the calculation
    /// produces a non-finite value.
    pub fn constraint(self, point: Vec3) -> Result<f32, NumericalError> {
        validation::finite_vector("plane test point", point)?;
        let dot = validation::finite_result("plane dot product", self.normal.dot(point))?;
        validation::finite_result("plane constraint", dot - self.offset)
    }

    /// ### Projects a point into the permitted halfspace
    ///
    /// Points already inside or exactly on the plane are returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError`] when the point or projection becomes
    /// ninfinite.
    pub fn project(self, point: Vec3) -> Result<Vec3, NumericalError> {
        let constraint = self.constraint(point)?;
        if constraint >= 0.0 {
            return Ok(point);
        }

        let correction = validation::finite_vector_result(
            "plane projection correction",
            self.normal * constraint,
        )?;
        validation::finite_vector_result("plane-projected position", point - correction)
    }

    /// Returns whether `point` is inside or exactly on the plane.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError`] under the same conditions as
    /// [`Plane::constraint`].
    pub fn contains(self, point: Vec3) -> Result<bool, NumericalError> {
        Ok(self.constraint(point)? >= 0.0)
    }
}

/// ### Projects a point through a list of planes
///
/// Each plane is applied exactly once in slice order. The output of one plane
/// becomes the input to the next.
///
/// # Errors
///
/// Returns [`NumericalError`] when the input or any projection becomes
/// non-finite.
pub fn project_in_order(mut point: Vec3, planes: &[Plane]) -> Result<Vec3, NumericalError> {
    validation::finite_vector("plane projection candidate", point)?;
    for &plane in planes {
        point = plane.project(point)?;
    }
    Ok(point)
}
