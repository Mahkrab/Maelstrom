//! # Shared numerical validation
//!
//! Small checks used by the public maths APIs to report consistent
//! [`NumericalError`] values.

use crate::{NumericalError, math::Vec3};

pub(crate) fn finite_scalar(quantity: &'static str, value: f32) -> Result<f32, NumericalError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalError::NonFiniteScalar { quantity, value })
    }
}

pub(crate) fn positive_scalar(quantity: &'static str, value: f32) -> Result<f32, NumericalError> {
    finite_scalar(quantity, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(NumericalError::NonPositiveScalar { quantity, value })
    }
}

pub(crate) fn non_negative_scalar(
    quantity: &'static str,
    value: f32,
) -> Result<f32, NumericalError> {
    finite_scalar(quantity, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(NumericalError::NegativeScalar { quantity, value })
    }
}

pub(crate) fn finite_vector(quantity: &'static str, value: Vec3) -> Result<Vec3, NumericalError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalError::NonFiniteVector { quantity })
    }
}

pub(crate) fn finite_result(quantity: &'static str, value: f32) -> Result<f32, NumericalError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalError::NonFiniteResult { quantity })
    }
}

pub(crate) fn finite_vector_result(
    quantity: &'static str,
    value: Vec3,
) -> Result<Vec3, NumericalError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalError::NonFiniteResult { quantity })
    }
}
