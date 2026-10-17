//! Sparse occupied cells prune discovery without changing binary32 membership.
//! Cell arithmetic uses exact power-of-two scaling in binary64; only the shared
//! checked binary32 distance decides which candidates enter an interaction row.

use super::{distance, reserve};
use crate::{NumericalError, fluid::InteractionSets, math::Vec3, validation};

type Cell = [i64; 3];

/// Builds strict-support rows using a serial uniform grid.
///
/// Indices name the supplied slice, so callers must keep it in particle-identity
/// order. Storage is proportional to particles and actual interactions, rather
/// than spatial volume. Dense input can still require quadratic work and storage.
/// Empty input and coincident particles are valid.
///
/// The cell width is a power of two at least the radius, at least `2^-72`
/// to cover squared-distance underflow, and large enough to bound cell indices
/// by `2^50`. The 27 adjacent cells then cover every binary32 neighbour, even at
/// negative cell boundaries. Enlarging cells adds candidates, never membership.
///
/// # Errors
///
/// Rejects invalid radius/positions, non-finite reference distance arithmetic,
/// unrepresentable indexing and failed storage reservations. All distant pairs
/// retain the reference's overflow checks; an unsafe bounding diagonal triggers
/// a full validation pass in reference order, not a geometric rejection.
pub fn uniform_grid(
    positions: &[Vec3],
    support_radius: f32,
) -> Result<InteractionSets, NumericalError> {
    validation::positive_scalar("support radius", support_radius)?;
    let mut lower = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut upper = Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX);
    let mut largest = 0.0_f64;
    for &p in positions {
        validation::finite_vector("neighbour search position", p)?;
        lower = Vec3::new(lower.x.min(p.x), lower.y.min(p.y), lower.z.min(p.z));
        upper = Vec3::new(upper.x.max(p.x), upper.y.max(p.y), upper.z.max(p.z));
        largest = largest.max(f64::from(p.x.abs().max(p.y.abs()).max(p.z.abs())));
    }
    // Non-negative binary32 products and sums are monotone. A finite bounding
    // diagonal therefore proves that every pair's distance arithmetic is finite.
    // If it cannot prove this, check the actual pairs so an empty box corner
    // cannot introduce a failure or change the reference's first error.
    if positions.len() > 1 && distance(upper, lower).is_err() {
        for (i, &p) in positions.iter().enumerate() {
            for (j, &q) in positions.iter().enumerate() {
                if i != j {
                    distance(p, q)?;
                }
            }
        }
    }
    let width = cell_width(support_radius, largest);
    let mut entries = Vec::new();
    reserve(&mut entries, positions.len())?;
    for (index, &p) in positions.iter().enumerate() {
        entries.push((cell(p, width)?, index));
    }
    entries.sort_unstable();
    let mut rows = Vec::new();
    reserve(&mut rows, positions.len())?;
    for (index, &p) in positions.iter().enumerate() {
        let origin = cell(p, width)?;
        let mut row = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let target = adjacent(origin, [x, y, z])?;
                    let start = entries.partition_point(|&(key, _)| key < target);
                    for &(key, neighbour) in &entries[start..] {
                        if key != target {
                            break;
                        }
                        if index != neighbour && distance(p, positions[neighbour])? < support_radius
                        {
                            reserve(&mut row, 1)?;
                            row.push(neighbour);
                        }
                    }
                }
            }
        }
        // Cells partition indices, so each candidate occurs once. Restore the
        // serial reduction order without allocating another row or a sort buffer.
        row.sort_unstable();
        rows.push(row);
    }
    InteractionSets::new(rows)
}

fn cell_width(radius: f32, largest: f64) -> f64 {
    // For a power-of-two width w >= 2^-72, w² is exact and nonzero in
    // binary32 (or overflows). If a real axis separation is >= w, monotone
    // rounding of subtraction, squaring, non-negative sums and sqrt cannot
    // produce a finite length < w. Thus every accepted pair differs by < w
    // on each axis, including when smaller squared distances underflow.
    let minimum = f64::from(radius)
        .max(2.0_f64.powi(-72))
        .max(largest * 2.0_f64.powi(-50));
    // All inputs are finite and this range lies well inside binary64. Repeated
    // doubling selects an exact power of two without a rounded logarithm.
    let mut width = 2.0_f64.powi(-72);
    while width < minimum {
        width *= 2.0;
    }
    width
}

fn cell(position: Vec3, width: f64) -> Result<Cell, NumericalError> {
    let mut key = [0; 3];
    for (value, output) in [position.x, position.y, position.z]
        .into_iter()
        .zip(&mut key)
    {
        let scaled = (f64::from(value) / width).floor();
        if !scaled.is_finite() || scaled.abs() > 2.0_f64.powi(50) {
            return Err(NumericalError::GridIndexOutOfRange);
        }
        *output = scaled as i64;
    }
    Ok(key)
}

fn adjacent(origin: Cell, offset: Cell) -> Result<Cell, NumericalError> {
    let mut target = [0; 3];
    for axis in 0..3 {
        target[axis] = origin[axis]
            .checked_add(offset[axis])
            .ok_or(NumericalError::GridIndexOutOfRange)?;
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_uses_floor_at_faces_edges_and_corners() {
        assert_eq!(cell(Vec3::new(-0.0, 0.0, 0.5), 0.5).unwrap(), [0, 0, 1]);
        assert_eq!(
            cell(Vec3::new(-0.5, -0.25, -f32::from_bits(1)), 0.5).unwrap(),
            [-1, -1, -1]
        );
        assert_eq!(cell(Vec3::new(0.5, 1.0, -1.0), 0.5).unwrap(), [1, 2, -2]);
        assert_eq!(cell_width(0.1, 0.0), 0.125);
        assert_eq!(cell_width(f32::from_bits(1), 0.0), 2.0_f64.powi(-72));
    }

    #[test]
    fn invalid_index_and_capacity_operations_are_errors() {
        assert_eq!(
            cell(Vec3::new(f32::MAX, 0.0, 0.0), 1.0),
            Err(NumericalError::GridIndexOutOfRange)
        );
        assert_eq!(
            adjacent([i64::MAX, 0, 0], [1, 0, 0]),
            Err(NumericalError::GridIndexOutOfRange)
        );
        assert_eq!(
            adjacent([0, i64::MIN, 0], [0, -1, 0]),
            Err(NumericalError::GridIndexOutOfRange)
        );
        assert_eq!(
            reserve(&mut Vec::<usize>::new(), usize::MAX),
            Err(NumericalError::SearchAllocation)
        );
    }
}
