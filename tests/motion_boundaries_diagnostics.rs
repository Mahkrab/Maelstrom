//! Tests for motion integration, plane boundaries and CFL diagnostics.
//!
//! The cases cover exact arithmetic, invalid numerical state, ordered boundary
//! projection and the warning-only timestep diagnostic.

mod support;

use maelstrom::{
    NumericalError,
    boundary::{Plane, project_in_order},
    diagnostics::{CflLimit, cfl_diagnostic},
    integration::{post_acceleration_velocity, predict_position, reconstruct_velocity},
    math::Vec3,
};
use support::assert_vec_absolute;

#[test]
fn acceleration_prediction_and_reconstruction_match_exact_binary_examples() {
    // Quarter-step arithmetic keeps every expected component exactly
    // representable in `f32`.
    let velocity =
        post_acceleration_velocity(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, -10.0, 0.0), 0.25)
            .unwrap();
    assert_eq!(velocity, Vec3::new(1.0, -0.5, 3.0));

    let accepted = Vec3::new(1.0, 1.0, 1.0);
    let predicted = predict_position(accepted, velocity, 0.25).unwrap();
    assert_eq!(predicted, Vec3::new(1.25, 0.875, 1.75));
    assert_eq!(
        reconstruct_velocity(accepted, predicted, 0.25),
        Ok(velocity)
    );
}

#[test]
fn motion_primitives_reject_invalid_inputs_and_non_finite_outputs() {
    // A timestep must be finite and positive, and all input vectors must be
    // finite before integration starts.
    assert!(matches!(
        post_acceleration_velocity(Vec3::ZERO, Vec3::ZERO, 0.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        post_acceleration_velocity(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::ZERO, 1.0),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        post_acceleration_velocity(Vec3::ZERO, Vec3::new(0.0, f32::INFINITY, 0.0), 1.0),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    // Valid finite inputs may still produce an overflowing result.
    assert!(matches!(
        post_acceleration_velocity(
            Vec3::new(f32::MAX, 0.0, 0.0),
            Vec3::new(f32::MAX, 0.0, 0.0),
            1.0,
        ),
        Err(NumericalError::NonFiniteResult { .. })
    ));
    assert!(matches!(
        predict_position(Vec3::ZERO, Vec3::ZERO, -1.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        predict_position(Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::ZERO, 1.0),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        reconstruct_velocity(Vec3::ZERO, Vec3::ZERO, f32::NAN),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        reconstruct_velocity(Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0), 1.0),
        Err(NumericalError::NonFiniteVector { .. })
    ));
}

#[test]
fn plane_projection_handles_permitted_penetrated_and_contact_points() {
    let floor = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0).unwrap();
    let permitted = Vec3::new(2.0, 3.0, -4.0);
    let contact = Vec3::new(2.0, 0.0, -4.0);

    // Points on or above the floor stay unchanged. A penetrated point moves
    // along the normal to the closest contact point.
    assert_eq!(floor.project(permitted), Ok(permitted));
    assert_eq!(floor.project(contact), Ok(contact));
    assert_eq!(floor.project(Vec3::new(2.0, -3.0, -4.0)), Ok(contact));
    assert_eq!(floor.contains(permitted), Ok(true));
    assert_eq!(
        floor.contains(Vec3::new(0.0, -f32::EPSILON, 0.0)),
        Ok(false)
    );
}

#[test]
fn axis_aligned_projection_recovers_exact_contact_across_signs_and_scales() {
    for axis in 0..3 {
        for direction in [-1.0_f32, 1.0] {
            let mut normal = [0.0; 3];
            normal[axis] = direction;
            for offset in [
                -1024.25_f32,
                -1.4,
                -0.1,
                -1.0e-38,
                -0.0,
                0.0,
                1.0e-38,
                0.1,
                1.4,
                1024.25,
            ] {
                let wall =
                    Plane::new(Vec3::new(normal[0], normal[1], normal[2]), offset, 0.0).unwrap();
                for separation in [0.125_f32, 4.0276384, 1.0e20, f32::MAX / 4.0] {
                    let mut candidate = [0.25, 0.5, 0.75];
                    candidate[axis] = direction * (offset - separation);
                    let point = Vec3::new(candidate[0], candidate[1], candidate[2]);
                    let projected = wall.project(point).unwrap();
                    let actual = [projected.x, projected.y, projected.z];
                    // An axis-aligned orthogonal projection fixes one coordinate
                    // to the signed offset, independently of candidate magnitude.
                    assert_eq!(actual[axis].to_bits(), (direction * offset).to_bits());
                    assert_eq!(wall.constraint(projected).unwrap(), 0.0);
                    assert!(wall.contains(projected).unwrap());
                    for transverse in 0..3 {
                        if transverse != axis {
                            assert_eq!(
                                actual[transverse].to_bits(),
                                candidate[transverse].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn axis_aligned_projection_repairs_the_observed_one_ulp_penetration() {
    let wall = Plane::new(Vec3::new(-1.0, 0.0, 0.0), -1.4, 0.0).unwrap();
    let point = Vec3::new(4.0276384, 0.46484208, 1.4125106);
    let projected = wall.project(point).unwrap();
    assert_eq!(projected.x.to_bits(), 1.4_f32.to_bits());
    assert_eq!(projected.y.to_bits(), point.y.to_bits());
    assert_eq!(projected.z.to_bits(), point.z.to_bits());
    let outside = Vec3::new(f32::from_bits(1.4_f32.to_bits() + 1), point.y, point.z);
    assert!(!wall.contains(outside).unwrap());
    assert_eq!(wall.project(projected).unwrap(), projected);
}

#[test]
fn nearly_axis_aligned_normals_keep_the_original_vector_projection() {
    let normal = Vec3::new(-1.0, 0.0001, 0.0);
    let wall = Plane::new(normal, -1.4, 1.0e-6).unwrap();
    let point = Vec3::new(4.0276384, 0.46484208, 1.4125106);
    let constraint = normal.x * point.x + normal.y * point.y + normal.z * point.z - wall.offset();
    let expected = [
        point.x - normal.x * constraint,
        point.y - normal.y * constraint,
        point.z - normal.z * constraint,
    ];
    let projected = wall.project(point).unwrap();
    assert_eq!(
        [projected.x, projected.y, projected.z].map(f32::to_bits),
        expected.map(f32::to_bits)
    );
    assert_ne!(projected.x.to_bits(), 1.4_f32.to_bits());
}

#[test]
fn plane_validation_uses_explicit_tolerance_without_normalising() {
    let diagonal_component = std::f32::consts::FRAC_1_SQRT_2;
    let diagonal = Vec3::new(diagonal_component, diagonal_component, 0.0);
    let tolerance = 2.0 * f32::EPSILON;
    let plane = Plane::new(diagonal, 1.0, tolerance).unwrap();

    // Construction validates the supplied normal but preserves its components.
    assert_eq!(plane.normal(), diagonal);
    assert_eq!(plane.offset(), 1.0);
    assert!(matches!(
        Plane::new(Vec3::ZERO, 0.0, tolerance),
        Err(NumericalError::ZeroPlaneNormal)
    ));
    assert!(matches!(
        Plane::new(Vec3::new(1.1, 0.0, 0.0), 0.0, tolerance),
        Err(NumericalError::NonUnitPlaneNormal { .. })
    ));
    assert!(matches!(
        Plane::new(Vec3::new(f32::NAN, 0.0, 0.0), 0.0, tolerance),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        Plane::new(Vec3::new(1.0, 0.0, 0.0), f32::INFINITY, tolerance),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    for invalid_tolerance in [-f32::EPSILON, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, invalid_tolerance),
            Err(NumericalError::InvalidUnitTolerance { .. })
        ));
    }
}

#[test]
fn ordered_plane_projection_is_deterministic_and_order_sensitive() {
    let x_positive = Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 0.0).unwrap();
    let component = std::f32::consts::FRAC_1_SQRT_2;
    let above_diagonal = Plane::new(
        Vec3::new(-component, component, 0.0),
        0.0,
        2.0 * f32::EPSILON,
    )
    .unwrap();
    let point = Vec3::new(-1.0, -2.0, 0.0);

    let first_order = project_in_order(point, &[x_positive, above_diagonal]).unwrap();
    let repeated = project_in_order(point, &[x_positive, above_diagonal]).unwrap();
    let reverse_order = project_in_order(point, &[above_diagonal, x_positive]).unwrap();

    // Sequential projections do not commute, so both determinism and the
    // caller-supplied order are observable parts of the result.
    // These are direct plane projections. `5e-7` only allows for the `f32`
    // representation of `1 / sqrt(2)`.
    assert_vec_absolute(first_order, [-1.0, -1.0, 0.0], 5.0e-7);
    assert_vec_absolute(reverse_order, [0.0, -1.5, 0.0], 5.0e-7);
    assert_eq!(first_order, repeated);
    assert_ne!(first_order, reverse_order);
}

#[test]
fn plane_projection_rejects_non_finite_points_and_results() {
    let plane = Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 0.0).unwrap();
    assert!(matches!(
        plane.project(Vec3::new(f32::NAN, 0.0, 0.0)),
        Err(NumericalError::NonFiniteVector { .. })
    ));

    // A valid plane and point can still overflow during projection.
    let offset_plane = Plane::new(Vec3::new(1.0, 0.0, 0.0), f32::MAX, 0.0).unwrap();
    assert!(matches!(
        offset_plane.project(Vec3::new(-f32::MAX, 0.0, 0.0)),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}

#[test]
fn cfl_reports_below_equality_and_above_without_changing_timestep() {
    let velocities = [Vec3::new(3.0, 4.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];

    // Maximum speed is `5`, giving a limit of `0.1`. Equality is accepted;
    // only a larger requested timestep raises the warning.
    let below = cfl_diagnostic(&velocities, 0.05, 1.0, 0.5).unwrap();
    let equal = cfl_diagnostic(&velocities, 0.1, 1.0, 0.5).unwrap();
    let above = cfl_diagnostic(&velocities, 0.2, 1.0, 0.5).unwrap();

    assert_eq!(below.maximum_speed, 5.0);
    assert_eq!(below.limit, CflLimit::Finite(0.1));
    assert!(!below.warning);
    assert!(!equal.warning);
    assert!(above.warning);
    assert_eq!(equal.limit, above.limit);
}

#[test]
fn cfl_zero_speed_and_empty_input_are_semantically_unbounded() {
    // With no motion there is no finite CFL upper bound.
    for velocities in [&[][..], &[Vec3::ZERO, Vec3::ZERO][..]] {
        let result = cfl_diagnostic(velocities, 0.1, 1.0, 0.5).unwrap();
        assert_eq!(result.maximum_speed, 0.0);
        assert_eq!(result.limit, CflLimit::Unbounded);
        assert!(!result.warning);
    }
}

#[test]
fn cfl_rejects_invalid_domains_and_non_finite_speeds_or_limits() {
    // Timestep, support radius and CFL factor must all be finite and positive.
    for result in [
        cfl_diagnostic(&[], 0.0, 1.0, 0.5),
        cfl_diagnostic(&[], 0.1, 0.0, 0.5),
        cfl_diagnostic(&[], 0.1, 1.0, 0.0),
        cfl_diagnostic(&[], f32::NAN, 1.0, 0.5),
    ] {
        assert!(matches!(
            result,
            Err(NumericalError::NonPositiveScalar { .. })
                | Err(NumericalError::NonFiniteScalar { .. })
        ));
    }
    // The remaining cases distinguish invalid velocity input, speed overflow
    // and overflow while calculating the limit.
    assert!(matches!(
        cfl_diagnostic(&[Vec3::new(f32::INFINITY, 0.0, 0.0)], 0.1, 1.0, 0.5),
        Err(NumericalError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        cfl_diagnostic(&[Vec3::new(f32::MAX, f32::MAX, 0.0)], 0.1, 1.0, 0.5),
        Err(NumericalError::NonFiniteResult { .. })
    ));
    assert!(matches!(
        cfl_diagnostic(&[Vec3::new(0.5, 0.0, 0.0)], 0.1, f32::MAX, 1.0),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}

#[test]
fn inclined_contacts_repair_roundoff_without_relaxing_containment() {
    let mut repaired = 0;
    for normal in [
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(-0.6, 0.8, 0.0),
        Vec3::new(0.0, 0.8, -0.6),
    ] {
        for offset in [-0.36, 0.0, 0.4] {
            let plane = Plane::new(normal, offset, 1e-6).unwrap();
            for i in 0..1024 {
                let point = Vec3::new(i as f32 * 0.013 - 4.0, i as f32 * -0.007, 0.37);
                let constraint = plane.constraint(point).unwrap();
                let original = if constraint < 0.0 {
                    point - normal * constraint
                } else {
                    point
                };
                let actual = plane.project(point).unwrap();
                assert!(
                    plane.contains(actual).unwrap(),
                    "{normal:?}, {offset}, {point:?}"
                );
                assert_eq!(plane.project(actual).unwrap(), actual);
                if plane.contains(original).unwrap() {
                    assert_eq!(actual, original);
                } else {
                    repaired += 1;
                }
            }
        }
    }
    assert!(
        repaired > 0,
        "the grid must exercise the observed round-off defect"
    );
}
