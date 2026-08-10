//! Tests for interaction lists and the numerical parts of a PBF Jacobi stage.
//!
//! Small particle sets make ordering, support membership and expected values
//! explicit.

mod support;

use maelstrom::{
    NumericalError,
    fluid::{
        ArtificialPressure, InteractionSets, PbfParameters, active_interactions,
        compute_jacobi_stage, density_constraint, estimate_density, relaxed_multiplier,
        substituted_directions,
    },
    math::Vec3,
};
use support::{assert_absolute, assert_vec_absolute, poly6_f64, spiky_magnitude_f64};

fn complete_four_particle_sets() -> InteractionSets {
    // Every particle lists all other particles in ascending index order.
    InteractionSets::new(vec![
        vec![1, 2, 3],
        vec![0, 2, 3],
        vec![0, 1, 3],
        vec![0, 1, 2],
    ])
    .unwrap()
}

#[test]
fn interaction_sets_reject_self_duplicates_order_and_invalid_indices() {
    // A particle cannot list itself.
    assert!(matches!(
        InteractionSets::new(vec![vec![0]]),
        Err(NumericalError::SelfInteraction { particle: 0 })
    ));
    assert!(matches!(
        InteractionSets::new(vec![vec![1, 1], vec![0]]),
        Err(NumericalError::InteractionOrder { .. })
    ));
    // Ascending order also prevents non-deterministic reductions.
    assert!(matches!(
        InteractionSets::new(vec![vec![2, 1], vec![0], vec![0]]),
        Err(NumericalError::InteractionOrder { .. })
    ));
    assert!(matches!(
        InteractionSets::new(vec![vec![1], vec![2]]),
        Err(NumericalError::InteractionIndexOutOfBounds { .. })
    ));
}

#[test]
fn active_interactions_exclude_self_and_apply_strict_current_support() {
    let positions = [
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    ];
    let sets = complete_four_particle_sets();

    // Particle 1 shares the same position and remains active. Particle 3 sits
    // exactly at `h`, so it is filtered out of the fixed list.
    assert_eq!(
        active_interactions(0, &positions, &sets, 1.0).unwrap(),
        vec![1, 2]
    );
}

#[test]
fn density_includes_self_and_coincident_particles_but_not_exact_support() {
    let positions = [
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    ];
    let density =
        estimate_density(0, &positions, &complete_four_particle_sets(), 1.0, 2.0).unwrap();
    let expected_normalised = 2.0 * poly6_f64(0.0, 1.0) + poly6_f64(0.5, 1.0);

    // The density adds three kernel values. Physical density then multiplies
    // that sum by the exactly represented mass `2.0`.
    assert_absolute(density.normalised, expected_normalised, 6.0e-6);
    assert_absolute(density.physical, 2.0 * expected_normalised, 1.2e-5);
}

#[test]
fn single_particle_density_contains_exactly_one_self_weight() {
    let positions = [Vec3::new(4.0, -2.0, 1.0)];
    let sets = InteractionSets::new(vec![vec![]]).unwrap();
    let density = estimate_density(0, &positions, &sets, 1.0, 1.0).unwrap();

    assert_absolute(density.normalised, poly6_f64(0.0, 1.0), 2.0e-6);
    assert_eq!(density.normalised, density.physical);
}

#[test]
fn density_constraint_is_signed_unclamped_and_validated() {
    // Values above and below rest density keep their positive and negative
    // signs; this primitive does not clamp under-density.
    assert_eq!(density_constraint(150.0, 100.0), Ok(0.5));
    assert_eq!(density_constraint(50.0, 100.0), Ok(-0.5));
    assert!(matches!(
        density_constraint(-1.0, 100.0),
        Err(NumericalError::NegativeScalar { .. })
    ));
    assert!(matches!(
        density_constraint(1.0, 0.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        density_constraint(f32::NAN, 1.0),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
}

#[test]
fn substituted_directions_and_multiplier_match_independent_equations() {
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let sets = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let directions = substituted_directions(0, &positions, &sets, 1.0, 2.0).unwrap();
    let expected_direction = spiky_magnitude_f64(0.5, 1.0) / 2.0;

    // Owner and neighbour directions must be equal and opposite for this
    // symmetric pair.
    assert_eq!(directions.owner(), 0);
    assert_vec_absolute(
        directions.owner_direction(),
        [expected_direction, 0.0, 0.0],
        2.0e-6,
    );
    assert_eq!(directions.neighbour_directions()[0].0, 1);
    assert_vec_absolute(
        directions.neighbour_directions()[0].1,
        [-expected_direction, 0.0, 0.0],
        2.0e-6,
    );

    let constraint = 0.25;
    let relaxation = 0.01;
    // Two directions of equal magnitude contribute to the denominator.
    let expected_multiplier =
        -f64::from(constraint) / (2.0 * expected_direction.powi(2) + f64::from(relaxation));
    assert_absolute(
        relaxed_multiplier(constraint, &directions, relaxation).unwrap(),
        expected_multiplier,
        2.0e-6,
    );
}

#[test]
fn multiplier_rejects_invalid_relaxation_and_non_finite_constraint() {
    let positions = [Vec3::ZERO];
    let sets = InteractionSets::new(vec![vec![]]).unwrap();
    let directions = substituted_directions(0, &positions, &sets, 1.0, 1.0).unwrap();

    // Relaxation keeps the denominator positive, and the constraint must be
    // finite before division.
    assert!(matches!(
        relaxed_multiplier(0.0, &directions, 0.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        relaxed_multiplier(f32::INFINITY, &directions, 1.0),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
}

#[test]
fn artificial_pressure_matches_reference_point_and_support_boundaries() {
    let pressure = ArtificialPressure::new(0.1, 0.2, 4.0, 1.0).unwrap();
    // The configured reference separation evaluates to `-k`; strict support
    // makes values at and beyond `h` exactly zero.
    assert_eq!(pressure.evaluate(0.2), Ok(-0.1));
    assert_eq!(pressure.evaluate(1.0), Ok(0.0));
    assert_eq!(pressure.evaluate(1.5), Ok(0.0));

    let expected_zero = -0.1_f64 * (poly6_f64(0.0, 1.0) / poly6_f64(0.2, 1.0)).powf(4.0);
    // Raising the kernel ratio to the fourth power increases its rounding error.
    // `3e-6` is still less than `0.01%` of this result.
    assert_absolute(pressure.evaluate(0.0).unwrap(), expected_zero, 3.0e-6);

    let disabled = ArtificialPressure::new(0.0, 0.2, 4.0, 1.0).unwrap();
    assert_eq!(disabled.evaluate(0.0), Ok(0.0));
}

#[test]
fn artificial_pressure_rejects_invalid_parameters_and_distances() {
    // Pressure strength is finite and non-negative.
    assert!(matches!(
        ArtificialPressure::new(-0.1, 0.2, 4.0, 1.0),
        Err(NumericalError::NegativeScalar { .. })
    ));
    assert!(matches!(
        ArtificialPressure::new(f32::NAN, 0.2, 4.0, 1.0),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    // Reference separation must lie strictly inside the support radius.
    for separation in [0.0, 1.0, 2.0] {
        assert!(matches!(
            ArtificialPressure::new(0.1, separation, 4.0, 1.0),
            Err(NumericalError::InvalidReferenceSeparation { .. })
        ));
    }
    assert!(matches!(
        ArtificialPressure::new(0.1, f32::NAN, 4.0, 1.0),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    // Exponent and support radius are both finite and positive.
    assert!(matches!(
        ArtificialPressure::new(0.1, 0.2, 0.0, 1.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        ArtificialPressure::new(0.1, 0.2, f32::INFINITY, 1.0),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        ArtificialPressure::new(0.1, 0.2, 4.0, 0.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));

    let pressure = ArtificialPressure::new(0.1, 0.2, 4.0, 1.0).unwrap();
    // Evaluation accepts finite non-negative distances only.
    assert!(matches!(
        pressure.evaluate(-0.1),
        Err(NumericalError::NegativeScalar { .. })
    ));
    assert!(matches!(
        pressure.evaluate(f32::NAN),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
}

#[test]
fn complete_correction_uses_one_immutable_multiplier_and_position_snapshot() {
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let original_positions = positions;
    let sets = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let parameters = PbfParameters::new(1.0, 3.0, 0.01, 0.1, 0.2, 4.0).unwrap();

    let stage = compute_jacobi_stage(&positions, &sets, parameters).unwrap();
    let repeated = compute_jacobi_stage(&positions, &sets, parameters).unwrap();

    // Rebuild the two-particle result from independent scalar equations. This
    // catches changes to density, gradients, relaxation or artificial pressure.
    let density = poly6_f64(0.0, 1.0) + poly6_f64(0.5, 1.0);
    let constraint = density / 3.0 - 1.0;
    let gradient = spiky_magnitude_f64(0.5, 1.0);
    let direction = gradient / 3.0;
    let multiplier = -constraint / (2.0 * direction.powi(2) + 0.01);
    let pressure = -0.1 * (poly6_f64(0.5, 1.0) / poly6_f64(0.2, 1.0)).powf(4.0);
    let correction = (2.0 * multiplier + pressure) * gradient / 3.0;

    assert_absolute(stage.normalised_densities[0], density, 4.0e-6);
    assert_absolute(stage.constraints[0], constraint, 2.0e-6);
    assert_absolute(stage.multipliers[0], multiplier, 4.0e-6);
    // The correction combines the kernel, multiplier and pressure calculations.
    // `2e-5` allows their rounding while still catching a stale position, missing
    // neighbour multiplier or reversed sign.
    assert_vec_absolute(stage.corrections[0], [correction, 0.0, 0.0], 2.0e-5);
    assert_vec_absolute(stage.corrections[1], [-correction, 0.0, 0.0], 2.0e-5);

    // The stage must not mutate its input, and the same snapshot must produce
    // identical symmetric results on every run.
    assert_eq!(positions, original_positions);
    assert_eq!(stage, repeated);
    assert_eq!(stage.multipliers[0], stage.multipliers[1]);
    assert_eq!(stage.corrections[1], -stage.corrections[0]);
}

#[test]
fn coincident_distinct_particles_have_density_but_zero_correction() {
    let positions = [Vec3::ZERO, Vec3::ZERO];
    let sets = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let parameters = PbfParameters::new(1.0, 3.0, 0.01, 0.1, 0.2, 4.0).unwrap();
    let stage = compute_jacobi_stage(&positions, &sets, parameters).unwrap();

    // Coincident particles contribute Poly6 density but have no Spiky direction.
    let expected_density = 2.0 * poly6_f64(0.0, 1.0);
    assert_absolute(stage.normalised_densities[0], expected_density, 4.0e-6);
    assert_eq!(stage.corrections, vec![Vec3::ZERO, Vec3::ZERO]);
}

#[test]
fn fluid_boundaries_reject_invalid_lengths_state_and_parameters() {
    let positions = [Vec3::ZERO];
    let empty_sets = InteractionSets::new(vec![]).unwrap();
    // Position and interaction-set counts must match.
    assert!(matches!(
        active_interactions(0, &positions, &empty_sets, 1.0),
        Err(NumericalError::InteractionSetCount { .. })
    ));

    let one_set = InteractionSets::new(vec![vec![]]).unwrap();
    // Particle indices and support radii are checked at the API boundary.
    assert!(matches!(
        active_interactions(1, &positions, &one_set, 1.0),
        Err(NumericalError::ParticleIndexOutOfBounds { .. })
    ));
    for support_radius in [0.0, -1.0, f32::NAN] {
        assert!(matches!(
            active_interactions(0, &positions, &one_set, support_radius),
            Err(NumericalError::NonPositiveScalar { .. })
                | Err(NumericalError::NonFiniteScalar { .. })
        ));
    }
    assert!(matches!(
        estimate_density(0, &positions, &one_set, 1.0, 0.0),
        Err(NumericalError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        estimate_density(0, &positions, &one_set, 1.0, f32::INFINITY),
        Err(NumericalError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        estimate_density(0, &[Vec3::new(f32::NAN, 0.0, 0.0)], &one_set, 1.0, 1.0),
        Err(NumericalError::NonFiniteVector { .. })
    ));

    // Each required PBF scalar is checked for a finite positive value.
    for parameters in [
        PbfParameters::new(0.0, 1.0, 1.0, 0.1, 0.2, 4.0),
        PbfParameters::new(1.0, 0.0, 1.0, 0.1, 0.2, 4.0),
        PbfParameters::new(1.0, 1.0, 0.0, 0.1, 0.2, 4.0),
        PbfParameters::new(f32::NAN, 1.0, 1.0, 0.1, 0.2, 4.0),
        PbfParameters::new(1.0, f32::INFINITY, 1.0, 0.1, 0.2, 4.0),
        PbfParameters::new(1.0, 1.0, f32::NAN, 0.1, 0.2, 4.0),
    ] {
        assert!(matches!(
            parameters,
            Err(NumericalError::NonPositiveScalar { .. })
                | Err(NumericalError::NonFiniteScalar { .. })
        ));
    }
    assert!(matches!(
        substituted_directions(0, &positions, &one_set, 1.0, f32::NAN),
        Err(NumericalError::NonFiniteScalar { .. })
    ));

    let two_sets = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let parameters = PbfParameters::new(1.0, 1.0, 0.01, 0.1, 0.2, 4.0).unwrap();
    // Finite positions can still overflow when their separation is calculated.
    assert!(matches!(
        compute_jacobi_stage(
            &[
                Vec3::new(f32::MAX, 0.0, 0.0),
                Vec3::new(-f32::MAX, 0.0, 0.0)
            ],
            &two_sets,
            parameters,
        ),
        Err(NumericalError::NonFiniteResult { .. })
    ));
}
