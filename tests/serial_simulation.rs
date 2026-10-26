//! Small, independently evaluated systems expose timestep ordering and rollback.
//! Exact bit comparisons test repeatability separately from numerical accuracy.

#[path = "support/serial_oracle.rs"]
mod oracle;
mod support;

use maelstrom::{
    NumericalError,
    boundary::Plane,
    config::{ResolvedConfiguration, parse_configuration_overlay, resolve_effective_configuration},
    diagnostics::CflLimit,
    fluid::{active_interactions, normalised_density},
    math::Vec3,
    neighbours::brute_force,
    scene::{Particle, load_resolved_scene},
    simulation::{SerialSimulation, Snapshot, StepError},
};
use oracle::{Variant, maximum_difference};
use support::{assert_absolute, assert_vec_absolute, poly6_f64};

fn config(overlay: &str) -> ResolvedConfiguration {
    let mut overlay = parse_configuration_overlay(overlay).unwrap();
    // Historical arithmetic fixtures retain their explicit four-pass inputs.
    overlay.numerical.solver_iterations.get_or_insert(4);
    overlay.numerical.correction_scale.get_or_insert(1.0);
    resolve_effective_configuration(None, None, Some(&overlay)).unwrap()
}

fn particle(identity: u64, position_m: Vec3, velocity_m_s: Vec3) -> Particle {
    Particle {
        identity,
        position_m,
        velocity_m_s,
    }
}

fn stationary(identity: u64, x: f32, y: f32, z: f32) -> Particle {
    particle(identity, Vec3::new(x, y, z), Vec3::ZERO)
}

fn plane(normal: Vec3, offset: f32) -> Plane {
    Plane::new(normal, offset, 1.0e-6).unwrap()
}

fn bits(snapshot: &Snapshot) -> (u64, Vec<(u64, [u32; 6])>) {
    (
        snapshot.completed_steps(),
        snapshot
            .particles()
            .iter()
            .map(|p| {
                (
                    p.identity,
                    [
                        p.position_m.x,
                        p.position_m.y,
                        p.position_m.z,
                        p.velocity_m_s.x,
                        p.velocity_m_s.y,
                        p.velocity_m_s.z,
                    ]
                    .map(f32::to_bits),
                )
            })
            .collect(),
    )
}

#[test]
fn search_has_strict_support_sorted_unique_rows_and_no_self() {
    let inside = f32::from_bits(1.0_f32.to_bits() - 1);
    let outside = f32::from_bits(1.0_f32.to_bits() + 1);
    let positions = [
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(inside, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(outside, 0.0, 0.0),
    ];
    let found = brute_force(&positions, 1.0).unwrap();
    assert_eq!(found.for_particle(0), Some([1, 2].as_slice()));
    assert_eq!(found.for_particle(1), Some([0, 2].as_slice()));
    assert_eq!(found.for_particle(2), Some([0, 1, 3, 4].as_slice()));
    assert_eq!(found.for_particle(3), Some([2, 4].as_slice()));
    assert_eq!(found.for_particle(4), Some([2, 3].as_slice()));
    assert_eq!(found.for_particle(5), None);
    assert_eq!(brute_force(&[], 1.0).unwrap().particle_count(), 0);
    assert_eq!(
        brute_force(&[Vec3::ZERO], 1.0).unwrap().for_particle(0),
        Some([].as_slice())
    );
}

#[test]
fn search_rejects_invalid_support_positions_and_overflowing_distances() {
    for h in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(brute_force(&[], h).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(brute_force(&[Vec3::new(value, 0.0, 0.0)], 1.0).is_err());
    }
    assert!(
        brute_force(
            &[
                Vec3::new(f32::MAX, 0.0, 0.0),
                Vec3::new(-f32::MAX, 0.0, 0.0)
            ],
            1.0
        )
        .is_err()
    );
    assert!(brute_force(&[Vec3::ZERO, Vec3::new(1.0e20, 0.0, 0.0)], 1.0).is_err());
}

#[test]
fn fixed_membership_filters_current_distances_without_adding_new_particles() {
    let initial = [
        Vec3::ZERO,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ];
    let fixed = brute_force(&initial, 1.0).unwrap();
    let current = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.25, 0.0, 0.0),
    ];
    assert_eq!(fixed.for_particle(0), Some([1].as_slice()));
    assert!(
        active_interactions(0, &current, &fixed, 1.0)
            .unwrap()
            .is_empty()
    );
    assert_absolute(
        normalised_density(0, &current, &fixed, 1.0).unwrap(),
        poly6_f64(0.0, 1.0),
        2.0e-6,
    );
    // A retained member becomes active again if it returns inside support.
    assert_eq!(active_interactions(0, &initial, &fixed, 1.0).unwrap(), [1]);
}

#[test]
fn empty_steps_are_successful_unbounded_and_counted() {
    let mut solver = SerialSimulation::new(&config(""), vec![], vec![]).unwrap();
    assert!(solver.run_steps(0).unwrap().is_empty());
    let outcomes = solver.run_steps(3).unwrap();
    for (index, outcome) in outcomes.iter().enumerate() {
        assert_eq!(outcome.completed_steps, index as u64 + 1);
        assert_eq!(outcome.cfl.maximum_speed.to_bits(), 0.0_f32.to_bits());
        assert_eq!(outcome.cfl.limit, CflLimit::Unbounded);
        assert!(!outcome.cfl.warning);
    }
    assert_eq!(solver.snapshot().completed_steps(), 3);
    assert!(solver.snapshot().particles().is_empty());
}

#[test]
fn single_particle_accelerates_projects_and_reconstructs_without_reflection() {
    let configuration = config(
        "[numerical]\nfixed_timestep_s = 0.25\nexternal_acceleration_m_s2 = [0.0, -8.0, 0.0]",
    );
    let initial = particle(73, Vec3::new(0.0, 0.125, 0.0), Vec3::new(2.0, 0.0, 0.0));
    let mut solver = SerialSimulation::new(
        &configuration,
        vec![initial],
        vec![plane(Vec3::new(0.0, 1.0, 0.0), 0.0)],
    )
    .unwrap();
    let checkpoint = solver.snapshot().clone();
    let outcome = solver.step().unwrap();
    let accepted = solver.snapshot().particles()[0];
    assert_eq!(accepted.identity, 73);
    assert_eq!(accepted.position_m, Vec3::new(0.5, 0.0, 0.0));
    assert_eq!(accepted.velocity_m_s, Vec3::new(2.0, -0.5, 0.0));
    assert!(outcome.cfl.warning);
    assert_absolute(outcome.cfl.maximum_speed, 8.0_f64.sqrt(), 3.0e-7);
    assert_eq!(checkpoint.particles(), &[initial]);
    assert_eq!(checkpoint.completed_steps(), 0);
}

#[test]
fn cfl_equality_is_allowed_and_stationary_on_plane_is_unchanged() {
    let configuration = config(
        "[numerical]\nfixed_timestep_s = 0.125\ncfl_factor = 0.5\nparticle_spacing_m = 0.25\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]",
    );
    let mut solver = SerialSimulation::new(
        &configuration,
        vec![particle(0, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0))],
        vec![],
    )
    .unwrap();
    let outcome = solver.step().unwrap();
    assert_eq!(outcome.cfl.limit, CflLimit::Finite(0.125));
    assert!(!outcome.cfl.warning);
    let mut still = SerialSimulation::new(
        &configuration,
        vec![stationary(0, 0.0, 0.0, 0.0)],
        vec![plane(Vec3::new(0.0, 1.0, 0.0), 0.0)],
    )
    .unwrap();
    assert_eq!(still.step().unwrap().cfl.limit, CflLimit::Unbounded);
    assert_eq!(
        still.snapshot().particles()[0],
        stationary(0, 0.0, 0.0, 0.0)
    );
}

#[test]
fn coincident_particles_keep_identity_and_receive_no_invented_separation() {
    let initial = vec![
        stationary(u64::MAX, 0.0, 1.0, 0.0),
        stationary(0, 0.0, 1.0, 0.0),
    ];
    let mut pair = SerialSimulation::new(&config(""), initial.clone(), vec![]).unwrap();
    let mut single = SerialSimulation::new(&config(""), vec![initial[0]], vec![]).unwrap();
    pair.run_steps(4).unwrap();
    single.run_steps(4).unwrap();
    for p in pair.snapshot().particles() {
        assert_eq!(p.position_m, single.snapshot().particles()[0].position_m);
        assert_eq!(
            p.velocity_m_s,
            single.snapshot().particles()[0].velocity_m_s
        );
    }
    assert_eq!(pair.snapshot().particles()[0].identity, 0);
    assert_eq!(pair.snapshot().particles()[1].identity, u64::MAX);
}

fn assert_oracle(solver: &SerialSimulation, expected: &oracle::Oracle) {
    for ((actual, &position), &velocity) in solver
        .snapshot()
        .particles()
        .iter()
        .zip(&expected.positions)
        .zip(&expected.velocities)
    {
        // Four passes amplify binary32 kernel/reduction rounding. The position
        // allowance is 5 micrometres (0.005% of support); reconstruction divides
        // that displacement error by dt, requiring a separate velocity allowance.
        assert_vec_absolute(actual.position_m, position, 5.0e-6);
        assert_vec_absolute(
            actual.velocity_m_s,
            velocity,
            5.0e-6 / solver.configuration().numerical.fixed_timestep_s + 2.0e-6,
        );
    }
}

#[test]
fn four_jacobi_iterations_match_independent_asymmetric_system() {
    // Stronger relaxation keeps this fixture interacting across all four passes,
    // so three or five passes cannot accidentally produce the same endpoint.
    let configuration = config(
        "[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]\nrelaxation_scale = 10.0",
    );
    let initial = vec![
        stationary(3, 0.0, 0.0, 0.0),
        stationary(19, 0.04, 0.015, 0.0),
        stationary(800, 0.07, -0.01, 0.02),
    ];
    let expected = oracle::step(&initial, &configuration, &[], 4, Variant::default());
    for iterations in [3, 5] {
        let wrong = oracle::step(
            &initial,
            &configuration,
            &[],
            iterations,
            Variant::default(),
        );
        assert!(maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    }
    for variant in [
        Variant {
            in_place: true,
            ..Variant::default()
        },
        Variant {
            stale_distances: true,
            ..Variant::default()
        },
    ] {
        let wrong = oracle::step(&initial, &configuration, &[], 4, variant);
        assert!(maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    }
    let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
    assert_eq!(solver.step().unwrap().completed_steps, 1);
    assert_oracle(&solver, &expected);
}

#[test]
fn configured_iteration_counts_match_independent_asymmetric_systems() {
    for iterations in [1, 2, 4, 8] {
        let configuration = config(&format!(
            "[numerical]\nexternal_acceleration_m_s2=[0.0,0.0,0.0]\nrelaxation_scale=10.0\nsolver_iterations={iterations}"
        ));
        let initial = vec![
            stationary(3, 0.0, 0.0, 0.0),
            stationary(19, 0.04, 0.015, 0.0),
            stationary(800, 0.07, -0.01, 0.02),
        ];
        let expected = oracle::step(
            &initial,
            &configuration,
            &[],
            iterations,
            Variant::default(),
        );
        let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
        let _ = solver.step().unwrap();
        assert_oracle(&solver, &expected);
    }
}

#[test]
fn solver_keeps_newly_close_particles_out_until_next_step() {
    let configuration = config("[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]");
    let initial = vec![stationary(0, -0.2, 0.0, 0.0), stationary(1, 0.04, 0.0, 0.0)];
    let planes = vec![plane(Vec3::new(1.0, 0.0, 0.0), 0.0)];
    let expected = oracle::step(
        &initial,
        &configuration,
        &[([1.0, 0.0, 0.0], 0.0)],
        4,
        Variant::default(),
    );
    let wrong = oracle::step(
        &initial,
        &configuration,
        &[([1.0, 0.0, 0.0], 0.0)],
        4,
        Variant {
            rebuild: true,
            ..Variant::default()
        },
    );
    assert!(maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    let mut solver = SerialSimulation::new(&configuration, initial, planes).unwrap();
    assert_eq!(solver.step().unwrap().completed_steps, 1);
    assert_oracle(&solver, &expected);
    assert_eq!(
        solver.snapshot().particles()[1].position_m.x.to_bits(),
        0.04_f32.to_bits()
    );
}

#[test]
fn original_neighbours_stop_contributing_after_leaving_support() {
    // Weak regularisation deliberately drives this pair out of support. Pin the
    // input so the membership regression does not depend on the current defaults.
    let configuration = config(
        "[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]\nrelaxation_scale = 0.000001",
    );
    let initial = vec![stationary(0, 0.0, 0.0, 0.0), stationary(1, 0.05, 0.0, 0.0)];
    let expected = oracle::step(&initial, &configuration, &[], 4, Variant::default());
    let departed = expected
        .stages
        .iter()
        .position(|positions| {
            (positions[0][0] - positions[1][0]).abs()
                >= f64::from(configuration.derived.support_radius_m)
        })
        .expect("fixture must leave support before the last iteration");
    assert!(departed < 4);
    assert_eq!(expected.stages[departed], expected.positions);
    let wrong = oracle::step(
        &initial,
        &configuration,
        &[],
        4,
        Variant {
            stale_distances: true,
            ..Variant::default()
        },
    );
    assert!(maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
    assert_eq!(solver.step().unwrap().completed_steps, 1);
    assert_oracle(&solver, &expected);
}

#[test]
fn neighbours_are_built_from_prediction_and_rebuilt_on_subsequent_steps() {
    let configuration = config(
        "[numerical]\nfixed_timestep_s = 0.125\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]\nrelaxation_scale = 10.0",
    );
    // The first step ends outside support. The second prediction brings the
    // pair inside, so retaining the previous step's empty rows would miss it.
    let initial = vec![
        stationary(3, 0.0, 0.0, 0.0),
        particle(5, Vec3::new(0.25, 0.0, 0.0), Vec3::new(-0.75, 0.0, 0.0)),
    ];
    let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
    assert_eq!(solver.step().unwrap().completed_steps, 1);
    assert_eq!(solver.snapshot().particles()[1].position_m.x, 0.15625);
    let before = solver.snapshot().particles().to_vec();
    let expected = oracle::step(&before, &configuration, &[], 4, Variant::default());
    assert!((expected.positions[1][0] - 0.0625).abs() > 1.0e-4);
    assert_eq!(solver.step().unwrap().completed_steps, 2);
    assert_oracle(&solver, &expected);
}

#[test]
fn projection_uses_configuration_order_on_every_iteration() {
    let configuration = config("[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]");
    let initial = vec![stationary(0, -1.0, -1.0, 0.0)];
    let x = plane(Vec3::new(1.0, 0.0, 0.0), 0.0);
    let diagonal = plane(Vec3::new(0.6, 0.8, 0.0), 0.0);
    let mut forward =
        SerialSimulation::new(&configuration, initial.clone(), vec![x, diagonal]).unwrap();
    let mut reverse = SerialSimulation::new(&configuration, initial, vec![diagonal, x]).unwrap();
    assert_eq!(forward.step().unwrap().completed_steps, 1);
    assert_eq!(reverse.step().unwrap().completed_steps, 1);
    assert_vec_absolute(
        forward.snapshot().particles()[0].position_m,
        [0.48, -0.36, 0.0],
        2.0e-7,
    );
    assert_vec_absolute(
        reverse.snapshot().particles()[0].position_m,
        [0.0, 0.12, 0.0],
        2.0e-7,
    );

    let configuration = config(
        "[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]\nrelaxation_scale = 10.0",
    );
    let initial = vec![
        stationary(0, -0.02, 0.0, 0.0),
        stationary(1, 0.025, 0.02, 0.0),
    ];
    let expected = oracle::step(
        &initial,
        &configuration,
        &[([1.0, 0.0, 0.0], 0.0)],
        4,
        Variant::default(),
    );
    let wrong = oracle::step(
        &initial,
        &configuration,
        &[([1.0, 0.0, 0.0], 0.0)],
        4,
        Variant {
            final_projection_only: true,
            ..Variant::default()
        },
    );
    assert!(maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    let mut solver = SerialSimulation::new(&configuration, initial, vec![x]).unwrap();
    assert_eq!(solver.step().unwrap().completed_steps, 1);
    assert_oracle(&solver, &expected);
}

#[test]
fn invalid_particles_configuration_and_permissively_constructed_planes_are_rejected() {
    let valid = config("");
    assert!(matches!(
        SerialSimulation::new(
            &valid,
            vec![stationary(7, 0.0, 0.0, 0.0), stationary(7, 1.0, 0.0, 0.0)],
            vec![]
        ),
        Err(StepError::IdentityOrder { .. })
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for p in [
            stationary(0, bad, 0.0, 0.0),
            particle(0, Vec3::ZERO, Vec3::new(0.0, bad, 0.0)),
        ] {
            assert!(matches!(
                SerialSimulation::new(&valid, vec![p], vec![]),
                Err(StepError::Numerical(_))
            ));
        }
    }
    let permissive = Plane::new(Vec3::new(2.0, 0.0, 0.0), 0.0, 3.0).unwrap();
    assert!(matches!(
        SerialSimulation::new(&valid, vec![], vec![permissive]),
        Err(StepError::Numerical(
            NumericalError::NonUnitPlaneNormal { .. }
        ))
    ));
    let mutations: &[fn(&mut ResolvedConfiguration)] = &[
        |c| c.numerical.fixed_timestep_s = 0.0,
        |c| c.numerical.external_acceleration_m_s2.x = f32::NAN,
        |c| c.numerical.solver_iterations = 0,
        |c| c.numerical.support_policy = "inclusive".into(),
        |c| c.numerical.xsph_viscosity_coefficient = f32::NAN,
        |c| c.derived.particle_mass_kg = f32::INFINITY,
        |c| c.derived.normalised_rest_density_per_m3 = 1.0,
        |c| c.derived.relaxation_per_m2 = 0.0,
        |c| c.derived.lattice_offset_count = 0,
        |c| c.schema_version = 99,
    ];
    for mutate in mutations {
        let mut invalid = valid.clone();
        mutate(&mut invalid);
        assert!(matches!(
            SerialSimulation::new(&invalid, vec![], vec![]),
            Err(StepError::Configuration(_))
        ));
    }
}

#[test]
fn numerical_failure_and_final_plane_failure_roll_back_every_particle() {
    let configuration = config("[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]");
    let mut overflow = SerialSimulation::new(
        &configuration,
        vec![
            stationary(0, 0.0, 0.0, 0.0),
            stationary(1, 1.0e20, 0.0, 0.0),
        ],
        vec![],
    )
    .unwrap();
    let before = bits(overflow.snapshot());
    assert!(matches!(overflow.step(), Err(StepError::Numerical(_))));
    assert_eq!(bits(overflow.snapshot()), before);

    // Feasible oblique halfspaces can still leave a penetration after four
    // ordered passes. A fifth repair pass would violate the prescribed work.
    let mut penetrating = SerialSimulation::new(
        &configuration,
        vec![stationary(2, 0.0, 3.0, 0.0), stationary(9, -1.0, -1.0, 0.0)],
        vec![
            plane(Vec3::new(1.0, 0.0, 0.0), 0.0),
            plane(Vec3::new(-0.6, 0.8, 0.0), 0.0),
        ],
    )
    .unwrap();
    let before = bits(penetrating.snapshot());
    assert!(matches!(
        penetrating.step(),
        Err(StepError::OutsidePlane {
            identity: 9,
            plane: 0
        })
    ));
    assert_eq!(bits(penetrating.snapshot()), before);
}

#[test]
fn cancellation_at_every_poll_rolls_back_and_a_retry_matches_an_uninterrupted_step() {
    let configuration = config("");
    let initial = vec![stationary(0, 0.0, 1.0, 0.0), stationary(1, 0.04, 1.01, 0.0)];
    let mut clean = SerialSimulation::new(&configuration, initial.clone(), vec![]).unwrap();
    let mut polls = 0;
    let outcome = clean
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    assert!(polls >= 10);
    for cancel_at in 1..=polls {
        let mut solver = SerialSimulation::new(&configuration, initial.clone(), vec![]).unwrap();
        let before = bits(solver.snapshot());
        let mut seen = 0;
        assert!(matches!(
            solver.step_with_cancellation(|| {
                seen += 1;
                seen == cancel_at
            }),
            Err(StepError::Cancelled)
        ));
        assert_eq!(bits(solver.snapshot()), before);
        assert_eq!(solver.step().unwrap(), outcome);
        assert_eq!(bits(solver.snapshot()), bits(clean.snapshot()));
    }
}

#[test]
fn incomplete_runs_retain_only_fully_accepted_steps_and_warnings() {
    let configuration = config("[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]");
    let initial = vec![particle(0, Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0))];
    let mut expected = SerialSimulation::new(&configuration, initial.clone(), vec![]).unwrap();
    let mut polls = 0;
    let first = expected
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
    let mut seen = 0;
    let failure = solver
        .run_steps_with_cancellation(3, || {
            seen += 1;
            seen > polls + 3
        })
        .unwrap_err();
    assert!(matches!(failure.error, StepError::Cancelled));
    assert_eq!(failure.completed, vec![first]);
    assert!(failure.completed[0].cfl.warning);
    assert_eq!(bits(solver.snapshot()), bits(expected.snapshot()));
    assert_eq!(solver.step().unwrap(), expected.step().unwrap());
    assert_eq!(bits(solver.snapshot()), bits(expected.snapshot()));
}

#[test]
fn numerical_failure_in_a_later_step_retains_the_previous_acceptance() {
    let configuration = config(
        "[numerical]\nfixed_timestep_s = 1.0\nexternal_acceleration_m_s2 = [1.0e19, 0.0, 0.0]",
    );
    let initial = vec![stationary(0, 0.0, 0.0, 0.0)];
    let mut expected = SerialSimulation::new(&configuration, initial.clone(), vec![]).unwrap();
    let first = expected.step().unwrap();
    let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
    let failure = solver.run_steps(3).unwrap_err();
    assert!(matches!(
        failure.error,
        StepError::Numerical(NumericalError::NonFiniteResult { .. })
    ));
    assert_eq!(failure.completed, [first]);
    assert_eq!(bits(solver.snapshot()), bits(expected.snapshot()));
}

#[test]
fn late_numerical_errors_discard_completed_work() {
    // These inputs remain finite until physical density and velocity
    // reconstruction respectively, after prediction and constraint work.
    let cases = [
        (
            config("[numerical]\nphysical_rest_density_kg_m3 = 3.0e38"),
            (0..10).map(|id| stationary(id, 0.0, 0.0, 0.0)).collect(),
            "physical density",
        ),
        (
            config("[numerical]\nfixed_timestep_s = 1.0e-40"),
            vec![stationary(0, 0.0, 0.0, 0.0)],
            "reciprocal timestep",
        ),
    ];
    for (configuration, initial, expected_quantity) in cases {
        let mut solver = SerialSimulation::new(&configuration, initial, vec![]).unwrap();
        let before = bits(solver.snapshot());
        match solver.step().unwrap_err() {
            StepError::Numerical(NumericalError::NonFiniteResult { quantity }) => {
                assert_eq!(quantity, expected_quantity)
            }
            other => panic!("unexpected failure: {other}"),
        }
        assert_eq!(bits(solver.snapshot()), before);
    }
}

#[test]
fn predicate_panic_cannot_commit_working_state() {
    let mut solver =
        SerialSimulation::new(&config(""), vec![stationary(0, 0.0, 1.0, 0.0)], vec![]).unwrap();
    let before = bits(solver.snapshot());
    let mut polls = 0;
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solver.step_with_cancellation(|| {
            polls += 1;
            assert!(polls < 6, "observer failed after constraint work");
            false
        })
    }));
    assert!(failed.is_err());
    assert_eq!(bits(solver.snapshot()), before);
    assert_eq!(solver.step().unwrap().completed_steps, 1);
}

#[test]
fn grid_retains_prediction_membership_and_rebuilds_after_acceptance() {
    use maelstrom::neighbours::SearchBackend;
    let configuration = config("[numerical]\nexternal_acceleration_m_s2 = [0.0,0.0,0.0]");
    // Projection brings a previously distant particle inside support during the
    // first step. Rebuilding inside Jacobi would move particle 1 immediately.
    let initial = vec![stationary(0, -0.2, 0.0, 0.0), stationary(1, 0.04, 0.0, 0.0)];
    let planes = vec![plane(Vec3::new(1.0, 0.0, 0.0), 0.0)];
    let mut grid = SerialSimulation::new_with_backend(
        &configuration,
        initial.clone(),
        planes.clone(),
        SearchBackend::UniformGrid,
    )
    .unwrap();
    let mut reference = SerialSimulation::new(&configuration, initial, planes).unwrap();
    for step in 1..=3 {
        assert_eq!(grid.step().unwrap(), reference.step().unwrap());
        assert_eq!(bits(grid.snapshot()), bits(reference.snapshot()));
        if step == 1 {
            assert_eq!(
                grid.snapshot().particles()[1].position_m.x.to_bits(),
                0.04_f32.to_bits()
            );
        }
    }
    // Projection above reconstructs a velocity that separates the pair next
    // step. Use an approaching pair instead to distinguish a fresh prediction
    // build from stale membership, crossing the x=0.25 cell face on step two.
    let configuration =
        config("[numerical]\nfixed_timestep_s = 1.0\nexternal_acceleration_m_s2 = [0.0,0.0,0.0]");
    let initial = vec![
        particle(0, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.1, 0.0, 0.0)),
        stationary(1, 0.34, 0.0, 0.0),
    ];
    let mut reference = SerialSimulation::new(&configuration, initial.clone(), vec![]).unwrap();
    let mut grid = SerialSimulation::new_with_backend(
        &configuration,
        initial,
        vec![],
        SearchBackend::UniformGrid,
    )
    .unwrap();
    for step in 1..=2 {
        assert_eq!(grid.step().unwrap(), reference.step().unwrap());
        assert_eq!(bits(grid.snapshot()), bits(reference.snapshot()));
        if step == 1 {
            assert_eq!(
                grid.snapshot().particles()[1].position_m.x.to_bits(),
                0.34_f32.to_bits()
            );
        } else {
            assert_ne!(
                grid.snapshot().particles()[1].position_m.x.to_bits(),
                0.34_f32.to_bits()
            );
        }
    }
    // A pair that leaves support must stop contributing despite fixed rows.
    let configuration = config(
        "[numerical]\nexternal_acceleration_m_s2 = [0.0,0.0,0.0]\nrelaxation_scale = 0.000001",
    );
    let initial = vec![stationary(0, 0.0, 0.0, 0.0), stationary(1, 0.05, 0.0, 0.0)];
    let expected = oracle::step(&initial, &configuration, &[], 4, Variant::default());
    assert!((expected.positions[0][0] - expected.positions[1][0]).abs() > 0.1);
    let mut grid = SerialSimulation::new_with_backend(
        &configuration,
        initial,
        vec![],
        SearchBackend::UniformGrid,
    )
    .unwrap();
    let _ = grid.step().unwrap();
    assert_oracle(&grid, &expected);
}

#[test]
fn grid_cancellation_at_every_stage_discards_work_and_retries_exactly() {
    use maelstrom::neighbours::SearchBackend;
    let configuration = config("[numerical]\nsolver_iterations = 8");
    let initial = vec![
        stationary(u64::MAX, 0.0, 0.5, 0.0),
        stationary(42, 0.05, 0.5, 0.0),
        stationary(0, 0.0, 0.55, 0.0),
    ];
    let mut reference = SerialSimulation::new(&configuration, initial.clone(), vec![]).unwrap();
    let mut polls = 0;
    let expected = reference
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    for stages in [None, Some(false), Some(true)] {
        for cancel_at in 1..=polls {
            let mut grid = SerialSimulation::new_with_backend(
                &configuration,
                initial.clone(),
                vec![],
                SearchBackend::UniformGrid,
            )
            .unwrap();
            assert_eq!(grid.backend(), SearchBackend::UniformGrid);
            let before = bits(grid.snapshot());
            let mut seen = 0;
            let mut cancelled = || {
                seen += 1;
                seen == cancel_at
            };
            let error = match stages {
                None => grid.step_with_cancellation(&mut cancelled).unwrap_err(),
                Some(stages) => grid
                    .step_measured_with_cancellation(stages, &mut cancelled)
                    .unwrap_err(),
            };
            assert!(matches!(error, StepError::Cancelled));
            assert_eq!(seen, cancel_at);
            assert_eq!(bits(grid.snapshot()), before);
            assert_eq!(grid.step().unwrap(), expected);
            assert_eq!(bits(grid.snapshot()), bits(reference.snapshot()));
        }
    }
}

#[test]
fn grid_numerical_and_plane_failures_preserve_every_accepted_bit() {
    use maelstrom::neighbours::SearchBackend;
    let still = "[numerical]\nexternal_acceleration_m_s2 = [0.0,0.0,0.0]";
    let cases = [
        (
            config(still),
            vec![
                stationary(0, 0.0, 0.0, 0.0),
                stationary(1, 1.0e20, 0.0, 0.0),
            ],
            vec![],
        ),
        (
            config(still),
            vec![stationary(2, 0.0, 3.0, 0.0), stationary(9, -1.0, -1.0, 0.0)],
            vec![
                plane(Vec3::new(1.0, 0.0, 0.0), 0.0),
                plane(Vec3::new(-0.6, 0.8, 0.0), 0.0),
            ],
        ),
        (
            config("[numerical]\nphysical_rest_density_kg_m3 = 3.0e38"),
            (0..10).map(|id| stationary(id, 0.0, 0.0, 0.0)).collect(),
            vec![],
        ),
        (
            config("[numerical]\nfixed_timestep_s = 1.0e-40"),
            vec![stationary(0, 0.0, 0.0, 0.0)],
            vec![],
        ),
    ];
    for (configuration, initial, planes) in cases {
        let mut reference =
            SerialSimulation::new(&configuration, initial.clone(), planes.clone()).unwrap();
        let expected = reference.step().unwrap_err().to_string();
        for profile in [None, Some(false), Some(true)] {
            let mut grid = SerialSimulation::new_with_backend(
                &configuration,
                initial.clone(),
                planes.clone(),
                SearchBackend::UniformGrid,
            )
            .unwrap();
            let before = bits(grid.snapshot());
            for _ in 0..2 {
                let error = match profile {
                    None => grid.step().unwrap_err(),
                    Some(stages) => grid
                        .step_measured_with_cancellation(stages, || false)
                        .unwrap_err(),
                };
                assert_eq!(error.to_string(), expected);
                assert_eq!(bits(grid.snapshot()), before);
            }
        }
    }
}

#[test]
fn repeated_scene_runs_and_input_permutations_are_bitwise_identical() {
    let scene = load_resolved_scene(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes/falling-block-04.toml"),
        None,
    )
    .unwrap();
    let mut a = SerialSimulation::from_scene(&scene).unwrap();
    let mut b = SerialSimulation::from_scene(&scene).unwrap();
    let mut reversed = scene.particles.clone();
    reversed.reverse();
    let mut c =
        SerialSimulation::new(&scene.configuration, reversed, scene.planes.clone()).unwrap();
    let original = scene.clone();
    let mut a_outcomes = Vec::new();
    let b_outcomes = b.run_steps(12).unwrap();
    for expected in &b_outcomes {
        let a_outcome = a.step().unwrap();
        let actual = c.step().unwrap();
        assert_eq!(actual, *expected);
        assert_eq!(a_outcome, *expected);
        assert_eq!(
            actual.cfl.maximum_speed.to_bits(),
            expected.cfl.maximum_speed.to_bits()
        );
        if let (CflLimit::Finite(x), CflLimit::Finite(y)) = (actual.cfl.limit, expected.cfl.limit) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(bits(a.snapshot()), bits(c.snapshot()));
        a_outcomes.push(a_outcome);
    }
    assert_eq!(a_outcomes, b_outcomes);
    assert_eq!(bits(a.snapshot()), bits(b.snapshot()));
    assert_eq!(bits(a.snapshot()), bits(c.snapshot()));
    assert_eq!(scene, original);
}
