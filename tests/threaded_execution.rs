//! Threaded work must preserve stage snapshots, ordered failures and accepted bits.
//! Independent equations and deliberately wrong variants distinguish numerical
//! correctness from simply agreeing with another execution of the same code.

#[path = "support/serial_oracle.rs"]
mod oracle;
mod support;

use maelstrom::{
    boundary::Plane,
    config::{ResolvedConfiguration, parse_configuration_overlay, resolve_effective_configuration},
    execution::{Execution, ExecutionError, ExecutionMode, MAX_WORKERS},
    fluid::{
        InteractionSets, PbfParameters, compute_jacobi_stage, compute_jacobi_stage_with_execution,
    },
    math::Vec3,
    neighbours::{SearchBackend, brute_force, uniform_grid},
    scene::{Particle, load_resolved_scene},
    simulation::{Simulation, Snapshot, StepError},
    workflow::{self, ExecutionOptions, MeasurementMode, RunRequest},
};

const WORKERS: [usize; 4] = [1, 2, 3, 7];

fn configuration(source: &str) -> ResolvedConfiguration {
    resolve_effective_configuration(
        None,
        None,
        Some(&parse_configuration_overlay(source).unwrap()),
    )
    .unwrap()
}

fn particle(identity: u64, position_m: Vec3) -> Particle {
    Particle {
        identity,
        position_m,
        velocity_m_s: Vec3::ZERO,
    }
}

fn bits(snapshot: &Snapshot) -> Vec<(u64, [u32; 6])> {
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
        .collect()
}

fn threaded(
    config: &ResolvedConfiguration,
    particles: Vec<Particle>,
    planes: Vec<Plane>,
    workers: usize,
) -> Simulation {
    Simulation::new_with_execution(
        config,
        particles,
        planes,
        SearchBackend::UniformGrid,
        ExecutionMode::Threaded { workers },
    )
    .unwrap()
}

fn physical(result: &workflow::RunResult) -> String {
    let mut value: serde_json::Value = serde_json::from_str(&result.to_json().unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("execution");
    support::serial_document(value)
}

#[test]
fn independent_equations_check_every_stage_quantity_with_more_workers_than_particles() {
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let rows = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let parameters = PbfParameters::new(1.0, 3.0, 0.01, 0.1, 0.2, 4.0).unwrap();
    let density = support::poly6_f64(0.0, 1.0) + support::poly6_f64(0.5, 1.0);
    let constraint = density / 3.0 - 1.0;
    let gradient = support::spiky_magnitude_f64(0.5, 1.0);
    let multiplier = -constraint / (2.0 * (gradient / 3.0).powi(2) + 0.01);
    let pressure = -0.1 * (support::poly6_f64(0.5, 1.0) / support::poly6_f64(0.2, 1.0)).powi(4);
    let correction = (2.0 * multiplier + pressure) * gradient / 3.0;
    for workers in WORKERS {
        let execution = Execution::new(ExecutionMode::Threaded { workers }).unwrap();
        for _ in 0..4 {
            let stage =
                compute_jacobi_stage_with_execution(&positions, &rows, parameters, &execution)
                    .unwrap();
            for i in 0..2 {
                support::assert_absolute(stage.normalised_densities[i], density, 4.0e-6);
                support::assert_absolute(stage.constraints[i], constraint, 2.0e-6);
                support::assert_absolute(stage.multipliers[i], multiplier, 4.0e-6);
                support::assert_vec_absolute(
                    stage.corrections[i],
                    [if i == 0 { correction } else { -correction }, 0.0, 0.0],
                    2.0e-5,
                );
            }
        }
    }
}

#[test]
fn asymmetric_timestep_rejects_stale_distances_in_place_updates_and_missing_passes() {
    let config = configuration(
        "[numerical]\nexternal_acceleration_m_s2=[0.0,0.0,0.0]\nrelaxation_scale=10.0\nsolver_iterations=4",
    );
    let initial = vec![
        particle(3, Vec3::ZERO),
        particle(19, Vec3::new(0.04, 0.015, 0.0)),
        particle(800, Vec3::new(0.07, -0.01, 0.02)),
    ];
    let expected = oracle::step(&initial, &config, &[], 4, oracle::Variant::default());
    assert_eq!(expected.stages.len(), 5);
    assert!(
        expected
            .stages
            .windows(2)
            .all(|pair| oracle::maximum_difference(&pair[0], &pair[1]) > 1.0e-6)
    );
    for variant in [
        oracle::Variant {
            in_place: true,
            ..Default::default()
        },
        oracle::Variant {
            stale_distances: true,
            ..Default::default()
        },
    ] {
        let wrong = oracle::step(&initial, &config, &[], 4, variant);
        assert!(oracle::maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    }
    for iterations in [3, 5] {
        let wrong = oracle::step(&initial, &config, &[], iterations, Default::default());
        assert!(oracle::maximum_difference(&expected.positions, &wrong.positions) > 1.0e-4);
    }
    let mut serial = Simulation::new(&config, initial.clone(), vec![]).unwrap();
    let _ = serial.step().unwrap();
    for workers in WORKERS {
        let mut solver = threaded(&config, initial.clone(), vec![], workers);
        let _ = solver.step().unwrap();
        assert_eq!(bits(solver.snapshot()), bits(serial.snapshot()));
        for ((actual, &position), &velocity) in solver
            .snapshot()
            .particles()
            .iter()
            .zip(&expected.positions)
            .zip(&expected.velocities)
        {
            support::assert_vec_absolute(actual.position_m, position, 5.0e-6);
            support::assert_vec_absolute(
                actual.velocity_m_s,
                velocity,
                5.0e-6 / config.numerical.fixed_timestep_s + 2.0e-6,
            );
        }
    }
}

#[test]
fn dense_sparse_boundary_and_empty_stages_keep_all_ordered_bits() {
    let h = 0.125_f32;
    let parameters = PbfParameters::new(h, 8000.0, 2000.0, 0.00001, h * 0.3, 4.0).unwrap();
    let cases = [
        vec![],
        vec![Vec3::new(-0.0, 0.0, 0.0)],
        vec![Vec3::ZERO; 37],
        (0..31)
            .map(|i| Vec3::new(i as f32 * 2.0 - 30.0, 0.0, 0.0))
            .collect(),
        vec![
            Vec3::ZERO,
            Vec3::new(-h, 0.0, 0.0),
            Vec3::new(f32::from_bits(h.to_bits() - 1), 0.0, 0.0),
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(f32::from_bits(h.to_bits() + 1), 0.0, 0.0),
            Vec3::new(-h, -h, 0.0),
        ],
        (0..53)
            .map(|i| {
                Vec3::new(
                    (i % 7) as f32 * 0.03 - 0.1,
                    (i % 5) as f32 * 0.04 - 0.1,
                    (i % 3) as f32 * 0.02,
                )
            })
            .collect(),
        vec![Vec3::new(f32::MAX, 0.0, 0.0)],
    ];
    for workers in WORKERS {
        let execution = Execution::new(ExecutionMode::Threaded { workers }).unwrap();
        for positions in &cases {
            let rows = uniform_grid(positions, h).unwrap();
            assert_eq!(rows, brute_force(positions, h).unwrap());
            let serial = compute_jacobi_stage(positions, &rows, parameters).unwrap();
            for _ in 0..3 {
                let actual =
                    compute_jacobi_stage_with_execution(positions, &rows, parameters, &execution)
                        .unwrap();
                for (a, b) in actual
                    .normalised_densities
                    .iter()
                    .chain(&actual.constraints)
                    .chain(&actual.multipliers)
                    .zip(
                        serial
                            .normalised_densities
                            .iter()
                            .chain(&serial.constraints)
                            .chain(&serial.multipliers),
                    )
                {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in actual.corrections.iter().zip(&serial.corrections) {
                    assert_eq!(
                        [a.x, a.y, a.z].map(f32::to_bits),
                        [b.x, b.y, b.z].map(f32::to_bits)
                    );
                }
            }
        }
    }
}

#[test]
fn current_filtering_keeps_fixed_rows_and_does_not_admit_new_neighbours() {
    let original = [
        Vec3::ZERO,
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::new(0.2, 0.0, 0.0),
    ];
    let current = [
        Vec3::ZERO,
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::new(0.025, 0.0, 0.0),
    ];
    let rows = uniform_grid(&original, 0.1).unwrap();
    let parameters = configuration("").pbf_parameters().unwrap();
    let expected = compute_jacobi_stage(&current, &rows, parameters).unwrap();
    let wrong =
        compute_jacobi_stage(&current, &uniform_grid(&current, 0.1).unwrap(), parameters).unwrap();
    assert_ne!(expected, wrong);
    for workers in WORKERS {
        let execution = Execution::new(ExecutionMode::Threaded { workers }).unwrap();
        assert_eq!(
            compute_jacobi_stage_with_execution(&current, &rows, parameters, &execution).unwrap(),
            expected
        );
    }
}

#[test]
fn competing_errors_use_pass_then_particle_precedence_and_do_not_poison_retry() {
    let parameters = configuration("").pbf_parameters().unwrap();
    let rows = InteractionSets::new(vec![vec![1], vec![0], vec![3], vec![2]]).unwrap();
    let bad = [
        Vec3::ZERO,
        Vec3::new(1.0e20, 0.0, 0.0),
        Vec3::new(f32::MAX, 0.0, 0.0),
        Vec3::new(-f32::MAX, 0.0, 0.0),
    ];
    let good = [Vec3::ZERO; 4];
    let expected = compute_jacobi_stage(&bad, &rows, parameters).unwrap_err();
    // The later pair fails at subtraction; the first fails only at squared length.
    assert_ne!(
        compute_jacobi_stage(
            &bad[2..],
            &InteractionSets::new(vec![vec![1], vec![0]]).unwrap(),
            parameters
        )
        .unwrap_err(),
        expected
    );
    for workers in WORKERS {
        let execution = Execution::new(ExecutionMode::Threaded { workers }).unwrap();
        for _ in 0..16 {
            assert_eq!(
                compute_jacobi_stage_with_execution(&bad, &rows, parameters, &execution)
                    .unwrap_err(),
                expected
            );
            assert_eq!(
                compute_jacobi_stage_with_execution(&good, &rows, parameters, &execution).unwrap(),
                compute_jacobi_stage(&good, &rows, parameters).unwrap()
            );
        }
        let invalid = [Vec3::new(f32::NAN, 0.0, 0.0)];
        assert!(
            compute_jacobi_stage_with_execution(
                &invalid,
                &InteractionSets::new(vec![vec![]]).unwrap(),
                parameters,
                &execution
            )
            .is_err()
        );
        assert!(
            compute_jacobi_stage_with_execution(
                &good,
                &InteractionSets::new(vec![]).unwrap(),
                parameters,
                &execution
            )
            .is_err()
        );
    }
}

#[test]
fn every_cancellation_poll_runs_on_caller_and_retry_matches_uninterrupted_state() {
    let config = configuration("");
    let initial = vec![
        particle(u64::MAX, Vec3::new(0.0, 0.5, 0.0)),
        particle(42, Vec3::new(0.05, 0.5, 0.0)),
        particle(0, Vec3::new(0.0, 0.55, 0.0)),
    ];
    let mut serial = Simulation::new(&config, initial.clone(), vec![]).unwrap();
    let mut polls = 0;
    let expected = serial
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    let caller = std::thread::current().id();
    for workers in WORKERS {
        for profile in [None, Some(false), Some(true)] {
            for cancel_at in 1..=polls {
                let mut solver = threaded(&config, initial.clone(), vec![], workers);
                let before = bits(solver.snapshot());
                // Rc<Cell> deliberately cannot be sent or shared with a worker.
                let seen = std::rc::Rc::new(std::cell::Cell::new(0));
                let mut cancelled = || {
                    assert_eq!(std::thread::current().id(), caller);
                    seen.set(seen.get() + 1);
                    seen.get() == cancel_at
                };
                let error = match profile {
                    None => solver.step_with_cancellation(&mut cancelled).unwrap_err(),
                    Some(stages) => solver
                        .step_measured_with_cancellation(stages, &mut cancelled)
                        .unwrap_err(),
                };
                assert!(matches!(error, StepError::Cancelled));
                assert_eq!(seen.get(), cancel_at);
                assert_eq!(bits(solver.snapshot()), before);
                assert_eq!(solver.snapshot().completed_steps(), 0);
                assert_eq!(solver.step().unwrap(), expected);
                assert_eq!(bits(solver.snapshot()), bits(serial.snapshot()));
            }
        }
    }
}

#[test]
fn numerical_and_projection_failures_preserve_state_and_classification_on_repeat() {
    let cases = [
        (
            configuration(""),
            vec![
                particle(0, Vec3::ZERO),
                particle(1, Vec3::new(1.0e20, 0.0, 0.0)),
            ],
            vec![],
        ),
        (
            configuration(""),
            vec![
                particle(2, Vec3::new(0.0, 3.0, 0.0)),
                particle(9, Vec3::new(-1.0, -1.0, 0.0)),
            ],
            vec![
                Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 1.0e-6).unwrap(),
                Plane::new(Vec3::new(-0.6, 0.8, 0.0), 0.0, 1.0e-6).unwrap(),
            ],
        ),
        (
            configuration("[numerical]\nphysical_rest_density_kg_m3=3.0e38"),
            (0..10).map(|id| particle(id, Vec3::ZERO)).collect(),
            vec![],
        ),
        (
            configuration("[numerical]\nfixed_timestep_s=1.0e-40"),
            vec![particle(0, Vec3::ZERO)],
            vec![],
        ),
    ];
    for (config, initial, planes) in cases {
        let mut serial = Simulation::new(&config, initial.clone(), planes.clone()).unwrap();
        let expected = serial.step().unwrap_err();
        for workers in WORKERS {
            let mut solver = threaded(&config, initial.clone(), planes.clone(), workers);
            let before = bits(solver.snapshot());
            for profile in [None, Some(false), Some(true)] {
                let actual = match profile {
                    None => solver.step().unwrap_err(),
                    Some(stages) => solver
                        .step_measured_with_cancellation(stages, || false)
                        .unwrap_err(),
                };
                assert_eq!(
                    std::mem::discriminant(&actual),
                    std::mem::discriminant(&expected)
                );
                assert_eq!(actual.to_string(), expected.to_string());
                assert_eq!(bits(solver.snapshot()), before);
                assert_eq!(solver.snapshot().completed_steps(), 0);
            }
        }
    }
}

#[test]
fn invalid_resources_and_backend_pairings_fail_before_state_is_created() {
    let config = configuration("");
    for workers in [0, MAX_WORKERS + 1, usize::MAX] {
        assert!(matches!(
            Execution::new(ExecutionMode::Threaded { workers }),
            Err(ExecutionError::InvalidWorkers { .. })
        ));
        assert!(
            Simulation::new_with_execution(
                &config,
                vec![],
                vec![],
                SearchBackend::UniformGrid,
                ExecutionMode::Threaded { workers }
            )
            .is_err()
        );
    }
    assert!(matches!(
        Simulation::new_with_execution(
            &config,
            vec![],
            vec![],
            SearchBackend::BruteForce,
            ExecutionMode::Threaded { workers: 1 }
        ),
        Err(StepError::Execution(ExecutionError::RequiresUniformGrid))
    ));
    for workers in WORKERS {
        let mut empty = threaded(&config, vec![], vec![], workers);
        assert_eq!(empty.execution(), ExecutionMode::Threaded { workers });
        assert_eq!(empty.run_steps(3).unwrap().len(), 3);
        assert!(empty.snapshot().particles().is_empty());
    }
}

#[test]
fn independent_simultaneous_solvers_keep_their_own_pools_inputs_and_observations() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/small-lattice.toml");
    let scene = load_resolved_scene(&path, None).unwrap();
    let before = scene.clone();
    let request = RunRequest {
        steps: 12,
        checkpoints: vec![0, 1, 4, 12],
    };
    let reference = physical(&workflow::run(&scene, request.clone()));
    std::thread::scope(|scope| {
        let handles: Vec<_> = WORKERS
            .into_iter()
            .map(|workers| {
                let scene = &scene;
                let request = request.clone();
                scope.spawn(move || {
                    let result = workflow::run_with_options(
                        scene,
                        request,
                        ExecutionOptions {
                            backend: SearchBackend::UniformGrid,
                            execution: ExecutionMode::Threaded { workers },
                            measurement: Some(MeasurementMode::Stages),
                        },
                        || false,
                    );
                    assert_eq!(result.execution(), ExecutionMode::Threaded { workers });
                    physical(&result)
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), reference);
        }
    });
    assert_eq!(scene, before);
}

#[test]
fn failures_after_acceptance_and_predicate_panics_preserve_prior_work() {
    let config = configuration(
        "[numerical]\nfixed_timestep_s=1.0\nexternal_acceleration_m_s2=[1.0e19,0.0,0.0]",
    );
    let initial = vec![particle(0, Vec3::ZERO)];
    let mut serial = Simulation::new(&config, initial.clone(), vec![]).unwrap();
    let first = serial.step().unwrap();
    assert!(first.cfl.warning);
    let expected_error = serial.step().unwrap_err().to_string();
    for workers in WORKERS {
        let mut solver = threaded(&config, initial.clone(), vec![], workers);
        let failure = solver.run_steps(3).unwrap_err();
        assert_eq!(failure.completed, vec![first]);
        assert_eq!(failure.error.to_string(), expected_error);
        assert_eq!(bits(solver.snapshot()), bits(serial.snapshot()));
        assert_eq!(solver.snapshot().completed_steps(), 1);
        let mut seen = 0;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                solver.step_with_cancellation(|| {
                    seen += 1;
                    panic!("injected caller panic")
                })
            }))
            .is_err()
        );
        assert_eq!(seen, 1);
        assert_eq!(bits(solver.snapshot()), bits(serial.snapshot()));
    }
}

#[test]
fn a_later_active_row_failure_precedes_an_earlier_particle_multiplier_failure() {
    let parameters = PbfParameters::new(0.1, f32::MIN_POSITIVE, 1.0, 0.0, 0.02, 4.0).unwrap();
    let positions = [Vec3::ZERO, Vec3::ZERO, Vec3::new(1.0e20, 0.0, 0.0)];
    let rows = InteractionSets::new(vec![vec![], vec![2], vec![1]]).unwrap();
    let expected = compute_jacobi_stage(&positions, &rows, parameters).unwrap_err();
    let premature = compute_jacobi_stage(
        &positions[..1],
        &InteractionSets::new(vec![vec![]]).unwrap(),
        parameters,
    )
    .unwrap_err();
    assert_ne!(
        expected, premature,
        "the fixture must distinguish fused passes"
    );
    for workers in WORKERS {
        let execution = Execution::new(ExecutionMode::Threaded { workers }).unwrap();
        assert_eq!(
            compute_jacobi_stage_with_execution(&positions, &rows, parameters, &execution)
                .unwrap_err(),
            expected
        );
    }
}
