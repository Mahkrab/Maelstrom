//! Device stepping is checked against independent equations, incorrect controls
//! and CPU transactions. Hardware cases must be selected explicitly and cannot
//! pass by skipping an absent device.

#[cfg(feature = "cuda")]
#[path = "support/serial_oracle.rs"]
mod oracle;
#[cfg(feature = "cuda")]
mod support;

use maelstrom::{
    config::{ResolvedConfiguration, parse_configuration_overlay, resolve_effective_configuration},
    execution::{Execution, ExecutionMode},
    neighbours::SearchBackend,
    simulation::{Simulation, StepError},
};

fn configuration(source: &str) -> ResolvedConfiguration {
    resolve_effective_configuration(
        None,
        None,
        Some(&parse_configuration_overlay(source).unwrap()),
    )
    .unwrap()
}

#[test]
fn invalid_device_settings_and_cpu_scheduler_pairings_fail_without_a_driver() {
    for mode in [
        ExecutionMode::Cuda {
            device: u32::MAX,
            threads: 128,
        },
        ExecutionMode::Cuda {
            device: 0,
            threads: 0,
        },
        ExecutionMode::Cuda {
            device: 0,
            threads: 33,
        },
    ] {
        assert!(mode.validate().is_err());
    }
    let mode = ExecutionMode::Cuda {
        device: 0,
        threads: 128,
    };
    assert!(Execution::new(mode).is_err());
    assert!(matches!(
        Simulation::new_with_execution(
            &configuration(""),
            vec![],
            vec![],
            SearchBackend::BruteForce,
            mode
        ),
        Err(StepError::Execution(_))
    ));
    #[cfg(not(feature = "cuda"))]
    assert!(matches!(
        Simulation::new_with_execution(
            &configuration(""),
            vec![],
            vec![],
            SearchBackend::UniformGrid,
            mode
        ),
        Err(StepError::Cuda(maelstrom::cuda::CudaError::FeatureDisabled))
    ));
}

#[cfg(feature = "cuda")]
mod hardware {
    use super::*;
    use maelstrom::{
        boundary::Plane,
        simulation::Snapshot,
        workflow::{self, ExecutionOptions, MeasurementMode, RunRequest},
    };
    use maelstrom::{math::Vec3, scene::Particle};

    fn particle(identity: u64, p: Vec3) -> Particle {
        Particle {
            identity,
            position_m: p,
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
    fn gpu(
        config: &ResolvedConfiguration,
        particles: Vec<Particle>,
        planes: Vec<Plane>,
        threads: u32,
    ) -> Simulation {
        Simulation::new_with_execution(
            config,
            particles,
            planes,
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda { device: 0, threads },
        )
        .unwrap()
    }
    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn asymmetric_stages_reject_stale_membership_in_place_updates_and_wrong_work() {
        let config = configuration(
            "[numerical]\nexternal_acceleration_m_s2=[0.0,0.0,0.0]\nrelaxation_scale=10.0\nsolver_iterations=4",
        );
        let initial = vec![
            particle(3, Vec3::ZERO),
            particle(19, Vec3::new(0.04, 0.015, 0.0)),
            particle(800, Vec3::new(0.07, -0.01, 0.02)),
        ];
        let expected = oracle::step(&initial, &config, &[], 4, Default::default());
        assert_eq!(expected.stages.len(), 5);
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
            assert!(oracle::maximum_difference(&expected.positions, &wrong.positions) > 1e-4);
        }
        for iterations in [3, 5] {
            assert!(
                oracle::maximum_difference(
                    &expected.positions,
                    &oracle::step(&initial, &config, &[], iterations, Default::default()).positions
                ) > 1e-4
            );
        }
        let mut cpu = Simulation::new(&config, initial.clone(), vec![]).unwrap();
        let _ = cpu.step().unwrap();
        for threads in [32, 64, 128, 256] {
            let mut solver = gpu(&config, initial.clone(), vec![], threads);
            let _ = solver.step().unwrap();
            assert_eq!(bits(solver.snapshot()), bits(cpu.snapshot()));
            for ((p, &position), &velocity) in solver
                .snapshot()
                .particles()
                .iter()
                .zip(&expected.positions)
                .zip(&expected.velocities)
            {
                support::assert_vec_absolute(p.position_m, position, 5e-6);
                support::assert_vec_absolute(
                    p.velocity_m_s,
                    velocity,
                    5e-6 / config.numerical.fixed_timestep_s + 2e-6,
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn published_full_corrections_and_contact_match_independent_equations() {
        let mut overlay = maelstrom::config::ConfigurationOverlay::default();
        maelstrom::config::NumericalMode::PaperBaseline.fill_overrides(&mut overlay.numerical);
        // The independent position oracle ends at velocity reconstruction. The
        // enabled-effect path is compared bitwise in the cancellation test below.
        overlay.numerical.xsph_viscosity_enabled = Some(false);
        let config = resolve_effective_configuration(None, None, Some(&overlay)).unwrap();
        let initial = vec![
            particle(0, Vec3::new(0.0, 0.005, 0.0)),
            particle(1, Vec3::new(0.04, 0.015, 0.0)),
            particle(2, Vec3::new(0.07, 0.005, 0.02)),
        ];
        let expected = oracle::step(
            &initial,
            &config,
            &[([0.0, 1.0, 0.0], 0.0)],
            3,
            Default::default(),
        );
        let planes = vec![Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0).unwrap()];
        let mut cpu = Simulation::new(&config, initial.clone(), planes.clone()).unwrap();
        let _ = cpu.step().unwrap();
        let mut device = gpu(&config, initial, planes, 128);
        let _ = device.step().unwrap();
        assert_eq!(bits(cpu.snapshot()), bits(device.snapshot()));
        for ((p, &position), &velocity) in device
            .snapshot()
            .particles()
            .iter()
            .zip(&expected.positions)
            .zip(&expected.velocities)
        {
            support::assert_vec_absolute(p.position_m, position, 5e-6);
            support::assert_vec_absolute(p.velocity_m_s, velocity, 0.002);
            assert!(p.position_m.y >= 0.0);
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn all_cancellation_polls_rollback_after_acceptance_and_retry_with_no_stale_buffers() {
        for mode in [
            None,
            Some(maelstrom::config::NumericalMode::Reference),
            Some(maelstrom::config::NumericalMode::Fast),
            Some(maelstrom::config::NumericalMode::Experimental),
            Some(maelstrom::config::NumericalMode::PaperBaseline),
        ] {
            let mut overlay = maelstrom::config::ConfigurationOverlay::default();
            if let Some(mode) = mode {
                mode.fill_overrides(&mut overlay.numerical);
            }
            let config = resolve_effective_configuration(None, None, Some(&overlay)).unwrap();
            let initial = vec![
                particle(u64::MAX, Vec3::new(0.0, 0.5, 0.0)),
                particle(42, Vec3::new(0.05, 0.5, 0.0)),
                particle(0, Vec3::new(0.0, 0.55, 0.0)),
            ];
            let mut cpu = Simulation::new(&config, initial.clone(), vec![]).unwrap();
            let mut polls = 0;
            let expected = cpu
                .step_with_cancellation(|| {
                    polls += 1;
                    false
                })
                .unwrap();
            for profile in [None, Some(false), Some(true)] {
                for cancel_at in 1..=polls {
                    let mut solver = gpu(&config, initial.clone(), vec![], 128);
                    let _ = solver.step().unwrap();
                    let before = bits(solver.snapshot());
                    let caller = std::thread::current().id();
                    let mut seen = 0;
                    let mut cancelled = || {
                        assert_eq!(std::thread::current().id(), caller);
                        seen += 1;
                        seen == cancel_at
                    };
                    let error = match profile {
                        None => solver.step_with_cancellation(&mut cancelled).unwrap_err(),
                        Some(stages) => solver
                            .step_measured_with_cancellation(stages, &mut cancelled)
                            .unwrap_err(),
                    };
                    assert!(matches!(error, StepError::Cancelled));
                    assert_eq!(seen, cancel_at);
                    assert_eq!(bits(solver.snapshot()), before);
                    assert_eq!(solver.snapshot().completed_steps(), 1);
                    let _ = solver.step().unwrap();
                    let mut control = Simulation::new(&config, initial.clone(), vec![]).unwrap();
                    let _ = control.run_steps(2).unwrap();
                    assert_eq!(bits(solver.snapshot()), bits(control.snapshot()));
                    assert_eq!(expected.completed_steps, 1);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn numerical_error_precedence_projection_and_rollback_match_serial() {
        let cases = [
            (
                configuration(""),
                vec![
                    particle(0, Vec3::ZERO),
                    particle(1, Vec3::new(1e20, 0.0, 0.0)),
                    particle(2, Vec3::new(f32::MAX, 0.0, 0.0)),
                    particle(3, Vec3::new(-f32::MAX, 0.0, 0.0)),
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
                    Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 1e-6).unwrap(),
                    Plane::new(Vec3::new(-0.6, 0.8, 0.0), 0.0, 1e-6).unwrap(),
                ],
            ),
            (
                configuration("[numerical]\nphysical_rest_density_kg_m3=3e38"),
                (0..10).map(|id| particle(id, Vec3::ZERO)).collect(),
                vec![],
            ),
            (
                configuration("[numerical]\nfixed_timestep_s=1e-40"),
                vec![particle(0, Vec3::ZERO)],
                vec![],
            ),
        ];
        for (config, initial, planes) in cases {
            let mut cpu = Simulation::new(&config, initial.clone(), planes.clone()).unwrap();
            let expected = cpu.step().unwrap_err();
            for threads in [32, 256] {
                let mut solver = gpu(&config, initial.clone(), planes.clone(), threads);
                let before = bits(solver.snapshot());
                for _ in 0..3 {
                    let actual = solver.step().unwrap_err();
                    assert_eq!(actual.to_string(), expected.to_string());
                    assert_eq!(
                        std::mem::discriminant(&actual),
                        std::mem::discriminant(&expected)
                    );
                    assert_eq!(bits(solver.snapshot()), before);
                    assert_eq!(solver.snapshot().completed_steps(), 0);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn empty_single_coincident_and_independent_lifetimes_keep_accepted_bits() {
        let config = configuration("");
        for initial in [
            vec![],
            vec![particle(u64::MAX, Vec3::new(-0.0, 0.0, 0.0))],
            (0..129).map(|i| particle(i, Vec3::ZERO)).collect(),
        ] {
            let mut first = gpu(&config, initial.clone(), vec![], 32);
            let mut peer = gpu(&config, initial.clone(), vec![], 256);
            let mut cpu = Simulation::new(&config, initial, vec![]).unwrap();
            for _ in 0..3 {
                assert_eq!(first.step().unwrap(), cpu.step().unwrap());
                let _ = peer.step().unwrap();
                assert_eq!(bits(first.snapshot()), bits(cpu.snapshot()));
                assert_eq!(bits(peer.snapshot()), bits(cpu.snapshot()));
            }
            drop(first);
            let _ = peer.step().unwrap();
            let _ = cpu.step().unwrap();
            assert_eq!(bits(peer.snapshot()), bits(cpu.snapshot()));
            let before = bits(peer.snapshot());
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                peer.step_with_cancellation(|| panic!("injected cancellation predicate panic"))
            }));
            assert!(panic.is_err());
            assert_eq!(bits(peer.snapshot()), before);
            let _ = peer.step().unwrap();
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn workflow_restores_replay_settings_and_rejects_incompatible_options() {
        use maelstrom::scene::load_resolved_scene;
        let scene = load_resolved_scene(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/small-lattice.toml"),
            None,
        )
        .unwrap();
        let request = RunRequest {
            steps: 12,
            checkpoints: vec![0, 1, 4, 12],
        };
        let options = ExecutionOptions {
            execution: ExecutionMode::Cuda {
                device: 0,
                threads: 64,
            },
            backend: SearchBackend::UniformGrid,
            measurement: Some(MeasurementMode::Stages),
        };
        let result = workflow::run_with_options(&scene, request, options, || false);
        assert!(result.status().is_complete());
        assert!(result.metadata().unwrap().device.is_some());
        let json = result.to_json().unwrap();
        let (restored, request, replay) = workflow::replay_options(&json).unwrap();
        assert_eq!(replay.execution, options.execution);
        assert_eq!(replay.measurement, None);
        assert!(workflow::replay_execution(&json).is_err());
        let repeated = workflow::run_with_options(&restored, request, replay, || false);
        assert_eq!(
            support::serial_document(serde_json::from_str(&json).unwrap()),
            support::serial_document(serde_json::from_str(&repeated.to_json().unwrap()).unwrap())
        );
        let mut altered: serde_json::Value = serde_json::from_str(&json).unwrap();
        altered["execution"]["threads"] = serde_json::json!(33);
        assert!(workflow::replay_options(&altered.to_string()).is_err());
        altered["execution"] = serde_json::json!({"mode":"threaded","workers":2});
        assert!(workflow::replay_options(&altered.to_string()).is_err());
    }

    #[test]
    #[ignore = "requires an explicitly selected CUDA hardware test run"]
    fn workflow_keeps_empty_audits_warnings_partial_failure_and_cancellation() {
        use maelstrom::scene::{RandomnessPolicy, ResolvedScene};
        use workflow::CompletionStatus;
        let scene = |config: ResolvedConfiguration,
                     particles: Vec<Particle>,
                     planes: Vec<Plane>| ResolvedScene {
            schema_version: 1,
            name: "device workflow fixture".into(),
            description: "independent inputs".into(),
            randomness: RandomnessPolicy::None,
            configuration: config,
            particles,
            planes,
        };
        let mut fast = particle(7, Vec3::ZERO);
        fast.velocity_m_s.y = -10.0;
        let cases = [
            (
                scene(configuration(""), vec![], vec![]),
                2,
                vec![0, 1, 2],
                CompletionStatus::Completed,
            ),
            (
                scene(
                    configuration(
                        "[numerical]\nfixed_timestep_s=0.125\nexternal_acceleration_m_s2=[0.0,0.0,0.0]",
                    ),
                    vec![fast],
                    vec![Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0).unwrap()],
                ),
                3,
                vec![0, 1, 3],
                CompletionStatus::CompletedWithWarnings,
            ),
            (
                scene(
                    configuration(
                        "[numerical]\nfixed_timestep_s=1.0\nexternal_acceleration_m_s2=[1e19,0.0,0.0]",
                    ),
                    vec![particle(0, Vec3::ZERO)],
                    vec![],
                ),
                3,
                vec![0, 1, 2, 3],
                CompletionStatus::SimulationFailure,
            ),
            (
                scene(
                    configuration(""),
                    vec![
                        particle(0, Vec3::ZERO),
                        particle(1, Vec3::new(1e20, 0.0, 0.0)),
                    ],
                    vec![],
                ),
                0,
                vec![0],
                CompletionStatus::AuditFailure,
            ),
        ];
        let physical = |result: &workflow::RunResult| {
            let mut value: serde_json::Value =
                serde_json::from_str(&result.to_json().unwrap()).unwrap();
            if let Some(metadata) = value["metadata"].as_object_mut() {
                metadata.remove("device");
            }
            support::serial_document(value)
        };
        for (input, steps, checkpoints, status) in cases {
            let request = RunRequest { steps, checkpoints };
            let reference = workflow::run(&input, request.clone());
            assert_eq!(reference.status(), status);
            for measurement in [
                None,
                Some(MeasurementMode::Total),
                Some(MeasurementMode::Stages),
            ] {
                let actual = workflow::run_with_options(
                    &input,
                    request.clone(),
                    ExecutionOptions {
                        backend: SearchBackend::UniformGrid,
                        execution: ExecutionMode::Cuda {
                            device: 0,
                            threads: 128,
                        },
                        measurement,
                    },
                    || false,
                );
                assert_eq!(actual.status(), status);
                assert_eq!(physical(&actual), physical(&reference));
                if let Some(times) = actual.measurements() {
                    assert_eq!(times.steps.len(), actual.completed_steps() as usize);
                    assert_eq!(
                        times.failed_attempt_s.is_some(),
                        status == CompletionStatus::SimulationFailure
                    );
                }
            }
        }
        let input = scene(configuration(""), vec![particle(0, Vec3::ZERO)], vec![]);
        let mut control = Simulation::from_scene(&input).unwrap();
        let mut polls = 0;
        let _ = control
            .step_with_cancellation(|| {
                polls += 1;
                false
            })
            .unwrap();
        let mut seen = 0;
        let result = workflow::run_with_options(
            &input,
            RunRequest {
                steps: 3,
                checkpoints: vec![0, 1, 3],
            },
            ExecutionOptions {
                backend: SearchBackend::UniformGrid,
                execution: ExecutionMode::Cuda {
                    device: 0,
                    threads: 128,
                },
                measurement: Some(MeasurementMode::Stages),
            },
            || {
                seen += 1;
                seen == polls + 3
            },
        );
        assert_eq!(result.status(), CompletionStatus::Cancelled);
        assert_eq!(result.completed_steps(), 1);
        assert_eq!(result.final_snapshot().unwrap().completed_steps(), 1);
        assert!(result.measurements().unwrap().failed_attempt_s.is_some());
    }
}
