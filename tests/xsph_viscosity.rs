//! Analytical kernel values and an independent f64 sum check velocity smoothing.
//! Complete solver comparisons check staging, rollback and disabled compatibility.

use maelstrom::{
    config::{ConfigurationOverlay, NumericalOverrides, resolve_effective_configuration},
    execution::ExecutionMode,
    fluid::InteractionSets,
    math::Vec3,
    neighbours::SearchBackend,
    scene::load_resolved_scene,
    simulation::{Simulation, Snapshot, StepError},
    velocity::xsph_viscosity,
};
use std::path::Path;

fn reference_weight(distance: f64, radius: f64, density: f64) -> f64 {
    if distance >= radius {
        0.
    } else {
        315.0 / (64.0 * std::f64::consts::PI * radius.powi(9))
            * (radius * radius - distance * distance).powi(3)
            / density
    }
}
fn assert_close(actual: f32, expected: f64) {
    assert!(
        (f64::from(actual) - expected).abs() <= 2e-6 * expected.abs().max(1.),
        "{actual} != {expected}"
    );
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

fn interaction_scene() -> maelstrom::scene::ResolvedScene {
    let mut scene = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    scene.configuration.numerical.solver_iterations = 2;
    for (index, p) in scene.particles.iter_mut().enumerate() {
        p.velocity_m_s = Vec3::new(if index % 2 == 0 { 0.125 } else { -0.125 }, 0.0625, 0.0);
    }
    scene
}

#[test]
fn analytical_pair_damps_relative_motion_and_preserves_equal_mass_momentum() {
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0., 0.)];
    let velocities = [Vec3::new(2., -1., 3.), Vec3::new(-2., 1., -3.)];
    let rows = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let result = xsph_viscosity(&positions, &velocities, &rows, 1., 4., 0.25).unwrap();
    let w = reference_weight(0.5, 1., 4.);
    for axis in 0..3 {
        let initial = [2., -1., 3.][axis];
        assert_close(
            [result[0].x, result[0].y, result[0].z][axis],
            initial * (1. - 0.5 * w),
        );
    }
    assert_eq!(result[0] + result[1], Vec3::ZERO);
    assert!(result[0].length_squared() < velocities[0].length_squared());
    assert_eq!(velocities[0], Vec3::new(2., -1., 3.));
}

#[test]
fn asymmetric_rows_read_one_immutable_velocity_snapshot() {
    let positions = [Vec3::ZERO, Vec3::new(0.25, 0., 0.), Vec3::new(0.5, 0., 0.)];
    let velocities = [
        Vec3::new(2., 0., 0.),
        Vec3::new(-1., 0., 0.),
        Vec3::new(4., 0., 0.),
    ];
    let rows = InteractionSets::new(vec![vec![1, 2], vec![0, 2], vec![0, 1]]).unwrap();
    let actual = xsph_viscosity(&positions, &velocities, &rows, 1., 4., 0.5).unwrap();
    for i in 0..3 {
        let expected = f64::from(velocities[i].x)
            + 0.5
                * (0..3)
                    .filter(|&j| j != i)
                    .map(|j| {
                        f64::from(velocities[j].x - velocities[i].x)
                            * reference_weight((j as f64 - i as f64).abs() * 0.25, 1., 4.)
                    })
                    .sum::<f64>();
        assert_close(actual[i].x, expected);
    }
}

#[test]
fn strict_support_coincidence_fixed_rows_empty_and_zero_behaviour() {
    let rows = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let velocities = [Vec3::ZERO, Vec3::new(2., 0., 0.)];
    for distance in [
        0.,
        f32::from_bits(1.0f32.to_bits() - 1),
        1.,
        f32::from_bits(1.0f32.to_bits() + 1),
    ] {
        let result = xsph_viscosity(
            &[Vec3::ZERO, Vec3::new(distance, 0., 0.)],
            &velocities,
            &rows,
            1.,
            4.,
            1.,
        )
        .unwrap();
        assert_close(
            result[0].x,
            2. * reference_weight(f64::from(distance), 1., 4.),
        );
        if distance >= 1. {
            assert_eq!(result, velocities);
        }
    }
    let no_rows = InteractionSets::new(vec![vec![], vec![]]).unwrap();
    assert_eq!(
        xsph_viscosity(&[Vec3::ZERO; 2], &velocities, &no_rows, 1., 4., 1.).unwrap(),
        velocities
    );
    let uniform = [Vec3::new(2., 3., 4.); 2];
    assert_eq!(
        xsph_viscosity(&[Vec3::ZERO; 2], &uniform, &rows, 1., 4., 1.).unwrap(),
        uniform
    );
    assert!(
        xsph_viscosity(&[], &[], &InteractionSets::new(vec![]).unwrap(), 1., 4., 1.)
            .unwrap()
            .is_empty()
    );
    let v = [Vec3::new(-0., f32::from_bits(1), -3.)];
    let result = xsph_viscosity(
        &[Vec3::ZERO],
        &v,
        &InteractionSets::new(vec![vec![]]).unwrap(),
        1.,
        4.,
        0.,
    )
    .unwrap();
    assert_eq!(result[0].x.to_bits(), v[0].x.to_bits());
    assert_eq!(result[0].y.to_bits(), 1);
}

#[test]
fn invalid_inputs_and_intermediate_overflow_are_errors_without_mutation() {
    let rows = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    for coefficient in [-1., 1.01, f32::NAN, f32::INFINITY] {
        assert!(
            xsph_viscosity(
                &[Vec3::ZERO; 2],
                &[Vec3::ZERO; 2],
                &rows,
                1.,
                4.,
                coefficient
            )
            .is_err()
        );
    }
    for scalar in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(xsph_viscosity(&[Vec3::ZERO; 2], &[Vec3::ZERO; 2], &rows, scalar, 4., 1.).is_err());
        assert!(xsph_viscosity(&[Vec3::ZERO; 2], &[Vec3::ZERO; 2], &rows, 1., scalar, 1.).is_err());
    }
    assert!(xsph_viscosity(&[Vec3::ZERO], &[], &rows, 1., 4., 1.).is_err());
    assert!(xsph_viscosity(&[], &[], &rows, 1., 4., 1.).is_err());
    let velocities = [Vec3::new(f32::MAX, 0., 0.), Vec3::new(-f32::MAX, 0., 0.)];
    assert!(xsph_viscosity(&[Vec3::ZERO; 2], &velocities, &rows, 1., 4., 1.).is_err());
    assert!(
        xsph_viscosity(
            &[Vec3::ZERO; 2],
            &[Vec3::ZERO; 2],
            &rows,
            1.,
            f32::from_bits(1),
            1.
        )
        .is_err()
    );
    assert_eq!(velocities[0].x, f32::MAX);
    assert!(
        xsph_viscosity(
            &[Vec3::new(f32::NAN, 0., 0.); 2],
            &[Vec3::ZERO; 2],
            &rows,
            1.,
            4.,
            0.
        )
        .is_err()
    );
}

#[test]
fn coefficient_validation_provenance_serialisation_and_identity_are_explicit() {
    let base = resolve_effective_configuration(None, None, None).unwrap();
    assert_eq!(base.numerical.xsph_viscosity_coefficient, 0.01);
    assert!(
        !serde_json::to_string(&base.numerical)
            .unwrap()
            .contains("xsph_viscosity_coefficient")
    );
    for c in [0., 0.01, 1.] {
        let selected = ConfigurationOverlay {
            numerical: NumericalOverrides {
                xsph_viscosity_enabled: Some(true),
                xsph_viscosity_coefficient: Some(c),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolved = resolve_effective_configuration(None, None, Some(&selected)).unwrap();
        assert!(
            resolved
                .comparison_key()
                .contains("xsph-rest-volume-poly6-v1")
        );
        assert_ne!(resolved.comparison_key(), base.comparison_key());
        assert!(
            resolved
                .canonical_toml()
                .contains("xsph_viscosity_coefficient")
        );
    }
    for c in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        let selected = ConfigurationOverlay {
            numerical: NumericalOverrides {
                xsph_viscosity_coefficient: Some(c),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(resolve_effective_configuration(None, None, Some(&selected)).is_err());
    }
}

#[test]
fn cpu_backends_match_and_zero_or_disabled_smoothing_preserves_every_bit() {
    let mut scene = interaction_scene();
    let mut original = Simulation::from_scene(&scene).unwrap();
    scene.configuration.numerical.xsph_viscosity_coefficient = 0.;
    scene.configuration.numerical.xsph_viscosity_enabled = true;
    let mut zero = Simulation::from_scene(&scene).unwrap();
    for _ in 0..8 {
        let _ = original.step().unwrap();
        let _ = zero.step().unwrap();
        assert_eq!(bits(original.snapshot()), bits(zero.snapshot()));
    }
    scene.configuration.numerical.xsph_viscosity_coefficient = 0.1;
    let mut brute = Simulation::from_scene(&scene).unwrap();
    let mut grid = Simulation::from_scene_with_backend(&scene, SearchBackend::UniformGrid).unwrap();
    let mut threaded = Simulation::from_scene_with_execution(
        &scene,
        SearchBackend::UniformGrid,
        ExecutionMode::Threaded { workers: 3 },
    )
    .unwrap();
    for _ in 0..16 {
        let _ = brute.step().unwrap();
        let _ = grid.step().unwrap();
        let _ = threaded.step().unwrap();
        assert_eq!(bits(brute.snapshot()), bits(grid.snapshot()));
        assert_eq!(bits(brute.snapshot()), bits(threaded.snapshot()));
    }
}

#[test]
fn enabled_smoothing_cancellation_is_atomic_at_every_boundary_and_retry_is_identical() {
    let mut scene = interaction_scene();
    scene.configuration.numerical.xsph_viscosity_enabled = true;
    let mut control = Simulation::from_scene(&scene).unwrap();
    let mut polls = 0;
    let _ = control
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    for stop in 1..=polls {
        let mut solver = Simulation::from_scene(&scene).unwrap();
        let before = bits(solver.snapshot());
        let mut count = 0;
        assert!(matches!(
            solver.step_with_cancellation(|| {
                count += 1;
                count >= stop
            }),
            Err(StepError::Cancelled)
        ));
        assert_eq!(bits(solver.snapshot()), before);
        assert_eq!(solver.snapshot().completed_steps(), 0);
        let _ = solver.step().unwrap();
        assert_eq!(bits(solver.snapshot()), bits(control.snapshot()));
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn cuda_smoothing_matches_cpu_at_all_block_sizes_and_retries_cancelled_stages() {
    for threads in [32, 64, 128, 256] {
        for count in [0, 1, 2, 33, 257] {
            let mut scene = interaction_scene();
            scene.planes.clear();
            scene.configuration.numerical.xsph_viscosity_enabled = true;
            scene.configuration.numerical.xsph_viscosity_coefficient = 0.125;
            scene.particles = (0..count)
                .map(|i| maelstrom::scene::Particle {
                    identity: i * 3,
                    position_m: Vec3::new(
                        (i % 7) as f32 * 0.0625,
                        ((i / 7) % 7) as f32 * 0.0625,
                        (i / 49) as f32 * 0.0625,
                    ),
                    velocity_m_s: Vec3::new(if i % 2 == 0 { 0.125 } else { -0.125 }, 0.0625, 0.),
                })
                .collect();
            let mut cpu =
                Simulation::from_scene_with_backend(&scene, SearchBackend::UniformGrid).unwrap();
            let mut gpu = Simulation::from_scene_with_execution(
                &scene,
                SearchBackend::UniformGrid,
                ExecutionMode::Cuda { device: 0, threads },
            )
            .unwrap();
            for _ in 0..3 {
                let _ = cpu.step().unwrap();
                let _ = gpu.step().unwrap();
                assert_eq!(
                    bits(cpu.snapshot()),
                    bits(gpu.snapshot()),
                    "count {count}, block {threads}"
                );
            }
        }
        let mut scene = interaction_scene();
        scene.configuration.numerical.xsph_viscosity_enabled = true;
        let mut control = Simulation::from_scene_with_execution(
            &scene,
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda { device: 0, threads },
        )
        .unwrap();
        let mut polls = 0;
        let _ = control
            .step_with_cancellation(|| {
                polls += 1;
                false
            })
            .unwrap();
        for stop in 1..=polls {
            let mut gpu = Simulation::from_scene_with_execution(
                &scene,
                SearchBackend::UniformGrid,
                ExecutionMode::Cuda { device: 0, threads },
            )
            .unwrap();
            let before = bits(gpu.snapshot());
            let mut count = 0;
            assert!(matches!(
                gpu.step_with_cancellation(|| {
                    count += 1;
                    count >= stop
                }),
                Err(StepError::Cancelled)
            ));
            assert_eq!(bits(gpu.snapshot()), before);
            let _ = gpu.step().unwrap();
            assert_eq!(bits(gpu.snapshot()), bits(control.snapshot()));
        }
    }
}
