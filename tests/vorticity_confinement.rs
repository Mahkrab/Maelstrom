//! A double-precision oracle checks curl sign and confinement independently of
//! production kernels. Solver checks exercise complete stages and atomic retry.

use maelstrom::{
    config::{ConfigurationOverlay, NumericalOverrides, resolve_effective_configuration},
    execution::ExecutionMode,
    fluid::InteractionSets,
    math::Vec3,
    neighbours::SearchBackend,
    scene::load_resolved_scene,
    simulation::{Simulation, Snapshot, StepError},
    velocity::{VorticityParameters, vorticity_confinement},
};
use std::path::Path;

fn params() -> VorticityParameters {
    VorticityParameters {
        radius_m: 1.,
        rest_density_per_m3: 4.,
        strength_m_s: 0.125,
        timestep_s: 0.0625,
    }
}
fn fixture() -> (Vec<Vec3>, Vec<Vec3>, InteractionSets) {
    let positions = vec![
        Vec3::ZERO,
        Vec3::new(0.25, 0., 0.),
        Vec3::new(0.5, 0.125, 0.),
        Vec3::new(0., 0.25, 0.125),
    ];
    let velocities = vec![
        Vec3::new(0., 0.125, 0.),
        Vec3::new(0., 0.25, 0.125),
        Vec3::new(-0.25, 0.5, 0.),
        Vec3::new(0.125, 0., -0.125),
    ];
    let rows = InteractionSets::new(
        (0..4)
            .map(|i| (0..4).filter(|&j| j != i).collect())
            .collect(),
    )
    .unwrap();
    (positions, velocities, rows)
}
fn array(v: Vec3) -> [f64; 3] {
    [v.x as f64, v.y as f64, v.z as f64]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn oracle(
    positions: &[Vec3],
    velocities: &[Vec3],
    rows: &InteractionSets,
    p: VorticityParameters,
) -> Vec<[f64; 3]> {
    let gradients: Vec<Vec<[f64; 3]>> = positions
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            positions
                .iter()
                .enumerate()
                .map(|(j, &b)| {
                    let a = array(a);
                    let b = array(b);
                    let d = std::array::from_fn::<_, 3, _>(|k| a[k] - b[k]);
                    let r = norm(d);
                    if i == j
                        || !rows.for_particle(i).unwrap().contains(&j)
                        || r == 0.
                        || r >= p.radius_m as f64
                    {
                        return [0.; 3];
                    }
                    let scale = -45. / (std::f64::consts::PI * (p.radius_m as f64).powi(6))
                        * (p.radius_m as f64 - r).powi(2)
                        / r
                        / p.rest_density_per_m3 as f64;
                    d.map(|x| x * scale)
                })
                .collect()
        })
        .collect();
    let curls: Vec<[f64; 3]> = gradients
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut sum = [0.; 3];
            for (j, &g) in row.iter().enumerate() {
                let a = array(velocities[i]);
                let b = array(velocities[j]);
                let c = cross(g, std::array::from_fn(|k| b[k] - a[k]));
                for k in 0..3 {
                    sum[k] += c[k];
                }
            }
            sum
        })
        .collect();
    gradients
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut eta = [0.; 3];
            for (j, g) in row.iter().enumerate() {
                for k in 0..3 {
                    eta[k] += (norm(curls[j]) - norm(curls[i])) * g[k];
                }
            }
            let n = norm(eta);
            let u = array(velocities[i]);
            if n == 0. {
                return u;
            }
            let force = cross(eta.map(|x| x / n), curls[i]);
            std::array::from_fn(|k| u[k] + p.timestep_s as f64 * p.strength_m_s as f64 * force[k])
        })
        .collect()
}
fn vector_bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}
fn bits(s: &Snapshot) -> Vec<(u64, [u32; 3], [u32; 3])> {
    s.particles()
        .iter()
        .map(|p| {
            (
                p.identity,
                vector_bits(p.position_m),
                vector_bits(p.velocity_m_s),
            )
        })
        .collect()
}
fn scene() -> maelstrom::scene::ResolvedScene {
    let mut scene = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    scene.configuration.numerical.solver_iterations = 2;
    for (i, p) in scene.particles.iter_mut().enumerate() {
        p.velocity_m_s = Vec3::new(
            if i % 2 == 0 { 0.125 } else { -0.125 },
            (i % 3) as f32 * 0.125,
            (i % 5) as f32 * 0.0625,
        );
    }
    scene
}

#[test]
fn independent_curl_sign_gradient_normalisation_and_timestep_match() {
    let (positions, velocities, rows) = fixture();
    let original = velocities.clone();
    for strength in [0.0625, 0.125, 0.25] {
        for dt in [0.03125, 0.0625] {
            let p = VorticityParameters {
                strength_m_s: strength,
                timestep_s: dt,
                ..params()
            };
            let actual = vorticity_confinement(&positions, &velocities, &rows, p).unwrap();
            let expected = oracle(&positions, &velocities, &rows, p);
            for (a, e) in actual.iter().zip(expected) {
                for (a, e) in array(*a).into_iter().zip(e) {
                    assert!((a - e).abs() < 3e-6 * e.abs().max(1.), "{a} != {e}");
                }
            }
            assert_ne!(actual, velocities);
        }
    }
    assert_eq!(original, velocities);
    // A quadratic y velocity along x gives positive z curl, increasing towards x.
    // N points towards +x, so confinement turns towards -y at the left endpoint.
    let p = [Vec3::ZERO, Vec3::new(0.25, 0., 0.), Vec3::new(0.5, 0., 0.)];
    let v = [
        Vec3::ZERO,
        Vec3::new(0., 0.0625, 0.),
        Vec3::new(0., 0.25, 0.),
    ];
    let rows = InteractionSets::new(vec![vec![1, 2], vec![0, 2], vec![0, 1]]).unwrap();
    let result = vorticity_confinement(&p, &v, &rows, params()).unwrap();
    assert!(result[0].y < 0.);
}

#[test]
fn uniform_motion_zero_curl_constant_curl_and_zero_strength_do_not_change_bits() {
    let (p, _, rows) = fixture();
    let uniform = vec![Vec3::new(0.25, -0.125, 0.5); 4];
    assert_eq!(
        vorticity_confinement(&p, &uniform, &rows, params()).unwrap(),
        uniform
    );
    let velocities = vec![Vec3::new(-0., 0., -0.); 4];
    let result = vorticity_confinement(
        &p,
        &velocities,
        &rows,
        VorticityParameters {
            strength_m_s: 0.,
            ..params()
        },
    )
    .unwrap();
    assert_eq!(
        result.iter().copied().map(vector_bits).collect::<Vec<_>>(),
        velocities
            .iter()
            .copied()
            .map(vector_bits)
            .collect::<Vec<_>>()
    );
    let rows = InteractionSets::new(vec![vec![1], vec![0]]).unwrap();
    let p = [Vec3::ZERO, Vec3::new(0.25, 0., 0.)];
    let v = [Vec3::ZERO, Vec3::new(0., 1., 0.)];
    assert_eq!(vorticity_confinement(&p, &v, &rows, params()).unwrap(), v);
    for n in [0, 1] {
        let rows = InteractionSets::new(vec![vec![]; n]).unwrap();
        let v = vec![Vec3::new(-0., 0.25, 0.); n];
        assert_eq!(
            vorticity_confinement(&vec![Vec3::ZERO; n], &v, &rows, params()).unwrap(),
            v
        );
    }
}

#[test]
fn strict_support_coincidence_and_fixed_row_omissions_are_respected() {
    let (mut p, v, rows) = fixture();
    for distance in [
        0.,
        f32::from_bits(1f32.to_bits() - 1),
        1.,
        f32::from_bits(1f32.to_bits() + 1),
    ] {
        p[1] = Vec3::new(distance, 0., 0.);
        let a = vorticity_confinement(&p, &v, &rows, params()).unwrap();
        let e = oracle(&p, &v, &rows, params());
        for (a, e) in a.iter().zip(e) {
            for (a, e) in array(*a).into_iter().zip(e) {
                assert!((a - e).abs() < 3e-6 * e.abs().max(1.));
            }
        }
    }
    let empty = InteractionSets::new(vec![vec![]; 4]).unwrap();
    assert_eq!(vorticity_confinement(&p, &v, &empty, params()).unwrap(), v);
}

#[test]
fn invalid_parameters_lengths_non_finite_state_and_overflow_fail() {
    let (p, v, rows) = fixture();
    for bad in [-1., f32::NAN, f32::INFINITY] {
        assert!(
            vorticity_confinement(
                &p,
                &v,
                &rows,
                VorticityParameters {
                    strength_m_s: bad,
                    ..params()
                }
            )
            .is_err()
        );
    }
    for bad in [0., -1., f32::NAN, f32::INFINITY] {
        for invalid in [
            VorticityParameters {
                radius_m: bad,
                ..params()
            },
            VorticityParameters {
                rest_density_per_m3: bad,
                ..params()
            },
            VorticityParameters {
                timestep_s: bad,
                ..params()
            },
        ] {
            assert!(vorticity_confinement(&p, &v, &rows, invalid).is_err());
        }
    }
    assert!(vorticity_confinement(&p, &v[..3], &rows, params()).is_err());
    assert!(
        vorticity_confinement(&p, &v, &InteractionSets::new(vec![]).unwrap(), params()).is_err()
    );
    let mut bad = v.clone();
    bad[0].x = f32::INFINITY;
    assert!(
        vorticity_confinement(
            &p,
            &bad,
            &rows,
            VorticityParameters {
                strength_m_s: 0.,
                ..params()
            }
        )
        .is_err()
    );
    bad[0].x = f32::MAX;
    bad[1].x = -f32::MAX;
    assert!(vorticity_confinement(&p, &bad, &rows, params()).is_err());
    assert!(
        vorticity_confinement(
            &p,
            &v,
            &rows,
            VorticityParameters {
                strength_m_s: f32::MAX,
                timestep_s: f32::MAX,
                ..params()
            }
        )
        .is_err()
    );
}

#[test]
fn controls_provenance_records_and_comparison_identity_are_explicit() {
    let base = resolve_effective_configuration(None, None, None).unwrap();
    assert_eq!(base.numerical.vorticity_confinement_strength_m_s, 0.01);
    assert!(
        !serde_json::to_string(&base.numerical)
            .unwrap()
            .contains("vorticity_confinement_strength_m_s")
    );
    for strength in [0., 0.01, 0.5, f32::MAX] {
        let overlay = ConfigurationOverlay {
            numerical: NumericalOverrides {
                vorticity_confinement_enabled: Some(true),
                vorticity_confinement_strength_m_s: Some(strength),
                ..Default::default()
            },
            ..Default::default()
        };
        let config = resolve_effective_configuration(None, None, Some(&overlay)).unwrap();
        assert!(
            config
                .comparison_key()
                .contains("vorticity-rest-volume-spiky-v1")
        );
        assert_ne!(config.comparison_key(), base.comparison_key());
        assert!(
            serde_json::to_string(&config.numerical)
                .unwrap()
                .contains("vorticity_confinement_strength_m_s")
        );
        assert!(
            config
                .canonical_toml()
                .contains("vorticity_confinement_strength_m_s")
        );
    }
    for strength in [-1., f32::NAN, f32::INFINITY] {
        let overlay = ConfigurationOverlay {
            numerical: NumericalOverrides {
                vorticity_confinement_strength_m_s: Some(strength),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(resolve_effective_configuration(None, None, Some(&overlay)).is_err());
    }
}

#[test]
fn cpu_backends_agree_for_each_effect_and_zero_strength_preserves_legacy_bits() {
    let mut s = scene();
    let mut disabled = Simulation::from_scene(&s).unwrap();
    s.configuration.numerical.vorticity_confinement_enabled = true;
    s.configuration.numerical.vorticity_confinement_strength_m_s = 0.;
    let mut zero = Simulation::from_scene(&s).unwrap();
    for _ in 0..8 {
        let _ = disabled.step().unwrap();
        let _ = zero.step().unwrap();
        assert_eq!(bits(disabled.snapshot()), bits(zero.snapshot()));
    }
    s.configuration.numerical.vorticity_confinement_strength_m_s = 0.125;
    for xsph in [false, true] {
        s.configuration.numerical.xsph_viscosity_enabled = xsph;
        let mut serial = Simulation::from_scene(&s).unwrap();
        let mut grid = Simulation::from_scene_with_backend(&s, SearchBackend::UniformGrid).unwrap();
        let mut threaded = Simulation::from_scene_with_execution(
            &s,
            SearchBackend::UniformGrid,
            ExecutionMode::Threaded { workers: 3 },
        )
        .unwrap();
        for _ in 0..8 {
            let _ = serial.step().unwrap();
            let _ = grid.step().unwrap();
            let _ = threaded.step().unwrap();
            assert_eq!(bits(serial.snapshot()), bits(grid.snapshot()));
            assert_eq!(bits(serial.snapshot()), bits(threaded.snapshot()));
        }
    }
}

#[test]
fn cancellation_at_every_boundary_is_atomic_and_retry_is_identical() {
    for xsph in [false, true] {
        let mut s = scene();
        s.configuration.numerical.vorticity_confinement_enabled = true;
        s.configuration.numerical.xsph_viscosity_enabled = xsph;
        let mut control = Simulation::from_scene(&s).unwrap();
        let mut polls = 0;
        let _ = control
            .step_with_cancellation(|| {
                polls += 1;
                false
            })
            .unwrap();
        for stop in 1..=polls {
            let mut solver = Simulation::from_scene(&s).unwrap();
            let before = bits(solver.snapshot());
            let mut calls = 0;
            assert!(matches!(
                solver.step_with_cancellation(|| {
                    calls += 1;
                    calls >= stop
                }),
                Err(StepError::Cancelled)
            ));
            assert_eq!(before, bits(solver.snapshot()));
            let _ = solver.step().unwrap();
            assert_eq!(bits(control.snapshot()), bits(solver.snapshot()));
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn cuda_confinement_and_combined_effects_match_cpu_and_retry_cancellation() {
    for threads in [32, 64, 128, 256] {
        for xsph in [false, true] {
            for count in [0, 1, 2, 33, 257] {
                let mut scene = scene();
                scene.planes.clear();
                scene.configuration.numerical.vorticity_confinement_enabled = true;
                scene.configuration.numerical.xsph_viscosity_enabled = xsph;
                scene
                    .configuration
                    .numerical
                    .vorticity_confinement_strength_m_s = 0.125;
                scene.particles = (0..count)
                    .map(|i| maelstrom::scene::Particle {
                        identity: i * 3,
                        position_m: Vec3::new(
                            (i % 7) as f32 * 0.0625,
                            ((i / 7) % 7) as f32 * 0.0625,
                            (i / 49) as f32 * 0.0625,
                        ),
                        velocity_m_s: Vec3::new(
                            if i % 2 == 0 { 0.125 } else { -0.125 },
                            0.0625,
                            0.,
                        ),
                    })
                    .collect();
                let mut cpu =
                    Simulation::from_scene_with_backend(&scene, SearchBackend::UniformGrid)
                        .unwrap();
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
        }
        let mut scene = scene();
        scene.configuration.numerical.vorticity_confinement_enabled = true;
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

#[test]
fn confinement_precedes_smoothing_and_never_changes_the_same_steps_positions() {
    let s = scene();
    let mut baseline = Simulation::from_scene(&s).unwrap();
    let _ = baseline.step().unwrap();
    let positions: Vec<_> = baseline
        .snapshot()
        .particles()
        .iter()
        .map(|p| p.position_m)
        .collect();
    let reconstructed: Vec<_> = baseline
        .snapshot()
        .particles()
        .iter()
        .map(|p| p.velocity_m_s)
        .collect();
    let n = &s.configuration.numerical;
    let d = s.configuration.derived;
    let predicted: Vec<_> = s
        .particles
        .iter()
        .map(|p| {
            p.position_m
                + (p.velocity_m_s + n.external_acceleration_m_s2 * n.fixed_timestep_s)
                    * n.fixed_timestep_s
        })
        .collect();
    let rows = SearchBackend::BruteForce
        .discover(&predicted, d.support_radius_m)
        .unwrap();
    let controls = VorticityParameters {
        radius_m: d.support_radius_m,
        rest_density_per_m3: d.normalised_rest_density_per_m3,
        strength_m_s: 0.25,
        timestep_s: n.fixed_timestep_s,
    };
    let confined = oracle(&positions, &reconstructed, &rows, controls);
    let expected: Vec<[f64; 3]> = confined
        .iter()
        .enumerate()
        .map(|(i, &u)| {
            let mut sum = [0.; 3];
            for &j in rows.for_particle(i).unwrap() {
                let a = array(positions[i]);
                let b = array(positions[j]);
                let radius = norm(std::array::from_fn(|k| a[k] - b[k]));
                let h = d.support_radius_m as f64;
                if radius >= h {
                    continue;
                }
                let weight = 315. / (64. * std::f64::consts::PI * h.powi(9))
                    * (h * h - radius * radius).powi(3)
                    / d.normalised_rest_density_per_m3 as f64;
                for k in 0..3 {
                    sum[k] += (confined[j][k] - u[k]) * weight;
                }
            }
            std::array::from_fn(|k| u[k] + 0.125 * sum[k])
        })
        .collect();
    let mut selected = s.clone();
    selected
        .configuration
        .numerical
        .vorticity_confinement_enabled = true;
    selected
        .configuration
        .numerical
        .vorticity_confinement_strength_m_s = 0.25;
    selected.configuration.numerical.xsph_viscosity_enabled = true;
    selected.configuration.numerical.xsph_viscosity_coefficient = 0.125;
    let mut solver = Simulation::from_scene(&selected).unwrap();
    let _ = solver.step().unwrap();
    for ((p, position), expected) in solver
        .snapshot()
        .particles()
        .iter()
        .zip(positions)
        .zip(expected)
    {
        assert_eq!(vector_bits(p.position_m), vector_bits(position));
        for (a, e) in array(p.velocity_m_s).into_iter().zip(expected) {
            assert!((a - e).abs() < 3e-5 * e.abs().max(1.), "{a} != {e}");
        }
    }
    assert_ne!(bits(solver.snapshot()), bits(baseline.snapshot()));
}

#[test]
fn numerical_failure_does_not_accept_partial_confinement() {
    let mut s = scene();
    s.configuration.numerical.vorticity_confinement_enabled = true;
    s.configuration.numerical.vorticity_confinement_strength_m_s = f32::MAX;
    let mut solver = Simulation::from_scene(&s).unwrap();
    let before = bits(solver.snapshot());
    for _ in 0..2 {
        assert!(matches!(solver.step(), Err(StepError::Numerical(_))));
        assert_eq!(bits(solver.snapshot()), before);
        assert_eq!(solver.snapshot().completed_steps(), 0);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn cuda_zero_strength_is_bitwise_legacy_and_failed_confinement_never_accepts() {
    for threads in [32, 64, 128, 256] {
        let s = scene();
        let make = |s: &maelstrom::scene::ResolvedScene| {
            Simulation::from_scene_with_execution(
                s,
                SearchBackend::UniformGrid,
                ExecutionMode::Cuda { device: 0, threads },
            )
            .unwrap()
        };
        let mut legacy = make(&s);
        let mut zero_scene = s.clone();
        zero_scene
            .configuration
            .numerical
            .vorticity_confinement_enabled = true;
        zero_scene
            .configuration
            .numerical
            .vorticity_confinement_strength_m_s = 0.;
        let mut zero = make(&zero_scene);
        for _ in 0..3 {
            let _ = legacy.step().unwrap();
            let _ = zero.step().unwrap();
            assert_eq!(bits(legacy.snapshot()), bits(zero.snapshot()));
        }
        zero_scene
            .configuration
            .numerical
            .vorticity_confinement_strength_m_s = f32::MAX;
        let mut failing = make(&zero_scene);
        let before = bits(failing.snapshot());
        for _ in 0..2 {
            assert!(matches!(failing.step(), Err(StepError::Numerical(_))));
            assert_eq!(bits(failing.snapshot()), before);
            assert_eq!(failing.snapshot().completed_steps(), 0);
        }
    }
}
