//! Hardware checks inspect private stage results without widening the solver API.

use super::*;
#[test]
fn capacity_arithmetic_rejects_unrepresentable_workspace_without_allocating() {
    assert!(
        storage_sizes(Settings {
            count: u32::MAX,
            words: u32::MAX,
            ..Default::default()
        })
        .is_err()
    );
    let c = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    assert!(settings(&c, usize::MAX, 0).is_err());
    assert!(settings(&c, 1, usize::MAX).is_err());
    assert_eq!(
        storage_sizes(settings(&c, 33, 2).unwrap()).unwrap()[11],
        264
    );
}

fn discovered(
    gpu: &mut CudaSolver,
    positions: &[Vec3],
    radius: f32,
) -> Result<Vec<Vec<usize>>, StepError> {
    let configuration = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let mut s = settings(&configuration, positions.len(), 0)?;
    s.radius = radius;
    let particles: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(i, &p)| Particle {
            identity: i as u64,
            position_m: p,
            velocity_m_s: Vec3::ZERO,
        })
        .collect();
    gpu.prepare(s, &particles, &[])?;
    gpu.context
        .inner
        .upload(2, &positions.iter().copied().map(array).collect::<Vec<_>>())?;
    gpu.discover(s, positions)?;
    let bits: Vec<u32> = gpu
        .context
        .inner
        .download(11, positions.len() * s.words as usize)?;
    Ok((0..positions.len())
        .map(|i| {
            (0..positions.len())
                .filter(|&j| bits[i * s.words as usize + j / 32] & (1 << (j % 32)) != 0)
                .collect()
        })
        .collect())
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_grid_covers_rounded_boundaries_dense_rows_and_extreme_domain() {
    use crate::neighbours::brute_force;
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        let points = vec![
            Vec3::ZERO,
            Vec3::new(-0.0, 0.0, 0.0),
            Vec3::new(f32::from_bits(1.0_f32.to_bits() - 1), 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(f32::from_bits(1.0_f32.to_bits() + 1), 0.0, 0.0),
        ];
        assert_eq!(
            discovered(&mut gpu, &points, 1.0).unwrap(),
            vec![
                vec![1, 2],
                vec![0, 2],
                vec![0, 1, 3, 4],
                vec![2, 4],
                vec![2, 3]
            ]
        );
        for positions in [
            vec![],
            vec![Vec3::new(f32::MAX, -f32::MAX, 0.0)],
            vec![Vec3::ZERO; 257],
            (0..129)
                .map(|i| Vec3::new(i as f32 * 2.0 - 128.0, 0.0, 0.0))
                .collect(),
            (0..67)
                .map(|i| {
                    Vec3::new(
                        (i % 7) as f32 * 0.03 - 0.1,
                        (i % 5) as f32 * 0.04 - 0.1,
                        (i % 3) as f32 * 0.02,
                    )
                })
                .collect(),
        ] {
            let expected = brute_force(&positions, 0.1).unwrap();
            let actual = discovered(&mut gpu, &positions, 0.1).unwrap();
            for (i, row) in actual.iter().enumerate() {
                assert_eq!(row, expected.for_particle(i).unwrap());
            }
        }
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if [x, y, z] == [0, 0, 0] {
                        continue;
                    }
                    let a = [x, y, z].map(|d| match d {
                        -1 => 0.01,
                        1 => 0.115,
                        _ => 0.0625,
                    });
                    let b = std::array::from_fn(|axis| a[axis] + [x, y, z][axis] as f32 * 0.04);
                    for shift in [0.0, -0.25] {
                        let a = vector(a.map(|v| v + shift));
                        let b = vector(b.map(|v| v + shift));
                        assert_eq!(
                            discovered(&mut gpu, &[a, b], 0.1).unwrap(),
                            vec![vec![1], vec![0]]
                        );
                    }
                }
            }
        }
        for exponent in (-149..=120).step_by(7) {
            let radius = 2.0_f32.powi(exponent);
            if radius == 0.0 {
                continue;
            }
            let positions = [
                Vec3::ZERO,
                Vec3::new(-radius, 0.0, 0.0),
                Vec3::new(f32::from_bits(radius.to_bits() - 1), 0.0, 0.0),
                Vec3::new(radius, 0.0, 0.0),
            ];
            match brute_force(&positions, radius) {
                Ok(expected) => {
                    let actual = discovered(&mut gpu, &positions, radius).unwrap();
                    for (i, row) in actual.iter().enumerate() {
                        assert_eq!(
                            row,
                            expected.for_particle(i).unwrap(),
                            "exponent {exponent}"
                        );
                    }
                }
                Err(expected) => assert_eq!(
                    discovered(&mut gpu, &positions, radius)
                        .unwrap_err()
                        .to_string(),
                    StepError::from(expected).to_string()
                ),
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_fixed_membership_and_independent_two_particle_equations() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    let mut s = settings(&config, 2, 0).unwrap();
    s.radius = 1.0;
    s.rest_density = 3.0;
    s.relaxation = 0.01;
    s.strength = 0.1;
    s.reference_weight = crate::kernel::poly6(0.2, 1.0).unwrap();
    s.exponent = 4.0;
    s.correction_scale = 1.0;
    let points = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let particles: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| Particle {
            identity: i as u64,
            position_m: p,
            velocity_m_s: Vec3::ZERO,
        })
        .collect();
    gpu.prepare(s, &particles, &[]).unwrap();
    gpu.context.inner.upload(2, &points.map(array)).unwrap();
    gpu.discover(s, &points).unwrap();
    for op in [5, 6, 7] {
        gpu.operation(s, op).unwrap();
    }
    let poly = |r: f64| 315.0 / (64.0 * std::f64::consts::PI) * (1.0 - r * r).powi(3);
    let density = poly(0.0) + poly(0.5);
    let constraint = density / 3.0 - 1.0;
    let gradient = 45.0 / std::f64::consts::PI * 0.5_f64.powi(2);
    let multiplier = -constraint / (2.0 * (gradient / 3.0).powi(2) + 0.01);
    let pressure = -0.1 * (poly(0.5) / poly(0.2)).powi(4);
    let correction = (2.0 * multiplier + pressure) * gradient / 3.0;
    for (slot, expected, tolerance) in [
        (4, density, 4e-6),
        (5, constraint, 2e-6),
        (6, multiplier, 4e-6),
    ] {
        let values: Vec<f32> = gpu.context.inner.download(slot, 2).unwrap();
        for value in values {
            assert!((f64::from(value) - expected).abs() <= tolerance);
        }
    }
    let values: Vec<[f32; 3]> = gpu.context.inner.download(7, 2).unwrap();
    assert!((f64::from(values[0][0]) - correction).abs() < 2e-5);
    assert!((f64::from(values[1][0]) + correction).abs() < 2e-5);
    for exponent in [0.5, 1.0, 1.25, 4.0, 16.0] {
        s.exponent = exponent;
        gpu.operation(s, 7).unwrap();
        let expected = crate::fluid::compute_jacobi_stage(
            &points,
            &crate::neighbours::brute_force(&points, 1.0).unwrap(),
            crate::fluid::PbfParameters::new(1.0, 3.0, 0.01, 0.1, 0.2, exponent).unwrap(),
        )
        .unwrap();
        let actual: Vec<[f32; 3]> = gpu.context.inner.download(7, 2).unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>(),
            expected
                .corrections
                .into_iter()
                .map(array)
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>()
        );
    }
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
    let _ = discovered(&mut gpu, &original, 0.1).unwrap();
    let s = settings(&config, 3, 0).unwrap();
    gpu.context.inner.upload(2, &current.map(array)).unwrap();
    for op in [5, 6, 7] {
        gpu.operation(s, op).unwrap();
    }
    let fixed = crate::neighbours::brute_force(&original, 0.1).unwrap();
    let expected =
        crate::fluid::compute_jacobi_stage(&current, &fixed, config.pbf_parameters().unwrap())
            .unwrap();
    let wrong = crate::fluid::compute_jacobi_stage(
        &current,
        &crate::neighbours::brute_force(&current, 0.1).unwrap(),
        config.pbf_parameters().unwrap(),
    )
    .unwrap();
    assert_ne!(expected, wrong);
    let actual: Vec<[f32; 3]> = gpu.context.inner.download(7, 3).unwrap();
    assert_eq!(
        actual,
        expected
            .corrections
            .into_iter()
            .map(array)
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_components_preserve_ordered_cpu_arithmetic() {
    use crate::{
        config::parse_configuration_overlay, fluid::compute_jacobi_stage, neighbours::brute_force,
        scene::load_resolved_scene,
    };
    let overlay = parse_configuration_overlay("[numerical]\nrefine_timestep=4").unwrap();
    let scene = load_resolved_scene(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes/falling-block-04.toml"),
        Some(&overlay),
    )
    .unwrap();
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    let s = settings(
        &scene.configuration,
        scene.particles.len(),
        scene.planes.len(),
    )
    .unwrap();
    let mut serial = crate::simulation::Simulation::from_scene(&scene).unwrap();
    let mut order_sensitive = false;
    for step in 0..480 {
        gpu.prepare(s, serial.snapshot().particles(), &scene.planes)
            .unwrap();
        gpu.operation(s, 0).unwrap();
        gpu.operation(s, 1).unwrap();
        let mut positions: Vec<Vec3> = gpu
            .context
            .inner
            .download(2, s.count as usize)
            .unwrap()
            .into_iter()
            .map(vector)
            .collect();
        gpu.discover(s, &positions).unwrap();
        let rows = brute_force(&positions, s.radius).unwrap();
        if step == 0 {
            for i in 0..positions.len() {
                let weights: Vec<f32> = (0..positions.len())
                    .filter(|&j| i == j || rows.for_particle(i).unwrap().contains(&j))
                    .map(|j| {
                        crate::kernel::poly6(
                            (positions[i] - positions[j])
                                .checked_length("independent order control")
                                .unwrap(),
                            s.radius,
                        )
                        .unwrap()
                    })
                    .collect();
                let forward = weights.iter().fold(0.0_f32, |sum, w| sum + w);
                let reverse = weights.iter().rev().fold(0.0_f32, |sum, w| sum + w);
                order_sensitive |= forward.to_bits() != reverse.to_bits();
            }
        }
        for iteration in 0..8 {
            let cpu = compute_jacobi_stage(
                &positions,
                &rows,
                scene.configuration.pbf_parameters().unwrap(),
            )
            .unwrap();
            for op in [5, 6, 7, 8] {
                gpu.operation(s, op).unwrap();
            }
            for (slot, expected) in [
                (4, &cpu.normalised_densities),
                (5, &cpu.constraints),
                (6, &cpu.multipliers),
            ] {
                let values: Vec<f32> = gpu.context.inner.download(slot, positions.len()).unwrap();
                for (i, (a, b)) in values.iter().zip(expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "iteration {iteration} slot {slot} particle {i}: {a} versus {b}"
                    );
                }
            }
            let corrections: Vec<[f32; 3]> =
                gpu.context.inner.download(7, positions.len()).unwrap();
            for (i, (a, b)) in corrections.iter().zip(&cpu.corrections).enumerate() {
                assert_eq!(
                    a.map(f32::to_bits),
                    array(*b).map(f32::to_bits),
                    "step {step} iteration {iteration} correction {i}: {a:?} versus {b:?}"
                );
            }
            let expected: Vec<_> = positions
                .iter()
                .zip(&cpu.corrections)
                .map(|(p, d)| crate::boundary::project_in_order(*p + *d, &scene.planes).unwrap())
                .collect();
            gpu.operation(s, 9).unwrap();
            gpu.operation(s, 10).unwrap();
            positions = gpu
                .context
                .inner
                .download(2, positions.len())
                .unwrap()
                .into_iter()
                .map(vector)
                .collect();
            assert_eq!(
                positions
                    .iter()
                    .copied()
                    .map(array)
                    .map(|p| p.map(f32::to_bits))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .copied()
                    .map(array)
                    .map(|p| p.map(f32::to_bits))
                    .collect::<Vec<_>>()
            );
        }
        let _ = serial.step().unwrap();
    }
    assert!(
        order_sensitive,
        "the fixture must detect a reversed floating-point reduction"
    );
}
