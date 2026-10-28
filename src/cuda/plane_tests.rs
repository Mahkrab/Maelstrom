//! Containment certificates must agree with the host predicate and remain private
//! until velocity effects, host state validation and cancellation reach acceptance.

use super::*;
use crate::{execution::ExecutionMode, neighbours::SearchBackend, simulation::Simulation};

fn trailer(gpu: &mut CudaSolver, count: usize) -> (u64, Vec<u32>) {
    let words = if count == 0 { 0 } else { 3 * count + 4 };
    let values = gpu.context.inner.download::<u32>(3, words).unwrap();
    let index = 3 * count + (count & 1);
    let record = if count == 0 {
        u64::MAX
    } else {
        u64::from(values[index]) | (u64::from(values[index + 1]) << 32)
    };
    (record, values)
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_plane_certificate_matches_host_and_survives_velocity_replacement() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let plane_sets = [
        vec![
            Plane::new(Vec3::new(1., 0., 0.), 0., 0.).unwrap(),
            Plane::new(Vec3::new(-1., 0., 0.), -1., 0.).unwrap(),
        ],
        vec![
            Plane::new(Vec3::new(0.6, 0.8, 0.), 0., 1e-6).unwrap(),
            Plane::new(Vec3::new(-0.6, -0.8, 0.), -1., 1e-6).unwrap(),
        ],
        vec![Plane::new(Vec3::new(1., 0., 0.), -f32::MAX, 0.).unwrap()],
    ];
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 2, 3, 31, 32, 33, 257, 513, 17] {
            for planes in &plane_sets {
                for mode in [0, 1, 2, 0] {
                    let points: Vec<_> = (0..count)
                        .map(|i| match mode {
                            0 => Vec3::new(-0.0, 0.0, 0.0),
                            2 if i + 1 == count => Vec3::new(f32::MAX, f32::MAX, 0.),
                            2 => Vec3::ZERO,
                            _ => match i % 5 {
                                0 => Vec3::new(-f32::from_bits(1), 0., 0.),
                                1 => Vec3::new(f32::from_bits(1.0_f32.to_bits() + 1), 0., 0.),
                                2 => Vec3::new(-0.8, 0.6, 0.),
                                3 => Vec3::new(0.8, -0.6, 0.),
                                _ => Vec3::new(0.5, 0., 0.),
                            },
                        })
                        .collect();
                    let particles: Vec<_> = points
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| Particle {
                            identity: (3 * i + 7) as u64,
                            position_m: p,
                            velocity_m_s: Vec3::ZERO,
                        })
                        .collect();
                    let mut s = settings(&config, count, planes.len()).unwrap();
                    s.timestep = 1.;
                    s.xsph = 0.01;
                    s.vorticity = 0.1;
                    gpu.prepare(s, &particles, planes).unwrap();
                    if count != 0 {
                        gpu.context
                            .inner
                            .upload(3, &vec![0xdeadbeef_u32; 3 * count + 4])
                            .unwrap();
                    }
                    gpu.context
                        .inner
                        .upload(8, &vec![[0.0_f32; 3]; count])
                        .unwrap();
                    gpu.operation(s, 10).unwrap();
                    gpu.context
                        .inner
                        .upload(2, &points.iter().copied().map(array).collect::<Vec<_>>())
                        .unwrap();
                    gpu.context
                        .inner
                        .upload(13, &vec![0_u64; count + 1])
                        .unwrap();
                    let mut expected = u64::MAX;
                    for (i, &point) in points.iter().enumerate() {
                        for (k, &plane) in planes.iter().enumerate() {
                            if !matches!(plane.contains(point), Ok(true)) {
                                expected = ((i as u64) << 32) | k as u64;
                                break;
                            }
                        }
                        if expected != u64::MAX {
                            break;
                        }
                    }
                    assert_eq!(
                        gpu.context.inner.checked_launch(s, 24, threads).unwrap(),
                        u64::MAX
                    );
                    let (actual, words) = trailer(&mut gpu, count);
                    assert_eq!(
                        actual, expected,
                        "block {threads}, count {count}, mode {mode}"
                    );
                    assert_eq!(&words[..3 * count], vec![0_u32; 3 * count]);
                    if count != 0 {
                        let record_offset = count & 1;
                        for (index, &value) in words[3 * count..].iter().enumerate() {
                            if index != record_offset && index != record_offset + 1 {
                                assert_eq!(value, 0);
                            }
                        }
                    }
                    for operation in [16, 17, 15, 14, 15] {
                        gpu.operation(s, operation).unwrap();
                        assert_eq!(trailer(&mut gpu, count).0, expected);
                    }
                    let replacement = vec![[3.0_f32, -4.0, 1.0]; count];
                    gpu.context.inner.upload(8, &replacement).unwrap();
                    gpu.operation(s, 15).unwrap();
                    assert_eq!(trailer(&mut gpu, count).0, expected);
                    assert_eq!(
                        gpu.context.inner.download::<[f32; 3]>(3, count).unwrap(),
                        replacement
                    );
                    assert_eq!(
                        gpu.context.inner.download::<u32>(2, 3 * count).unwrap(),
                        points
                            .iter()
                            .copied()
                            .map(array)
                            .flat_map(|v| v.map(f32::to_bits))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_deferred_plane_failures_keep_error_priority_cancellation_and_retry() {
    for threads in [32, 64, 128, 256] {
        for effects in [false, true] {
            for case in 0..4 {
                let timestep = if case == 3 { 0.5 } else { 1.0 };
                let text = format!(
                    "[numerical]\nsolver_iterations=1\nfixed_timestep_s={timestep}\nexternal_acceleration_m_s2=[0.0,0.0,0.0]\nxsph_viscosity_enabled={effects}\nxsph_viscosity_coefficient=0.01\nvorticity_confinement_enabled={effects}\nvorticity_confinement_strength_m_s=0.1\n"
                );
                let overlay = crate::config::parse_configuration_overlay(&text).unwrap();
                let config =
                    crate::config::resolve_effective_configuration(None, Some(&overlay), None)
                        .unwrap();
                let planes = match case {
                    0 => vec![
                        Plane::new(Vec3::new(1., 0., 0.), 1., 0.).unwrap(),
                        Plane::new(Vec3::new(-1., 0., 0.), 0., 0.).unwrap(),
                    ],
                    2 => vec![
                        Plane::new(Vec3::new(1., 0., 0.), -f32::MAX, 0.).unwrap(),
                        Plane::new(Vec3::new(1., 0., 0.), f32::MAX, 0.).unwrap(),
                    ],
                    _ => vec![
                        Plane::new(Vec3::new(0.6, 0.8, 0.), 0., 1e-6).unwrap(),
                        Plane::new(Vec3::new(1., 0., 0.), f32::MAX, 0.).unwrap(),
                        Plane::new(Vec3::new(0., 1., 0.), f32::MAX, 0.).unwrap(),
                    ],
                };
                let expected = match case {
                    0 => StepError::OutsidePlane {
                        identity: 7,
                        plane: 0,
                    },
                    1 => NumericalError::NonFiniteResult {
                        quantity: "plane dot product",
                    }
                    .into(),
                    2 => NumericalError::NonFiniteResult {
                        quantity: "plane constraint",
                    }
                    .into(),
                    _ => NumericalError::NonFiniteResult {
                        quantity: "reconstructed velocity",
                    }
                    .into(),
                };
                let particles = vec![
                    Particle {
                        identity: 49,
                        position_m: Vec3::ZERO,
                        velocity_m_s: Vec3::ZERO,
                    },
                    Particle {
                        identity: 7,
                        position_m: Vec3::ZERO,
                        velocity_m_s: Vec3::ZERO,
                    },
                ];
                let mut cpu = Simulation::new_with_backend(
                    &config,
                    particles.clone(),
                    planes.clone(),
                    SearchBackend::UniformGrid,
                )
                .unwrap();
                assert_eq!(
                    format!("{:?}", cpu.step().unwrap_err()),
                    format!("{expected:?}")
                );
                let mut gpu = Simulation::new_with_execution(
                    &config,
                    particles,
                    planes,
                    SearchBackend::UniformGrid,
                    ExecutionMode::Cuda { device: 0, threads },
                )
                .unwrap();
                let before = gpu.snapshot().clone();
                for _ in 0..2 {
                    assert_eq!(
                        format!("{:?}", gpu.step().unwrap_err()),
                        format!("{expected:?}")
                    );
                    assert_eq!(gpu.snapshot(), &before);
                }
                if case != 3 {
                    let last_poll = if effects { 11 } else { 6 };
                    for cancel_at in 1..=last_poll + 1 {
                        let mut polls = 0;
                        let result = gpu.step_with_cancellation(|| {
                            polls += 1;
                            polls == cancel_at
                        });
                        if cancel_at <= last_poll {
                            assert!(matches!(result, Err(StepError::Cancelled)));
                            assert_eq!(polls, cancel_at);
                        } else {
                            assert_eq!(
                                format!("{:?}", result.unwrap_err()),
                                format!("{expected:?}")
                            );
                            assert_eq!(polls, last_poll);
                        }
                        assert_eq!(gpu.snapshot(), &before);
                    }
                    assert_eq!(
                        format!("{:?}", gpu.step().unwrap_err()),
                        format!("{expected:?}")
                    );
                }
            }
        }
    }
}
