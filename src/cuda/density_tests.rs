//! Fused traversal must preserve each ordered reduction and the original error
//! selection even when density, owner and prefix failures compete.

use super::*;
use crate::{fluid::PbfParameters, neighbours::brute_force};

fn scalar_bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn density_owner_fusion_matches_independent_intermediates_for_fixed_rows() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 33, 65, 257, 513, 17] {
            let mut original: Vec<_> = (0..count)
                .map(|i| {
                    Vec3::new(
                        (i % 11) as f32 * 0.015 - 0.08,
                        ((i / 11) % 9) as f32 * 0.015 - 0.06,
                        (i / 99) as f32 * 0.015 - 0.04,
                    )
                })
                .collect();
            original.reverse();
            original.rotate_left(count / 3);
            let fixed = brute_force(&original, 0.2).unwrap();
            let particles: Vec<_> = original
                .iter()
                .enumerate()
                .map(|(i, &position_m)| Particle {
                    identity: (3 * i + 7) as u64,
                    position_m,
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let mut s = settings(&config, count, 0).unwrap();
            s.radius = 0.2;
            s.relaxation = 0.01;
            s.strength = 0.1;
            s.reference_weight = crate::kernel::poly6(0.04, 0.2).unwrap();
            s.correction_scale = 1.0;
            gpu.prepare(s, &particles, &[]).unwrap();
            gpu.context
                .inner
                .upload(2, &original.iter().copied().map(array).collect::<Vec<_>>())
                .unwrap();
            gpu.discover(&mut s, &original).unwrap();
            for changed in [false, true, false] {
                let mut positions = original.clone();
                if changed {
                    for (i, point) in positions.iter_mut().enumerate() {
                        if i % 7 == 0 {
                            point.x += 0.4;
                        } else if i % 11 == 0 {
                            *point = Vec3::new(-0.0, 0.0, 0.0);
                        }
                    }
                }
                let input: Vec<_> = positions.iter().copied().map(array).collect();
                gpu.context.inner.upload(2, &input).unwrap();
                for density in [3.0, 10_000.0] {
                    s.rest_density = density;
                    assert_eq!(
                        gpu.context.inner.checked_group(s, 5, threads).unwrap(),
                        u64::MAX
                    );
                    let expected = crate::fluid::compute_jacobi_stage(
                        &positions,
                        &fixed,
                        PbfParameters::new(0.2, density, 0.01, 0.1, 0.04, 4.0).unwrap(),
                    )
                    .unwrap();
                    for (slot, expected) in [
                        (4, expected.normalised_densities),
                        (5, expected.constraints),
                        (6, expected.multipliers),
                    ] {
                        assert_eq!(
                            scalar_bits(gpu.context.inner.download(slot, count).unwrap()),
                            scalar_bits(expected),
                            "block {threads}, count {count}, changed {changed}, slot {slot}"
                        );
                    }
                    assert_eq!(
                        gpu.context
                            .inner
                            .download::<[f32; 3]>(7, count)
                            .unwrap()
                            .into_iter()
                            .map(|v| v.map(f32::to_bits))
                            .collect::<Vec<_>>(),
                        expected
                            .corrections
                            .into_iter()
                            .map(array)
                            .map(|v| v.map(f32::to_bits))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        gpu.context.inner.download::<u32>(12, count).unwrap(),
                        vec![0; count]
                    );
                    assert_eq!(
                        gpu.context.inner.download::<[f32; 3]>(2, count).unwrap(),
                        input
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn density_errors_precede_speculative_owner_and_reciprocal_checks() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for (radius, density, distance, quantity) in [
            (1.0, 2.0_f32.powi(-130), 0.5, 24),
            (1.0e-10, 1.0, 0.0, 9),
            (0.0, f32::from_bits(1), 0.0, 26),
            (1.0, 1.0, f32::MAX, 2),
        ] {
            let mut s = settings(&config, 2, 0).unwrap();
            s.radius = radius;
            s.rest_density = density;
            s.entries = 2;
            let particles = [
                Particle {
                    identity: 0,
                    position_m: Vec3::ZERO,
                    velocity_m_s: Vec3::ZERO,
                },
                Particle {
                    identity: 1,
                    position_m: Vec3::new(distance, 0., 0.),
                    velocity_m_s: Vec3::ZERO,
                },
            ];
            gpu.prepare(s, &particles, &[]).unwrap();
            gpu.context
                .inner
                .upload(2, &[[0.0, 0., 0.], [distance, 0., 0.]])
                .unwrap();
            gpu.context.inner.upload(13, &[0_u64, 1, 2]).unwrap();
            gpu.context.inner.upload(11, &[1_u32, 0]).unwrap();
            assert_eq!(
                gpu.context.inner.checked_launch(s, 6, threads).unwrap(),
                quantity,
                "block {threads}, radius {radius}, density {density}"
            );
            assert_eq!(
                gpu.context.inner.download::<u32>(12, 2).unwrap(),
                vec![quantity as u32; 2]
            );
            assert!(gpu.accept().is_err());
        }
        assert_eq!(
            tests::discovered(&mut gpu, &[Vec3::ZERO; 3], 0.1).unwrap(),
            vec![vec![1, 2], vec![0, 2], vec![0, 1]]
        );
    }
}
