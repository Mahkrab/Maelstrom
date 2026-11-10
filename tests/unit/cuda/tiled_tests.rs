//! Interleaving rows must preserve the independent CPU intermediates even for
//! dense, tail and changed-position inputs with fixed prediction membership.

use super::*;
use crate::{fluid::PbfParameters, neighbours::brute_force};

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn tiled_empty_rows_need_no_neighbour_allocation_and_keep_the_self_term() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        for count in [1, 31, 32, 33, 65] {
            let particles: Vec<_> = (0..count)
                .map(|i| Particle {
                    identity: i as u64,
                    position_m: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let points: Vec<_> = particles.iter().map(|p| p.position_m).collect();
            let mut gpu = CudaSolver::new(0, threads).unwrap();
            let mut s = settings(&config, count, 0).unwrap();
            s.neighbour_layout = 1;
            gpu.prepare(s, &particles, &[]).unwrap();
            gpu.context
                .inner
                .upload(2, &points.iter().copied().map(array).collect::<Vec<_>>())
                .unwrap();
            gpu.discover(&mut s, &points).unwrap();
            assert_eq!(s.entries, 0);
            assert_eq!(gpu.neighbour_capacity, 0);
            gpu.group(s, 5).unwrap();
            let expected = crate::kernel::poly6(0.0, s.radius).unwrap().to_bits();
            assert!(
                gpu.context
                    .inner
                    .download::<f32>(4, count)
                    .unwrap()
                    .iter()
                    .all(|x| x.to_bits() == expected)
            );
            assert_eq!(
                gpu.context.inner.download::<[f32; 3]>(7, count).unwrap(),
                vec![[0.0; 3]; count]
            );
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn tiled_prefix_rejects_invalid_counts_before_reusing_sort_scratch() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        let particles: Vec<_> = (0..33)
            .map(|i| Particle {
                identity: i,
                position_m: Vec3::ZERO,
                velocity_m_s: Vec3::ZERO,
            })
            .collect();
        let mut s = settings(&config, particles.len(), 0).unwrap();
        s.neighbour_layout = 1;
        gpu.prepare(s, &particles, &[]).unwrap();
        let scratch = vec![0xa5a5a5a5_u32; storage_sizes(s).unwrap()[10] / 4];
        gpu.context.inner.upload(10, &scratch).unwrap();
        let mut counts = vec![0_u64; 34];
        counts[1] = 33; // A row can contain at most the other 32 particles.
        gpu.context.inner.upload(13, &counts).unwrap();
        assert_eq!(gpu.context.inner.prefix(s, threads, false).unwrap(), 52);
        assert_eq!(
            gpu.context
                .inner
                .download::<u32>(10, scratch.len())
                .unwrap(),
            scratch
        );
        gpu.context.inner.begin_attempt().unwrap();
        counts[1] = 32;
        gpu.context.inner.upload(13, &counts).unwrap();
        assert_eq!(
            gpu.context.inner.prefix(s, threads, false).unwrap(),
            u64::MAX
        );
        assert_eq!(
            gpu.context
                .inner
                .download_at::<u64>(10, s.padded as usize * 4, 3)
                .unwrap(),
            vec![0, 32 * 32, 32 * 32]
        );
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn tiled_attempts_keep_every_cancellation_boundary_and_retry_from_accepted_state() {
    use crate::{execution::ExecutionMode, neighbours::SearchBackend, simulation::Simulation};

    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let particles: Vec<_> = (0..32_000)
        .map(|i| Particle {
            identity: i,
            position_m: Vec3::new(
                (i % 40) as f32 * 0.05,
                ((i / 40) % 40) as f32 * 0.05 + 0.4,
                (i / 1600) as f32 * 0.05,
            ),
            velocity_m_s: Vec3::ZERO,
        })
        .collect();
    let build = || {
        Simulation::new_with_execution(
            &config,
            particles.clone(),
            vec![],
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda {
                device: 0,
                threads: 128,
            },
        )
        .unwrap()
    };
    let mut reference = build();
    let mut candidate = build();
    let mut polls = 0;
    let _ = reference
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    for boundary in 1..=polls {
        if boundary != 1 {
            let _ = reference.step().unwrap();
        }
        let old = candidate.snapshot().clone();
        let mut observed = 0;
        assert!(matches!(
            candidate.step_with_cancellation(|| {
                observed += 1;
                observed == boundary
            }),
            Err(StepError::Cancelled)
        ));
        assert_eq!(candidate.snapshot(), &old);
        let _ = candidate.step().unwrap();
        assert_eq!(candidate.snapshot(), reference.snapshot());
    }
}

fn scalar_bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn tiled_density_matches_independent_intermediates_for_fixed_rows() {
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
            s.neighbour_layout = 1;
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
            // Derive physical storage independently from brute-force row
            // lengths. Check only members, never unspecified padding words.
            let mut tile_offsets = vec![0_u64];
            for first in (0..count).step_by(32) {
                let longest = (first..(first + 32).min(count))
                    .map(|i| fixed.for_particle(i).unwrap().len())
                    .max()
                    .unwrap();
                tile_offsets.push(tile_offsets.last().unwrap() + longest as u64 * 32);
            }
            assert_eq!(s.entries, *tile_offsets.last().unwrap());
            if count != 0 {
                assert_eq!(
                    gpu.context
                        .inner
                        .download_at::<u64>(10, s.padded as usize * 4, tile_offsets.len())
                        .unwrap(),
                    tile_offsets
                );
                let indices = gpu
                    .context
                    .inner
                    .download::<u32>(11, s.entries as usize)
                    .unwrap();
                for i in 0..count {
                    let row = fixed.for_particle(i).unwrap();
                    let actual: Vec<_> = (0..row.len())
                        .map(|k| indices[tile_offsets[i / 32] as usize + k * 32 + i % 32] as usize)
                        .collect();
                    assert_eq!(actual, row, "block {threads}, row {i}");
                }
            }
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
