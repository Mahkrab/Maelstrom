//! Hardware checks inspect private stage results without widening the solver API.

use super::*;

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn xsph_candidate_is_separate_and_numerical_failure_can_retry_without_partial_commit() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        let particles = vec![
            Particle {
                identity: 0,
                position_m: Vec3::ZERO,
                velocity_m_s: Vec3::new(2., 0., 0.),
            },
            Particle {
                identity: 1,
                position_m: Vec3::new(0.5, 0., 0.),
                velocity_m_s: Vec3::new(-2., 0., 0.),
            },
        ];
        let mut s = settings(&config, 2, 0).unwrap();
        s.radius = 1.;
        s.rest_density = 4.;
        s.xsph = 0.25;
        for (distance, overflow) in [
            (0.5, false),
            (0.5, true),
            (0.5, false),
            (0.0, false),
            (1.0, false),
            (f32::from_bits(1.0_f32.to_bits() - 1), false),
            (f32::from_bits(1.0_f32.to_bits() + 1), false),
            (1.0, true),
        ] {
            gpu.prepare(s, &particles, &[]).unwrap();
            let mut positions: Vec<_> = particles.iter().map(|p| p.position_m).collect();
            gpu.context
                .inner
                .upload(2, &positions.iter().copied().map(array).collect::<Vec<_>>())
                .unwrap();
            gpu.discover(&mut s, &positions).unwrap();
            // Keep the predicted row, but independently change the final distance.
            positions[1].x = distance;
            gpu.context
                .inner
                .upload(2, &positions.iter().copied().map(array).collect::<Vec<_>>())
                .unwrap();
            let velocities = if overflow {
                vec![Vec3::new(f32::MAX, 0., 0.), Vec3::new(-f32::MAX, 0., 0.)]
            } else {
                particles.iter().map(|p| p.velocity_m_s).collect()
            };
            let input: Vec<_> = velocities.iter().copied().map(array).collect();
            gpu.context.inner.upload(3, &input).unwrap();
            let result = gpu.operation(s, 14);
            let after: Vec<[f32; 3]> = gpu.context.inner.download(3, 2).unwrap();
            assert_eq!(after, input, "smoothing must not overwrite its own input");
            if overflow && distance < 1.0 {
                assert!(matches!(
                    result,
                    Err(StepError::Numerical(NumericalError::NonFiniteResult {
                        quantity: "XSPH velocity difference"
                    }))
                ));
            } else {
                result.unwrap();
                let expected = crate::velocity::xsph_viscosity(
                    &positions,
                    &velocities,
                    &crate::fluid::InteractionSets::new(vec![vec![1], vec![0]]).unwrap(),
                    1.,
                    4.,
                    0.25,
                )
                .unwrap();
                let candidate: Vec<[f32; 3]> = gpu.context.inner.download(8, 2).unwrap();
                assert_eq!(
                    candidate,
                    expected.into_iter().map(array).collect::<Vec<_>>()
                );
                gpu.operation(s, 15).unwrap();
                assert_eq!(
                    gpu.context.inner.download::<[f32; 3]>(3, 2).unwrap(),
                    candidate
                );
            }
        }
    }
}

#[test]
fn packed_diagnostics_preserve_quantities_and_reject_invalid_records() {
    assert!(diagnostic_error(u64::MAX, 0).is_ok());
    assert!(diagnostic_error(u64::MAX, 9).is_ok());
    assert!(matches!(
        diagnostic_error((3_u64 << 32) | 22, 9),
        Err(StepError::Numerical(NumericalError::NonFiniteResult {
            quantity: "predicted position"
        }))
    ));
    assert!(matches!(
        diagnostic_error((3_u64 << 32) | 52, 9),
        Err(StepError::Cuda(CudaError::NeighbourStorage { particle: 3 }))
    ));
    for (particle, quantity, count) in [(0, 0, 1), (0, 71, 1), (1, 22, 1), (0, 22, 0)] {
        assert!(matches!(
            diagnostic_error((particle as u64) << 32 | u64::from(quantity), count),
            Err(StepError::Cuda(CudaError::Diagnostic { particle: actual_particle, quantity: actual_quantity }))
                if actual_particle == particle && actual_quantity == quantity
        ));
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_diagnostic_selection_preserves_particle_priority_and_clears_old_failures() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [1, 31, 32, 33, 257, 1025] {
            let mut s = settings(&config, count, 0).unwrap();
            s.timestep = 2.0;
            for failure in [None, Some(count - 1), Some(count / 2), None] {
                let mut positions = vec![Vec3::ZERO; count];
                let mut velocities = vec![Vec3::ZERO; count];
                if let Some(first) = failure {
                    positions[first].x = f32::MAX;
                    velocities[first].x = f32::MAX / 2.0;
                    if first + 1 < count {
                        // A later particle fails earlier within its expression
                        // (code 21). Identity must still select this one's code 22.
                        velocities[count - 1].x = f32::MAX;
                    }
                }
                let particles: Vec<_> = positions
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| Particle {
                        identity: i as u64,
                        position_m: p,
                        velocity_m_s: Vec3::ZERO,
                    })
                    .collect();
                gpu.prepare(s, &particles, &[]).unwrap();
                gpu.context
                    .inner
                    .upload(3, &velocities.into_iter().map(array).collect::<Vec<_>>())
                    .unwrap();
                gpu.context.inner.upload(12, &vec![52_u32; count]).unwrap();
                let result = gpu.context.inner.checked_launch(s, 1, threads).unwrap();
                let expected = failure.map_or(u64::MAX, |index| (index as u64) << 32 | 22);
                assert_eq!(
                    result, expected,
                    "count {count}, block {threads}, first {failure:?}"
                );
                let errors: Vec<u32> = gpu.context.inner.download(12, count).unwrap();
                assert_eq!(errors.iter().position(|&code| code != 0), failure);
                if let Some(first) = failure {
                    assert_eq!(errors[first], 22);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_groups_preserve_stage_priority_and_skip_dependent_writes() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [2, 31, 32, 33, 257, 1025] {
            for failure in 0..4 {
                let mut s = settings(&config, count, 0).unwrap();
                let particles: Vec<_> = (0..count)
                    .map(|i| Particle {
                        identity: i as u64,
                        position_m: Vec3::ZERO,
                        velocity_m_s: Vec3::ZERO,
                    })
                    .collect();
                gpu.prepare(s, &particles, &[]).unwrap();
                let mut positions = vec![[0.0_f32; 3]; count];
                let mut offsets = vec![0_u64; count + 1];
                let mut indices = Vec::new();
                match failure {
                    0 => {
                        // Only the last row contains a distant pair. Its squared
                        // length overflows in the first pass, while rest_density
                        // would make the NEXT pass fail at particle zero.
                        positions[count - 1][0] = f32::MAX;
                        offsets[count] = 1;
                        indices.push(0_u32);
                        s.rest_density = 0.0;
                    }
                    1 => s.rest_density = 0.0,
                    _ => {
                        for i in 0..count {
                            offsets[i + 1] = (i + 1) as u64;
                            indices.push(((i + 1) % count) as u32);
                        }
                        if failure == 2 {
                            s.reference_weight = 0.0;
                        } else {
                            s.mass = f32::MAX;
                        }
                    }
                }
                s.entries = indices.len() as u64;
                gpu.context.inner.reserve(11, indices.len() * 4).unwrap();
                gpu.context.inner.upload(11, &indices).unwrap();
                gpu.context.inner.upload(13, &offsets).unwrap();
                gpu.context.inner.upload(2, &positions).unwrap();
                gpu.context.inner.upload(4, &vec![99.0_f32; count]).unwrap();
                let untouched = vec![[77.0_f32, 78.0, 79.0]; count];
                gpu.context.inner.upload(7, &untouched).unwrap();
                let record = gpu.context.inner.checked_group(s, 5, threads).unwrap();
                let expected = match failure {
                    0 => ((count as u64 - 1) << 32) | 2,
                    1 => 24,
                    2 => 34,
                    _ => 43,
                };
                assert_eq!(
                    record, expected,
                    "count {count}, block {threads}, pass {failure}"
                );
                if failure <= 1 {
                    let correction: Vec<[f32; 3]> = gpu.context.inner.download(7, count).unwrap();
                    assert_eq!(correction, untouched);
                }
                if failure == 0 {
                    let density: Vec<f32> = gpu.context.inner.download(4, count).unwrap();
                    assert_eq!(density, vec![99.0_f32; count]);
                }
                // Reset and reuse the same resource with an independently simple
                // no-neighbour case. No failed slot may leak into this attempt.
                let clean = settings(&config, count, 0).unwrap();
                gpu.prepare(clean, &particles, &[]).unwrap();
                gpu.context
                    .inner
                    .upload(2, &vec![[0.0_f32; 3]; count])
                    .unwrap();
                gpu.context
                    .inner
                    .upload(13, &vec![0_u64; count + 1])
                    .unwrap();
                assert_eq!(
                    gpu.context.inner.checked_group(clean, 5, threads).unwrap(),
                    u64::MAX
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_projection_group_keeps_candidate_failure_before_plane_failure() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let plane = Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 0.0).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [1, 31, 32, 33, 257, 1025] {
            let particles: Vec<_> = (0..count)
                .map(|i| Particle {
                    identity: i as u64,
                    position_m: Vec3::ZERO,
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let s = settings(&config, count, 1).unwrap();
            for candidate_failure in [true, false] {
                gpu.prepare(s, &particles, &[plane]).unwrap();
                let mut positions = vec![[0.0_f32; 3]; count];
                let mut corrections = positions.clone();
                if candidate_failure {
                    positions[count - 1][0] = f32::MAX;
                    corrections[count - 1][0] = f32::MAX;
                }
                gpu.context.inner.upload(2, &positions).unwrap();
                gpu.context.inner.upload(7, &corrections).unwrap();
                // Private fault injection: public scene validation rejects this.
                gpu.context
                    .inner
                    .upload(9, &[[1.0_f32, 0.0, 0.0, f32::NAN]])
                    .unwrap();
                let record = gpu.context.inner.checked_group(s, 9, threads).unwrap();
                assert_eq!(
                    record,
                    if candidate_failure {
                        ((count as u64 - 1) << 32) | 44
                    } else {
                        46
                    }
                );
                if candidate_failure {
                    let actual: Vec<[f32; 3]> = gpu.context.inner.download(2, count).unwrap();
                    assert_eq!(actual, positions, "projection must not touch any position");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_multiplier_errors_keep_owner_checks_before_prefix_squares() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let h = 2.0_f32.powi(-10);
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for (positions, neighbours, reciprocal_power, owner, quantity) in [
            // Each substituted direction is about 1.52e38: finite individually,
            // but their owner sum overflows. Prefix squares also overflow; they
            // must not mask the earlier owner-direction check (28).
            (
                vec![
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Vec3::new(h / 2.0, 0.0, 0.0),
                ],
                vec![0_u32, 1, 2],
                85,
                3,
                28,
            ),
            // Opposite directions cancel in the owner. Individual squares are
            // about 2.73e38; their ordered prefix sum overflows (31).
            (
                vec![
                    Vec3::new(-h / 2.0, 0.0, 0.0),
                    Vec3::new(h / 2.0, 0.0, 0.0),
                    Vec3::ZERO,
                ],
                vec![0_u32, 1],
                22,
                2,
                31,
            ),
            // With self first, the same cancellation leaves the suffix's first
            // individual squared direction to fail (30).
            (
                vec![
                    Vec3::ZERO,
                    Vec3::new(-h / 2.0, 0.0, 0.0),
                    Vec3::new(h / 2.0, 0.0, 0.0),
                ],
                vec![1_u32, 2],
                24,
                0,
                30,
            ),
        ] {
            let count = positions.len();
            let mut s = settings(&config, count, 0).unwrap();
            s.radius = h;
            s.rest_density = 2.0_f32.powi(-reciprocal_power);
            s.relaxation = 1.0;
            s.entries = neighbours.len() as u64;
            let particles: Vec<_> = positions
                .iter()
                .enumerate()
                .map(|(i, &position_m)| Particle {
                    identity: i as u64,
                    position_m,
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            gpu.prepare(s, &particles, &[]).unwrap();
            let offsets: Vec<u64> = (0..=count)
                .map(|i| if i <= owner { 0 } else { s.entries })
                .collect();
            gpu.context.inner.upload(11, &neighbours).unwrap();
            gpu.context.inner.upload(13, &offsets).unwrap();
            gpu.context
                .inner
                .upload(2, &positions.into_iter().map(array).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(
                gpu.context.inner.checked_launch(s, 6, threads).unwrap(),
                (owner as u64) << 32 | quantity
            );
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_multiplier_prefix_keeps_self_order_and_rechecks_current_support() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 3, 31, 32, 33, 129] {
            let initial: Vec<_> = (0..count)
                .map(|i| {
                    Vec3::new(
                        (i % 7) as f32 * 0.021,
                        ((i / 7) % 5) as f32 * 0.019,
                        (i / 35) as f32 * 0.017,
                    )
                })
                .collect();
            let rows =
                crate::neighbours::brute_force(&initial, config.derived.support_radius_m).unwrap();
            let particles: Vec<_> = initial
                .iter()
                .enumerate()
                .map(|(i, &position_m)| Particle {
                    identity: i as u64,
                    position_m,
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let mut s = settings(&config, count, 0).unwrap();
            gpu.prepare(s, &particles, &[]).unwrap();
            gpu.context
                .inner
                .upload(2, &initial.iter().copied().map(array).collect::<Vec<_>>())
                .unwrap();
            gpu.discover(&mut s, &initial).unwrap();
            for displaced in [false, true, false] {
                let current: Vec<_> = initial
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| {
                        if displaced && i % 3 == 1 {
                            p + Vec3::new(1.0, 0.0, 0.0)
                        } else {
                            p
                        }
                    })
                    .collect();
                let expected = crate::fluid::compute_jacobi_stage(
                    &current,
                    &rows,
                    config.pbf_parameters().unwrap(),
                )
                .unwrap();
                gpu.context
                    .inner
                    .upload(2, &current.into_iter().map(array).collect::<Vec<_>>())
                    .unwrap();
                gpu.operation(s, 6).unwrap();
                for (slot, values) in [
                    (4, expected.normalised_densities),
                    (5, expected.constraints),
                    (6, expected.multipliers),
                ] {
                    let actual: Vec<f32> = gpu.context.inner.download(slot, count).unwrap();
                    assert_eq!(
                        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                        values.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                        "count {count}, block {threads}, displaced {displaced}, slot {slot}"
                    );
                }
            }
        }
    }
}

#[test]
fn capacity_arithmetic_rejects_unrepresentable_workspace_without_allocating() {
    assert!(
        storage_sizes(Settings {
            count: u32::MAX,
            entries: u64::MAX,
            ..Default::default()
        })
        .is_err()
    );
    let c = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    assert!(settings(&c, usize::MAX, 0).is_err());
    assert!(settings(&c, 1, usize::MAX).is_err());
    assert_eq!(
        storage_sizes(settings(&c, 33, 2).unwrap()).unwrap()[13],
        272
    );
    assert_eq!(
        storage_sizes(settings(&c, 8000, 6).unwrap()).unwrap()[11],
        0
    );
}

#[test]
fn neighbour_prefixes_preserve_empty_rows_and_reject_invalid_counts() {
    let mut counts = [0, 2, 0, 1, 3];
    assert_eq!(prefix_offsets(&mut counts, 4).unwrap(), 6);
    assert_eq!(counts, [0, 2, 2, 3, 6]);
    assert_eq!(prefix_offsets(&mut [0], 0).unwrap(), 0);
    assert!(prefix_offsets(&mut [0, 4, 0, 0, 0], 4).is_err());
    assert!(prefix_offsets(&mut [1, 0], 1).is_err());
    assert!(prefix_offsets(&mut [0], 1).is_err());
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_batched_sort_matches_integer_keys_and_individual_passes() {
    let configuration = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [
            1, 3, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1025, 8193,
        ] {
            // Integral coordinates make the expected unit-width cell keys
            // independent of the device's floating-point cell calculation.
            let keys: Vec<_> = (0..count)
                .map(|i| ((i * 7 % 11) as i64 - 5, (i * 3 % 7) as i64 - 3, 0_i64, i))
                .collect();
            let particles: Vec<_> = keys
                .iter()
                .map(|&(x, y, z, i)| Particle {
                    identity: i as u64,
                    position_m: Vec3::new(x as f32, y as f32, z as f32),
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let mut expected = keys;
            expected.sort_unstable();
            let mut s = settings(&configuration, count, 0).unwrap();
            s.width = 1.0;
            gpu.prepare(s, &particles, &[]).unwrap();
            gpu.context
                .inner
                .upload(
                    2,
                    &particles
                        .iter()
                        .map(|p| array(p.position_m))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            gpu.operation(s, 2).unwrap();
            gpu.context.inner.launch(s, 13, threads).unwrap();
            let batched: Vec<u64> = gpu
                .context
                .inner
                .download(10, s.padded as usize * 4)
                .unwrap();
            for (cell, &(x, y, z, index)) in batched.chunks_exact(4).zip(&expected) {
                assert_eq!(
                    (cell[0] as i64, cell[1] as i64, cell[2] as i64, cell[3]),
                    (x, y, z, index as u64)
                );
            }
            for cell in batched.chunks_exact(4).skip(count) {
                assert_eq!(
                    cell,
                    &[
                        i64::MAX as u64,
                        i64::MAX as u64,
                        i64::MAX as u64,
                        u32::MAX as u64
                    ]
                );
            }
            gpu.operation(s, 2).unwrap();
            let mut span = 2_u64;
            while span <= u64::from(s.padded) {
                s.span = span as u32;
                s.stride = s.span / 2;
                while s.stride != 0 {
                    gpu.operation(s, 3).unwrap();
                    s.stride /= 2;
                }
                span *= 2;
            }
            let separate: Vec<u64> = gpu
                .context
                .inner
                .download(10, s.padded as usize * 4)
                .unwrap();
            assert_eq!(batched, separate, "count {count}, block {threads}");
        }
    }
}

pub(super) fn discovered(
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
    gpu.discover(&mut s, positions)?;
    let offsets: Vec<u64> = gpu.context.inner.download(13, positions.len() + 1)?;
    let indices: Vec<u32> = gpu.context.inner.download(11, s.entries as usize)?;
    Ok((0..positions.len())
        .map(|i| {
            indices[offsets[i] as usize..offsets[i + 1] as usize]
                .iter()
                .map(|&j| j as usize)
                .collect()
        })
        .collect())
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_identity_rows_sort_network_edges_dense_fallback_and_tail_warps() {
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        // Every pair is inside support, but cell order interleaves identities.
        // Each row's exact answer is simply all identities except its owner.
        for count in [
            0, 1, 2, 3, 32, 33, 34, 64, 65, 66, 128, 129, 130, 256, 257, 258, 514, 1026,
        ] {
            let positions: Vec<_> = (0..count)
                .map(|i| {
                    Vec3::new(
                        if i & 1 == 0 { -0.01 } else { 0.01 },
                        if i & 2 == 0 { -0.01 } else { 0.01 },
                        if i & 4 == 0 { -0.01 } else { 0.01 },
                    )
                })
                .collect();
            let rows = discovered(&mut gpu, &positions, 0.1).unwrap();
            for (owner, row) in rows.iter().enumerate() {
                let expected: Vec<_> = (0..count).filter(|&i| i != owner).collect();
                assert_eq!(
                    row, &expected,
                    "particles {count}, owner {owner}, block {threads}"
                );
            }
        }
        // Spatially separated clusters give a single launch different network
        // widths, empty rows and a final row beyond the dense-row threshold.
        let mut points = Vec::new();
        let mut expected = Vec::new();
        for (cluster, count) in [1, 2, 34, 66, 130, 258].into_iter().enumerate() {
            let first = points.len();
            for local in 0..count {
                points.push(Vec3::new(
                    cluster as f32 * 2.0,
                    if local & 1 == 0 { -0.01 } else { 0.01 },
                    0.0,
                ));
                expected.push(
                    (first..first + count)
                        .filter(|&i| i != first + local)
                        .collect::<Vec<_>>(),
                );
            }
        }
        assert_eq!(discovered(&mut gpu, &points, 0.1).unwrap(), expected);
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_failed_fill_does_not_sort_invalid_rows_and_can_retry() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        let points = [Vec3::ZERO; 3];
        discovered(&mut gpu, &points, 0.1).unwrap();
        let mut s = settings(&config, 3, 0).unwrap();
        s.radius = 0.1;
        s.width = 0.125;
        s.entries = 6;
        let before = vec![99_u32; 6];
        gpu.context.inner.upload(11, &before).unwrap();
        // Every range is reversed or beyond storage. No lane may access it.
        gpu.context.inner.upload(13, &[7_u64, 6, 5, 4]).unwrap();
        assert!(matches!(
            gpu.operation(s, 12),
            Err(StepError::Cuda(CudaError::NeighbourStorage { particle: 0 }))
        ));
        assert_eq!(gpu.context.inner.download::<u32>(11, 6).unwrap(), before);
        assert_eq!(
            discovered(&mut gpu, &points, 0.1).unwrap(),
            vec![vec![1, 2], vec![0, 2], vec![0, 1]]
        );
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_compact_rows_grow_shrink_and_reject_inconsistent_offsets() {
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    let dense = vec![Vec3::ZERO; 513];
    let rows = discovered(&mut gpu, &dense, 0.1).unwrap();
    for (owner, row) in rows.iter().enumerate() {
        assert_eq!(row, &(0..513).filter(|&j| j != owner).collect::<Vec<_>>());
    }
    let sparse: Vec<_> = (0..513).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
    assert!(
        discovered(&mut gpu, &sparse, 0.1)
            .unwrap()
            .iter()
            .all(Vec::is_empty)
    );
    assert_eq!(
        discovered(&mut gpu, &[Vec3::ZERO; 2], 0.1).unwrap(),
        vec![vec![1], vec![0]]
    );
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let mut s = settings(&config, 2, 0).unwrap();
    s.width = 0.125;
    s.entries = 2;
    // A bad row must fail before writing past its reserved range. This private
    // injection does not expose arbitrary native buffers through the public API.
    gpu.context.inner.upload(13, &[0_u64, 0, 2]).unwrap();
    assert!(matches!(
        gpu.operation(s, 12),
        Err(StepError::Cuda(CudaError::NeighbourStorage { particle: 0 }))
    ));
    // Rebuilding from fresh prediction data must not retain the injected layout.
    assert_eq!(
        discovered(&mut gpu, &[Vec3::ZERO; 2], 0.1).unwrap(),
        vec![vec![1], vec![0]]
    );
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_large_lattice_count_and_fill_match_independent_grid_rows() {
    let overlay =
        crate::config::parse_configuration_overlay("[numerical]\nrefine_timestep=4").unwrap();
    let config =
        crate::config::resolve_effective_configuration(None, None, Some(&overlay)).unwrap();
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    for side in [32, 48] {
        let mut points = Vec::new();
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let p = Vec3::new(0.2, 0.4, 0.2)
                        + Vec3::new(x as f32 * 0.05, y as f32 * 0.05, z as f32 * 0.05);
                    let velocity = config.numerical.external_acceleration_m_s2
                        * config.numerical.fixed_timestep_s;
                    points.push(p + velocity * config.numerical.fixed_timestep_s);
                }
            }
        }
        let expected = crate::neighbours::uniform_grid(&points, 0.1).unwrap();
        let mut s = settings(&config, points.len(), 0).unwrap();
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
        gpu.context
            .inner
            .upload(2, &points.iter().copied().map(array).collect::<Vec<_>>())
            .unwrap();
        let result = gpu.discover(&mut s, &points);
        if result.is_err() {
            let errors: Vec<u32> = gpu.context.inner.download(12, points.len()).unwrap();
            let offsets: Vec<u64> = gpu.context.inner.download(13, points.len() + 1).unwrap();
            let failing = errors.iter().position(|&e| e != 0).unwrap_or(0);
            let indices: Vec<u32> = gpu.context.inner.download(11, s.entries as usize).unwrap();
            panic!(
                "side {side}: {result:?}, total {}, row {failing} bounds {:?}, actual {:?}, expected {:?}",
                s.entries,
                &offsets[failing..=failing + 1],
                &indices[offsets[failing] as usize..offsets[failing + 1] as usize],
                expected.for_particle(failing)
            );
        }
        let offsets: Vec<u64> = gpu.context.inner.download(13, points.len() + 1).unwrap();
        let indices: Vec<u32> = gpu.context.inner.download(11, s.entries as usize).unwrap();
        for i in 0..points.len() {
            let actual: Vec<usize> = indices[offsets[i] as usize..offsets[i + 1] as usize]
                .iter()
                .map(|&j| j as usize)
                .collect();
            assert_eq!(
                actual,
                expected.for_particle(i).unwrap(),
                "side {side}, row {i}"
            );
        }
    }
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
    gpu.discover(&mut s, &points).unwrap();
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
    let mut s = settings(&config, 3, 0).unwrap();
    // Retain the previously discovered membership rather than rebuilding it.
    s.entries = 2;
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
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("reference/scenes/falling-block-04.toml"),
        Some(&overlay),
    )
    .unwrap();
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    let mut s = settings(
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
        gpu.discover(&mut s, &positions).unwrap();
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

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_spatial_schedule_keeps_identity_rows_and_error_priority() {
    use crate::neighbours::{brute_force, uniform_grid};

    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [33, 257, 4099, 65] {
            let mut points: Vec<_> = (0..count)
                .map(|i| {
                    Vec3::new(
                        (i % 37) as f32 * 0.04 - 0.8,
                        ((i / 37) % 11) as f32 * 0.04 - 0.3,
                        (i / (37 * 11)) as f32 * 0.04 - 0.2,
                    )
                })
                .collect();
            // Break the relationship between slice identity and spatial order.
            // The oracle still enumerates neighbours in ascending slice order.
            for i in (1..count).rev() {
                points.swap(i, (i * 1664525 + 1013904223) % (i + 1));
            }
            let expected = if count < 300 {
                brute_force(&points, 0.1).unwrap()
            } else {
                uniform_grid(&points, 0.1).unwrap()
            };
            let actual = discovered(&mut gpu, &points, 0.1).unwrap();
            for (owner, row) in actual.iter().enumerate() {
                assert_eq!(row, expected.for_particle(owner).unwrap());
            }
        }

        // Spatial order visits the negative extreme first, but particle zero's
        // displacement error must win. The unsafe-domain all-pairs check retains
        // its identity-ordered traversal before any cell-local distance checks.
        let points = [
            Vec3::new(f32::MAX, 0., 0.),
            Vec3::new(-f32::MAX, 0., 0.),
            Vec3::ZERO,
        ];
        assert!(matches!(
            discovered(&mut gpu, &points, 0.1),
            Err(StepError::Numerical(NumericalError::NonFiniteResult {
                quantity: "neighbour search displacement"
            }))
        ));
        assert_eq!(
            gpu.context.inner.download::<u32>(12, 3).unwrap(),
            vec![1, 1, 2]
        );
        assert!(gpu.accept().is_err());
        assert_eq!(
            discovered(&mut gpu, &[Vec3::ZERO; 3], 0.1).unwrap(),
            vec![vec![1, 2], vec![0, 2], vec![0, 1]]
        );
    }
}
