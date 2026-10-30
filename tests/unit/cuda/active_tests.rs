//! Distance validation remains a global boundary even when numerical work is
//! calculated speculatively from the same immutable position snapshot.

use super::*;

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn active_errors_precede_all_numerical_errors_without_publishing_results() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [3, 33, 257, 1025, 17] {
            for case in 0..4 {
                let mut s = settings(&config, count, 0).unwrap();
                let particles: Vec<_> = (0..count)
                    .map(|i| Particle {
                        identity: (3 * i + 7) as u64,
                        position_m: Vec3::ZERO,
                        velocity_m_s: Vec3::ZERO,
                    })
                    .collect();
                let mut positions = vec![[0.0_f32; 3]; count];
                positions[count - 1][0] = f32::MAX;
                let (indices, expected, earlier) = match case {
                    // A numerical failure at particle zero cannot displace a
                    // later particle's invalid distance, even after block tails.
                    0 => {
                        s.rest_density = 0.0;
                        (vec![0_u32], 2_u32, None)
                    }
                    // The first neighbour's square overflows before the second
                    // neighbour's displacement does. Poly6 also fails earlier
                    // within the speculative numerical calculation.
                    1 => {
                        s.radius = 1.0e-10;
                        positions[1][0] = -f32::MAX;
                        (vec![0, 1], 2, None)
                    }
                    2 => {
                        s.radius = 1.0e-10;
                        positions[0][0] = -f32::MAX;
                        (vec![0], 18, None)
                    }
                    _ => {
                        s.rest_density = 0.0;
                        positions[0][0] = -f32::MAX;
                        positions[1][0] = f32::MAX;
                        positions[count - 1][0] = 0.0;
                        // Row one wins particle priority with displacement 18;
                        // the final row independently fails squared length 2.
                        (vec![0, 1], 2, Some((1, 18_u32)))
                    }
                };
                let mut offsets = vec![0_u64; count + 1];
                if earlier.is_some() {
                    offsets[2..count].fill(1);
                }
                offsets[count] = indices.len() as u64;
                s.entries = indices.len() as u64;
                gpu.prepare(s, &particles, &[]).unwrap();
                gpu.context.inner.upload(2, &positions).unwrap();
                gpu.context.inner.reserve(11, indices.len() * 4).unwrap();
                gpu.context.inner.upload(11, &indices).unwrap();
                gpu.context.inner.upload(13, &offsets).unwrap();
                for slot in [4, 5, 6] {
                    gpu.context
                        .inner
                        .upload(slot, &vec![99.0_f32; count])
                        .unwrap();
                }
                let untouched = vec![[77.0_f32, 78.0, 79.0]; count];
                gpu.context.inner.upload(7, &untouched).unwrap();
                let (owner, quantity) = earlier.unwrap_or((count - 1, expected));
                assert_eq!(
                    gpu.context.inner.checked_group(s, 5, threads).unwrap(),
                    ((owner as u64) << 32) | u64::from(quantity),
                    "block {threads}, count {count}, case {case}"
                );
                for slot in [4, 5, 6] {
                    assert_eq!(
                        gpu.context.inner.download::<f32>(slot, count).unwrap(),
                        vec![99.0; count]
                    );
                }
                assert_eq!(
                    gpu.context.inner.download::<[f32; 3]>(7, count).unwrap(),
                    untouched
                );
                assert_eq!(
                    gpu.context.inner.download::<[f32; 3]>(2, count).unwrap(),
                    positions
                );
                let mut errors = vec![0_u32; count];
                errors[count - 1] = expected;
                if let Some((owner, quantity)) = earlier {
                    errors[owner] = quantity;
                }
                assert_eq!(
                    gpu.context.inner.download::<u32>(12, count).unwrap(),
                    errors
                );
                assert!(gpu.accept().is_err());
                // Reuse after failure must reset diagnostics and overwrite all
                // speculative scratch, including when the next row is empty.
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
                assert_eq!(
                    gpu.context.inner.download::<u32>(12, count).unwrap(),
                    vec![0; count]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn active_group_matches_separate_stages_at_support_and_coincident_positions() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 3, 33, 257, 17] {
            let mut s = settings(&config, count, 0).unwrap();
            s.radius = 0.2;
            s.rest_density = 10_000.0;
            let positions: Vec<_> = (0..count)
                .map(|i| match i % 5 {
                    0 => [-0.0, 0.0, 0.0],
                    1 => [s.radius, 0.0, 0.0],
                    2 => [f32::from_bits(s.radius.to_bits() - 1), 0.0, 0.0],
                    3 => [f32::from_bits(s.radius.to_bits() + 1), 0.0, 0.0],
                    _ => [0.0, 0.0, 0.0],
                })
                .collect();
            let particles: Vec<_> = positions
                .iter()
                .enumerate()
                .map(|(i, &p)| Particle {
                    identity: (3 * i + 7) as u64,
                    position_m: vector(p),
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let mut indices = Vec::new();
            let mut offsets = vec![0_u64];
            // Fixed rows deliberately include neighbours outside current support.
            for i in 0..count {
                indices.extend((0..count as u32).filter(|&j| j as usize != i));
                offsets.push(indices.len() as u64);
            }
            s.entries = indices.len() as u64;
            let mut reference = None;
            for grouped in [false, true, true] {
                gpu.prepare(s, &particles, &[]).unwrap();
                gpu.context.inner.upload(2, &positions).unwrap();
                gpu.context.inner.reserve(11, indices.len() * 4).unwrap();
                gpu.context.inner.upload(11, &indices).unwrap();
                gpu.context.inner.upload(13, &offsets).unwrap();
                if grouped {
                    assert_eq!(
                        gpu.context.inner.checked_group(s, 5, threads).unwrap(),
                        u64::MAX
                    );
                } else {
                    for operation in 5..=8 {
                        gpu.operation(s, operation).unwrap();
                    }
                }
                let mut bits = Vec::new();
                for slot in [4, 5, 6] {
                    bits.extend(
                        gpu.context
                            .inner
                            .download::<f32>(slot, count)
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                }
                for v in gpu.context.inner.download::<[f32; 3]>(7, count).unwrap() {
                    bits.extend(v.map(f32::to_bits));
                }
                assert_eq!(
                    gpu.context.inner.download::<u32>(12, count).unwrap(),
                    vec![0; count]
                );
                if let Some(expected) = &reference {
                    assert_eq!(&bits, expected);
                } else {
                    reference = Some(bits);
                }
            }
        }
    }
}
