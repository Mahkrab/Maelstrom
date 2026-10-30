//! Compact observations must agree with the CPU's binary32 diagnostics and leave
//! accepted vectors untouched. Prefix fixtures are independent of neighbour search.

use super::*;
use crate::diagnostics::cfl_diagnostic;

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_compact_summaries_keep_lengths_extrema_and_stage_priority() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 2, 31, 32, 33, 127, 128, 129, 1003] {
            let values = [
                0.0_f32,
                -0.0,
                f32::from_bits(1),
                1.0e-20,
                -0.125,
                0.75,
                1.0e19,
            ];
            let particles: Vec<_> = (0..count)
                .map(|i| Particle {
                    identity: i as u64,
                    position_m: Vec3::new((i as f32 - 7.0) * 1.0e30, -0.0, 0.25),
                    velocity_m_s: Vec3::new(values[i % values.len()], 0.0, 0.0),
                })
                .collect();
            let mut s = settings(&config, count, 0).unwrap();
            s.acceleration = [0.0; 3];
            gpu.prepare(s, &particles, &[]).unwrap();
            let records = gpu.context.inner.observe(s, 0, threads).unwrap();
            assert_eq!(records, [u64::MAX; 2]);
            let velocities: Vec<_> = particles.iter().map(|p| p.velocity_m_s).collect();
            let cpu = cfl_diagnostic(
                &velocities,
                s.timestep,
                config.derived.particle_diameter_m,
                config.numerical.cfl_factor,
            )
            .unwrap();
            let blocks = count.div_ceil(threads as usize);
            let observed: Vec<[f32; 8]> = gpu.context.inner.download(7, blocks).unwrap();
            let maximum = observed.iter().map(|s| s[0]).fold(0.0_f32, f32::max);
            assert_eq!(maximum.to_bits(), cpu.maximum_speed.to_bits());
            assert_eq!(
                cfl_from_maximum_speed(
                    maximum,
                    s.timestep,
                    config.derived.particle_diameter_m,
                    config.numerical.cfl_factor
                )
                .unwrap(),
                cpu
            );
            assert_eq!(
                gpu.context.inner.observe(s, 1, threads).unwrap(),
                [u64::MAX; 2]
            );
            let predicted: Vec<_> = particles
                .iter()
                .map(|p| p.position_m + p.velocity_m_s * s.timestep)
                .collect();
            let summaries: Vec<[f32; 8]> = gpu.context.inner.download(7, blocks).unwrap();
            for (summary, points) in summaries.iter().zip(predicted.chunks(threads as usize)) {
                let maximum = points
                    .iter()
                    .map(|p| p.x.abs().max(p.y.abs()).max(p.z.abs()))
                    .fold(0.0_f32, f32::max);
                assert_eq!(summary[0].to_bits(), maximum.to_bits());
                for axis in 0..3 {
                    let minimum = points
                        .iter()
                        .map(|p| array(*p)[axis])
                        .fold(f32::MAX, f32::min);
                    let maximum = points
                        .iter()
                        .map(|p| array(*p)[axis])
                        .fold(-f32::MAX, f32::max);
                    assert_eq!(summary[axis + 1], minimum);
                    assert_eq!(summary[axis + 4], maximum);
                }
            }
            let accepted: Vec<[f32; 3]> = gpu.context.inner.download(1, count).unwrap();
            assert_eq!(
                accepted,
                particles
                    .iter()
                    .map(|p| array(p.velocity_m_s))
                    .collect::<Vec<_>>()
            );
        }
        // An acceleration error at a later particle must beat an earlier CFL
        // squared-length overflow. The two error slots must survive readback.
        let particles = vec![
            Particle {
                identity: 0,
                position_m: Vec3::ZERO,
                velocity_m_s: Vec3::ZERO,
            },
            Particle {
                identity: 1,
                position_m: Vec3::ZERO,
                velocity_m_s: Vec3::new(f32::MAX, 0.0, 0.0),
            },
        ];
        let mut s = settings(&config, 2, 0).unwrap();
        s.timestep = 1.0;
        s.acceleration = [f32::MAX / 2.0, 0.0, 0.0];
        gpu.prepare(s, &particles, &[]).unwrap();
        let records = gpu.context.inner.observe(s, 0, threads).unwrap();
        assert_eq!(records, [(1_u64 << 32) | 20, 2]);
        assert!(gpu.accept().is_err());
        gpu.context.inner.begin_attempt().unwrap();
        s.acceleration = [0.0; 3];
        let records = gpu.context.inner.observe(s, 0, threads).unwrap();
        assert_eq!(records, [u64::MAX, (1_u64 << 32) | 2]);
        assert!(gpu.accept().is_err());
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_checked_prefix_matches_integer_oracle_and_rejects_bad_counts() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [
            0, 1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257, 4097, 110592, 33,
        ] {
            let particles = vec![
                Particle {
                    identity: 0,
                    position_m: Vec3::ZERO,
                    velocity_m_s: Vec3::ZERO
                };
                count
            ];
            let s = settings(&config, count, 0).unwrap();
            gpu.prepare(s, &particles, &[]).unwrap();
            let mut counts = vec![0_u64];
            counts.extend((0..count).map(|i| ((i * 7) % count) as u64));
            let mut expected = Vec::with_capacity(count + 1);
            let mut sum = 0;
            for value in &counts {
                sum += value;
                expected.push(sum);
            }
            gpu.context.inner.upload(13, &counts).unwrap();
            assert_eq!(
                gpu.context.inner.prefix(s, threads, false).unwrap(),
                u64::MAX
            );
            let actual: Vec<u64> = gpu.context.inner.download(13, count + 1).unwrap();
            assert_eq!(actual, expected, "count {count}, block {threads}");
            assert_eq!(
                gpu.context.inner.download_at::<u64>(13, count, 1).unwrap(),
                vec![sum]
            );
        }
        let empty = settings(&config, 0, 0).unwrap();
        gpu.prepare(empty, &[], &[]).unwrap();
        gpu.context.inner.upload(13, &[1_u64]).unwrap();
        assert_eq!(gpu.context.inner.prefix(empty, threads, false).unwrap(), 52);
        let count = 33;
        let particles = vec![
            Particle {
                identity: 0,
                position_m: Vec3::ZERO,
                velocity_m_s: Vec3::ZERO
            };
            count
        ];
        let s = settings(&config, count, 0).unwrap();
        for index in [0, 1, 17, 33] {
            gpu.prepare(s, &particles, &[]).unwrap();
            let mut counts = vec![0_u64; count + 1];
            counts[index] = u64::MAX;
            gpu.context.inner.upload(13, &counts).unwrap();
            let record = gpu.context.inner.prefix(s, threads, false).unwrap();
            assert_eq!(record, ((index.saturating_sub(1) as u64) << 32) | 52);
            assert!(gpu.accept().is_err());
        }
    }
}
