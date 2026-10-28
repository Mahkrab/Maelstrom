//! Mapped error observations must preserve both record slots across contention,
//! failed attempts and resets. Expected records come from the injected rows.

use super::*;

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_mapped_diagnostics_keep_minima_slots_and_reset_after_rejection() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 2, 31, 32, 33, 257, 513, 8000] {
            let particles: Vec<_> = (0..count)
                .map(|i| Particle {
                    identity: i as u64,
                    position_m: Vec3::ZERO,
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let mut s = settings(&config, count, 0).unwrap();
            s.acceleration = [0.; 3];
            s.timestep = 2.;
            gpu.prepare(s, &particles, &[]).unwrap();
            for first in [count / 3, 0, count.saturating_sub(1)] {
                let mut velocities = vec![[0.0_f32; 3]; count];
                for velocity in &mut velocities[first..] {
                    *velocity = [f32::MAX, 0., 0.];
                }
                gpu.context.inner.upload(1, &velocities).unwrap();
                let result = gpu.context.inner.observe(s, 0, threads).unwrap();
                let expected = if count == 0 {
                    u64::MAX
                } else {
                    ((first as u64) << 32) | 2
                };
                assert_eq!(result, [u64::MAX, expected]);
                gpu.context.inner.begin_attempt().unwrap();
                let mut position = vec![[0.0_f32; 3]; count];
                if count != 0 {
                    position[first] = [f32::NAN, 0., 0.];
                }
                gpu.context.inner.upload(0, &position).unwrap();
                gpu.context
                    .inner
                    .upload(3, &vec![[0.0_f32; 3]; count])
                    .unwrap();
                let result = gpu.context.inner.observe(s, 1, threads).unwrap();
                let expected = if count == 0 {
                    u64::MAX
                } else {
                    ((first as u64) << 32) | 22
                };
                assert_eq!(result, [expected, u64::MAX]);
                gpu.context.inner.begin_attempt().unwrap();
                gpu.context
                    .inner
                    .upload(0, &vec![[0.0_f32; 3]; count])
                    .unwrap();
                gpu.context
                    .inner
                    .upload(1, &vec![[0.0_f32; 3]; count])
                    .unwrap();
                assert_eq!(
                    gpu.context.inner.observe(s, 0, threads).unwrap(),
                    [u64::MAX; 2]
                );
                assert_eq!(
                    gpu.context.inner.observe(s, 1, threads).unwrap(),
                    [u64::MAX; 2]
                );
            }
        }
    }
}
