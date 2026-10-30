//! Warp-varying pressure inputs must preserve the ordered CPU correction,
//! including fixed neighbours that leave support or become coincident.

use super::*;

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_corrections_match_ordered_cpu_for_divergent_power_inputs() {
    use crate::{fluid::compute_jacobi_stage, neighbours::brute_force};

    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for pairs in [1, 16, 17, 65] {
            let original: Vec<_> = (0..pairs)
                .flat_map(|pair| {
                    let x = (pair * 3) as f32;
                    let distance = (pair % 31 + 1) as f32 / 32.0;
                    [Vec3::new(x, 0.0, 0.0), Vec3::new(x + distance, 0.0, 0.0)]
                })
                .collect();
            let particles: Vec<_> = original
                .iter()
                .enumerate()
                .map(|(i, &position)| Particle {
                    identity: i as u64,
                    position_m: position,
                    velocity_m_s: Vec3::ZERO,
                })
                .collect();
            let mut s = settings(&config, particles.len(), 0).unwrap();
            s.radius = 1.0;
            s.rest_density = 3.0;
            s.relaxation = 0.01;
            s.strength = 0.1;
            s.reference_weight = crate::kernel::poly6(0.2, 1.0).unwrap();
            gpu.prepare(s, &particles, &[]).unwrap();
            gpu.context
                .inner
                .upload(2, &original.iter().copied().map(array).collect::<Vec<_>>())
                .unwrap();
            gpu.discover(&mut s, &original).unwrap();
            let fixed = brute_force(&original, 1.0).unwrap();
            for altered in [false, true, false] {
                let mut positions = original.clone();
                if altered {
                    for pair in 0..pairs {
                        if pair % 5 == 0 {
                            positions[pair * 2 + 1].x = positions[pair * 2].x + 1.0;
                        } else if pair % 7 == 0 {
                            positions[pair * 2 + 1] = positions[pair * 2];
                        }
                    }
                }
                gpu.context
                    .inner
                    .upload(2, &positions.iter().copied().map(array).collect::<Vec<_>>())
                    .unwrap();
                for exponent in [0.5, 1.0, 1.25, 4.0, 16.0, 64.0] {
                    s.exponent = exponent;
                    for op in [5, 6, 7] {
                        gpu.operation(s, op).unwrap();
                    }
                    let expected = compute_jacobi_stage(
                        &positions,
                        &fixed,
                        crate::fluid::PbfParameters::new(1.0, 3.0, 0.01, 0.1, 0.2, exponent)
                            .unwrap(),
                    )
                    .unwrap();
                    let actual: Vec<[f32; 3]> =
                        gpu.context.inner.download(7, positions.len()).unwrap();
                    assert_eq!(
                        actual
                            .into_iter()
                            .map(|v| v.map(f32::to_bits))
                            .collect::<Vec<_>>(),
                        expected
                            .corrections
                            .into_iter()
                            .map(array)
                            .map(|v| v.map(f32::to_bits))
                            .collect::<Vec<_>>(),
                        "block {threads}, pairs {pairs}, altered {altered}, exponent {exponent}"
                    );
                }
            }
        }
    }
}
