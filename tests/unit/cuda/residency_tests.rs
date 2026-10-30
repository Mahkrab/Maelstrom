//! Pointer promotion must follow acceptance, while failed attempts retain the
//! preceding device inputs. The serial solver supplies an independent trajectory.

use super::*;
use crate::simulation::Simulation;

fn bits(particles: &[Particle]) -> Vec<(u64, [u32; 3], [u32; 3])> {
    particles
        .iter()
        .map(|p| {
            (
                p.identity,
                array(p.position_m).map(f32::to_bits),
                array(p.velocity_m_s).map(f32::to_bits),
            )
        })
        .collect()
}

fn device_vectors(gpu: &mut CudaSolver, slot: u32, count: usize) -> Vec<[u32; 3]> {
    gpu.context
        .inner
        .download::<[f32; 3]>(slot, count)
        .unwrap()
        .into_iter()
        .map(|v| v.map(f32::to_bits))
        .collect()
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_resident_inputs_promote_after_acceptance_and_survive_retry() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let plane = Plane::new(Vec3::new(0., 1., 0.), 0., 0.).unwrap();
    for threads in [32, 64, 128, 256] {
        for count in [0, 1, 33] {
            let mut particles: Vec<_> = (0..count)
                .map(|i| Particle {
                    identity: 2 * i as u64 + 1,
                    position_m: Vec3::new(i as f32, 0.01, 0.),
                    velocity_m_s: Vec3::new(0.25, -0.5, 0.),
                })
                .collect();
            let mut cpu = Simulation::new(&config, particles.clone(), vec![plane]).unwrap();
            let mut gpu = CudaSolver::new(0, threads).unwrap();
            for step in 0..6 {
                let old = bits(&particles);
                let output = gpu
                    .step(
                        &config,
                        &particles,
                        &[plane],
                        &mut || false,
                        &mut StageClock::<false>::new(),
                    )
                    .unwrap()
                    .particles;
                let _ = cpu.step().unwrap();
                assert_eq!(bits(&output), bits(cpu.snapshot().particles()));
                assert_eq!(
                    device_vectors(&mut gpu, 0, count),
                    old.iter().map(|p| p.1).collect::<Vec<_>>()
                );
                assert_eq!(
                    device_vectors(&mut gpu, 1, count),
                    old.iter().map(|p| p.2).collect::<Vec<_>>()
                );
                gpu.accept().unwrap();
                let accepted = bits(&output);
                assert_eq!(
                    device_vectors(&mut gpu, 0, count),
                    accepted.iter().map(|p| p.1).collect::<Vec<_>>()
                );
                assert_eq!(
                    device_vectors(&mut gpu, 1, count),
                    accepted.iter().map(|p| p.2).collect::<Vec<_>>()
                );
                assert_eq!(
                    device_vectors(&mut gpu, 2, count),
                    old.iter().map(|p| p.1).collect::<Vec<_>>()
                );
                assert_eq!(
                    device_vectors(&mut gpu, 3, count),
                    old.iter().map(|p| p.2).collect::<Vec<_>>()
                );
                assert!(gpu.accept().is_err(), "a second promotion must be rejected");
                let failed = gpu.step(
                    &config,
                    &output,
                    &[plane],
                    &mut || true,
                    &mut StageClock::<false>::new(),
                );
                assert!(matches!(failed, Err(StepError::Cancelled)));
                assert_eq!(
                    device_vectors(&mut gpu, 0, count),
                    accepted.iter().map(|p| p.1).collect::<Vec<_>>()
                );
                assert_eq!(
                    device_vectors(&mut gpu, 1, count),
                    accepted.iter().map(|p| p.2).collect::<Vec<_>>()
                );
                if step == 0 {
                    let mut polls = 0;
                    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        gpu.step(
                            &config,
                            &output,
                            &[plane],
                            &mut || {
                                polls += 1;
                                assert_ne!(polls, 2, "injected panic after prediction");
                                false
                            },
                            &mut StageClock::<false>::new(),
                        )
                    }));
                    assert!(panic.is_err());
                    assert_eq!(
                        device_vectors(&mut gpu, 0, count),
                        accepted.iter().map(|p| p.1).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        device_vectors(&mut gpu, 1, count),
                        accepted.iter().map(|p| p.2).collect::<Vec<_>>()
                    );
                }
                particles = output;
            }
        }
    }
}
