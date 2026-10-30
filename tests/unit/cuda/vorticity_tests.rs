//! Direct device stages check separate output and retry after numerical failure,
//! including final-position changes that the fixed prediction rows cannot know.

use super::*;
use crate::{
    fluid::InteractionSets,
    velocity::{VorticityParameters, vorticity_confinement},
};

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn confinement_stages_preserve_input_and_retry_both_numerical_failure_paths() {
    for threads in [32, 64, 128, 256] {
        let configuration =
            crate::config::resolve_effective_configuration(None, None, None).unwrap();
        let initial = [
            Vec3::ZERO,
            Vec3::new(0.25, 0., 0.),
            Vec3::new(0.5, 0.125, 0.),
            Vec3::new(0., 0.25, 0.125),
        ];
        let velocities = [
            Vec3::ZERO,
            Vec3::new(0., 0.25, 0.125),
            Vec3::new(-0.25, 0.5, 0.),
            Vec3::new(0.125, 0., -0.125),
        ];
        let particles: Vec<_> = initial
            .iter()
            .enumerate()
            .map(|(i, &p)| Particle {
                identity: i as u64,
                position_m: p,
                velocity_m_s: velocities[i],
            })
            .collect();
        let rows = InteractionSets::new(
            (0..4)
                .map(|i| (0..4).filter(|&j| i != j).collect())
                .collect(),
        )
        .unwrap();
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        let mut s = settings(&configuration, 4, 0).unwrap();
        s.radius = 1.;
        s.rest_density = 4.;
        s.vorticity = 0.125;
        s.timestep = 0.0625;
        gpu.prepare(s, &particles, &[]).unwrap();
        gpu.context.inner.upload(2, &initial.map(array)).unwrap();
        gpu.discover(&mut s, &initial).unwrap();
        for distance in [
            0.,
            0.25,
            f32::from_bits(1f32.to_bits() - 1),
            1.,
            f32::from_bits(1f32.to_bits() + 1),
        ] {
            let mut positions = initial;
            positions[1] = Vec3::new(distance, 0., 0.);
            gpu.context.inner.upload(2, &positions.map(array)).unwrap();
            gpu.context.inner.upload(3, &velocities.map(array)).unwrap();
            gpu.operation(s, 16).unwrap();
            gpu.operation(s, 17).unwrap();
            let before: Vec<[f32; 3]> = gpu.context.inner.download(3, 4).unwrap();
            assert_eq!(before, velocities.map(array));
            let actual: Vec<[f32; 3]> = gpu.context.inner.download(8, 4).unwrap();
            let expected = vorticity_confinement(
                &positions,
                &velocities,
                &rows,
                VorticityParameters {
                    radius_m: 1.,
                    rest_density_per_m3: 4.,
                    strength_m_s: s.vorticity,
                    timestep_s: s.timestep,
                },
            )
            .unwrap();
            for (a, e) in actual.iter().zip(&expected) {
                assert_eq!(a.map(f32::to_bits), array(*e).map(f32::to_bits));
            }
            gpu.operation(s, 15).unwrap();
            let copied: Vec<[f32; 3]> = gpu.context.inner.download(3, 4).unwrap();
            assert_eq!(copied, actual);
        }
        for failing_operation in [16, 17] {
            gpu.context.inner.upload(2, &initial.map(array)).unwrap();
            let mut bad = velocities;
            if failing_operation == 16 {
                bad[0].x = f32::MAX;
                bad[1].x = -f32::MAX;
            }
            gpu.context.inner.upload(3, &bad.map(array)).unwrap();
            let mut invalid = s;
            if failing_operation == 17 {
                invalid.vorticity = f32::MAX;
                invalid.timestep = f32::MAX;
                gpu.operation(invalid, 16).unwrap();
            }
            assert!(matches!(
                gpu.operation(invalid, failing_operation),
                Err(StepError::Numerical(_))
            ));
            let unchanged: Vec<[f32; 3]> = gpu.context.inner.download(3, 4).unwrap();
            assert_eq!(unchanged, bad.map(array));
            gpu.context.inner.begin_attempt().unwrap();
            gpu.context.inner.upload(3, &velocities.map(array)).unwrap();
            gpu.operation(s, 16).unwrap();
            gpu.operation(s, 17).unwrap();
            gpu.operation(s, 15).unwrap();
            let expected = vorticity_confinement(
                &initial,
                &velocities,
                &rows,
                VorticityParameters {
                    radius_m: 1.,
                    rest_density_per_m3: 4.,
                    strength_m_s: s.vorticity,
                    timestep_s: s.timestep,
                },
            )
            .unwrap();
            let actual: Vec<[f32; 3]> = gpu.context.inner.download(3, 4).unwrap();
            for (a, e) in actual.iter().zip(expected) {
                assert_eq!(a.map(f32::to_bits), array(e).map(f32::to_bits));
            }
        }
    }
}
