//! A hand-calculated particle trajectory makes skipped or duplicated worker
//! steps visible without using another solver invocation as the oracle.

use maelstrom::{
    application::{Action, Controller, RunSpec, RunStatus},
    config::{parse_configuration_overlay, resolve_effective_configuration},
    execution::ExecutionMode,
    math::Vec3,
    neighbours::SearchBackend,
    scene::{Particle, RandomnessPolicy, ResolvedScene},
};

#[test]
fn worker_single_steps_match_exact_constant_acceleration_motion() {
    let configuration = resolve_effective_configuration(
        None,
        None,
        Some(
            &parse_configuration_overlay(
                "[numerical]\nfixed_timestep_s=0.25\nexternal_acceleration_m_s2=[0.0,-8.0,0.0]",
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let scene = ResolvedScene {
        schema_version: 1,
        name: "one particle".into(),
        description: String::new(),
        randomness: RandomnessPolicy::None,
        configuration,
        planes: vec![],
        particles: vec![Particle {
            identity: u64::MAX,
            position_m: Vec3::new(0.0, 0.5, 0.0),
            velocity_m_s: Vec3::new(2.0, 0.0, 0.0),
        }],
    };
    let mut controller = Controller::new().unwrap();
    let (handle, start) = controller
        .start(RunSpec {
            scene,
            backend: SearchBackend::BruteForce,
            execution: ExecutionMode::Serial,
            steps: 2,
            initially_paused: true,
        })
        .unwrap();
    assert_eq!(start.wait().unwrap(), RunStatus::Paused);
    for (step, position, velocity, status) in [
        (
            1,
            [0.5_f32, 0.0, 0.0],
            [2.0_f32, -2.0, 0.0],
            RunStatus::Paused,
        ),
        (
            2,
            [1.0_f32, -1.0, 0.0],
            [2.0_f32, -4.0, 0.0],
            RunStatus::Completed,
        ),
    ] {
        assert_eq!(
            controller
                .control(&handle, Action::SingleStep)
                .unwrap()
                .wait()
                .unwrap(),
            status
        );
        let observation = handle.observe();
        let snapshot = observation.snapshot.unwrap();
        assert_eq!(snapshot.completed_steps(), step);
        let particle = snapshot.particles()[0];
        assert_eq!(particle.identity, u64::MAX);
        assert_eq!(
            [
                particle.position_m.x,
                particle.position_m.y,
                particle.position_m.z
            ]
            .map(f32::to_bits),
            position.map(f32::to_bits)
        );
        assert_eq!(
            [
                particle.velocity_m_s.x,
                particle.velocity_m_s.y,
                particle.velocity_m_s.z
            ]
            .map(f32::to_bits),
            velocity.map(f32::to_bits)
        );
    }
    controller.join().unwrap();
}
