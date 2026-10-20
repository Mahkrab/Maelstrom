//! Controls must change when work runs, never the accepted numerical request.
//! Barriers and acknowledgements determine ordering; timeouts only detect a
//! stranded worker and are not used to make a scheduling assertion pass.

use std::{path::Path, sync::Arc, time::Duration};

use maelstrom::{
    application::{
        Action, ControlError, Controller, FailureKind, Observation, RunHandle, RunSpec, RunStatus,
    },
    config::{parse_configuration_overlay, resolve_effective_configuration},
    execution::ExecutionMode,
    math::Vec3,
    neighbours::SearchBackend,
    scene::{Particle, RandomnessPolicy, ResolvedScene, load_resolved_scene},
    simulation::{Simulation, Snapshot},
};

fn spec(steps: u64) -> RunSpec {
    RunSpec {
        scene: load_resolved_scene(Path::new("scenes/falling-block-04.toml"), None).unwrap(),
        backend: SearchBackend::UniformGrid,
        execution: ExecutionMode::Serial,
        steps,
        initially_paused: true,
    }
}

fn terminal(handle: &RunHandle) -> Observation {
    handle
        .wait_for_terminal(Duration::from_secs(30))
        .expect("worker did not reach a terminal state")
}

fn bits(snapshot: &Snapshot) -> Vec<(u64, [u32; 6])> {
    snapshot
        .particles()
        .iter()
        .map(|p| {
            (
                p.identity,
                [
                    p.position_m.x,
                    p.position_m.y,
                    p.position_m.z,
                    p.velocity_m_s.x,
                    p.velocity_m_s.y,
                    p.velocity_m_s.z,
                ]
                .map(f32::to_bits),
            )
        })
        .collect()
}

fn direct(spec: &RunSpec, steps: u64) -> Simulation {
    let mut solver =
        Simulation::from_scene_with_execution(&spec.scene, spec.backend, spec.execution).unwrap();
    for _ in 0..steps {
        let _ = solver.step().unwrap();
    }
    solver
}

fn assert_rejected(controller: &Controller, handle: &RunHandle, action: Action, state: RunStatus) {
    assert!(matches!(controller.control(handle, action).unwrap().wait(),
        Err(ControlError::InvalidTransition { status, action: rejected }) if status == state && rejected == action));
}

#[test]
fn pause_step_resume_and_completion_preserve_every_accepted_bit() {
    for (backend, execution) in [
        (SearchBackend::BruteForce, ExecutionMode::Serial),
        (SearchBackend::UniformGrid, ExecutionMode::Serial),
        (
            SearchBackend::UniformGrid,
            ExecutionMode::Threaded { workers: 2 },
        ),
    ] {
        let mut request = spec(5);
        request.backend = backend;
        request.execution = execution;
        let expected = direct(&request, 5);
        let mut controller = Controller::new().unwrap();
        let (handle, started) = controller.start(request.clone()).unwrap();
        assert_eq!(started.wait().unwrap(), RunStatus::Paused);
        let initial = handle.observe();
        assert_eq!(initial.snapshot.as_ref().unwrap().completed_steps(), 0);
        assert_eq!(initial.simulated_time_s(), Some(0.0));
        assert_eq!(initial.info.as_ref().unwrap().execution, execution);
        assert_eq!(initial.info.as_ref().unwrap().backend, backend);
        assert!(initial.info.as_ref().unwrap().device.is_none());
        assert_rejected(&controller, &handle, Action::Pause, RunStatus::Paused);
        for index in 1..=3 {
            assert_eq!(
                controller
                    .control(&handle, Action::SingleStep)
                    .unwrap()
                    .wait()
                    .unwrap(),
                RunStatus::Paused
            );
            let observed = handle.observe();
            assert_eq!(observed.snapshot.as_ref().unwrap().completed_steps(), index);
            let prefix = direct(&request, index);
            assert_eq!(
                bits(observed.snapshot.as_ref().unwrap()),
                bits(prefix.snapshot())
            );
            assert_eq!(
                observed.simulated_time_s(),
                Some(
                    index as f64
                        * f64::from(request.scene.configuration.numerical.fixed_timestep_s)
                )
            );
        }
        assert_eq!(initial.snapshot.as_ref().unwrap().completed_steps(), 0);
        assert_eq!(
            controller
                .control(&handle, Action::Resume)
                .unwrap()
                .wait()
                .unwrap(),
            RunStatus::Running
        );
        let final_state = terminal(&handle);
        assert_eq!(final_state.status, RunStatus::Completed);
        assert_eq!(final_state.snapshot.as_ref().unwrap().completed_steps(), 5);
        assert_eq!(
            bits(final_state.snapshot.as_ref().unwrap()),
            bits(expected.snapshot())
        );
        for action in [Action::Pause, Action::Resume, Action::SingleStep] {
            assert_rejected(&controller, &handle, action, RunStatus::Completed);
        }
        controller.join().unwrap();
        assert_eq!(handle.observe().status, RunStatus::Completed);
    }
}

#[test]
fn pause_acknowledgement_is_a_stable_boundary_and_running_step_is_rejected() {
    let mut request = spec(u64::MAX);
    request.scene.particles.clear();
    let mut controller = Controller::new().unwrap();
    let (handle, started) = controller.start(request).unwrap();
    started.wait().unwrap();
    controller
        .control(&handle, Action::Resume)
        .unwrap()
        .wait()
        .unwrap();
    assert_rejected(&controller, &handle, Action::Resume, RunStatus::Running);
    assert_rejected(&controller, &handle, Action::SingleStep, RunStatus::Running);
    assert_eq!(
        controller
            .control(&handle, Action::Pause)
            .unwrap()
            .wait()
            .unwrap(),
        RunStatus::Paused
    );
    let paused = handle.observe();
    // A controlled display clock jumps backwards, forwards and repeats. Reads
    // cannot advance a paused solver, regardless of the presentation timestamps.
    for _display_time_ns in [0_u128, 16_000_000, 16_000_000, u128::MAX, 1] {
        assert_eq!(handle.observe().snapshot, paused.snapshot);
    }
    controller
        .control(&handle, Action::SingleStep)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        handle.observe().snapshot.unwrap().completed_steps(),
        paused.snapshot.unwrap().completed_steps() + 1
    );
    controller.join().unwrap();
    assert_eq!(terminal(&handle).status, RunStatus::Cancelled);
}

#[test]
fn slow_or_absent_display_cannot_backpressure_solver_or_lose_final_diagnostics() {
    let mut request = spec(20);
    request.initially_paused = false;
    let mut expected =
        Simulation::from_scene_with_backend(&request.scene, request.backend).unwrap();
    let outcomes = expected.run_steps(20).unwrap();
    let warnings = outcomes.iter().filter(|o| o.cfl.warning).count() as u64;
    let mut controller = Controller::new().unwrap();
    let (handle, started) = controller.start(request).unwrap();
    // Deliberately consume no display observation until the whole run finishes.
    drop(started);
    let end = terminal(&handle);
    assert_eq!(end.status, RunStatus::Completed);
    assert_eq!(end.warning_count, warnings);
    assert_eq!(end.last_cfl, Some(outcomes.last().unwrap().cfl));
    assert_eq!(end.snapshot.as_ref().unwrap().completed_steps(), 20);
    assert_eq!(
        bits(end.snapshot.as_ref().unwrap()),
        bits(expected.snapshot())
    );
    assert!(Arc::ptr_eq(
        end.snapshot.as_ref().unwrap(),
        handle.observe().snapshot.as_ref().unwrap()
    ));
    controller.join().unwrap();
}

#[test]
fn restart_isolates_old_snapshots_commands_and_cancellation() {
    let mut controller = Controller::new().unwrap();
    let first_spec = spec(5);
    let (old, started) = controller.start(first_spec).unwrap();
    started.wait().unwrap();
    controller
        .control(&old, Action::SingleStep)
        .unwrap()
        .wait()
        .unwrap();
    let delayed = old.observe();
    let mut replacement = spec(2);
    replacement.execution = ExecutionMode::Threaded { workers: 1 };
    let expected = direct(&replacement, 2);
    let (new, started) = controller.start(replacement).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Paused);
    assert_eq!(terminal(&old).status, RunStatus::Cancelled);
    assert!(matches!(
        controller.control(&old, Action::Resume),
        Err(ControlError::StaleGeneration)
    ));
    old.cancel();
    assert_eq!(new.observe().generation, delayed.generation + 1);
    assert_eq!(new.observe().snapshot.unwrap().completed_steps(), 0);
    controller
        .control(&new, Action::Resume)
        .unwrap()
        .wait()
        .unwrap();
    let end = terminal(&new);
    assert_eq!(end.status, RunStatus::Completed);
    assert_eq!(
        bits(end.snapshot.as_ref().unwrap()),
        bits(expected.snapshot())
    );
    assert_eq!(delayed.snapshot.unwrap().completed_steps(), 1);
    assert_eq!(old.observe().snapshot.unwrap().completed_steps(), 1);
    controller.join().unwrap();
}

#[test]
fn cancellation_wakes_paused_worker_is_latched_and_retains_an_exact_prefix() {
    for running in [false, true] {
        let mut request = spec(u64::MAX);
        request.initially_paused = !running;
        let mut controller = Controller::new().unwrap();
        let (handle, started) = controller.start(request.clone()).unwrap();
        started.wait().unwrap();
        handle.cancel();
        handle.cancel();
        let cancelled = terminal(&handle);
        assert_eq!(cancelled.status, RunStatus::Cancelled);
        let snapshot = cancelled.snapshot.as_ref().unwrap();
        assert_eq!(
            bits(snapshot),
            bits(direct(&request, snapshot.completed_steps()).snapshot())
        );
        assert_rejected(&controller, &handle, Action::Resume, RunStatus::Cancelled);
        assert_eq!(handle.observe().snapshot, cancelled.snapshot);
        controller.join().unwrap();
    }
}

#[test]
fn invalid_setup_retains_previous_outcome_and_never_selects_a_fallback() {
    let mut controller = Controller::new().unwrap();
    let (old, started) = controller.start(spec(0)).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Completed);
    for invalid in [
        {
            let mut s = spec(1);
            s.execution = ExecutionMode::Threaded { workers: 0 };
            s
        },
        {
            let mut s = spec(1);
            s.backend = SearchBackend::BruteForce;
            s.execution = ExecutionMode::Threaded { workers: 2 };
            s
        },
        {
            let mut s = spec(1);
            s.scene.particles[0].position_m.x = f32::NAN;
            s
        },
        {
            let mut s = spec(1);
            s.scene.schema_version = 99;
            s
        },
    ] {
        let (failed, started) = controller.start(invalid).unwrap();
        assert_eq!(started.wait().unwrap(), RunStatus::Failed);
        let outcome = terminal(&failed);
        assert!(outcome.snapshot.is_none());
        assert!(outcome.info.is_none());
        assert_eq!(outcome.failure.unwrap().kind, FailureKind::Construction);
        assert_eq!(old.observe().status, RunStatus::Completed);
        assert_rejected(&controller, &failed, Action::Resume, RunStatus::Failed);
    }
    controller.join().unwrap();
}

#[test]
fn numerical_failure_retains_accepted_state_and_failure_across_controls_and_shutdown() {
    let configuration = resolve_effective_configuration(
        None,
        None,
        Some(
            &parse_configuration_overlay(
                "[numerical]\nfixed_timestep_s=1.0\nexternal_acceleration_m_s2=[1.0e19,0.0,0.0]",
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let scene = ResolvedScene {
        schema_version: 1,
        name: "overflow".into(),
        description: String::new(),
        randomness: RandomnessPolicy::None,
        configuration,
        planes: vec![],
        particles: vec![Particle {
            identity: 99,
            position_m: Vec3::ZERO,
            velocity_m_s: Vec3::ZERO,
        }],
    };
    let request = RunSpec {
        scene,
        backend: SearchBackend::BruteForce,
        execution: ExecutionMode::Serial,
        steps: 3,
        initially_paused: true,
    };
    let mut controller = Controller::new().unwrap();
    let (handle, started) = controller.start(request).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Paused);
    assert_eq!(
        controller
            .control(&handle, Action::SingleStep)
            .unwrap()
            .wait()
            .unwrap(),
        RunStatus::Paused
    );
    let before = handle.observe();
    assert_eq!(before.snapshot.as_ref().unwrap().completed_steps(), 1);
    assert_eq!(
        controller
            .control(&handle, Action::SingleStep)
            .unwrap()
            .wait()
            .unwrap(),
        RunStatus::Failed
    );
    let failed = terminal(&handle);
    assert_eq!(failed.snapshot, before.snapshot);
    assert_eq!(failed.warning_count, 1);
    assert_eq!(failed.last_cfl, before.last_cfl);
    assert_eq!(
        failed.failure.as_ref().unwrap().kind,
        FailureKind::Simulation
    );
    assert_rejected(&controller, &handle, Action::SingleStep, RunStatus::Failed);
    handle.cancel();
    controller.join().unwrap();
    assert_eq!(handle.observe().failure, failed.failure);
    assert_eq!(handle.observe().status, RunStatus::Failed);
}

#[test]
fn zero_step_empty_input_and_last_single_step_have_explicit_completion() {
    let mut controller = Controller::new().unwrap();
    let mut request = spec(0);
    request.scene.particles.clear();
    let (empty, started) = controller.start(request).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Completed);
    assert!(terminal(&empty).snapshot.unwrap().particles().is_empty());
    let (one, started) = controller.start(spec(1)).unwrap();
    started.wait().unwrap();
    assert_eq!(
        controller
            .control(&one, Action::SingleStep)
            .unwrap()
            .wait()
            .unwrap(),
        RunStatus::Completed
    );
    assert_eq!(terminal(&one).snapshot.unwrap().completed_steps(), 1);
    controller.join().unwrap();
}

#[test]
fn shutdown_and_drop_join_idle_paused_running_and_queued_work() {
    Controller::new().unwrap().join().unwrap();
    for paused in [false, true] {
        let mut request = spec(u64::MAX);
        request.initially_paused = paused;
        let mut controller = Controller::new().unwrap();
        let (handle, started) = controller.start(request).unwrap();
        started.wait().unwrap();
        controller.request_shutdown();
        assert!(matches!(
            controller.start(spec(0)),
            Err(ControlError::Stopped)
        ));
        controller.join().unwrap();
        assert_eq!(terminal(&handle).status, RunStatus::Cancelled);
    }
    let mut controller = Controller::new().unwrap();
    let (queued, receipt) = controller.start(spec(u64::MAX)).unwrap();
    drop(receipt);
    drop(controller);
    assert_eq!(terminal(&queued).status, RunStatus::Cancelled);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_selection_on_cpu_build_fails_explicitly_on_the_worker() {
    let mut request = spec(1);
    request.execution = ExecutionMode::Cuda {
        device: 0,
        threads: 128,
    };
    let mut controller = Controller::new().unwrap();
    let (handle, started) = controller.start(request).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Failed);
    let failure = terminal(&handle).failure.unwrap();
    assert_eq!(failure.kind, FailureKind::Construction);
    assert!(failure.message.contains("cuda"));
    controller.join().unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an available qualified CUDA device"]
fn cuda_context_lives_entirely_on_worker_across_step_restart_cancel_and_teardown() {
    let mut request = spec(3);
    request.execution = ExecutionMode::Cuda {
        device: 0,
        threads: 128,
    };
    let mut cpu = request.clone();
    cpu.execution = ExecutionMode::Serial;
    let expected = direct(&cpu, 3);
    let mut controller = Controller::new().unwrap();
    let (old, started) = controller.start(request.clone()).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Paused);
    controller
        .control(&old, Action::SingleStep)
        .unwrap()
        .wait()
        .unwrap();
    let (new, started) = controller.start(request).unwrap();
    assert_eq!(started.wait().unwrap(), RunStatus::Paused);
    assert_eq!(terminal(&old).status, RunStatus::Cancelled);
    controller
        .control(&new, Action::Resume)
        .unwrap()
        .wait()
        .unwrap();
    let outcome = terminal(&new);
    assert_eq!(outcome.status, RunStatus::Completed);
    assert!(outcome.info.as_ref().unwrap().device.is_some());
    assert_eq!(
        bits(outcome.snapshot.as_ref().unwrap()),
        bits(expected.snapshot())
    );
    controller.join().unwrap();
}
