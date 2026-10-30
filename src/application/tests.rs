//! Hold dispatch at a barrier to exercise full queues and obsolete commands
//! deterministically, without relying on worker speed or sleep intervals.

use std::{path::Path, time::Duration};

use super::*;

fn spec() -> RunSpec {
    RunSpec {
        scene: crate::scene::load_resolved_scene(
            Path::new("reference/scenes/falling-block-04.toml"),
            None,
        )
        .unwrap(),
        backend: SearchBackend::UniformGrid,
        execution: ExecutionMode::Serial,
        steps: 2,
        initially_paused: true,
    }
}

#[test]
fn supported_scene_versions_construct_and_unknown_versions_fail_explicitly() {
    let mut controller = Controller::new().unwrap();
    for path in [
        "reference/scenes/falling-block-04.toml",
        "reference/scenes/two-blocks.toml",
    ] {
        let scene = crate::scene::load_resolved_scene(Path::new(path), None).unwrap();
        let mut expected = crate::simulation::Simulation::from_scene_with_execution(
            &scene,
            SearchBackend::UniformGrid,
            ExecutionMode::Serial,
        )
        .unwrap();
        let request = RunSpec { scene, ..spec() };
        let (handle, receipt) = controller.start(request).unwrap();
        assert_eq!(receipt.wait().unwrap(), RunStatus::Paused);
        let before = handle.observe().snapshot.unwrap();
        assert_eq!(before.particles(), expected.snapshot().particles());
        assert_eq!(expected.step().unwrap().completed_steps, 1);
        assert_eq!(
            controller
                .control(&handle, Action::SingleStep)
                .unwrap()
                .wait()
                .unwrap(),
            RunStatus::Paused
        );
        assert_eq!(
            handle.observe().snapshot.unwrap().particles(),
            expected.snapshot().particles()
        );
    }
    for schema in [0, 3, u32::MAX] {
        let mut request = spec();
        request.scene.schema_version = schema;
        let (handle, receipt) = controller.start(request).unwrap();
        assert_eq!(receipt.wait().unwrap(), RunStatus::Failed);
        let observation = handle.observe();
        assert!(observation.snapshot.is_none());
        assert_eq!(observation.failure.unwrap().kind, FailureKind::Construction);
    }
    controller.join().unwrap();
}

fn gated() -> (Controller, mpsc::Sender<()>) {
    let (sender, receiver) = mpsc::sync_channel(COMMAND_CAPACITY);
    let (release, gate) = mpsc::channel();
    let stopping = Arc::new(AtomicBool::new(false));
    let admission = Arc::new(Mutex::new(()));
    let stop = stopping.clone();
    let worker_admission = admission.clone();
    let thread = thread::spawn(move || {
        gate.recv().unwrap();
        worker::run(receiver, stop, worker_admission);
    });
    (
        Controller {
            sender: Some(sender),
            thread: Some(thread),
            stopping,
            admission,
            current: None,
            generation: 0,
        },
        release,
    )
}

#[test]
fn full_queue_rejects_restart_atomically_and_cancellation_bypasses_backpressure() {
    let (mut controller, release) = gated();
    let (first, start) = controller.start(spec()).unwrap();
    let receipts: Vec<_> = (1..COMMAND_CAPACITY)
        .map(|_| controller.control(&first, Action::SingleStep).unwrap())
        .collect();
    assert!(matches!(
        controller.start(spec()),
        Err(ControlError::QueueFull)
    ));
    assert!(!first.shared.cancelled.load(Ordering::Acquire));
    assert_eq!(controller.generation, first.observe().generation);
    assert_eq!(Arc::strong_count(&first.shared), 3);
    first.cancel();
    assert_eq!(first.observe().status, RunStatus::Queued);
    release.send(()).unwrap();
    assert_eq!(start.wait().unwrap(), RunStatus::Cancelled);
    for receipt in receipts {
        assert!(matches!(
            receipt.wait(),
            Err(ControlError::InvalidTransition {
                status: RunStatus::Cancelled,
                ..
            })
        ));
    }
    assert!(first.observe().snapshot.is_none());
    controller.join().unwrap();
}

#[test]
fn queued_old_steps_are_discarded_and_new_generation_stays_paused_at_zero() {
    let (mut controller, release) = gated();
    let (old, start_old) = controller.start(spec()).unwrap();
    let obsolete = controller.control(&old, Action::SingleStep).unwrap();
    let (new, start_new) = controller.start(spec()).unwrap();
    release.send(()).unwrap();
    assert_eq!(start_old.wait().unwrap(), RunStatus::Cancelled);
    assert!(obsolete.wait().is_err());
    assert_eq!(start_new.wait().unwrap(), RunStatus::Paused);
    assert!(old.observe().snapshot.is_none());
    assert_eq!(new.observe().snapshot.unwrap().completed_steps(), 0);
    controller.join().unwrap();
}

#[test]
fn shutdown_with_full_queue_resolves_unconstructed_handles_and_receipts() {
    let (mut controller, release) = gated();
    let (handle, start) = controller.start(spec()).unwrap();
    let receipts: Vec<_> = (1..COMMAND_CAPACITY)
        .map(|_| controller.control(&handle, Action::Resume).unwrap())
        .collect();
    controller.request_shutdown();
    release.send(()).unwrap();
    controller.join().unwrap();
    assert_eq!(start.wait().unwrap(), RunStatus::Cancelled);
    for receipt in receipts {
        assert!(matches!(receipt.wait(), Err(ControlError::Stopped)));
    }
    assert_eq!(handle.observe().status, RunStatus::Cancelled);
}

#[test]
fn panic_retains_active_snapshot_marks_queued_runs_and_join_reports_failure() {
    let mut controller = Controller::new().unwrap();
    let (active, start) = controller.start(spec()).unwrap();
    start.wait().unwrap();
    let accepted = active.observe().snapshot.unwrap();
    // Enqueue both requests before dispatch using the same barrier as above.
    // The observation lock holds panic reporting until the queue is ready.
    let lock = active.shared.observation.lock().unwrap();
    controller.send(Command::Panic).unwrap();
    let pending = Arc::new(SharedRun {
        observation: Mutex::new(Observation {
            generation: 2,
            status: RunStatus::Queued,
            snapshot: None,
            info: None,
            last_cfl: None,
            warning_count: 0,
            failure: None,
        }),
        changed: Condvar::new(),
        cancelled: AtomicBool::new(false),
    });
    let (reply, receiver) = mpsc::sync_channel(1);
    controller
        .send(Command::Start {
            spec: Box::new(spec()),
            shared: pending.clone(),
            reply,
        })
        .unwrap();
    drop(lock);
    let failure = active.wait_for_terminal(Duration::from_secs(5)).unwrap();
    assert_eq!(failure.failure.unwrap().kind, FailureKind::WorkerPanic);
    assert_eq!(failure.snapshot.unwrap(), accepted);
    assert_eq!(
        receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap(),
        RunStatus::Failed
    );
    assert_eq!(
        pending
            .observation
            .lock()
            .unwrap()
            .failure
            .as_ref()
            .unwrap()
            .kind,
        FailureKind::WorkerPanic
    );
    assert!(matches!(
        controller.start(spec()),
        Err(ControlError::Stopped)
    ));
    assert!(matches!(controller.join(), Err(ControlError::WorkerPanic)));
}
