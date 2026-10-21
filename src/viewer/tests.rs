//! Camera operations, delayed consumers and draft edits must leave each accepted
//! trajectory identical to independently run headless execution.

use super::*;
use crate::{
    application::{RunSpec, RunStatus},
    execution::ExecutionMode,
    neighbours::SearchBackend,
    scene::load_resolved_scene,
    simulation::{Simulation, Snapshot},
};
use std::path::Path;

fn draft() -> Draft {
    Draft {
        scene: "falling-block-04".into(),
        config: String::new(),
        mode: "serial-grid".into(),
        workers: "2".into(),
        device: "0".into(),
        block_size: "128".into(),
        steps: "4".into(),
    }
}
fn spec(d: &Draft) -> Result<RunSpec, String> {
    if d.scene == "invalid" {
        return Err("selected scene failed validation".into());
    }
    Ok(RunSpec {
        scene: load_resolved_scene(Path::new("scenes/falling-block-04.toml"), None).unwrap(),
        steps: d.steps.parse().map_err(|_| "invalid steps")?,
        backend: if d.mode == "brute-force" {
            SearchBackend::BruteForce
        } else {
            SearchBackend::UniformGrid
        },
        execution: match d.mode.as_str() {
            "threaded-grid" => ExecutionMode::Threaded { workers: 2 },
            "cuda-grid" => ExecutionMode::Cuda {
                device: 0,
                threads: 128,
            },
            _ => ExecutionMode::Serial,
        },
        initially_paused: true,
    })
}
fn model() -> Model {
    Model::new(draft(), Default::default(), Box::new(spec)).unwrap()
}
fn acknowledge(m: &mut Model) -> RunStatus {
    m.receipts.pop().unwrap().wait().unwrap()
}
fn bits(s: &Snapshot) -> Vec<(u64, [u32; 6])> {
    s.particles()
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

#[test]
fn presentation_cadence_camera_draft_and_reset_preserve_all_cpu_modes() {
    for mode in ["brute-force", "serial-grid", "threaded-grid"] {
        let mut m = model();
        m.draft.mode = mode.into();
        m.apply().unwrap();
        assert_eq!(acknowledge(&mut m), RunStatus::Paused);
        let request = m.selected.as_ref().unwrap().spec.clone();
        let mut direct = Simulation::from_scene_with_execution(
            &request.scene,
            request.backend,
            request.execution,
        )
        .unwrap();
        for display_reads in [0, 1, 7, 19] {
            m.camera.orbit(0.17, -0.03).unwrap();
            m.camera.pan(0.001, 0.002).unwrap();
            m.camera.zoom(0.98).unwrap();
            m.draft.steps = "999".into(); // unapplied edits must not change work
            m.action(Action::SingleStep).unwrap();
            let _ = acknowledge(&mut m);
            let _ = direct.step().unwrap();
            let accepted = m.selected.as_ref().unwrap().handle.observe();
            for _ in 0..display_reads {
                let _ = m.selected.as_ref().unwrap().handle.observe();
                m.camera.project([0.3, 0.4, 0.3], 800, 600).unwrap();
            }
            assert_eq!(
                bits(accepted.snapshot.as_ref().unwrap()),
                bits(direct.snapshot())
            );
        }
        let old = m.selected.as_ref().unwrap().handle.clone();
        let final_bits = bits(old.observe().snapshot.as_ref().unwrap());
        m.reset().unwrap();
        assert_eq!(acknowledge(&mut m), RunStatus::Paused);
        let initial = m.selected.as_ref().unwrap().handle.observe();
        assert_eq!(initial.snapshot.unwrap().completed_steps(), 0);
        assert_eq!(initial.info.unwrap().requested_steps, 4);
        assert_eq!(bits(old.observe().snapshot.as_ref().unwrap()), final_bits);
        m.cancel();
        assert!(
            m.selected
                .as_ref()
                .unwrap()
                .handle
                .wait_for_terminal(Duration::from_secs(30))
                .is_some()
        );
        m.join().unwrap();
    }
}
#[test]
fn invalid_draft_preserves_selection_and_resource_failure_retains_prior() {
    let mut m = model();
    m.apply().unwrap();
    acknowledge(&mut m);
    m.action(Action::SingleStep).unwrap();
    acknowledge(&mut m);
    let old = m.selected.as_ref().unwrap().handle.clone();
    let generation = old.observe().generation;
    let before = bits(old.observe().snapshot.as_ref().unwrap());
    m.draft.scene = "invalid".into();
    assert!(m.apply().is_err());
    assert_eq!(
        m.selected.as_ref().unwrap().handle.observe().generation,
        generation
    );
    m.draft.scene = "falling-block-04".into();
    // Invalid worker input passes the resolver to exercise construction rejection
    // rather than relying on a particular machine's device availability.
    m.resolver = Box::new(|d| {
        let mut s = spec(d)?;
        s.scene.schema_version = u32::MAX;
        Ok(s)
    });
    m.apply().unwrap();
    assert_eq!(acknowledge(&mut m), RunStatus::Failed);
    assert!(
        m.selected
            .as_ref()
            .unwrap()
            .handle
            .observe()
            .snapshot
            .is_none()
    );
    assert_eq!(
        bits(
            m.retained
                .as_ref()
                .unwrap()
                .handle
                .observe()
                .snapshot
                .as_ref()
                .unwrap()
        ),
        before
    );
    m.show_retained = true;
    assert_eq!(
        m.displayed().unwrap().handle.observe().generation,
        generation
    );
    m.join().unwrap();
}
#[test]
fn close_during_running_work_retains_terminal_accepted_state() {
    let mut m = model();
    m.draft.steps = "1000000".into();
    m.apply().unwrap();
    acknowledge(&mut m);
    let handle = m.selected.as_ref().unwrap().handle.clone();
    m.action(Action::Resume).unwrap();
    acknowledge(&mut m);
    m.request_shutdown();
    m.join().unwrap();
    let observation = handle.observe();
    assert_eq!(observation.status, RunStatus::Cancelled);
    assert!(observation.snapshot.is_some());
}
#[cfg(not(feature = "cuda"))]
#[test]
fn explicit_cuda_selection_fails_without_fallback() {
    let mut m = model();
    m.draft.mode = "cuda-grid".into();
    m.apply().unwrap();
    assert_eq!(acknowledge(&mut m), RunStatus::Failed);
    let o = m.selected.as_ref().unwrap().handle.observe();
    assert!(o.snapshot.is_none());
    assert!(o.info.is_none());
    assert!(o.failure.unwrap().message.contains("CUDA"));
    m.join().unwrap();
}
#[test]
fn receipts_are_polled_and_removed_without_blocking() {
    let mut m = model();
    m.apply().unwrap();
    acknowledge(&mut m);
    m.action(Action::SingleStep).unwrap();
    // A pause receipt provides a deterministic barrier after the step request.
    let barrier = m
        .controller
        .control(&m.selected.as_ref().unwrap().handle, Action::Pause)
        .unwrap();
    let _ = barrier.wait();
    m.poll();
    assert!(m.receipts.is_empty());
    assert!(m.message.contains("Paused"));
    m.join().unwrap();
}
#[test]
fn overlay_is_bounded_to_the_drawable_and_has_complete_triangles() {
    for (width, height) in [(1, 1), (100, 50), (1280, 720)] {
        let vertices = overlay::panel(&vec!["A long diagnostic".into(); 3], width, height, 1.0);
        assert_eq!(vertices.len() % 15, 0);
        for v in vertices.as_chunks::<5>().0 {
            assert!((-1.0..=1.0).contains(&v[0]));
            assert!((-1.0..=1.0).contains(&v[1]));
        }
    }
}

#[test]
fn requested_audit_is_labelled_by_snapshot_and_does_not_change_acceptance() {
    let mut m = model();
    m.apply().unwrap();
    acknowledge(&mut m);
    m.action(Action::SingleStep).unwrap();
    acknowledge(&mut m);
    let handle = m.selected.as_ref().unwrap().handle.clone();
    let before = bits(handle.observe().snapshot.as_ref().unwrap());
    m.audit().unwrap();
    assert!(m.audit_message.contains("gen 1 step 1"));
    m.reset().unwrap();
    acknowledge(&mut m);
    m.join().unwrap();
    assert_eq!(bits(handle.observe().snapshot.as_ref().unwrap()), before);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_viewer_observations_preserve_exact_headless_results_and_teardown() {
    let mut m = model();
    m.draft.mode = "cuda-grid".into();
    m.apply().unwrap();
    assert_eq!(acknowledge(&mut m), RunStatus::Paused);
    let request = m.selected.as_ref().unwrap().spec.clone();
    let mut headless =
        Simulation::from_scene_with_execution(&request.scene, request.backend, request.execution)
            .unwrap();
    for _ in 0..4 {
        m.camera.orbit(0.2, 0.1).unwrap();
        m.camera.zoom(0.95).unwrap();
        m.action(Action::SingleStep).unwrap();
        acknowledge(&mut m);
        let _ = headless.step().unwrap();
        assert_eq!(
            bits(
                m.selected
                    .as_ref()
                    .unwrap()
                    .handle
                    .observe()
                    .snapshot
                    .as_ref()
                    .unwrap()
            ),
            bits(headless.snapshot())
        );
    }
    let old = m.selected.as_ref().unwrap().handle.clone();
    m.reset().unwrap();
    assert_eq!(acknowledge(&mut m), RunStatus::Paused);
    assert!(
        m.selected
            .as_ref()
            .unwrap()
            .handle
            .observe()
            .info
            .unwrap()
            .device
            .is_some()
    );
    m.join().unwrap();
    assert_eq!(old.observe().status, RunStatus::Completed);
}

#[test]
fn zero_extent_and_occlusion_skip_graphics_configuration_and_drawing() {
    for (width, height, occluded) in [
        (0, 0, false),
        (0, 720, false),
        (1280, 0, false),
        (1280, 720, true),
    ] {
        assert!(!drawable(width, height, occluded));
    }
    assert!(drawable(1, 1, false));
    assert!(drawable(1280, 720, false));
}
#[test]
fn graphics_failure_requests_worker_teardown_and_preserves_accepted_state() {
    let mut m = model();
    m.apply().unwrap();
    acknowledge(&mut m);
    m.action(Action::SingleStep).unwrap();
    acknowledge(&mut m);
    let handle = m.selected.as_ref().unwrap().handle.clone();
    let before = bits(handle.observe().snapshot.as_ref().unwrap());
    let mut viewer = Viewer::new(m, Settings::default(), GraphicsSelection::default());
    viewer.close(Some(RenderError("controlled graphics failure".into())));
    assert!(viewer.closing);
    assert!(viewer.error.is_some());
    viewer.model.join().unwrap();
    let o = handle.observe();
    assert_eq!(o.status, RunStatus::Cancelled);
    assert_eq!(bits(o.snapshot.as_ref().unwrap()), before);
}
