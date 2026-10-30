//! Qualified software Vulkan exports are decoded independently and compared with
//! headless observations. All device checks are explicit rather than implicit CI.
#![cfg(feature = "render")]
use maelstrom::{
    export::{ExportRequest, Source, export},
    render::{GraphicsSelection, Settings},
    scene::{ResolvedScene, load_resolved_scene},
    workflow::{ExecutionOptions, RunRequest, run_with_options},
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "maelstrom-export-device-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn settings(stride: u64) -> Settings {
    Settings {
        width: 127,
        height: 93,
        step_stride: stride,
        ..Settings::default()
    }
}
fn request<'a>(source: Source<'a>, path: &'a Path, stride: u64) -> ExportRequest<'a> {
    ExportRequest {
        source,
        settings: settings(stride),
        graphics: GraphicsSelection {
            index: 0,
            software: true,
        },
        fit_camera: true,
        destination: path,
    }
}
fn hash(bytes: &[u8]) -> String {
    format!(
        "sha256:{}",
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}
fn decode(path: &Path) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
    let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buffer).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    buffer.truncate(info.buffer_size());
    buffer
}
fn exercise(scene: &ResolvedScene, steps: u64, stride: u64, options: ExecutionOptions) {
    let schedule = maelstrom::export::FrameSchedule::new(steps, stride).unwrap();
    let reference = run_with_options(
        scene,
        RunRequest {
            steps,
            checkpoints: schedule.steps.clone(),
        },
        options,
        || false,
    );
    assert!(reference.status().is_complete());
    let json = reference.to_json().unwrap();
    let a = Directory::new();
    let b = Directory::new();
    let c = Directory::new();
    let direct = export(
        request(
            Source::Direct {
                scene,
                steps,
                options,
            },
            &a.0,
            stride,
        ),
        || false,
    )
    .unwrap();
    let saved = export(
        request(
            Source::Saved {
                json: &json,
                last_step: None,
            },
            &b.0,
            stride,
        ),
        || false,
    )
    .unwrap();
    let rerun = export(request(Source::Rerun { json: &json }, &c.0, stride), || {
        false
    })
    .unwrap();
    eprintln!("export graphics: {}", direct["graphics"]);
    for manifest in [&direct, &saved, &rerun] {
        let profile = &manifest["timing_profile"];
        assert_eq!(profile["schema_version"], 2);
        assert_eq!(profile["boundary"], "export_entry_to_before_manifest");
        let components = profile["components_s"].as_object().unwrap();
        assert_eq!(components.len(), 10);
        let seconds: Vec<_> = components.values().map(|v| v.as_f64().unwrap()).collect();
        assert!(seconds.iter().all(|v| v.is_finite() && *v >= 0.0));
        let total = profile["total_s"].as_f64().unwrap();
        assert!(total.is_finite() && total > 0.0);
        assert!((seconds.iter().sum::<f64>() - total).abs() <= total * 1e-12);
        for (legacy, component) in [
            ("solver_steps", "solver_steps"),
            ("graphics_readback", "graphics_readback"),
        ] {
            assert_eq!(manifest["timing_seconds"][legacy], components[component]);
        }
        assert_eq!(
            manifest["timing_seconds"]["encoding"],
            profile["worker_components_s"]["png_encoding"]
        );
        assert!(
            profile["worker_components_s"]
                .as_object()
                .unwrap()
                .values()
                .all(|v| v.as_f64().is_some_and(|s| s.is_finite() && s >= 0.0))
        );
        assert_eq!(manifest["frame_count"], schedule.steps.len());
        assert_eq!(
            manifest["input"]["comparison_key"],
            serde_json::to_value(reference.input().unwrap()).unwrap()["comparison_key"]
        );
    }
    assert_eq!(saved["timing_profile"]["components_s"]["solver_steps"], 0.0);
    for (i, checkpoint) in reference.checkpoints().iter().enumerate() {
        let name = format!("frame-{i:08}.png");
        let rgba = decode(&a.0.join(&name));
        assert_eq!(rgba.len(), 127 * 93 * 4);
        assert!(rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
        assert_eq!(rgba, decode(&b.0.join(&name)));
        assert_eq!(rgba, decode(&c.0.join(&name)));
        for (dir, m) in [(&a.0, &direct), (&b.0, &saved), (&c.0, &rerun)] {
            let f = &m["frames"][i];
            assert_eq!(f["step"], schedule.steps[i].to_string());
            assert_eq!(f["file"], name);
            assert_eq!(f["png_sha256"], hash(&fs::read(dir.join(&name)).unwrap()));
            assert_eq!(f["rgba_sha256"], hash(&rgba));
            assert_eq!(
                f["snapshot_sha256"],
                hash(checkpoint.snapshot().to_json().unwrap().as_bytes())
            );
            assert_eq!(
                f["simulated_seconds"],
                schedule.steps[i] as f64
                    * f64::from(scene.configuration.numerical.fixed_timestep_s)
            );
        }
    }
    let (restored, restored_request, restored_options) =
        maelstrom::workflow::replay_options(&direct.to_string()).unwrap();
    assert_eq!(restored.comparison_key(), scene.comparison_key());
    assert_eq!(restored_request.steps, steps);
    assert_eq!(restored_request.checkpoints, schedule.steps);
    assert_eq!(restored_options.execution, options.execution);
    assert!(a.0.join("manifest.json").exists());
    assert!(
        fs::read_dir(&a.0).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp"))
    );
    let before = fs::read(a.0.join("manifest.json")).unwrap();
    assert!(
        export(
            request(
                Source::Saved {
                    json: &json,
                    last_step: None
                },
                &a.0,
                stride
            ),
            || false
        )
        .is_err()
    );
    assert_eq!(before, fs::read(a.0.join("manifest.json")).unwrap());
}
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn exports_match_direct_observations_and_repeat_decoded_pixels() {
    let small = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    exercise(&small, 5, 2, ExecutionOptions::default());
    // Use the frozen falling-block request itself, with its original refinement.
    let json = fs::read_to_string("reference/falling-block-v1.json").unwrap();
    let (scene, request, options) = maelstrom::workflow::replay_options(&json).unwrap();
    assert_eq!(request.steps, 480);
    exercise(&scene, request.steps, 120, options);
    exercise(
        &small,
        5,
        2,
        ExecutionOptions {
            backend: maelstrom::neighbours::SearchBackend::UniformGrid,
            execution: maelstrom::execution::ExecutionMode::Threaded { workers: 2 },
            measurement: None,
        },
    );
    let mut empty = small.clone();
    empty.particles.clear();
    empty.planes.clear();
    exercise(&empty, 0, 1, ExecutionOptions::default());
    let mut single = small;
    single.particles.truncate(1);
    exercise(&single, 1, 1, ExecutionOptions::default());
}
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn failures_and_cancellation_retain_incomplete_sequences_and_prior_files() {
    let scene = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    let options = ExecutionOptions::default();
    for fault in [0, 1, 2, 3] {
        let dir = Directory::new();
        let mut calls = 0;
        let error = export(
            request(
                Source::Direct {
                    scene: &scene,
                    steps: 3,
                    options,
                },
                &dir.0,
                1,
            ),
            || {
                calls += 1;
                if dir.0.join("frame-00000000.png").exists() {
                    match fault {
                        0 => return true,
                        1 => {
                            let _ = fs::write(dir.0.join("frame-00000001.png"), b"user data");
                        }
                        2 => {
                            let _ = fs::create_dir(dir.0.join(".progress.tmp"));
                        }
                        _ => {
                            let _ = fs::write(dir.0.join("manifest.json"), b"user manifest");
                        }
                    }
                }
                false
            },
        )
        .unwrap_err();
        eprintln!("fault {fault}: {error}; cancellation polls {calls}");
        assert!(dir.0.join("frame-00000000.png").exists());
        if fault == 3 {
            assert_eq!(
                fs::read(dir.0.join("manifest.json")).unwrap(),
                b"user manifest"
            );
        } else {
            assert!(!dir.0.join("manifest.json").exists());
        }
        let progress: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.0.join("progress.json")).unwrap()).unwrap();
        assert_eq!(progress["complete"], false);
        if fault != 2 {
            assert!(!progress["failures"].as_array().unwrap().is_empty());
        }
        if fault == 1 {
            assert_eq!(
                fs::read(dir.0.join("frame-00000001.png")).unwrap(),
                b"user data"
            );
        }
    }
    // A rerun mismatch is retained as failure after the last agreeing frame.
    let source = run_with_options(
        &scene,
        RunRequest {
            steps: 3,
            checkpoints: vec![0, 1, 3],
        },
        options,
        || false,
    );
    let mut json: serde_json::Value = serde_json::from_str(&source.to_json().unwrap()).unwrap();
    json["checkpoints"][1]["snapshot"]["particles"][0]["position_m"]["x"] = 0.333.into();
    let dir = Directory::new();
    let error = export(
        request(
            Source::Rerun {
                json: &json.to_string(),
            },
            &dir.0,
            1,
        ),
        || false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("checkpoint mismatch"));
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.0.join("comparison-failure.json")).unwrap()).unwrap();
    assert_eq!(report["expected"]["completed_steps"], "1");
    assert_eq!(report["actual"]["completed_steps"], "1");
    assert_ne!(
        report["expected"]["particles"],
        report["actual"]["particles"]
    );
    assert_eq!(
        report["source"]["metadata"]["input_id"],
        report["metadata"]["input_id"]
    );
    assert!(dir.0.join("frame-00000000.png").exists());
    assert!(!dir.0.join("manifest.json").exists());
}

#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn exported_pngs_preserve_orientation_scale_depth_and_explicit_camera() {
    use maelstrom::{math::Vec3, render::Camera, scene::Particle};
    let mut scene = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    scene.planes.clear();
    scene.particles = vec![
        (0, 0.0, 0.0, 0.05),
        (1, 0.06, 0.06, 0.0),
        (2, 0.0, 0.0, -0.05),
    ]
    .into_iter()
    .map(|(identity, x, y, z)| Particle {
        identity,
        position_m: Vec3::new(x, y, z),
        velocity_m_s: Vec3::ZERO,
    })
    .collect();
    let mut settings = settings(1);
    settings.camera = Camera {
        eye_m: [0.0, 0.0, 0.25],
        target_m: [0.0; 3],
        up: [0.0, 1.0, 0.0],
        vertical_fov_deg: 90.0,
        near_m: 0.001,
        far_m: 10.0,
    };
    let draw = |scene: &ResolvedScene| {
        let dir = Directory::new();
        export(
            ExportRequest {
                source: Source::Direct {
                    scene,
                    steps: 0,
                    options: ExecutionOptions::default(),
                },
                settings: settings.clone(),
                graphics: GraphicsSelection {
                    index: 0,
                    software: true,
                },
                fit_camera: false,
                destination: &dir.0,
            },
            || false,
        )
        .unwrap();
        decode(&dir.0.join("frame-00000000.png"))
    };
    let both = draw(&scene);
    scene.particles.pop();
    assert_eq!(both, draw(&scene), "far particle must be fully occluded");
    let at = |x: usize, y: usize| &both[(y * 127 + x) * 4..(y * 127 + x) * 4 + 4];
    // At z=0, projection gives x=63.5+46.5*0.06/0.25,
    // y=46.5-46.5*0.06/0.25. These are independent camera-space probes.
    assert!(at(75, 35)[2] > at(75, 35)[0] + 30);
    assert_eq!(at(52, 58), [16, 24, 32, 255]);
    assert!(at(63, 46)[2] > at(63, 46)[0] + 30);
    // Centre radius 0.025 m at distance 0.2 m projects to about 6 pixels.
    assert!(at(68, 46)[2] > at(68, 46)[0] + 30);
    assert_eq!(at(72, 46), [16, 24, 32, 255]);
    scene.particles.clear();
    assert!(
        draw(&scene)
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [16, 24, 32, 255])
    );
}

#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn explicit_saved_prefix_and_process_interruption_cannot_claim_original_completion() {
    use std::{
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    let scene = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    let mut result = run_with_options(
        &scene,
        RunRequest {
            steps: 3,
            checkpoints: vec![0, 1, 3],
        },
        ExecutionOptions::default(),
        || false,
    );
    let existing = Directory::new();
    fs::create_dir(&existing.0).unwrap();
    let path = existing.0.join("user.txt");
    fs::write(&path, b"user data").unwrap();
    assert!(
        result
            .publish(&path, maelstrom::workflow::OutputFormat::Json)
            .is_err()
    );
    let dir = Directory::new();
    let manifest = export(
        request(
            Source::Saved {
                json: &result.to_json().unwrap(),
                last_step: Some(1),
            },
            &dir.0,
            1,
        ),
        || false,
    )
    .unwrap();
    assert_eq!(manifest["accepted_steps"], "1");
    assert_eq!(manifest["original"]["request"]["steps"], "3");
    assert_eq!(manifest["original"]["status"], "output_failure");
    assert_eq!(
        manifest["original"]["failures"].as_array().unwrap().len(),
        1
    );
    let interrupted = Directory::new();
    let mut child = Command::new(env!("CARGO_BIN_EXE_maelstrom"))
        .args([
            "export",
            "direct",
            "reference/small-lattice.toml",
            "--steps",
            "100000",
            "--output",
        ])
        .arg(&interrupted.0)
        .args([
            "--software",
            "--width",
            "127",
            "--height",
            "93",
            "--fit-camera",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while !interrupted.0.join("frame-00000000.png").exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("export ended before interruption: {status}");
        }
        if start.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("export did not publish its initial frame");
        }
        thread::sleep(Duration::from_millis(10));
    }
    child.kill().unwrap();
    assert!(!child.wait().unwrap().success());
    assert!(!interrupted.0.join("manifest.json").exists());
    let progress: serde_json::Value =
        serde_json::from_slice(&fs::read(interrupted.0.join("progress.json")).unwrap()).unwrap();
    assert_eq!(progress["complete"], false);
    assert!(interrupted.0.join("frame-00000000.png").exists());
}
