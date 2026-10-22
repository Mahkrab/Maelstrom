//! Offline schedules and saved histories are rejected before any graphics or
//! output acquisition. Deliberate record corruption must not become a new run.
#![cfg(feature = "render")]
use maelstrom::{
    export::{ExportRequest, FrameSchedule, Source, export},
    render::{GraphicsSelection, Settings},
    scene::load_resolved_scene,
    workflow::{RunRequest, run},
};
use std::path::Path;

#[test]
fn schedules_have_exact_endpoints_without_duplicates_or_overflow() {
    for (s, k, want) in [
        (0, 1, vec![0]),
        (1, 8, vec![0, 1]),
        (5, 2, vec![0, 2, 4, 5]),
        (6, 2, vec![0, 2, 4, 6]),
        (u64::MAX, u64::MAX, vec![0, u64::MAX]),
    ] {
        assert_eq!(FrameSchedule::new(s, k).unwrap().steps, want);
    }
    assert!(FrameSchedule::new(1, 0).is_err());
    assert!(FrameSchedule::new(u64::MAX, 1).is_err());
    assert!(FrameSchedule::new(1_000_000, 1).is_err());
    assert_eq!(
        FrameSchedule::new(999_999, 1).unwrap().steps.len(),
        1_000_000
    );
}
fn reject(json: &str, last_step: Option<u64>, stride: u64) -> String {
    let path =
        std::env::temp_dir().join(format!("maelstrom-rejected-export-{}", std::process::id()));
    assert!(!path.exists());
    let error = export(
        ExportRequest {
            source: Source::Saved { json, last_step },
            settings: Settings {
                step_stride: stride,
                ..Settings::default()
            },
            graphics: GraphicsSelection {
                index: usize::MAX,
                software: false,
            },
            fit_camera: false,
            destination: &path,
        },
        || false,
    )
    .unwrap_err();
    assert!(!path.exists());
    error.to_string()
}
#[test]
fn saved_mode_never_invents_missing_initial_or_intermediate_observations() {
    let scene = load_resolved_scene(Path::new("scenes/falling-block-04.toml"), None).unwrap();
    let result = run(
        &scene,
        RunRequest {
            steps: 5,
            checkpoints: vec![0, 2, 5],
        },
    );
    let json = result.to_json().unwrap();
    assert!(reject(&json, None, 2).contains("missing scheduled indices: [4]"));
    let no_initial = run(
        &scene,
        RunRequest {
            steps: 0,
            checkpoints: vec![],
        },
    );
    // Its final snapshot IS a stored observation at zero. Passing preflight
    // reaches adapter selection, unlike the missing-initial case below.
    assert!(reject(&no_initial.to_json().unwrap(), None, 1).contains("adapter"));
    let no_initial = run(
        &scene,
        RunRequest {
            steps: 1,
            checkpoints: vec![],
        },
    );
    assert!(reject(&no_initial.to_json().unwrap(), None, 1).contains("[0]"));
}
#[test]
fn saved_records_reject_malformed_and_conflicting_accepted_history() {
    let scene = load_resolved_scene(Path::new("scenes/falling-block-04.toml"), None).unwrap();
    let result = run(
        &scene,
        RunRequest {
            steps: 2,
            checkpoints: vec![0, 2],
        },
    );
    let original: serde_json::Value = serde_json::from_str(&result.to_json().unwrap()).unwrap();
    for mutate in [0, 1, 2, 3, 4, 5, 6, 7] {
        let mut v = original.clone();
        match mutate {
            0 => v["schema_version"] = 99.into(),
            1 => v["completed_steps"] = "1".into(),
            2 => v["checkpoints"][1]["snapshot"]["particles"][0]["position_m"]["x"] = 0.125.into(),
            3 => v["final_snapshot"]["particles"][0]["identity"] = "999".into(),
            4 => v["steps"][0]["completed_steps"] = "2".into(),
            5 => v["status"] = "invented".into(),
            6 => {
                v["checkpoints"][0]["snapshot"]["particles"][0]["velocity_m_s"]["x"] = (-0.0).into()
            }
            _ => v["input"]["comparison_key"] = "wrong".into(),
        }
        let error = reject(&v.to_string(), None, 2);
        assert!(
            !error.contains("adapter"),
            "corruption {mutate} passed source validation: {error}"
        );
    }
    let mut partial = original;
    partial["status"] = "cancelled".into();
    partial["failures"] =
        serde_json::json!([{"status":"cancelled","completed_steps":"2","message":"cancelled"}]);
    assert!(reject(&partial.to_string(), None, 2).contains("explicit --last-step"));
    assert!(reject(&partial.to_string(), Some(3), 2).contains("exceeds accepted count"));
    assert!(reject(&partial.to_string(), Some(2), 2).contains("adapter"));
}

#[test]
fn existing_files_directories_and_symlinks_are_rejected_before_graphics() {
    let scene = load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    let root =
        std::env::temp_dir().join(format!("maelstrom-export-existing-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let file = root.join("user.txt");
    std::fs::write(&file, b"user data").unwrap();
    let directory = root.join("directory");
    std::fs::create_dir(&directory).unwrap();
    let mut paths = vec![file.clone(), directory];
    #[cfg(unix)]
    {
        let link = root.join("link");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        paths.push(link);
        let dangling = root.join("dangling");
        std::os::unix::fs::symlink(root.join("missing"), &dangling).unwrap();
        paths.push(dangling);
    }
    for path in paths {
        let error = export(
            ExportRequest {
                source: Source::Direct {
                    scene: &scene,
                    steps: 0,
                    options: Default::default(),
                },
                settings: Settings::default(),
                graphics: GraphicsSelection {
                    index: usize::MAX,
                    software: false,
                },
                fit_camera: false,
                destination: &path,
            },
            || false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("destination already exists"));
    }
    assert_eq!(std::fs::read(&file).unwrap(), b"user data");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn export_cli_help_and_source_errors_are_usable_without_a_display() {
    use std::process::Command;
    let exe = env!("CARGO_BIN_EXE_maelstrom");
    let help = Command::new(exe)
        .args(["export", "--help"])
        .output()
        .unwrap();
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();
    assert!(text.contains("saved"));
    assert!(!text.contains("  rerun") && !text.contains("  direct"));
    let missing = Command::new(exe)
        .args(["export", "direct", "falling-block-04", "--steps", "0"])
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(2));
    assert!(
        String::from_utf8(missing.stderr)
            .unwrap()
            .contains("requires --output DIR")
    );
    let incompatible = Command::new(exe)
        .args([
            "export",
            "direct",
            "falling-block-04",
            "--steps",
            "0",
            "--workers",
            "2",
            "--backend",
            "serial-grid",
            "--output",
            "unused-directory",
        ])
        .output()
        .unwrap();
    assert_eq!(incompatible.status.code(), Some(2));
    assert!(!Path::new("unused-directory").exists());
}
