//! Subprocess checks exercise the installed-command boundary: arguments, channels,
//! file safety and reproducibility all go through the public binary.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::Value;

mod support;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "maelstrom-cli-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn command(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_maelstrom"));
    command
        .current_dir(directory)
        .stdin(Stdio::null())
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY");
    command
}

fn cli(args: &[&str]) -> Output {
    command(&std::env::temp_dir()).args(args).output().unwrap()
}

#[test]
fn saved_playback_command_is_optional_and_explains_its_non_numerical_input() {
    let output = cli(&["play", "--help"]);
    #[cfg(feature = "viewer")]
    {
        success(&output);
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains("no simulation"));
        for flag in ["--speed", "--paused", "--repeat", "--software"] {
            assert!(help.contains(flag));
        }
        assert_eq!(
            cli(&["play", "frames", "--backend", "uniform-grid"])
                .status
                .code(),
            Some(2)
        );
    }
    #[cfg(not(feature = "viewer"))]
    {
        success(&output);
        assert_eq!(cli(&["play", "frames"]).status.code(), Some(3));
    }
}

#[cfg(feature = "viewer")]
#[test]
fn playback_rejects_raw_results_and_incomplete_exports_without_writing_files() {
    let directory = Directory::new();
    for name in ["result.json", "progress.json", "manifest.json"] {
        let path = directory.write(name, r#"{"schema_version":1,"complete":false}"#);
        let output = command(&directory.0)
            .args(["play", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3));
        assert!(output.stdout.is_empty());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(!error.contains(&directory.0.to_string_lossy().to_string()));
        assert!(!error.contains("panicked"));
    }
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 3);
}

fn success(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn document(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{error}; stdout: {}; stderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn state_bits(snapshot: &Value) -> Vec<(String, [u32; 6])> {
    snapshot["particles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|particle| {
            let mut bits = [0; 6];
            for (index, (quantity, axis)) in [
                ("position_m", "x"),
                ("position_m", "y"),
                ("position_m", "z"),
                ("velocity_m_s", "x"),
                ("velocity_m_s", "y"),
                ("velocity_m_s", "z"),
            ]
            .into_iter()
            .enumerate()
            {
                bits[index] = (particle[quantity][axis].as_f64().unwrap() as f32).to_bits();
            }
            (particle["identity"].as_str().unwrap().to_owned(), bits)
        })
        .collect()
}

fn scene(configuration: &str, overrides: &str, velocity: &str) -> String {
    format!(
        r#"schema_version = 1
name = "one particle"
description = "CLI fixture with independent free-fall behaviour"
randomness = "none"
[configuration]
{configuration}
{overrides}
[particles]
construction = "explicit"
[[particles.items]]
identity = 7
position_m = [0.0, 0.0, 0.0]
velocity_m_s = {velocity}
"#
    )
}

#[test]
fn help_version_and_operation_help_explain_the_workflow() {
    let output = cli(&["--help"]);
    success(&output);
    let help = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "maelstrom",
        "scenes",
        "validate",
        "inspect",
        "run",
        "rerun",
        "compile",
        "doctor",
        "--quiet",
        "--verbose",
        "--format",
    ] {
        assert!(help.contains(expected), "missing {expected} in {help}");
    }
    assert!(output.stderr.is_empty());
    assert!(!help.contains("Examples:") && !help.contains("cuda-check"));
    let version = cli(&["--version"]);
    success(&version);
    assert_eq!(
        String::from_utf8(version.stdout).unwrap(),
        format!("maelstrom {}\n", env!("CARGO_PKG_VERSION"))
    );
    for operation in ["scenes", "validate", "inspect", "run", "replay"] {
        let output = cli(&[operation, "--help"]);
        success(&output);
        assert!(output.stderr.is_empty());
    }
    let output = cli(&["run", "--help"]);
    let help = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "--steps",
        "--duration",
        "--checkpoints",
        "--snapshot",
        "--output",
        "--config",
        "--set",
        "rounded up",
        "--unpaced",
    ] {
        assert!(help.contains(expected));
    }
}

#[test]
fn bundled_scenes_and_defaults_are_discoverable_outside_the_checkout() {
    let output = cli(&["scenes", "--format", "json"]);
    success(&output);
    let scenes = document(&output);
    assert_eq!(scenes["schema_version"], 1);
    let mut expected: Vec<_> = fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("scene.toml").is_file())
        .map(|path| path.file_name().unwrap().to_str().unwrap().to_owned())
        .collect();
    expected.sort();
    let discovered: Vec<_> = scenes["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|scene| scene["origin"] == "builtin")
        .map(|scene| scene["name"].as_str().unwrap())
        .collect();
    assert_eq!(discovered, expected);
    for name in [
        "falling-block-04",
        "falling-block-06",
        "falling-block-08",
        "falling-block-10",
        "falling-block-12",
        "falling-block-16",
        "falling-block-20",
    ] {
        let output = cli(&["inspect", name, "--format", "json"]);
        success(&output);
        let input = document(&output);
        let side: usize = name.rsplit('-').next().unwrap().parse().unwrap();
        assert_eq!(input["scene"]["particle_count"], side.pow(3));
        assert_eq!(input["configuration"]["name"], format!("{name}-model"));
        assert_eq!(input["configuration"]["numerical"]["relaxation_scale"], 2.0);
    }
    let defaults = cli(&["inspect", "--format", "json"]);
    success(&defaults);
    assert_eq!(
        document(&defaults)["configuration"]["numerical"]["solver_iterations"],
        8
    );
    assert!(document(&defaults)["scene"].is_null());
    assert_eq!(
        document(&defaults)["configuration"]["name"],
        "maelstrom-defaults-v4"
    );
    assert_eq!(
        document(&defaults)["configuration"]["numerical"]["relaxation_scale"],
        2.0
    );
}

#[test]
fn historical_configuration_replays_its_recorded_values_after_defaults_change() {
    for (source, name) in [
        (
            include_str!("../config/defaults-v1.toml"),
            "maelstrom-defaults-v1",
        ),
        (
            include_str!("../config/defaults-v2.toml"),
            "maelstrom-defaults-v2",
        ),
        (
            include_str!("../config/defaults-v3.toml"),
            "maelstrom-defaults-v3",
        ),
    ] {
        let dir = Directory::new();
        let config = dir.write("historical.toml", source);
        let legacy = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("reference/scenes/falling-block-04.toml");
        let original = command(&dir.0)
            .args([
                "run",
                legacy.to_str().unwrap(),
                "--steps",
                "4",
                "--config",
                "historical.toml",
                "--numerical-mode",
                "reference",
                "--format",
                "json",
                "--output",
                "historical.json",
            ])
            .output()
            .unwrap();
        success(&original);
        let original_bytes = fs::read(dir.path("historical.json")).unwrap();
        let historical: Value = serde_json::from_slice(&original_bytes).unwrap();
        assert_eq!(historical["input"]["configuration"]["name"], name);

        let current = command(&dir.0)
            .args([
                "run",
                "falling-block-04",
                "--steps",
                "4",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        success(&current);
        let current = document(&current);
        assert_eq!(
            current["input"]["configuration"]["numerical"]["relaxation_scale"],
            2.0
        );
        assert_ne!(
            state_bits(&current["final_snapshot"]),
            state_bits(&historical["final_snapshot"])
        );

        fs::remove_file(config).unwrap();
        let replay = command(&dir.0)
            .args(["replay", "historical.json", "--format", "json"])
            .output()
            .unwrap();
        success(&replay);
        assert_eq!(document(&replay), historical);
        assert_eq!(replay.stdout, [original_bytes.as_slice(), b"\n"].concat());
    }
}

#[test]
fn configuration_only_validation_never_executes_a_timestep() {
    let dir = Directory::new();
    dir.write("config.toml", "schema_version=1\nname='supplied'\ndescription='configuration fixture'\n[numerical]\nfixed_timestep_s=0.01\n");
    let output = command(&dir.0)
        .args(["validate", "--config", "config.toml", "--format", "json"])
        .output()
        .unwrap();
    success(&output);
    let result = document(&output);
    assert_eq!(result["operation"], "validate");
    assert_eq!(result["valid"], true);
    assert!(result.get("completed_steps").is_none());
    assert!(output.stderr.is_empty());
}

#[test]
fn timestep_refinement_matches_explicit_values_and_replays_without_the_recipe() {
    let dir = Directory::new();
    let refined = command(&dir.0)
        .args([
            "run",
            "falling-block-04",
            "--steps",
            "4",
            "--refine-timestep",
            "2",
            "--format",
            "json",
            "--output",
            "refined.json",
        ])
        .output()
        .unwrap();
    success(&refined);
    let record: Value =
        serde_json::from_slice(&fs::read(dir.path("refined.json")).unwrap()).unwrap();
    let explicit = cli(&[
        "run",
        "falling-block-04",
        "--steps",
        "4",
        "--set",
        "fixed_timestep_s=0.0020833334",
        "--set",
        "correction_scale=0.125",
        "--format",
        "json",
    ]);
    success(&explicit);
    assert_eq!(document(&explicit), record);
    let replay = command(&dir.0)
        .args(["replay", "refined.json", "--format", "json"])
        .output()
        .unwrap();
    success(&replay);
    assert_eq!(document(&replay), record);
    let inspected = cli(&[
        "inspect",
        "--set",
        "fixed_timestep_s=0.125",
        "--refine-timestep",
        "2",
        "--format",
        "json",
    ]);
    success(&inspected);
    assert_eq!(
        document(&inspected)["configuration"]["numerical"]["fixed_timestep_s"],
        0.0625
    );
    for value in ["0", "17", "2.5"] {
        let invalid = cli(&["inspect", "--refine-timestep", value]);
        assert_eq!(invalid.status.code(), Some(2));
        assert!(invalid.stdout.is_empty());
    }
    let duplicate = cli(&[
        "inspect",
        "--refine-timestep",
        "2",
        "--set",
        "refine_timestep=2",
    ]);
    assert_eq!(duplicate.status.code(), Some(2));
}

#[test]
fn configured_iteration_work_is_identified_and_replays_exactly() {
    let dir = Directory::new();
    let original = command(&dir.0)
        .args([
            "run",
            "--backend",
            "serial-grid",
            "falling-block-04",
            "--steps",
            "2",
            "--set",
            "solver_iterations=8",
            "--format",
            "json",
            "--output",
            "eight.json",
        ])
        .output()
        .unwrap();
    success(&original);
    let data: Value = serde_json::from_slice(&fs::read(dir.path("eight.json")).unwrap()).unwrap();
    assert_eq!(
        data["backend"],
        "serial-uniform-grid-pbf-binary32-8-jacobi-v2"
    );
    let replay = command(&dir.0)
        .args(["replay", "eight.json", "--format", "json"])
        .output()
        .unwrap();
    success(&replay);
    assert_eq!(document(&replay), data);
}

#[test]
fn external_scene_configuration_precedence_and_relative_paths_are_explicit() {
    let dir = Directory::new();
    dir.write("base.toml", "schema_version=1\nname='base'\ndescription='base fixture'\n[numerical]\nfixed_timestep_s=0.02\ncfl_factor=0.4\n");
    let path = dir.write(
        "scene.toml",
        &scene(
            "source='file'\npath='base.toml'",
            "[configuration.overrides.numerical]\nfixed_timestep_s=0.015",
            "[0.0, 0.0, 0.0]",
        ),
    );
    let output = cli(&[
        "inspect",
        path.to_str().unwrap(),
        "--set",
        "fixed_timestep_s=0.01",
        "--format",
        "json",
    ]);
    success(&output);
    let effective = document(&output);
    assert_eq!(
        effective["configuration"]["numerical"]["fixed_timestep_s"],
        serde_json::json!(0.01_f32)
    );
    assert_eq!(
        effective["configuration"]["provenance"]["fixed_timestep_s"],
        "command_line"
    );
    assert_eq!(
        effective["configuration"]["provenance"]["cfl_factor"],
        "configuration_file"
    );
    assert_eq!(
        effective["configuration"]["provenance"]["solver_iterations"],
        "command_line"
    );
    let without_cli = cli(&["inspect", path.to_str().unwrap(), "--format", "json"]);
    success(&without_cli);
    assert_eq!(
        document(&without_cli)["configuration"]["provenance"]["fixed_timestep_s"],
        "command_line"
    );
}

#[test]
fn explicit_config_replaces_scene_selection_but_preserves_scene_values() {
    let dir = Directory::new();
    dir.write("config.toml", "schema_version=1\nname='selected'\ndescription='selected fixture'\n[numerical]\ncfl_factor=0.6\n");
    for source in [
        "source='built_in_defaults'",
        "source='file'\npath='missing.toml'",
    ] {
        dir.write(
            "scene.toml",
            &scene(
                source,
                "[configuration.overrides.numerical]\nfixed_timestep_s=0.01",
                "[0.0, 0.0, 0.0]",
            ),
        );
        let output = command(&dir.0)
            .args([
                "inspect",
                "./scene.toml",
                "--config",
                "config.toml",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        success(&output);
        assert_eq!(document(&output)["configuration"]["name"], "selected");
        assert_eq!(
            document(&output)["configuration"]["provenance"]["fixed_timestep_s"],
            "command_line"
        );
    }
}

#[test]
fn named_and_external_inputs_produce_identical_physical_results() {
    let external = Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes/falling-block-04/scene.toml");
    let named = cli(&[
        "run",
        "falling-block-04",
        "--steps",
        "6",
        "--checkpoints",
        "0,3,6",
        "--format",
        "json",
    ]);
    let supplied = cli(&[
        "run",
        external.to_str().unwrap(),
        "--steps",
        "6",
        "--checkpoints",
        "0,3,6",
        "--format",
        "json",
    ]);
    success(&named);
    success(&supplied);
    assert_eq!(document(&named), document(&supplied));
    assert_eq!(document(&named)["completed_steps"], "6");
    assert_eq!(document(&named)["checkpoints"].as_array().unwrap().len(), 3);
}

#[test]
fn cli_run_matches_independently_known_free_fall_state() {
    let dir = Directory::new();
    dir.write(
        "scene.toml",
        &scene("source='built_in_defaults'", "", "[0.0, 0.0, 0.0]"),
    );
    let output = command(&dir.0)
        .args([
            "run",
            "./scene.toml",
            "--steps",
            "1",
            "--set",
            "fixed_timestep_s=0.125",
            "--set",
            "external_acceleration_m_s2=[0.0,-8.0,0.0]",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    success(&output);
    let result = document(&output);
    let particle = &result["final_snapshot"]["particles"][0];
    assert_eq!(particle["identity"], "7");
    assert_eq!(particle["position_m"]["y"], -0.125);
    assert_eq!(particle["velocity_m_s"]["y"], -1.0);
    assert_eq!(result["status"], "completed_with_warnings");
    assert!(String::from_utf8(output.stderr).unwrap().contains("CFL"));
}

#[test]
fn result_publication_snapshots_and_replay_reproduce_the_entire_run() {
    let dir = Directory::new();
    let first = command(&dir.0)
        .args([
            "run",
            "falling-block-04",
            "--steps",
            "6",
            "--checkpoints",
            "0,3,6",
            "--format",
            "json",
            "--output",
            "result.json",
            "--snapshot",
            "final.json",
        ])
        .output()
        .unwrap();
    success(&first);
    assert!(first.stdout.is_empty());
    let result: Value =
        serde_json::from_slice(&fs::read(dir.path("result.json")).unwrap()).unwrap();
    let snapshot: Value =
        serde_json::from_slice(&fs::read(dir.path("final.json")).unwrap()).unwrap();
    assert_eq!(snapshot, result["final_snapshot"]);
    let replay = command(&dir.0)
        .args(["replay", "result.json", "--format", "json"])
        .output()
        .unwrap();
    success(&replay);
    assert_eq!(document(&replay), result);
    assert_eq!(
        String::from_utf8(replay.stdout).unwrap(),
        format!("{}\n", fs::read_to_string(dir.path("result.json")).unwrap())
    );
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
}

#[test]
fn supplied_scene_and_selected_configuration_replay_without_original_files() {
    let dir = Directory::new();
    dir.write(
        "scene.toml",
        &scene("source='file'\npath='superseded.toml'", "", "[0.0,0.0,0.0]"),
    );
    dir.write("selected.toml", "schema_version=1\nname='explicit selection'\ndescription='exact free-fall configuration'\n[numerical]\nfixed_timestep_s=0.125\nexternal_acceleration_m_s2=[0.0,-8.0,0.0]\n");
    let output = command(&dir.0)
        .args([
            "run",
            "./scene.toml",
            "--config",
            "selected.toml",
            "--set",
            "fixed_timestep_s=0.125",
            "--steps",
            "1",
            "--checkpoints",
            "0,1",
            "--format",
            "json",
            "--output",
            "result.json",
        ])
        .output()
        .unwrap();
    success(&output);
    assert!(output.stdout.is_empty());
    let record = fs::read_to_string(dir.path("result.json")).unwrap();
    let result: Value = serde_json::from_str(&record).unwrap();
    assert_eq!(
        result["input"]["configuration"]["name"],
        "explicit selection"
    );
    assert_eq!(
        result["final_snapshot"]["particles"][0]["position_m"]["y"],
        -0.125
    );
    let legacy =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/scenes/falling-block-04.toml");
    let named = command(&dir.0)
        .args([
            "run",
            legacy.to_str().unwrap(),
            "--config",
            "selected.toml",
            "--set",
            "fixed_timestep_s=0.125",
            "--steps",
            "1",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    success(&named);
    assert_eq!(
        document(&named)["input"]["configuration"]["name"],
        "explicit selection"
    );
    fs::remove_file(dir.path("scene.toml")).unwrap();
    fs::remove_file(dir.path("selected.toml")).unwrap();
    let replay = command(&dir.0)
        .args(["replay", "result.json", "--format", "json"])
        .output()
        .unwrap();
    success(&replay);
    assert_eq!(
        String::from_utf8(replay.stdout).unwrap(),
        format!("{record}\n")
    );
}

#[test]
fn diagnostic_and_checkpoint_choices_preserve_accepted_state() {
    let dir = Directory::new();
    let mut results = Vec::new();
    let base = [
        "run",
        "falling-block-04",
        "--steps",
        "3",
        "--set",
        "cfl_factor=0.000001",
    ];
    for flags in [
        vec!["--format", "json"],
        vec!["--format", "json", "--quiet"],
        vec!["--format", "json", "--verbose", "--checkpoints", "0,3"],
    ] {
        let output = command(&dir.0).args(base).args(&flags).output().unwrap();
        success(&output);
        let result = document(&output);
        assert_eq!(result["status"], "completed_with_warnings");
        assert!(!output.stdout.contains(&0x1b));
        assert!(!output.stderr.contains(&0x1b));
        if flags.contains(&"--quiet") {
            assert!(output.stderr.is_empty());
        } else {
            assert!(String::from_utf8_lossy(&output.stderr).contains("CFL"));
        }
        if flags.contains(&"--verbose") {
            assert!(String::from_utf8_lossy(&output.stderr).contains("running 3 steps"));
        }
        results.push(result);
    }
    for result in &results[1..] {
        assert_eq!(
            state_bits(&result["final_snapshot"]),
            state_bits(&results[0]["final_snapshot"])
        );
        for key in ["metadata", "input", "final_snapshot", "steps", "status"] {
            assert_eq!(result[key], results[0][key]);
        }
    }
    let text = command(&dir.0)
        .args(base)
        .args(["--snapshot", "text-state.json"])
        .output()
        .unwrap();
    success(&text);
    assert!(String::from_utf8_lossy(&text.stdout).contains("CompletedWithWarnings"));
    let snapshot: Value =
        serde_json::from_slice(&fs::read(dir.path("text-state.json")).unwrap()).unwrap();
    assert_eq!(snapshot, results[0]["final_snapshot"]);
    assert_eq!(
        state_bits(&snapshot),
        state_bits(&results[0]["final_snapshot"])
    );
}

#[test]
fn duration_uses_the_effective_fixed_timestep_and_rounds_up() {
    for (duration, expected) in [("0", "0"), ("0.25", "2"), ("0.26", "3")] {
        let output = cli(&[
            "run",
            "falling-block-04",
            "--duration",
            duration,
            "--set",
            "fixed_timestep_s=0.125",
            "--format",
            "json",
            "--quiet",
        ]);
        success(&output);
        assert_eq!(document(&output)["request"]["steps"], expected);
        assert_eq!(document(&output)["completed_steps"], expected);
    }
}

#[test]
fn invalid_use_has_actionable_stderr_and_no_result_stdout() {
    for args in [
        vec![],
        vec!["rn"],
        vec!["run", "falling-block-04", "--steps", "1", "--duration", "1"],
        vec!["run", "--steps", "1"],
        vec![
            "run",
            "falling-block-04",
            "--steps",
            "1",
            "--duration",
            "0.1",
        ],
        vec!["run", "falling-block-04", "--steps", "-1"],
        vec!["run", "falling-block-04", "--steps", "18446744073709551616"],
        vec!["run", "falling-block-04", "--duration", "NaN"],
        vec!["run", "falling-block-04", "--duration", "inf"],
        vec!["run", "falling-block-04", "--duration=-0.1"],
        vec!["run", "falling-block-04", "--duration", "1e30"],
        vec![
            "run",
            "falling-block-04",
            "--steps",
            "2",
            "--checkpoints",
            "0,0",
        ],
        vec![
            "run",
            "falling-block-04",
            "--steps",
            "2",
            "--checkpoints",
            "2,1",
        ],
        vec![
            "run",
            "falling-block-04",
            "--steps",
            "2",
            "--checkpoints",
            "3",
        ],
        vec!["inspect", "--set", "missing=1"],
        vec!["inspect", "--set", "cfl_factor"],
        vec![
            "inspect",
            "--set",
            "cfl_factor=0.3",
            "--set",
            "cfl_factor=0.4",
        ],
        vec!["inspect", "--set", "cfl_factor=0.3\nfixed_timestep_s=1.0"],
        vec!["inspect", "--format", "yaml"],
        vec!["inspect", "--quiet", "--verbose"],
    ] {
        let output = cli(&args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "args {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn replay_rejects_the_previous_projection_contract_explicitly() {
    let dir = Directory::new();
    let run = cli(&[
        "run",
        "--backend",
        "serial-grid",
        "falling-block-04",
        "--steps",
        "0",
        "--set",
        "solver_iterations=4",
        "--format",
        "json",
    ]);
    success(&run);
    let mut previous = document(&run);
    assert_eq!(previous["schema_version"], 1);
    assert_eq!(
        previous["backend"],
        "serial-uniform-grid-pbf-binary32-four-jacobi-v2"
    );
    previous["backend"] = "serial-brute-force-pbf-binary32-four-jacobi-v1".into();
    dir.write("previous.json", &previous.to_string());
    let replay = command(&dir.0)
        .args(["replay", "previous.json", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(replay.status.code(), Some(3));
    assert!(replay.stdout.is_empty());
    let error = String::from_utf8(replay.stderr).unwrap();
    assert!(error.contains("four-jacobi-v1"));
    assert!(error.contains("four-jacobi-v2"));
    assert!(error.contains("original binary or rerun the scene"));
}

#[test]
fn invalid_inputs_and_malformed_records_are_distinct_from_usage_errors() {
    let dir = Directory::new();
    dir.write(
        "bad.toml",
        "schema_version=99\nname='bad'\ndescription='bad input'",
    );
    dir.write("bad.json", "{\"schema_version\":1}");
    for args in [
        vec!["validate", "--config", "bad.toml"],
        vec!["run", "missing.toml", "--steps", "1"],
        vec!["replay", "bad.json"],
        vec!["replay", "missing.json"],
        vec!["inspect", "--set", "fixed_timestep_s=0.0"],
        vec![
            "run",
            "falling-block-04",
            "--config",
            "missing.toml",
            "--steps",
            "1",
        ],
    ] {
        let output = command(&dir.0).args(&args).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(3),
            "args {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn referenced_configuration_errors_name_the_dependency_file() {
    let dir = Directory::new();
    dir.write(
        "scene.toml",
        &scene(
            "source='file'\npath='missing-config.toml'",
            "",
            "[0.0,0.0,0.0]",
        ),
    );
    let output = command(&dir.0)
        .args(["validate", "./scene.toml", "--quiet"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("scene.toml"));
    assert!(error.contains("missing-config.toml"));
    assert!(output.stdout.is_empty());
}

#[cfg(unix)]
#[test]
fn broken_stdout_returns_an_output_failure_without_panicking() {
    let (reader, writer) = std::os::unix::net::UnixStream::pair().unwrap();
    drop(reader);
    let output = command(&std::env::temp_dir())
        .args(["inspect", "--format", "json"])
        .stdout(Stdio::from(std::os::fd::OwnedFd::from(writer)))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(5));
    let diagnostics = String::from_utf8(output.stderr).unwrap();
    assert!(diagnostics.contains("stdout"));
    assert!(!diagnostics.contains("panicked"));
}

#[test]
fn simulation_failure_reports_the_last_accepted_state_and_returns_failure() {
    let dir = Directory::new();
    dir.write(
        "scene.toml",
        &scene("source='built_in_defaults'", "", "[3.0e38,0.0,0.0]"),
    );
    let output = command(&dir.0)
        .args([
            "run",
            "./scene.toml",
            "--steps",
            "2",
            "--format",
            "json",
            "--snapshot",
            "failed-state.json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = document(&output);
    assert_eq!(result["status"], "simulation_failure");
    assert_eq!(result["completed_steps"], "0");
    assert_eq!(
        result["final_snapshot"]["particles"][0]["position_m"]["x"],
        0.0
    );
    assert_eq!(
        result["input"]["particles"],
        result["final_snapshot"]["particles"]
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("SimulationFailure"));
    assert!(dir.path("failed-state.json").exists());
}

#[test]
fn output_paths_refuse_replacement_aliases_and_missing_directories() {
    let dir = Directory::new();
    dir.write("existing.json", "preserve me");
    for flags in [
        vec!["--output", "existing.json"],
        vec!["--snapshot", "existing.json"],
        vec!["--output", "new.json", "--snapshot", "./new.json"],
        vec!["--output", "missing/new.json"],
        vec!["--snapshot", "missing/new.json"],
        vec!["--output", "."],
    ] {
        let output = command(&dir.0)
            .args(["run", "falling-block-04", "--steps", "1"])
            .args(&flags)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(5),
            "flags {flags:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
        assert_eq!(
            fs::read_to_string(dir.path("existing.json")).unwrap(),
            "preserve me"
        );
        assert!(!dir.path("new.json").exists());
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
    }
}

#[cfg(unix)]
#[test]
fn dangling_symlinks_are_not_replaced() {
    let dir = Directory::new();
    std::os::unix::fs::symlink("missing.json", dir.path("link.json")).unwrap();
    let output = command(&dir.0)
        .args([
            "run",
            "falling-block-04",
            "--steps",
            "0",
            "--output",
            "link.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(5));
    assert_eq!(
        fs::read_link(dir.path("link.json")).unwrap(),
        Path::new("missing.json")
    );
    assert!(!dir.path("missing.json").exists());
}

#[cfg(unix)]
fn cancelled_cli_for(backend: &str) {
    use std::{
        io::{BufRead, BufReader},
        sync::mpsc,
        time::Duration,
    };
    let dir = Directory::new();
    let mut child = command(&dir.0)
        .args([
            "run",
            "falling-block-04",
            "--backend",
            backend,
            "--steps",
            "1000000",
            "--verbose",
            "--format",
            "json",
            "--output",
            "cancelled.json",
            "--snapshot",
            "cancelled-state.json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let (send, receive) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut diagnostics = String::new();
        for line in BufReader::new(stderr).lines() {
            let line = line.unwrap();
            if line.contains("running 1000000 steps") {
                send.send(()).unwrap();
            }
            diagnostics.push_str(&line);
            diagnostics.push('\n');
        }
        diagnostics
    });
    if receive.recv_timeout(Duration::from_secs(10)).is_err() {
        let _ = child.kill();
        let _ = child.wait();
        panic!("CLI never reported its installed signal handler");
    }
    let signal = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(signal.success());
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CLI did not cooperate with Ctrl-C");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(130));
    assert!(output.stdout.is_empty());
    assert!(reader.join().unwrap().contains("Cancelled"));
    let result: Value =
        serde_json::from_slice(&fs::read(dir.path("cancelled.json")).unwrap()).unwrap();
    assert_eq!(result["status"], "cancelled");
    assert_eq!(
        result["completed_steps"],
        result["final_snapshot"]["completed_steps"]
    );
    let steps: u64 = result["completed_steps"].as_str().unwrap().parse().unwrap();
    assert!(steps < 1000000);
    let expected = cli(&[
        "run",
        "falling-block-04",
        "--backend",
        backend,
        "--steps",
        &steps.to_string(),
        "--format",
        "json",
    ]);
    success(&expected);
    assert_eq!(
        result["final_snapshot"],
        document(&expected)["final_snapshot"]
    );
    let snapshot: Value =
        serde_json::from_slice(&fs::read(dir.path("cancelled-state.json")).unwrap()).unwrap();
    assert_eq!(snapshot, result["final_snapshot"]);
}

#[test]
fn measured_cli_retains_replay_identity_and_profiles_only_when_requested() {
    let args = [
        "run",
        "--unpaced",
        "falling-block-04",
        "--steps",
        "3",
        "--refine-timestep",
        "4",
        "--checkpoints",
        "0,1,3",
        "--format",
        "json",
        "--quiet",
    ];
    let normal = cli(&args);
    success(&normal);
    let expected = document(&normal);
    assert!(expected.get("measurements").is_none());
    for stages in [false, true] {
        let mut measured_args = args.to_vec();
        measured_args.push("--measure");
        if stages {
            measured_args.push("--stage-profile");
        }
        let output = cli(&measured_args);
        success(&output);
        let mut actual = document(&output);
        let timings = actual
            .as_object_mut()
            .unwrap()
            .remove("measurements")
            .unwrap();
        assert_eq!(timings["schema_version"], 1);
        assert_eq!(timings["mode"], if stages { "stages" } else { "total" });
        assert_eq!(timings["steps"].as_array().unwrap().len(), 3);
        assert_eq!(timings["steps"][0]["stages"].is_object(), stages);
        assert_eq!(
            serde_json::to_string(&actual).unwrap(),
            serde_json::to_string(&expected).unwrap()
        );
        let directory = Directory::new();
        let path = directory.write("measured.json", &String::from_utf8(output.stdout).unwrap());
        let replay = command(&directory.0)
            .args([
                "replay",
                path.to_str().unwrap(),
                "--format",
                "json",
                "--quiet",
            ])
            .output()
            .unwrap();
        success(&replay);
        assert_eq!(
            serde_json::to_string(&document(&replay)).unwrap(),
            serde_json::to_string(&expected).unwrap()
        );
    }
    let invalid = cli(&["run", "falling-block-04", "--steps", "1", "--stage-profile"]);
    assert_eq!(invalid.status.code(), Some(2));
    let text = cli(&[
        "run",
        "falling-block-04",
        "--steps",
        "0",
        "--measure",
        "--unpaced",
    ]);
    success(&text);
    assert!(
        String::from_utf8(text.stdout)
            .unwrap()
            .contains("Solver measurements Total: 0 accepted steps")
    );
}

#[cfg(unix)]
#[test]
fn ctrl_c_returns_a_cancelled_result_and_preserves_accepted_state() {
    cancelled_cli_for("brute-force");
}

#[cfg(unix)]
#[test]
fn grid_ctrl_c_publishes_its_last_accepted_state() {
    cancelled_cli_for("uniform-grid");
}

#[test]
fn explicit_discovery_selection_preserves_inputs_and_replays_every_work_count() {
    for iterations in [1, 4, 8, 64] {
        let dir = Directory::new();
        let work = format!("solver_iterations={iterations}");
        let base = [
            "run",
            "--unpaced",
            "falling-block-04",
            "--steps",
            "2",
            "--checkpoints",
            "0,1,2",
            "--set",
            &work,
            "--format",
            "json",
            "--quiet",
        ];
        let mut reference_args = base.to_vec();
        reference_args.extend(["--backend", "brute-force"]);
        let reference = cli(&reference_args);
        success(&reference);
        let explicit = command(&dir.0)
            .args(base)
            .args(["--backend", "brute-force"])
            .output()
            .unwrap();
        success(&explicit);
        assert_eq!(reference.stdout, explicit.stdout);
        let reference = document(&reference);
        for flags in [
            vec![],
            vec!["--measure"],
            vec!["--measure", "--stage-profile"],
        ] {
            let output = command(&dir.0)
                .args(base)
                .args(["--backend", "uniform-grid"])
                .args(&flags)
                .output()
                .unwrap();
            success(&output);
            let result = document(&output);
            let count = if iterations == 4 {
                "four".into()
            } else {
                iterations.to_string()
            };
            assert_eq!(
                result["backend"],
                format!("serial-uniform-grid-pbf-binary32-{count}-jacobi-v2")
            );
            assert_eq!(result["metadata"]["backend"], result["backend"]);
            assert_ne!(
                result["metadata"]["run_id"],
                reference["metadata"]["run_id"]
            );
            assert_eq!(
                support::serial_document(result.clone()),
                support::serial_document(reference.clone())
            );
            let file = dir.write("record.json", std::str::from_utf8(&output.stdout).unwrap());
            let replay = command(&dir.0)
                .args([
                    "replay",
                    file.to_str().unwrap(),
                    "--format",
                    "json",
                    "--quiet",
                ])
                .output()
                .unwrap();
            success(&replay);
            let mut expected = result;
            expected.as_object_mut().unwrap().remove("measurements");
            assert_eq!(
                serde_json::to_string(&document(&replay)).unwrap(),
                serde_json::to_string(&expected).unwrap()
            );
            fs::remove_file(file).unwrap();
        }
    }
}

#[test]
fn unsupported_discovery_and_replay_contracts_cannot_silently_substitute() {
    let output = cli(&[
        "run",
        "falling-block-04",
        "--steps",
        "1",
        "--backend",
        "unsupported",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let dir = Directory::new();
    let output = cli(&[
        "run",
        "falling-block-04",
        "--steps",
        "0",
        "--backend",
        "uniform-grid",
        "--format",
        "json",
    ]);
    success(&output);
    let result = document(&output);
    for backend in [
        "serial-uniform-grid-pbf-binary32-8-jacobi-v1",
        "serial-uniform-grid-pbf-binary32-four-jacobi-v2",
        "unknown",
    ] {
        let mut invalid = result.clone();
        invalid["backend"] = backend.into();
        dir.write("invalid.json", &invalid.to_string());
        let output = command(&dir.0)
            .args(["replay", "invalid.json", "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3));
        assert!(output.stdout.is_empty());
    }
    let output = command(&dir.0)
        .args(["replay", "invalid.json", "--backend", "brute-force"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn grid_publication_keeps_no_clobber_guarantees_and_snapshot_bits() {
    let dir = Directory::new();
    let args = [
        "run",
        "falling-block-04",
        "--backend",
        "uniform-grid",
        "--steps",
        "2",
        "--format",
        "json",
        "--output",
        "grid.json",
        "--snapshot",
        "state.json",
    ];
    let output = command(&dir.0).args(args).output().unwrap();
    success(&output);
    assert!(output.stdout.is_empty());
    let original = fs::read(dir.path("grid.json")).unwrap();
    let result: Value = serde_json::from_slice(&original).unwrap();
    let snapshot: Value =
        serde_json::from_slice(&fs::read(dir.path("state.json")).unwrap()).unwrap();
    assert_eq!(state_bits(&snapshot), state_bits(&result["final_snapshot"]));
    let repeated = command(&dir.0).args(args).output().unwrap();
    assert_eq!(repeated.status.code(), Some(5));
    assert_eq!(fs::read(dir.path("grid.json")).unwrap(), original);
    let help = cli(&["run", "--help"]);
    success(&help);
    let help = String::from_utf8(help.stdout).unwrap();
    for expected in ["--backend", "brute-force", "serial-grid"] {
        assert!(help.contains(expected));
    }
}

#[test]
fn threaded_selection_replay_and_environment_independence_are_public() {
    let directory = Directory::new();
    let args = [
        "run",
        "falling-block-04",
        "--backend",
        "uniform-grid",
        "--execution",
        "threaded",
        "--threads",
        "3",
        "--steps",
        "3",
        "--checkpoints",
        "0,1,3",
        "--format",
        "json",
    ];
    let output = command(&directory.0)
        .env("RAYON_NUM_THREADS", "17")
        .args(args)
        .output()
        .unwrap();
    success(&output);
    let result = document(&output);
    assert_eq!(
        result["execution"],
        serde_json::json!({"mode":"threaded","workers":3})
    );
    assert_eq!(
        result["backend"],
        "threaded-uniform-grid-pbf-binary32-8-jacobi-v2"
    );
    let path = directory.write(
        "threaded.json",
        std::str::from_utf8(&output.stdout).unwrap(),
    );
    let replay = command(&directory.0)
        .env("RAYON_NUM_THREADS", "1")
        .args(["replay", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    success(&replay);
    assert_eq!(
        support::serial_document(result.clone()),
        support::serial_document(document(&replay))
    );
    let serial = cli(&[
        "run",
        "--backend",
        "serial-grid",
        "falling-block-04",
        "--steps",
        "3",
        "--checkpoints",
        "0,1,3",
        "--format",
        "json",
    ]);
    success(&serial);
    let mut actual = result;
    actual.as_object_mut().unwrap().remove("execution");
    assert_eq!(
        support::serial_document(actual),
        support::serial_document(document(&serial))
    );
    let text = cli(&[
        "run",
        "falling-block-04",
        "--backend",
        "uniform-grid",
        "--execution",
        "threaded",
        "--threads",
        "2",
        "--steps",
        "0",
    ]);
    success(&text);
    assert!(
        String::from_utf8(text.stdout)
            .unwrap()
            .contains("requested workers: 2; effective workers: 2")
    );
}

#[test]
fn threaded_default_uses_available_capacity_and_invalid_combinations_fail() {
    let default = cli(&[
        "run",
        "falling-block-04",
        "--backend",
        "uniform-grid",
        "--execution",
        "threaded",
        "--steps",
        "0",
        "--format",
        "json",
    ]);
    success(&default);
    assert_eq!(
        document(&default)["execution"],
        serde_json::to_value(maelstrom::execution::ExecutionMode::threaded_default().unwrap())
            .unwrap()
    );
    let cases: &[&[&str]] = &[
        &["--backend", "uniform-grid", "--threads", "2"],
        &[
            "--backend",
            "uniform-grid",
            "--execution",
            "threaded",
            "--threads",
            "0",
        ],
        &[
            "--backend",
            "uniform-grid",
            "--execution",
            "threaded",
            "--threads",
            "257",
        ],
        &[
            "--backend",
            "uniform-grid",
            "--execution",
            "threaded",
            "--threads",
            "-1",
        ],
        &[
            "--backend",
            "uniform-grid",
            "--execution",
            "threaded",
            "--threads",
            "many",
        ],
    ];
    for extra in cases {
        let mut args = vec![
            "run",
            "falling-block-04",
            "--steps",
            "0",
            "--format",
            "json",
        ];
        args.extend_from_slice(extra);
        let output = cli(&args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let help = cli(&["run", "--help"]);
    success(&help);
    let text = String::from_utf8(help.stdout).unwrap();
    assert!(!text.contains("--execution") && text.contains("--threads"));
}

#[test]
fn replay_cannot_silently_discard_or_invent_worker_settings() {
    let output = cli(&[
        "run",
        "falling-block-04",
        "--backend",
        "uniform-grid",
        "--execution",
        "threaded",
        "--threads",
        "2",
        "--steps",
        "0",
        "--format",
        "json",
    ]);
    success(&output);
    let original = document(&output);
    let directory = Directory::new();
    for execution in [
        serde_json::Value::Null,
        serde_json::json!({"mode":"threaded","workers":0}),
        serde_json::json!({"mode":"serial"}),
        serde_json::json!({"mode":"threaded"}),
        serde_json::json!({"mode":"threaded","workers":2,"unknown":true}),
    ] {
        let mut value = original.clone();
        value["execution"] = execution;
        assert!(maelstrom::workflow::replay_options(&value.to_string()).is_err());
    }
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("execution");
    let path = directory.write("missing.json", &missing.to_string());
    let output = cli(&["replay", path.to_str().unwrap(), "--format", "json"]);
    assert_eq!(output.status.code(), Some(3));
    let mut serial = original;
    serial["backend"] = "serial-uniform-grid-pbf-binary32-8-jacobi-v2".into();
    assert!(maelstrom::workflow::replay_options(&serial.to_string()).is_err());
}
