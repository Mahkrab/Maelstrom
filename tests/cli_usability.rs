//! Public journeys use isolated libraries so a developer's scenes and preferences
//! cannot affect selection, publication or numerical replay.
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Library(PathBuf);
impl Library {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "maelstrom-usability-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn command(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_maelstrom"));
        c.current_dir(&self.0)
            .env("XDG_DATA_HOME", self.0.join("data"))
            .env("XDG_CONFIG_HOME", self.0.join("settings"))
            .env_remove("MAELSTROM_SCENES_DIR")
            .env_remove("MAELSTROM_RUNS_DIR")
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY");
        c
    }
    fn run(&self, args: &[&str]) -> Output {
        self.command().args(args).output().unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let out = self
            .command()
            .args(args)
            .args(["--format", "json"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    fn create(&self, name: &str) {
        let out = self.run(&["scenes", "create", name]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn help_is_example_free_and_legacy_commands_are_not_advertised() {
    let lib = Library::new();
    let top = lib.run(&["--help"]);
    let text = String::from_utf8(top.stdout).unwrap();
    for name in [
        "compile",
        "run",
        "play",
        "view",
        "doctor",
        "runs",
        "paths",
        "completions",
        "rerun",
    ] {
        assert!(text.contains(name));
    }
    assert!(!text.contains("cuda-check") && !text.contains("replay"));
    for args in [
        vec!["help"],
        vec!["help", "compile"],
        vec!["run", "--help"],
        vec!["doctor", "cuda", "--help"],
        vec!["scenes", "create", "--help"],
        vec!["play", "--help"],
    ] {
        let out = lib.run(&args);
        assert!(out.status.success());
        assert!(
            !String::from_utf8(out.stdout)
                .unwrap()
                .to_lowercase()
                .contains("example")
        );
    }
    assert!(!lib.0.join("data").exists());
}
#[test]
fn paths_obey_flags_then_environment_then_xdg_without_creating_libraries() {
    let lib = Library::new();
    let paths = lib.json(&["paths"]);
    assert_eq!(
        paths["scenes"],
        lib.0
            .join("data/maelstrom/scenes")
            .to_string_lossy()
            .as_ref()
    );
    let out = lib
        .command()
        .env("MAELSTROM_SCENES_DIR", lib.0.join("environment"))
        .args(["paths", "--scenes-dir", "explicit", "--format", "json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let paths: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        paths["scenes"],
        lib.0.join("explicit").to_string_lossy().as_ref()
    );
    assert!(!lib.0.join("explicit").exists());
    let invalid = lib
        .command()
        .env("MAELSTROM_SCENES_DIR", "relative")
        .arg("paths")
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(3));
    assert!(invalid.stdout.is_empty());
}
#[test]
fn scene_packages_are_editable_portable_and_do_not_shadow_bundled_names() {
    let lib = Library::new();
    let bundled_count = lib.json(&["scenes", "list"])["scenes"]
        .as_array()
        .unwrap()
        .len();
    lib.create("water");
    let path = lib.0.join("data/maelstrom/scenes/water");
    assert!(path.join("scene.toml").is_file() && path.join("config.toml").is_file());
    let selected = lib.json(&["inspect", "water", "--backend", "serial-grid"]);
    assert_eq!(selected["scene"]["particle_count"], 64);
    let moved = lib
        .command()
        .current_dir(std::env::temp_dir())
        .args([
            "inspect",
            "water",
            "--backend",
            "serial-grid",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        moved.status.success(),
        "{}",
        String::from_utf8_lossy(&moved.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&moved.stdout).unwrap(),
        selected
    );
    let before = fs::read(path.join("scene.toml")).unwrap();
    assert_eq!(
        lib.run(&["scenes", "create", "water"]).status.code(),
        Some(5)
    );
    assert_eq!(fs::read(path.join("scene.toml")).unwrap(), before);
    assert_eq!(
        lib.run(&["scenes", "create", "../escape"]).status.code(),
        Some(3)
    );
    lib.create("falling-block-04");
    assert_eq!(
        lib.run(&["inspect", "falling-block-04"]).status.code(),
        Some(3)
    );
    assert!(
        lib.run(&["validate", "builtin:falling-block-04"])
            .status
            .success()
    );
    assert!(
        lib.run(&["validate", "user:falling-block-04"])
            .status
            .success()
    );
    let first = lib.json(&["scenes", "list"]);
    assert_eq!(first, lib.json(&["scenes", "list"]));
    assert_eq!(first["scenes"].as_array().unwrap().len(), bundled_count + 2);
}

#[test]
fn every_bundled_scene_matches_its_file_and_copies_its_own_configuration() {
    let lib = Library::new();
    let rows = lib.json(&["scenes", "list"]);
    for entry in rows["scenes"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let selector = entry["selector"].as_str().unwrap();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("scenes")
            .join(format!("{name}.toml"));
        let bundled = lib.json(&["inspect", selector, "--backend", "serial-grid"]);
        let file = lib.json(&[
            "inspect",
            path.to_str().unwrap(),
            "--backend",
            "serial-grid",
        ]);
        assert_eq!(bundled, file, "{name}");
    }
    let source = lib.json(&[
        "inspect",
        "builtin:falling-block-10-playground",
        "--backend",
        "serial-grid",
    ]);
    assert_eq!(
        source["configuration"]["name"],
        "falling-block-10-playground-config"
    );
    let created = lib.run(&[
        "scenes",
        "create",
        "playground-copy",
        "--from",
        "builtin:falling-block-10-playground",
    ]);
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stderr)
    );
    let copy = lib.json(&[
        "inspect",
        "user:playground-copy",
        "--backend",
        "serial-grid",
    ]);
    for field in ["numerical", "derived", "comparison_key"] {
        assert!(!source["configuration"][field].is_null(), "{field}");
        assert_eq!(
            source["configuration"][field], copy["configuration"][field],
            "{field}"
        );
    }
    for field in ["particle_count", "planes"] {
        assert_eq!(source["scene"][field], copy["scene"][field], "{field}");
    }
    let replacement = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/defaults-v4.toml");
    let replaced = lib.json(&[
        "inspect",
        "builtin:falling-block-10-playground",
        "--config",
        replacement.to_str().unwrap(),
        "--backend",
        "serial-grid",
    ]);
    assert_eq!(replaced["configuration"]["name"], "maelstrom-defaults-v4");
    assert_eq!(
        replaced["configuration"]["numerical"]["relaxation_scale"],
        20.0
    );
}
#[test]
fn preferences_are_explicit_and_cannot_change_scene_physics() {
    let lib = Library::new();
    let directory = lib.0.join("settings/maelstrom");
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("config.toml"),
        "schema_version=1\nbackend='threaded-grid'\nthreads=2\n",
    )
    .unwrap();
    let selected = lib.json(&["inspect", "falling-block-04"]);
    assert_eq!(selected["execution"]["resources"]["workers"], 2);
    let explicit = lib.json(&["inspect", "falling-block-04", "--backend", "serial-grid"]);
    assert_eq!(explicit["execution"]["resources"]["mode"], "serial");
    assert_eq!(selected["configuration"], explicit["configuration"]);
    fs::write(
        directory.join("config.toml"),
        "schema_version=1\nfixed_timestep_s=1.0\n",
    )
    .unwrap();
    assert_eq!(lib.run(&["config", "show"]).status.code(), Some(3));
    for invalid in [
        "backend='unknown'",
        "threads=0",
        "threads=257",
        "block_size=33",
        "backend='serial-grid'\nthreads=2",
        "threads=2\ndevice=0",
    ] {
        fs::write(
            directory.join("config.toml"),
            format!("schema_version=1\n{invalid}\n"),
        )
        .unwrap();
        let out = lib.run(&["config", "show"]);
        assert_eq!(out.status.code(), Some(3), "{invalid}");
        assert!(out.stdout.is_empty());
    }
}
#[test]
fn paced_and_unpaced_execution_keep_identical_accepted_results() {
    let lib = Library::new();
    let args = [
        "run",
        "falling-block-04",
        "--steps",
        "2",
        "--backend",
        "serial-grid",
        "--quiet",
    ];
    let normal = lib.json(&args);
    let mut unpaced = args.to_vec();
    unpaced.push("--unpaced");
    assert_eq!(normal, lib.json(&unpaced));
    let output = lib.run(&["run", "falling-block-04", "--steps", "0", "--measure"]);
    assert_eq!(output.status.code(), Some(2));
    let measured = lib.json(&[
        "run",
        "falling-block-04",
        "--steps",
        "0",
        "--backend",
        "serial-grid",
        "--measure",
        "--unpaced",
    ]);
    assert_eq!(
        measured["measurements"]["steps"].as_array().unwrap().len(),
        0
    );
    for extra in [
        vec!["--backend", "serial-grid", "--threads", "2"],
        vec!["--threads", "0"],
        vec!["--threads", "2", "--device", "0"],
    ] {
        let mut args = vec!["run", "falling-block-04", "--steps", "0"];
        args.extend(extra);
        assert_eq!(lib.run(&args).status.code(), Some(2));
    }
}
#[test]
fn explicit_resources_are_recorded_and_rerun_without_the_scene_library() {
    let lib = Library::new();
    lib.create("water");
    let out = lib.run(&[
        "run",
        "water",
        "--steps",
        "1",
        "--threads",
        "2",
        "--unpaced",
        "--format",
        "json",
        "--output",
        "result.json",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let expected: Value =
        serde_json::from_slice(&fs::read(lib.0.join("result.json")).unwrap()).unwrap();
    assert_eq!(expected["execution"]["workers"], 2);
    fs::rename(lib.0.join("data/maelstrom/scenes"), lib.0.join("away")).unwrap();
    assert_eq!(lib.json(&["rerun", "./result.json"]), expected);
    assert_eq!(lib.json(&["replay", "./result.json"]), expected);
}
#[test]
fn diagnostic_discovery_and_completions_have_no_installation_side_effects() {
    let lib = Library::new();
    let report = lib.json(&["doctor"]);
    assert!(report["features"]["cuda"].is_boolean());
    for shell in ["bash", "zsh", "fish"] {
        let out = lib.run(&["completions", shell]);
        assert!(out.status.success());
        assert!(String::from_utf8(out.stdout).unwrap().contains("compile"));
    }
    assert_eq!(
        lib.json(&["runs", "list"])["runs"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert!(!lib.0.join("data").exists());
}
#[cfg(not(feature = "render"))]
#[test]
fn missing_graphics_is_actionable_and_does_not_create_output() {
    let lib = Library::new();
    for args in [
        vec!["compile", "falling-block-04", "--steps", "0"],
        vec!["doctor", "graphics"],
        vec!["play", "missing"],
    ] {
        let out = lib.run(&args);
        assert_eq!(out.status.code(), Some(3));
        assert!(String::from_utf8_lossy(&out.stderr).contains("--features"));
    }
    assert!(!lib.0.join("data").exists());
}
#[cfg(feature = "render")]
#[test]
#[ignore = "requires a Vulkan software adapter"]
fn compiled_runs_relocate_validate_and_rerun_without_original_scenes() {
    let lib = Library::new();
    lib.create("water");
    let bundle = lib.json(&[
        "compile",
        "water",
        "--steps",
        "0",
        "--backend",
        "serial-grid",
        "--software",
        "--width",
        "32",
        "--height",
        "32",
        "--name",
        "Water",
    ]);
    let original = PathBuf::from(bundle["path"].as_str().unwrap());
    let id = bundle["id"].as_str().unwrap();
    assert!(lib.run(&["runs", "validate", id]).status.success());
    let relocated = lib.0.join("moved");
    fs::rename(&original, &relocated).unwrap();
    fs::rename(lib.0.join("data/maelstrom/scenes"), lib.0.join("away")).unwrap();
    assert!(
        lib.run(&["runs", "validate", relocated.to_str().unwrap()])
            .status
            .success()
    );
    let rerun = lib.json(&["rerun", relocated.to_str().unwrap()]);
    assert_eq!(rerun["completed_steps"], "0");
    let manifest = fs::read(relocated.join("manifest.json")).unwrap();
    let frame = fs::read(relocated.join("frame-00000000.png")).unwrap();
    let sequence = maelstrom::playback::Sequence::open(&relocated).unwrap();
    assert_eq!(sequence.frame_count(), 1);
    sequence.decode(0).unwrap();
    assert_eq!(fs::read(relocated.join("manifest.json")).unwrap(), manifest);
    assert_eq!(
        fs::read(relocated.join("frame-00000000.png")).unwrap(),
        frame
    );
    let path = relocated.join("frame-00000000.png");
    fs::write(path, b"invalid").unwrap();
    assert_eq!(
        lib.run(&["runs", "validate", relocated.to_str().unwrap()])
            .status
            .code(),
        Some(3)
    );
}
