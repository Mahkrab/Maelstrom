//! Portable scene packages are exercised through separate processes so neither
//! cached inputs nor a developer's working directory can hide resolution errors.

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
        let path = std::env::temp_dir().join(format!(
            "maelstrom-saved-settings-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        // These fixtures exercise saved custom schedules, independently of the application preset.
        fs::create_dir_all(path.join("preferences/maelstrom")).unwrap();
        fs::write(
            path.join("preferences/maelstrom/config.toml"),
            "schema_version=1\nnumerical_mode='scene'\n",
        )
        .unwrap();
        Self(path)
    }
    fn scenes(&self) -> PathBuf {
        self.0.join("scenes")
    }
    fn package(&self) -> PathBuf {
        self.scenes().join("water")
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_maelstrom"));
        command
            .current_dir(&self.0)
            .env("MAELSTROM_SCENES_DIR", self.scenes())
            .env("MAELSTROM_RUNS_DIR", self.0.join("runs"))
            .env("XDG_CONFIG_HOME", self.0.join("preferences"))
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY");
        command
    }
    fn run(&self, args: &[&str]) -> Output {
        self.command()
            .args(args)
            .args(["--format", "json"])
            .output()
            .unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
    fn inspect(&self) -> Value {
        self.json(&["inspect", "user:water", "--backend", "serial-grid"])
    }
    fn seed(&self) {
        fs::create_dir_all(self.package()).unwrap();
        fs::write(self.package().join("scene.toml"),"schema_version=2\nname='water'\ndescription='one freely falling particle'\nrandomness='none'\n[configuration]\nsource='file'\npath='config.toml'\n[run]\nsteps=2\n[particles]\nconstruction='lattice'\nfirst_identity=0\norigin_m=[0.0,2.0,0.0]\ncounts=[1,1,1]\nendpoint_rule='counted_half_open'\norder='x_then_y_then_z'\ninitial_velocity_m_s=[0.0,0.0,0.0]").unwrap();
        self.config("fixed_timestep_s=0.125\ncorrection_scale=0.5\nrefine_timestep=2\nexternal_acceleration_m_s2=[0.0,-2.0,0.0]");
    }
    fn config(&self, values: &str) {
        fs::write(self.package().join("config.toml"),format!("schema_version=2\nname='model'\ndescription='saved numerical input'\n[numerical]\n{values}")).unwrap();
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn saved_limits_and_refinement_reach_execution_and_recorded_inputs_do_not_reapply_them() {
    let lib = Library::new();
    lib.seed();
    let inspected = lib.inspect();
    assert_eq!(inspected["run"]["steps"], 2);
    assert_eq!(inspected["run"]["source"], "scene");
    assert_eq!(
        inspected["configuration"]["numerical"]["fixed_timestep_s"],
        0.0625
    );
    assert_eq!(inspected["refinement"]["factor"], 2);
    let result = lib.json(&["run", "user:water", "--backend", "serial-grid", "--unpaced"]);
    assert_eq!(result["completed_steps"], "2");
    assert_eq!(
        result["final_snapshot"]["particles"][0]["position_m"]["y"],
        1.9765625
    );
    assert_eq!(
        result["final_snapshot"]["particles"][0]["velocity_m_s"]["y"],
        -0.25
    );
    let overridden = lib.json(&[
        "run",
        "user:water",
        "--backend",
        "serial-grid",
        "--unpaced",
        "--duration",
        "0.0625",
    ]);
    assert_eq!(overridden["completed_steps"], "1");
    let path = lib.0.join("record.json");
    fs::write(&path, serde_json::to_vec(&result).unwrap()).unwrap();
    fs::remove_dir_all(lib.package()).unwrap();
    let replay = lib.json(&["rerun", path.to_str().unwrap()]);
    assert_eq!(replay["input"], result["input"]);
    assert_eq!(replay["final_snapshot"], result["final_snapshot"]);
    let mut mismatched = result.clone();
    mismatched["input"]["configuration"]["schema_version"] = 1.into();
    fs::write(&path, serde_json::to_vec(&mismatched).unwrap()).unwrap();
    let rejected = lib.run(&["rerun", path.to_str().unwrap()]);
    assert_eq!(rejected.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("versions must match"));
}

#[test]
fn saved_edits_and_cli_overrides_have_visible_origins_and_do_not_rewrite_files() {
    let lib = Library::new();
    lib.seed();
    lib.config("particle_spacing_m=0.125\nrefine_timestep=4");
    let before = fs::read(lib.package().join("config.toml")).unwrap();
    let original = lib.inspect();
    assert_eq!(
        original["configuration"]["numerical"]["particle_spacing_m"],
        0.125
    );
    let changed = lib.json(&[
        "inspect",
        "user:water",
        "--backend",
        "serial-grid",
        "--set",
        "particle_spacing_m=0.25",
        "--refine-timestep",
        "1",
        "--steps",
        "0",
    ]);
    assert_eq!(
        changed["configuration"]["numerical"]["particle_spacing_m"],
        0.25
    );
    assert_eq!(
        changed["configuration"]["provenance"]["particle_spacing_m"],
        "command_line"
    );
    assert_eq!(changed["refinement"]["factor"], 1);
    assert_eq!(changed["run"]["steps"], 0);
    assert_eq!(fs::read(lib.package().join("config.toml")).unwrap(), before);
    lib.config("particle_spacing_m=0.25");
    let edited = lib.inspect();
    assert_eq!(
        edited["configuration"]["numerical"]["particle_spacing_m"],
        0.25
    );
    assert_ne!(original["documents"], edited["documents"]);
    let duplicate = lib.run(&[
        "inspect",
        "user:water",
        "--backend",
        "serial-grid",
        "--set",
        "particle_spacing_m=0.25",
        "--set",
        "particle_spacing_m=0.125",
    ]);
    assert_eq!(duplicate.status.code(), Some(2));
    assert_eq!(
        lib.run(&[
            "inspect",
            "user:water",
            "--refine-timestep",
            "2",
            "--set",
            "refine_timestep=4"
        ])
        .status
        .code(),
        Some(2)
    );
}

#[test]
fn presentation_discovery_replacement_and_relocation_work_without_graphics() {
    let lib = Library::new();
    lib.seed();
    fs::write(
        lib.package().join("render.toml"),
        "schema_version=2\nwidth=640\nheight=480\nstep_stride=3",
    )
    .unwrap();
    let original = lib.inspect();
    assert_eq!(
        original["presentation"]["effective"]["settings"]["width"],
        640
    );
    assert_eq!(original["presentation"]["effective"]["camera_mode"], "fit");
    assert_eq!(
        original["presentation"]["provenance"]["width"],
        "render.toml"
    );
    fs::write(
        lib.0.join("replacement.toml"),
        "schema_version=1\nheight=200",
    )
    .unwrap();
    let replaced = lib.json(&[
        "inspect",
        "user:water",
        "--backend",
        "serial-grid",
        "--render-config",
        "replacement.toml",
        "--width",
        "320",
        "--frame-stride",
        "5",
        "--fit-camera",
    ]);
    assert_eq!(
        replaced["presentation"]["effective"]["settings"]["width"],
        320
    );
    assert_eq!(
        replaced["presentation"]["effective"]["settings"]["height"],
        200
    );
    assert_eq!(
        replaced["presentation"]["effective"]["settings"]["step_stride"],
        5
    );
    assert_eq!(
        replaced["presentation"]["provenance"]["width"],
        "command_line"
    );
    let replacement_only = lib.json(&[
        "inspect",
        "user:water",
        "--backend",
        "serial-grid",
        "--render-config",
        "replacement.toml",
    ]);
    assert_eq!(
        replacement_only["presentation"]["effective"]["settings"]["width"],
        1280
    );
    assert_eq!(
        replacement_only["presentation"]["effective"]["camera_mode"],
        "fixed"
    );
    let moved = lib.0.join("moved");
    fs::rename(lib.package(), &moved).unwrap();
    let other = lib.0.join("unrelated");
    fs::create_dir(&other).unwrap();
    let output = lib
        .command()
        .current_dir(&other)
        .args([
            "inspect",
            moved.join("scene.toml").to_str().unwrap(),
            "--backend",
            "serial-grid",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let relocated: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(relocated["configuration"], original["configuration"]);
    assert_eq!(relocated["presentation"], original["presentation"]);
    assert_eq!(relocated["documents"], original["documents"]);
    assert!(
        !String::from_utf8(output.stdout)
            .unwrap()
            .contains(lib.0.to_str().unwrap())
    );
}

#[test]
fn malformed_packages_fail_without_changing_the_previously_accepted_inputs() {
    let lib = Library::new();
    lib.seed();
    let accepted = lib.inspect();
    let scene = fs::read_to_string(lib.package().join("scene.toml")).unwrap();
    for invalid in [
        "",
        "steps=1\nduration_s=1.0",
        "duration_s=nan",
        "duration_s=-1.0",
        "duration_s=1e100",
        "steps=-1",
        "steps=1\nunknown=2",
    ] {
        let bad = scene.replace("[run]\nsteps=2", &format!("[run]\n{invalid}"));
        fs::write(lib.package().join("scene.toml"), bad).unwrap();
        assert_eq!(
            lib.run(&["validate", "user:water"]).status.code(),
            Some(3),
            "{invalid}"
        );
    }
    fs::write(lib.package().join("scene.toml"), scene).unwrap();
    fs::write(
        lib.package().join("render.toml"),
        "schema_version=2\nwidth=0",
    )
    .unwrap();
    assert_eq!(lib.run(&["validate", "user:water"]).status.code(), Some(3));
    assert!(
        lib.run(&[
            "run",
            "user:water",
            "--backend",
            "serial-grid",
            "--steps",
            "0"
        ])
        .status
        .success()
    );
    fs::remove_file(lib.package().join("render.toml")).unwrap();
    assert_eq!(lib.inspect(), accepted);
    assert_eq!(
        lib.run(&["inspect", "user:water", "--render-config", "missing.toml"])
            .status
            .code(),
        Some(3)
    );
    let mismatched = lib.0.join("old.toml");
    fs::write(
        &mismatched,
        "schema_version=1\nname='old'\ndescription='old model'",
    )
    .unwrap();
    assert_eq!(
        lib.run(&[
            "inspect",
            "user:water",
            "--config",
            mismatched.to_str().unwrap()
        ])
        .status
        .code(),
        Some(3)
    );
}

#[test]
fn migration_preserves_effective_physics_and_created_spacing_is_editable() {
    let lib = Library::new();
    let before = lib.json(&[
        "inspect",
        "builtin:falling-block-04",
        "--backend",
        "serial-grid",
    ]);
    lib.json(&[
        "scenes",
        "create",
        "water",
        "--from",
        "builtin:falling-block-04",
    ]);
    let migrated = lib.inspect();
    for field in ["numerical", "derived", "comparison_key"] {
        assert_eq!(
            before["configuration"][field],
            migrated["configuration"][field]
        );
    }
    let text = fs::read_to_string(lib.package().join("scene.toml")).unwrap();
    assert!(!text.contains("spacing_m") && !text.contains("overrides"));
    assert_eq!(migrated["run_defaults"]["duration_s"], 3.0);
    assert!(lib.package().join("render.toml").is_file());
    lib.config("particle_spacing_m=0.04");
    assert!(lib.run(&["validate", "user:water"]).status.success());
    let original = fs::read(lib.package().join("scene.toml")).unwrap();
    assert_eq!(
        lib.run(&["scenes", "create", "water"]).status.code(),
        Some(5)
    );
    assert_eq!(
        fs::read(lib.package().join("scene.toml")).unwrap(),
        original
    );
}

#[cfg(feature = "render")]
#[test]
fn invalid_colour_requests_fail_before_graphics_or_output_creation() {
    let lib = Library::new();
    lib.seed();
    fs::write(lib.package().join("render.toml"),"schema_version=2\n[particles]\ncolour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=0.0").unwrap();
    let output = lib.run(&[
        "compile",
        "user:water",
        "--backend",
        "serial-grid",
        "--output",
        "bundle",
    ]);
    assert_eq!(output.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&output.stderr).contains("energy bounds"));
    assert!(!lib.0.join("bundle").exists());
    assert!(!lib.0.join("runs").exists());
}

#[test]
fn shared_config_and_scene_overlays_remain_visible_with_explicit_base_replacement() {
    let lib = Library::new();
    lib.seed();
    let path = lib.package().join("scene.toml");
    let text = fs::read_to_string(&path).unwrap().replace(
        "[run]",
        "[configuration.overrides.numerical]\nexternal_acceleration_m_s2=[0.0,-4.0,0.0]\n[run]",
    );
    fs::write(&path, text).unwrap();
    fs::write(lib.0.join("shared.toml"),"schema_version=2\nname='shared'\ndescription='replacement'\n[numerical]\nexternal_acceleration_m_s2=[0.0,-8.0,0.0]\nparticle_spacing_m=0.125").unwrap();
    let replaced = lib.json(&[
        "inspect",
        "user:water",
        "--backend",
        "serial-grid",
        "--config",
        "shared.toml",
    ]);
    assert_eq!(
        replaced["configuration"]["numerical"]["external_acceleration_m_s2"]["y"],
        -4.0
    );
    assert_eq!(
        replaced["configuration"]["numerical"]["particle_spacing_m"],
        0.125
    );
    assert_eq!(
        replaced["configuration"]["provenance"]["external_acceleration_m_s2"],
        "scene"
    );
    assert_eq!(replaced["documents"][0]["label"], "shared.toml");
    let overridden = lib.json(&[
        "inspect",
        "user:water",
        "--backend",
        "serial-grid",
        "--config",
        "shared.toml",
        "--set",
        "external_acceleration_m_s2=[0.0,-1.0,0.0]",
    ]);
    assert_eq!(
        overridden["configuration"]["numerical"]["external_acceleration_m_s2"]["y"],
        -1.0
    );
}

#[test]
fn legacy_compounding_and_source_assertions_are_migrated_without_reinterpreting_them() {
    let lib = Library::new();
    lib.seed();
    let scene = fs::read_to_string(lib.package().join("scene.toml"))
        .unwrap()
        .replace("schema_version=2", "schema_version=1")
        .replace(
            "[run]\nsteps=2\n",
            "[configuration.overrides.numerical]\nrefine_timestep=2\nparticle_spacing_m=0.25\n",
        )
        .replace("counts=[1,1,1]", "counts=[1,1,1]\nspacing_m=0.25");
    fs::write(lib.package().join("scene.toml"), &scene).unwrap();
    let config = "schema_version=1\nname='old'\ndescription='legacy layers'\n[numerical]\nfixed_timestep_s=0.125\nrefine_timestep=2\nparticle_spacing_m=0.125\n[derived]\nsupport_radius_m=0.25";
    fs::write(lib.package().join("config.toml"), config).unwrap();
    fs::write(
        lib.package().join("render.toml"),
        "schema_version=1\n[camera]\neye_m=[2.0,3.0,4.0]",
    )
    .unwrap();
    let before = lib.inspect();
    assert_eq!(
        before["configuration"]["numerical"]["fixed_timestep_s"],
        0.03125
    );
    let report = lib.json(&["scenes", "create", "copy", "--from", "user:water"]);
    assert_eq!(
        report["assertion_replacements"][0]["field"],
        "support_radius_m"
    );
    assert_eq!(report["assertion_replacements"][0]["previous"], 0.25);
    assert_eq!(report["assertion_replacements"][0]["effective"], 0.5);
    let migrated = lib.json(&["inspect", "user:copy", "--backend", "serial-grid"]);
    for field in ["numerical", "derived", "comparison_key"] {
        assert_eq!(
            before["configuration"][field],
            migrated["configuration"][field]
        );
    }
    assert_eq!(
        before["presentation"]["effective"]["settings"]["camera"],
        migrated["presentation"]["effective"]["settings"]["camera"]
    );
    assert_eq!(
        migrated["presentation"]["effective"]["camera_mode"],
        "fixed"
    );
    assert_eq!(migrated["refinement"]["factor"], 1);
    assert_eq!(
        fs::read_to_string(lib.package().join("scene.toml")).unwrap(),
        scene
    );
    assert_eq!(
        fs::read_to_string(lib.package().join("config.toml")).unwrap(),
        config
    );
    let copy = fs::read_to_string(lib.scenes().join("copy/config.toml")).unwrap();
    assert!(!copy.contains("refine_timestep"));
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn compilation_uses_saved_limit_dimensions_and_stride_and_publishes_playable_images() {
    let lib = Library::new();
    lib.seed();
    fs::write(
        lib.package().join("render.toml"),
        "schema_version=2\nwidth=32\nheight=24\nstep_stride=2",
    )
    .unwrap();
    let report = lib.json(&[
        "compile",
        "user:water",
        "--backend",
        "serial-grid",
        "--software",
        "--output",
        "bundle",
    ]);
    assert_eq!(report["frame_count"], 2);
    let manifest: Value =
        serde_json::from_slice(&fs::read(lib.0.join("bundle/manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["render"]["width"], 32);
    assert_eq!(manifest["render"]["height"], 24);
    assert_eq!(manifest["step_stride"], "2");
    assert_eq!(manifest["accepted_steps"], "2");
    let validation = lib.run(&["runs", "validate", "./bundle"]);
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn compilation_records_colours_with_identical_physical_observations() {
    let lib = Library::new();
    lib.seed();
    let mut manifests = Vec::new();
    for (directory, colours) in [
        ("fixed", "colour_mode='fixed'\ncolour='#00ff00'"),
        (
            "energy",
            "colour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=0.0001",
        ),
    ] {
        fs::write(
            lib.package().join("render.toml"),
            format!(
                "schema_version=2\nwidth=64\nheight=64\nstep_stride=2\n[particles]\n{colours}\n"
            ),
        )
        .unwrap();
        lib.json(&[
            "compile",
            "user:water",
            "--backend",
            "serial-grid",
            "--software",
            "--output",
            directory,
        ]);
        let manifest: Value =
            serde_json::from_slice(&fs::read(lib.0.join(directory).join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["render"]["schema_version"], 2);
        assert_eq!(manifest["schema_version"], 1);
        let validation = lib.run(&["runs", "validate", &format!("./{directory}")]);
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
        manifests.push(manifest);
    }
    assert_eq!(manifests[0]["render"]["particles"]["colour"], "#00ff00");
    assert_eq!(
        manifests[1]["render"]["particles"]["colour_mode"],
        "kinetic_energy"
    );
    for (first, second) in manifests[0]["frames"]
        .as_array()
        .unwrap()
        .iter()
        .zip(manifests[1]["frames"].as_array().unwrap())
    {
        assert_eq!(first["snapshot_sha256"], second["snapshot_sha256"]);
        assert_ne!(first["rgba_sha256"], second["rgba_sha256"]);
    }
}
