//! Named scenes retain their authoring form when copied, while recorded runs
//! need only the flattened particles and planes to reproduce accepted state.

use maelstrom::scene::{ParticleSource, parse_scene};
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
            "maelstrom-authoring-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_maelstrom"))
            .current_dir(&self.0)
            .env("MAELSTROM_SCENES_DIR", self.0.join("scenes"))
            .env("MAELSTROM_RUNS_DIR", self.0.join("runs"))
            .env("XDG_CONFIG_HOME", self.0.join("preferences"))
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
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
    fn create_fixture(&self, name: &str) {
        let source =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/two-blocks/scene.toml");
        self.json(&["scenes", "create", name, "--from", source.to_str().unwrap()]);
    }
    fn scene(&self, name: &str) -> PathBuf {
        self.0.join("scenes").join(name).join("scene.toml")
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn saved_xsph_controls_resolve_override_and_replay_without_the_original_package() {
    let library = Library::new();
    library.create_fixture("fluid");
    let config = library.scene("fluid").with_file_name("config.toml");
    let mut document: toml::Value = toml::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
    document["numerical"]["xsph_viscosity_enabled"] = toml::Value::Boolean(true);
    document["numerical"].as_table_mut().unwrap().insert(
        "xsph_viscosity_coefficient".into(),
        toml::Value::Float(0.125),
    );
    document["numerical"]["solver_iterations"] = toml::Value::Integer(1);
    fs::write(&config, toml::to_string(&document).unwrap()).unwrap();
    let before = fs::read(&config).unwrap();
    let inspected = library.json(&["inspect", "user:fluid", "--backend", "serial-grid"]);
    assert_eq!(
        inspected["configuration"]["numerical"]["xsph_viscosity_coefficient"],
        0.125
    );
    assert_eq!(
        inspected["configuration"]["provenance"]["xsph_viscosity_coefficient"],
        "configuration_file"
    );
    let overridden = library.json(&[
        "inspect",
        "user:fluid",
        "--backend",
        "serial-grid",
        "--set",
        "xsph_viscosity_coefficient=0.25",
    ]);
    assert_eq!(
        overridden["configuration"]["numerical"]["xsph_viscosity_coefficient"],
        0.25
    );
    assert_eq!(
        overridden["configuration"]["provenance"]["xsph_viscosity_coefficient"],
        "command_line"
    );
    assert_eq!(
        library
            .run(&[
                "validate",
                "user:fluid",
                "--set",
                "xsph_viscosity_coefficient=1.01"
            ])
            .status
            .code(),
        Some(3)
    );
    assert_eq!(fs::read(&config).unwrap(), before);
    let result = library.json(&[
        "run",
        "user:fluid",
        "--backend",
        "serial-grid",
        "--steps",
        "2",
        "--unpaced",
    ]);
    assert_eq!(
        result["input"]["configuration"]["numerical"]["xsph_viscosity_coefficient"],
        0.125
    );
    let record = library.0.join("xsph.json");
    fs::write(&record, serde_json::to_vec(&result).unwrap()).unwrap();
    fs::remove_dir_all(library.0.join("scenes")).unwrap();
    let replay = library.json(&["rerun", record.to_str().unwrap()]);
    assert_eq!(result["final_snapshot"], replay["final_snapshot"]);
}

#[test]
fn saved_vorticity_controls_resolve_override_and_replay_without_the_original_package() {
    let library = Library::new();
    library.create_fixture("fluid");
    let config = library.scene("fluid").with_file_name("config.toml");
    let mut document: toml::Value = toml::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
    document["numerical"]["vorticity_confinement_enabled"] = toml::Value::Boolean(true);
    document["numerical"].as_table_mut().unwrap().insert(
        "vorticity_confinement_strength_m_s".into(),
        toml::Value::Float(0.125),
    );
    document["numerical"]["solver_iterations"] = toml::Value::Integer(1);
    fs::write(&config, toml::to_string(&document).unwrap()).unwrap();
    let before = fs::read(&config).unwrap();
    let inspected = library.json(&["inspect", "user:fluid", "--backend", "serial-grid"]);
    assert_eq!(
        inspected["configuration"]["numerical"]["vorticity_confinement_strength_m_s"],
        0.125
    );
    assert_eq!(
        inspected["configuration"]["provenance"]["vorticity_confinement_strength_m_s"],
        "configuration_file"
    );
    let overridden = library.json(&[
        "inspect",
        "user:fluid",
        "--backend",
        "serial-grid",
        "--set",
        "vorticity_confinement_strength_m_s=0.25",
    ]);
    assert_eq!(
        overridden["configuration"]["numerical"]["vorticity_confinement_strength_m_s"],
        0.25
    );
    assert_eq!(
        overridden["configuration"]["provenance"]["vorticity_confinement_strength_m_s"],
        "command_line"
    );
    assert_eq!(
        library
            .run(&[
                "validate",
                "user:fluid",
                "--set",
                "vorticity_confinement_strength_m_s=-0.01"
            ])
            .status
            .code(),
        Some(3)
    );
    assert_eq!(fs::read(&config).unwrap(), before);
    let result = library.json(&[
        "run",
        "user:fluid",
        "--backend",
        "serial-grid",
        "--steps",
        "2",
        "--unpaced",
    ]);
    assert_eq!(
        result["input"]["configuration"]["numerical"]["vorticity_confinement_strength_m_s"],
        0.125
    );
    let record = library.0.join("vorticity.json");
    fs::write(&record, serde_json::to_vec(&result).unwrap()).unwrap();
    fs::remove_dir_all(library.0.join("scenes")).unwrap();
    let replay = library.json(&["rerun", record.to_str().unwrap()]);
    assert_eq!(result["final_snapshot"], replay["final_snapshot"]);
}

#[test]
fn editable_groups_and_container_survive_copying_and_records_survive_source_removal() {
    let library = Library::new();
    library.create_fixture("water");
    let source = library.scene("water");
    let mut document = parse_scene(&fs::read_to_string(&source).unwrap()).unwrap();
    let ParticleSource::Groups { groups } = &mut document.particles else {
        panic!()
    };
    for group in groups.iter_mut() {
        let ParticleSource::Lattice { spacing_m, .. } = &mut group.source else {
            panic!()
        };
        *spacing_m = Some(0.05);
    }
    let ParticleSource::Lattice { first_identity, .. } = &mut groups[0].source else {
        panic!()
    };
    *first_identity = Some(10);
    fs::write(&source, toml::to_string_pretty(&document).unwrap()).unwrap();
    let inspected = library.json(&["inspect", "user:water", "--backend", "serial-grid"]);
    assert_eq!(
        inspected["scene"]["particle_groups"],
        serde_json::json!(["left", "right"])
    );
    assert_eq!(inspected["scene"]["particle_count"], 128);
    assert_eq!(inspected["scene"]["plane_count"], 6);
    library.json(&["scenes", "create", "copy", "--from", "user:water"]);
    let copied = parse_scene(&fs::read_to_string(library.scene("copy")).unwrap()).unwrap();
    assert_eq!(copied.container, document.container);
    let ParticleSource::Groups { groups } = &copied.particles else {
        panic!()
    };
    assert!(groups.iter().all(|g| matches!(
        g.source,
        ParticleSource::Lattice {
            spacing_m: None,
            ..
        }
    )));
    let original = library.json(&[
        "run",
        "user:water",
        "--steps",
        "2",
        "--backend",
        "serial-grid",
        "--unpaced",
    ]);
    let copy = library.json(&[
        "run",
        "user:copy",
        "--steps",
        "2",
        "--backend",
        "serial-grid",
        "--unpaced",
    ]);
    assert_eq!(original["input"]["particles"], copy["input"]["particles"]);
    assert_eq!(original["input"]["planes"], copy["input"]["planes"]);
    assert_eq!(original["final_snapshot"], copy["final_snapshot"]);
    assert_eq!(copy["input"]["particles"][0]["identity"], "10");
    assert_eq!(copy["input"]["particles"][64]["identity"], "74");
    let record = library.0.join("run.json");
    fs::write(&record, serde_json::to_vec(&copy).unwrap()).unwrap();
    fs::remove_dir_all(library.0.join("scenes")).unwrap();
    let replay = library.json(&["rerun", record.to_str().unwrap()]);
    assert_eq!(replay["final_snapshot"], copy["final_snapshot"]);
}

#[test]
fn invalid_group_edits_do_not_create_a_destination_or_change_an_existing_record() {
    let library = Library::new();
    library.create_fixture("water");
    let before = library.json(&[
        "run",
        "user:water",
        "--steps",
        "0",
        "--backend",
        "serial-grid",
        "--unpaced",
    ]);
    let source = library.scene("water");
    let text = fs::read_to_string(&source).unwrap();
    fs::write(&source, text.replace("name = \"right\"", "name = \"left\"")).unwrap();
    assert_eq!(
        library.run(&["validate", "user:water"]).status.code(),
        Some(3)
    );
    assert_eq!(
        library
            .run(&["scenes", "create", "invalid-copy", "--from", "user:water"])
            .status
            .code(),
        Some(3)
    );
    assert!(!library.0.join("scenes/invalid-copy").exists());
    fs::write(&source, text).unwrap();
    let after = library.json(&[
        "run",
        "user:water",
        "--steps",
        "0",
        "--backend",
        "serial-grid",
        "--unpaced",
    ]);
    assert_eq!(before, after);
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn grouped_scene_compiles_to_a_portable_validated_image_sequence() {
    let library = Library::new();
    library.create_fixture("water");
    fs::write(
        library.0.join("scenes/water/render.toml"),
        "schema_version=2\nwidth=32\nheight=24",
    )
    .unwrap();
    let output = library.json(&[
        "compile",
        "user:water",
        "--steps",
        "1",
        "--backend",
        "serial-grid",
        "--software",
        "--output",
        "images",
    ]);
    assert_eq!(output["frame_count"], 2);
    fs::remove_dir_all(library.0.join("scenes")).unwrap();
    let validation = library.run(&["runs", "validate", "./images"]);
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(library.0.join("images/manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["input"]["particles"].as_array().unwrap().len(),
        128
    );
    assert_eq!(manifest["input"]["planes"].as_array().unwrap().len(), 6);
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn relocated_fluid_package_compiles_identical_state_and_survives_source_removal() {
    use maelstrom::workflow::SnapshotRecord;
    use sha2::{Digest, Sha256};
    let library = Library::new();
    library.create_fixture("water");
    let package = library.scene("water").parent().unwrap().to_owned();
    fs::write(package.join("config.toml"), "schema_version=2\nname='model'\ndescription='saved fluid controls'\n[numerical]\nparticle_spacing_m=0.05\nrefine_timestep=4\nxsph_viscosity_enabled=true\nxsph_viscosity_coefficient=0.01\nvorticity_confinement_enabled=true\nvorticity_confinement_strength_m_s=0.01\n").unwrap();
    let mut scene: toml::Value =
        toml::from_str(&fs::read_to_string(package.join("scene.toml")).unwrap()).unwrap();
    scene["run"] = toml::Value::Table(toml::map::Map::from_iter([(
        "steps".into(),
        toml::Value::Integer(4),
    )]));
    fs::write(package.join("scene.toml"), toml::to_string(&scene).unwrap()).unwrap();
    fs::write(package.join("render.toml"), "schema_version=2\nwidth=64\nheight=48\nstep_stride=2\n[particles]\ncolour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=0.001\n").unwrap();
    let before = library.json(&["inspect", "user:water", "--backend", "serial-grid"]);
    assert_eq!(before["scene"]["particle_count"], 128);
    assert_eq!(before["run"]["steps"], 4);
    let moved = library.0.join("moved");
    fs::rename(&package, &moved).unwrap();
    let files: Vec<_> = ["scene.toml", "config.toml", "render.toml"]
        .map(|f| fs::read(moved.join(f)).unwrap())
        .into();
    let selected = "./moved/scene.toml";
    let after = library.json(&["inspect", selected, "--backend", "serial-grid"]);
    assert_eq!(
        before["configuration"]["numerical"],
        after["configuration"]["numerical"]
    );
    assert_eq!(
        before["presentation"]["effective"],
        after["presentation"]["effective"]
    );
    let headless = library.json(&["run", selected, "--backend", "serial-grid", "--unpaced"]);
    let record = library.0.join("result.json");
    fs::write(&record, serde_json::to_vec(&headless).unwrap()).unwrap();
    library.json(&[
        "compile",
        selected,
        "--backend",
        "serial-grid",
        "--software",
        "--output",
        "images",
    ]);
    let manifest_path = library.0.join("images/manifest.json");
    let manifest_bytes = fs::read(&manifest_path).unwrap();
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest["accepted_steps"], "4");
    assert_eq!(manifest["frame_count"], 3);
    assert_eq!(
        manifest["render"]["particles"]["colour_mode"],
        "kinetic_energy"
    );
    assert_eq!(manifest["frames"][2]["step"], "4");
    let snapshot =
        SnapshotRecord::from_json(&serde_json::to_string(&headless["final_snapshot"]).unwrap())
            .unwrap();
    let expected_hash = format!(
        "sha256:{}",
        Sha256::digest(snapshot.to_json().unwrap().as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    assert_eq!(manifest["frames"][2]["snapshot_sha256"], expected_hash);
    // Inspection-style JSON widens f32 values, while result JSON uses shortest
    // round-tripping f32 text. Compare their actual numerical model values.
    let compiled_model: maelstrom::config::NumericalModel =
        serde_json::from_value(manifest["input"]["configuration"]["numerical"].clone()).unwrap();
    let headless_model: maelstrom::config::NumericalModel =
        serde_json::from_value(headless["input"]["configuration"]["numerical"].clone()).unwrap();
    assert_eq!(compiled_model, headless_model);
    let duplicate = library.run(&[
        "compile",
        selected,
        "--backend",
        "serial-grid",
        "--software",
        "--output",
        "images",
    ]);
    assert!(!duplicate.status.success());
    assert_eq!(fs::read(&manifest_path).unwrap(), manifest_bytes);
    for (name, bytes) in ["scene.toml", "config.toml", "render.toml"]
        .iter()
        .zip(files)
    {
        assert_eq!(fs::read(moved.join(name)).unwrap(), bytes);
    }
    fs::remove_dir_all(&moved).unwrap();
    library.json(&["runs", "validate", "./images"]);
    let replay = library.json(&["rerun", record.to_str().unwrap()]);
    assert_eq!(replay["final_snapshot"], headless["final_snapshot"]);
    let sequence = maelstrom::playback::Sequence::open(&library.0.join("images")).unwrap();
    assert_eq!(sequence.frame_count(), 3);
    for frame in 0..sequence.frame_count() {
        assert_eq!(sequence.decode(frame).unwrap().len(), 64 * 48 * 4);
    }
}
