//! Public commands exercise cache ownership, independent rendering and migration.
//! Hardware-independent checks run normally; software graphics checks are explicit.
use serde_json::Value;
#[cfg(feature = "render")]
use std::path::Path;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "maelstrom-cache-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn command(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_maelstrom"));
        c.current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("XDG_CONFIG_HOME", self.0.join("config"))
            .env("XDG_DATA_HOME", self.0.join("data"))
            .env("MAELSTROM_RUNS_DIR", self.0.join("runs"))
            .env("MAELSTROM_SCENES_DIR", self.0.join("scenes"))
            .arg("--format")
            .arg("json");
        c
    }
    fn ok(&self, args: &[&str]) -> Value {
        let output = self.command().args(args).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
    fn bake(&self) -> PathBuf {
        let cache = self.0.join("cache");
        self.ok(&[
            "cache",
            "bake",
            "builtin:falling-block-04",
            "--backend",
            "serial-grid",
            "--steps",
            "5",
            "--frame-stride",
            "2",
            "--output",
            cache.to_str().unwrap(),
        ]);
        cache
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn failed(output: Output) {
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty());
}

#[test]
fn cached_timing_and_partial_options_are_validated_before_viewer_startup() {
    let t = Temp::new();
    for args in [
        vec![
            "view",
            "builtin:falling-block-04",
            "--cache-playback",
            "frame-accurate",
        ],
        vec!["view", "builtin:falling-block-04", "--allow-partial"],
        vec![
            "view",
            "--cache",
            "unused",
            "--cache-playback",
            "interpolated",
        ],
    ] {
        let output = t.command().args(args).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn headless_cache_commands_validate_without_graphics_and_preserve_existing_output() {
    let t = Temp::new();
    let cache = t.bake();
    let value = t.ok(&["cache", "validate", cache.to_str().unwrap(), "--full"]);
    assert_eq!(value["complete"], true);
    assert_eq!(value["frames"], 4);
    assert_eq!(value["particle_count"], 64);
    let request = fs::read(cache.join("request.json")).unwrap();
    failed(
        t.command()
            .args([
                "cache",
                "bake",
                "builtin:falling-block-04",
                "--backend",
                "serial-grid",
                "--steps",
                "0",
                "--output",
                cache.to_str().unwrap(),
            ])
            .output()
            .unwrap(),
    );
    assert_eq!(fs::read(cache.join("request.json")).unwrap(), request);
    fs::remove_file(cache.join("cache.json")).unwrap();
    failed(
        t.command()
            .args(["cache", "inspect", cache.to_str().unwrap()])
            .output()
            .unwrap(),
    );
    let partial = t.ok(&[
        "cache",
        "inspect",
        cache.to_str().unwrap(),
        "--allow-partial",
    ]);
    assert_eq!(partial["complete"], false);
    assert_eq!(partial["frames"], 4);
    failed(
        t.command()
            .args([
                "render",
                cache.to_str().unwrap(),
                "--allow-partial",
                "--output",
                t.0.join("bad").to_str().unwrap(),
            ])
            .output()
            .unwrap(),
    );
}

#[cfg(feature = "render")]
fn manifest(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap()
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn cached_render_matches_direct_and_staged_compile_and_supports_sparse_views() {
    let t = Temp::new();
    let cache = t.bake();
    let render_config = t.0.join("render.toml");
    fs::write(&render_config,"schema_version = 2\nwidth = 80\nheight = 60\ncamera_mode = \"fit\"\n[particles]\ncolour_mode = \"fixed\"\ncolour = \"#258bdf\"\n").unwrap();
    let direct = t.0.join("direct");
    t.ok(&[
        "compile",
        "builtin:falling-block-04",
        "--backend",
        "serial-grid",
        "--steps",
        "5",
        "--frame-stride",
        "2",
        "--software",
        "--render-config",
        render_config.to_str().unwrap(),
        "--output",
        direct.to_str().unwrap(),
    ]);
    let rendered = t.0.join("rendered");
    let output = t
        .command()
        .env("CUDA_VISIBLE_DEVICES", "")
        .args([
            "render",
            cache.to_str().unwrap(),
            "--software",
            "--render-config",
            render_config.to_str().unwrap(),
            "--output",
            rendered.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let original = manifest(&direct);
    let actual = manifest(&rendered);
    assert_eq!(original["schema_version"], 1);
    assert_eq!(actual["schema_version"], 2);
    assert!(
        actual["build"]["rustflags_sha256"]
            .as_str()
            .is_some_and(|s| s.starts_with("sha256:"))
    );
    assert!(actual["graphics"]["vendor"].is_u64());
    assert!(actual["graphics"]["device"].is_u64());
    assert_eq!(original["frames"], actual["frames"]);
    assert_eq!(actual["timing_seconds"]["solver_steps"], 0.0);
    for path in [&direct, &rendered] {
        t.ok(&["runs", "validate", path.to_str().unwrap()]);
    }
    let staged = t.0.join("staged");
    let staged_cache = t.0.join("staged-cache");
    t.ok(&[
        "compile",
        "builtin:falling-block-04",
        "--backend",
        "serial-grid",
        "--steps",
        "5",
        "--frame-stride",
        "2",
        "--software",
        "--render-config",
        render_config.to_str().unwrap(),
        "--cache-output",
        staged_cache.to_str().unwrap(),
        "--output",
        staged.to_str().unwrap(),
    ]);
    assert_eq!(original["frames"], manifest(&staged)["frames"]);
    let still = t.0.join("still");
    t.ok(&[
        "render",
        cache.to_str().unwrap(),
        "--frame",
        "1",
        "--software",
        "--render-config",
        render_config.to_str().unwrap(),
        "--output",
        still.to_str().unwrap(),
    ]);
    let single = manifest(&still);
    assert_eq!(single["frames"][0]["step"], "2");
    assert_eq!(
        single["frames"][0]["png_sha256"],
        original["frames"][1]["png_sha256"]
    );
    assert_eq!(
        maelstrom::playback::Sequence::open(&still)
            .unwrap()
            .frame_count(),
        1
    );
    let sparse = t.0.join("sparse");
    t.ok(&[
        "render",
        cache.to_str().unwrap(),
        "--first-frame",
        "1",
        "--last-frame",
        "3",
        "--every",
        "2",
        "--software",
        "--render-config",
        render_config.to_str().unwrap(),
        "--output",
        sparse.to_str().unwrap(),
    ]);
    assert_eq!(
        maelstrom::playback::Sequence::open(&sparse)
            .unwrap()
            .frame_count(),
        2
    );
    t.ok(&["runs", "validate", sparse.to_str().unwrap()]);
    let before = maelstrom::cache::Cache::open(&cache, false)
        .unwrap()
        .identity()
        .to_owned();
    let alternate = t.0.join("alternate.toml");
    fs::write(&alternate,"schema_version = 2\nwidth = 96\nheight = 72\ncamera_mode = \"fixed\"\n[camera]\neye_m = [1.5, 1.2, 1.0]\ntarget_m = [0.5, 0.5, 0.5]\n[particles]\ncolour_mode = \"fixed\"\ncolour = \"#ff0000\"\n").unwrap();
    let changed = t.0.join("changed");
    t.ok(&[
        "render",
        cache.to_str().unwrap(),
        "--software",
        "--render-config",
        alternate.to_str().unwrap(),
        "--output",
        changed.to_str().unwrap(),
    ]);
    let changed_manifest = manifest(&changed);
    assert_ne!(
        changed_manifest["frames"][0]["png_sha256"],
        actual["frames"][0]["png_sha256"]
    );
    assert_eq!(
        changed_manifest["frames"][1]["snapshot_sha256"],
        actual["frames"][1]["snapshot_sha256"]
    );
    assert_eq!(
        maelstrom::cache::Cache::open(&cache, false)
            .unwrap()
            .identity(),
        before
    );
    fs::remove_dir_all(&cache).unwrap();
    t.ok(&["runs", "validate", rendered.to_str().unwrap()]);
    assert!(
        maelstrom::workflow::replay_options(
            &fs::read_to_string(rendered.join("manifest.json")).unwrap()
        )
        .is_ok()
    );
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn incomplete_prefix_requires_explicit_selection_and_corrupt_frames_never_complete() {
    let t = Temp::new();
    let cache = t.bake();
    fs::remove_file(cache.join("cache.json")).unwrap();
    let output = t.0.join("partial");
    t.ok(&[
        "render",
        cache.to_str().unwrap(),
        "--allow-partial",
        "--last-frame",
        "1",
        "--width",
        "32",
        "--height",
        "32",
        "--software",
        "--output",
        output.to_str().unwrap(),
    ]);
    let result = manifest(&output);
    assert_eq!(result["source_cache"]["complete"], false);
    assert_eq!(result["frame_count"], 2);
    t.ok(&["runs", "validate", output.to_str().unwrap()]);
    let frame = cache.join("frames/frame-00000000.mpc");
    let mut bytes = fs::read(&frame).unwrap();
    bytes[160] ^= 1;
    fs::write(&frame, bytes).unwrap();
    let bad = t.0.join("bad");
    failed(
        t.command()
            .args([
                "render",
                cache.to_str().unwrap(),
                "--allow-partial",
                "--last-frame",
                "1",
                "--width",
                "32",
                "--height",
                "32",
                "--software",
                "--output",
                bad.to_str().unwrap(),
            ])
            .output()
            .unwrap(),
    );
    assert!(!bad.join("manifest.json").exists());
    assert!(!bad.join("frame-00000000.png").exists());
}

#[cfg(feature = "render")]
#[test]
#[ignore = "requires software graphics, FFmpeg with libx264 and ffprobe"]
fn cached_video_and_png_conversion_preserve_selection_without_solver() {
    let t = Temp::new();
    let cache = t.bake();
    let config = t.0.join("camera.toml");
    fs::write(&config,"schema_version = 2\nwidth = 64\nheight = 48\ncamera_mode = 'fixed'\n[camera]\neye_m = [1.4, 1.2, 1.1]\ntarget_m = [0.1, 0.1, 0.1]\n[particles]\ncolour_mode = 'fixed'\ncolour = '#00ffaa'\n").unwrap();
    let video = t.0.join("cache.mp4");
    let v = t.ok(&[
        "video",
        "cache",
        cache.to_str().unwrap(),
        "--render-config",
        config.to_str().unwrap(),
        "--software",
        "-o",
        video.to_str().unwrap(),
        "--first-frame",
        "1",
        "--last-frame",
        "3",
        "--every",
        "2",
        "--fps",
        "24",
    ]);
    assert_eq!(v["frame_count"], 2);
    assert_eq!(v["source"]["solver_steps"], 0);
    assert_eq!(v["frames"][0]["source_index"], 1);
    assert_eq!(v["frames"][1]["source_index"], 3);
    assert_eq!(
        v["frames"][1]["simulated_seconds"],
        5.0 * f64::from(1.0f32 / 240.0)
    );
    let images = t.0.join("images");
    t.ok(&[
        "render",
        cache.to_str().unwrap(),
        "--render-config",
        config.to_str().unwrap(),
        "--software",
        "-o",
        images.to_str().unwrap(),
    ]);
    let sparse = t.0.join("sparse-images");
    t.ok(&[
        "render",
        cache.to_str().unwrap(),
        "--render-config",
        config.to_str().unwrap(),
        "--software",
        "--first-frame",
        "1",
        "--last-frame",
        "3",
        "--every",
        "2",
        "-o",
        sparse.to_str().unwrap(),
    ]);
    let sparse_video = t.0.join("sparse-images.mp4");
    let selected = t.ok(&[
        "video",
        "images",
        sparse.to_str().unwrap(),
        "--first-frame",
        "1",
        "--last-frame",
        "1",
        "-o",
        sparse_video.to_str().unwrap(),
    ]);
    assert_eq!(selected["frame_count"], 1);
    assert_eq!(selected["frames"][0]["source_index"], 1);
    assert_eq!(
        selected["frames"][0]["simulated_seconds"],
        5.0 * f64::from(1.0f32 / 240.0)
    );
    fs::remove_dir_all(&cache).unwrap();
    let png_video = t.0.join("images.mp4");
    let v = t.ok(&[
        "video",
        "images",
        images.to_str().unwrap(),
        "-o",
        png_video.to_str().unwrap(),
        "--omit-terminal",
        "--fps",
        "60",
    ]);
    assert_eq!(v["frame_count"], 3);
    assert_eq!(v["duration_seconds"], 0.05);
    assert_eq!(
        v["source"]["render"]["camera"]["eye_m"],
        serde_json::json!([1.4, 1.2, 1.1])
    );
    let png = images.join("frame-00000000.png");
    fs::write(png, b"corrupt").unwrap();
    let failed_video = t.0.join("corrupt.mp4");
    failed(
        t.command()
            .args([
                "video",
                "images",
                images.to_str().unwrap(),
                "-o",
                failed_video.to_str().unwrap(),
            ])
            .output()
            .unwrap(),
    );
    assert!(!failed_video.exists());
}
