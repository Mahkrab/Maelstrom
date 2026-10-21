//! A failed device/readback or PNG stage must leave the same incomplete sequence
//! evidence as a failed filesystem publication, including earlier closed frames.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
static READS: AtomicUsize = AtomicUsize::new(0);
static ENCODINGS: AtomicUsize = AtomicUsize::new(0);
fn lost_device(g: &Graphics, scene: &crate::render::DrawScene<'_>) -> Result<Vec<u8>, ExportError> {
    if READS.fetch_add(1, Ordering::SeqCst) > 0 {
        g.device.destroy();
    }
    offscreen::pixels(g, scene)
}
fn invalid_encoding(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, ExportError> {
    if ENCODINGS.fetch_add(1, Ordering::SeqCst) > 0 {
        offscreen::encode(width, height, &rgba[..rgba.len() - 1])
    } else {
        offscreen::encode(width, height, rgba)
    }
}
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn stage_failures_retain_closed_frames_and_incomplete_progress() {
    let mut scene =
        crate::scene::load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap();
    scene.particles.clear();
    scene.planes.clear();
    for (name, operations) in [
        (
            "device",
            offscreen::FrameOperations {
                read_pixels: lost_device,
                ..Default::default()
            },
        ),
        (
            "encoding",
            offscreen::FrameOperations {
                encode_png: invalid_encoding,
                ..Default::default()
            },
        ),
    ] {
        let directory = std::env::temp_dir().join(format!(
            "maelstrom-export-stage-{}-{name}",
            std::process::id()
        ));
        let error = export_using(
            ExportRequest {
                source: Source::Direct {
                    scene: &scene,
                    steps: 1,
                    options: Default::default(),
                },
                settings: Settings {
                    width: 17,
                    height: 19,
                    ..Default::default()
                },
                graphics: GraphicsSelection {
                    index: 0,
                    software: true,
                },
                fit_camera: true,
                destination: &directory,
            },
            || false,
            operations,
        )
        .unwrap_err();
        eprintln!("{name} failure: {error}");
        let bytes = fs::read(directory.join("frame-00000000.png")).unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(!directory.join("frame-00000001.png").exists());
        assert!(!directory.join("manifest.json").exists());
        let progress: Value =
            serde_json::from_slice(&fs::read(directory.join("progress.json")).unwrap()).unwrap();
        assert_eq!(progress["complete"], false);
        assert_eq!(progress["frames"].as_array().unwrap().len(), 1);
        assert_eq!(progress["failures"][0], error.to_string());
        assert_eq!(progress["frames"][0]["png_sha256"], hash(&bytes));
        fs::remove_dir_all(directory).unwrap();
    }
}
