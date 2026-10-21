//! Synthetic pixels and hand-calculated timestamps separate playback checks from
//! simulation and export. No device is needed for the input and clock contracts.

use super::{Sequence, clock::Clock};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    root: PathBuf,
    manifest: Value,
    png: Vec<u8>,
    rgba: Vec<u8>,
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
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "maelstrom-playback-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let rgba = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 31, 127, 211, 255,
        ];
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 2, 2);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgba).unwrap();
        }
        let frames: Vec<_> = [(0,0.0),(2,0.2),(4,0.4),(5,0.5)].into_iter().enumerate().map(|(i,(step,time))| {
            let file = format!("frame-{i:08}.png");
            fs::write(root.join(&file), &png).unwrap();
            json!({"index":i,"step":step.to_string(),"simulated_seconds":time,"file":file,"png_sha256":hash(&png),"rgba_sha256":hash(&rgba),"snapshot_sha256":hash(b"opaque provenance")})
        }).collect();
        let manifest = json!({"schema_version":1,"complete":true,"frame_count":4,"accepted_steps":"5","step_stride":"2","frames":frames,"render":{"width":2,"height":2,"format":"png"},"execution":{"mode":"cuda","device":999},"input":{"deliberately_not_a_scene":true}});
        let f = Self {
            root,
            manifest,
            png,
            rgba,
        };
        f.save();
        f
    }
    fn save(&self) {
        fs::write(
            self.root.join("manifest.json"),
            serde_json::to_vec(&self.manifest).unwrap(),
        )
        .unwrap();
    }
    fn open(&self) -> Result<Sequence, crate::render::RenderError> {
        self.save();
        Sequence::open(&self.root)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn pixels_load_without_physical_inputs_or_the_recorded_cuda_device() {
    let f = Fixture::new();
    let s = f.open().unwrap();
    assert_eq!(s.frame_count(), 4);
    assert_eq!(s.decode(0).unwrap(), f.rgba);
    assert_eq!(s.decode(3).unwrap(), f.rgba);
    assert_eq!(fs::read(f.root.join("frame-00000000.png")).unwrap(), f.png);
    assert!(s.decode(4).is_err());
    assert_eq!(
        Sequence::open(&f.root.join("manifest.json"))
            .unwrap()
            .frame_count(),
        4
    );
    assert_eq!(fs::read_dir(&f.root).unwrap().count(), 5);
}
#[test]
fn malformed_or_incomplete_manifests_do_not_become_playback_requests() {
    let f = Fixture::new();
    for (field, value) in [
        ("schema_version", json!(2)),
        ("complete", json!(false)),
        ("frame_count", json!(3)),
        ("accepted_steps", json!("6")),
        ("step_stride", json!("0")),
        ("frames", json!([])),
    ] {
        let mut m = f.manifest.clone();
        m[field] = value;
        fs::write(
            f.root.join("manifest.json"),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
        assert!(Sequence::open(&f.root).is_err(), "{field}");
    }
    for field in ["schema_version", "complete", "frames", "render"] {
        let mut m = f.manifest.clone();
        m.as_object_mut().unwrap().remove(field);
        fs::write(
            f.root.join("manifest.json"),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
        assert!(Sequence::open(&f.root).is_err(), "missing {field}");
    }
    for name in ["progress.json", "result.json"] {
        fs::write(f.root.join(name), serde_json::to_vec(&f.manifest).unwrap()).unwrap();
        assert!(Sequence::open(&f.root.join(name)).is_err());
    }
}
#[test]
fn unsafe_paths_order_steps_timestamps_hashes_and_dimensions_are_rejected() {
    let f = Fixture::new();
    for (field, value) in [
        ("file", json!("../outside.png")),
        ("file", json!("/tmp/image.png")),
        ("index", json!(0)),
        ("step", json!("3")),
        ("simulated_seconds", json!(0.0)),
        ("simulated_seconds", json!(-1.0)),
        ("simulated_seconds", json!(1e13)),
        ("png_sha256", json!("sha256:no")),
        ("rgba_sha256", json!("")),
        ("snapshot_sha256", json!("")),
    ] {
        let mut m = f.manifest.clone();
        m["frames"][1][field] = value;
        fs::write(
            f.root.join("manifest.json"),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
        assert!(Sequence::open(&f.root).is_err(), "{field}");
    }
    for (field, value) in [
        ("width", json!(0)),
        ("height", json!(8193)),
        ("format", json!("jpeg")),
    ] {
        let mut m = f.manifest.clone();
        m["render"][field] = value;
        fs::write(
            f.root.join("manifest.json"),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
        assert!(Sequence::open(&f.root).is_err());
    }
}
#[test]
fn missing_changed_invalid_and_wrong_sized_images_fail_without_regeneration() {
    let mut f = Fixture::new();
    fs::remove_file(f.root.join("frame-00000003.png")).unwrap();
    assert!(
        f.open()
            .unwrap()
            .decode(3)
            .unwrap_err()
            .to_string()
            .contains("frame 3")
    );
    assert!(!f.root.join("frame-00000003.png").exists());
    fs::write(f.root.join("frame-00000001.png"), b"not PNG").unwrap();
    assert!(
        f.open()
            .unwrap()
            .decode(1)
            .unwrap_err()
            .to_string()
            .contains("PNG hash mismatch")
    );
    f.manifest["frames"][1]["png_sha256"] = json!(hash(b"not PNG"));
    assert!(
        f.open()
            .unwrap()
            .decode(1)
            .unwrap_err()
            .to_string()
            .contains("PNG header")
    );
    f.manifest["render"]["width"] = json!(3);
    assert!(
        f.open()
            .unwrap()
            .decode(0)
            .unwrap_err()
            .to_string()
            .contains("dimensions")
    );
    f.manifest["render"]["width"] = json!(2);
    f.manifest["frames"][0]["rgba_sha256"] = json!(hash(b"wrong pixels"));
    assert!(
        f.open()
            .unwrap()
            .decode(0)
            .unwrap_err()
            .to_string()
            .contains("pixel hash mismatch")
    );
}
#[cfg(unix)]
#[test]
fn symlinks_are_not_followed() {
    let f = Fixture::new();
    let frame = f.root.join("frame-00000000.png");
    fs::remove_file(&frame).unwrap();
    std::os::unix::fs::symlink(f.root.join("frame-00000001.png"), &frame).unwrap();
    assert!(
        f.open()
            .unwrap()
            .decode(0)
            .unwrap_err()
            .to_string()
            .contains("regular file")
    );
    let manifest = f.root.join("manifest.json");
    fs::rename(&manifest, f.root.join("other.json")).unwrap();
    std::os::unix::fs::symlink(f.root.join("other.json"), manifest).unwrap();
    assert!(Sequence::open(&f.root).is_err());
}

#[test]
fn zero_step_export_is_one_static_frame() {
    let mut f = Fixture::new();
    f.manifest["frames"].as_array_mut().unwrap().truncate(1);
    f.manifest["frame_count"] = json!(1);
    f.manifest["accepted_steps"] = json!("0");
    assert_eq!(f.open().unwrap().decode(0).unwrap(), f.rgba);
    let mut c = Clock::new(vec![0.0], 1.0, false, true).unwrap();
    c.toggle();
    c.restart();
    c.advance(100.0);
    assert!(!c.playing);
    assert_eq!(c.frame(), 0);
}
#[test]
fn clock_uses_recorded_intervals_speed_pause_seek_and_final_hold() {
    let mut c = Clock::new(vec![0.0, 0.2, 0.4, 0.5], 2.0, false, false).unwrap();
    c.advance(0.099);
    assert_eq!(c.frame(), 0);
    c.advance(0.001);
    assert_eq!(c.frame(), 1);
    c.toggle();
    c.advance(5.0);
    assert_eq!(c.frame(), 1);
    c.seek(2);
    assert!(!c.playing);
    assert_eq!(c.position, 0.4);
    c.toggle();
    c.advance(1.0);
    assert_eq!(c.frame(), 3);
    assert!(!c.playing);
    assert_eq!(c.position, 0.5);
    c.advance(100.0);
    assert_eq!(c.frame(), 3);
    c.toggle();
    assert_eq!(c.frame(), 0);
    assert!(c.playing);
    c.seek(usize::MAX);
    assert_eq!(c.frame(), 3);
    c.restart();
    assert_eq!(c.frame(), 0);
    assert!(c.playing);
}
#[test]
fn looping_holds_the_final_image_for_the_previous_interval() {
    let mut c = Clock::new(vec![0.0, 0.2, 0.4, 0.5], 1.0, false, true).unwrap();
    c.advance(0.55);
    assert_eq!(c.frame(), 3);
    assert!(c.playing);
    c.advance(0.06);
    assert_eq!(c.frame(), 0);
    c.advance(60.0);
    assert_eq!(c.frame(), 0);
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY, 0.009, 100.01] {
        assert!(Clock::new(vec![0.0], speed, false, false).is_err());
    }
    assert!(Clock::new(vec![0.0], 0.01, true, false).is_ok());
    assert!(Clock::new(vec![0.0], 100.0, true, false).is_ok());
}
#[cfg(feature = "viewer")]
#[test]
fn letterboxing_preserves_aspect_and_centres_both_orientations() {
    assert_eq!(
        super::image::viewport(4, 2, 100, 100),
        [0.0, 25.0, 100.0, 50.0]
    );
    assert_eq!(
        super::image::viewport(2, 4, 100, 100),
        [25.0, 0.0, 50.0, 100.0]
    );
    assert_eq!(super::image::viewport(2, 2, 40, 40), [0.0, 0.0, 40.0, 40.0]);
}

#[cfg(feature = "viewer")]
#[test]
#[ignore = "requires explicitly available software Vulkan"]
fn saved_pixels_keep_orientation_and_colour_through_the_graphics_pass() {
    use crate::render::{Graphics, GraphicsSelection};
    let f = Fixture::new();
    let rgba = f.open().unwrap().decode(0).unwrap();
    let instance = Graphics::instance();
    let g = pollster::block_on(Graphics::new(
        &instance,
        GraphicsSelection {
            index: 0,
            software: true,
        },
        None,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
    .unwrap();
    let image = super::image::Image::new(&g, wgpu::TextureFormat::Rgba8UnormSrgb, 2, 2).unwrap();
    image.upload(&g, &rgba).unwrap();
    let extent = wgpu::Extent3d {
        width: 2,
        height: 2,
        depth_or_array_layers: 1,
    };
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("playback verification"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    image
        .draw(&g, &target.create_view(&Default::default()), 2, 2)
        .unwrap();
    let buffer = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("playback pixels"),
        size: 512,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = g.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(2),
            },
        },
        extent,
    );
    g.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let mapped = buffer.get_mapped_range(..).unwrap();
    let actual: Vec<_> = mapped[..8]
        .iter()
        .chain(&mapped[256..264])
        .copied()
        .collect();
    // Each sample lands at its source texel centre: no interpolation tolerance.
    assert_eq!(actual, f.rgba);
    drop(mapped);
    buffer.unmap();
    g.check().unwrap();
}
