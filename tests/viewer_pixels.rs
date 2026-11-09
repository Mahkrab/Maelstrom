//! Matched hardware images and accepted-state digests make renderer experiments
//! reviewable across executables without deriving an image oracle from a shader.
#![cfg(all(feature = "viewer", feature = "cuda"))]

use maelstrom::{
    config::{ConfigurationOverlay, NumericalMode},
    execution::ExecutionMode,
    neighbours::SearchBackend,
    render::{BoundaryLines, Camera, DrawScene, Graphics, GraphicsSelection},
    scene::load_resolved_scene,
    simulation::Simulation,
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, sync::mpsc, time::Duration};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
#[ignore = "requires CUDA, hardware graphics, MAELSTROM_PIXEL_INPUTS and a new MAELSTROM_PIXEL_OUTPUT directory"]
fn hardware_images_and_accepted_states_for_matched_viewer_inputs() {
    let inputs = PathBuf::from(std::env::var_os("MAELSTROM_PIXEL_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var_os("MAELSTROM_PIXEL_OUTPUT").unwrap());
    fs::create_dir(&output).unwrap();
    let instance = Graphics::instance();
    let g = pollster::block_on(Graphics::new(
        &instance,
        GraphicsSelection::default(),
        None,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
    .unwrap();
    eprintln!("PIXEL_ADAPTER {:?}", g.adapter_info);
    let extent = wgpu::Extent3d {
        width: 1280,
        height: 720,
        depth_or_array_layers: 1,
    };
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("matched viewer image"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = g.depth(1280, 720).unwrap();
    let colours = serde_json::from_value(serde_json::json!({"colour_mode":"kinetic_energy",
        "min_energy_j":0.0,"max_energy_j":1.962}))
    .unwrap();
    let mut records = Vec::new();
    for count in [8000, 32000, 110592] {
        let mut overlay = ConfigurationOverlay::default();
        NumericalMode::Experimental.fill_overrides(&mut overlay.numerical);
        let scene =
            load_resolved_scene(&inputs.join(format!("{count}.toml")), Some(&overlay)).unwrap();
        assert_eq!(scene.particles.len(), count);
        let spacing = f64::from(scene.configuration.numerical.particle_spacing_m);
        let boundaries = BoundaryLines::new(&scene.particles, &scene.planes, spacing).unwrap();
        let mut solver = Simulation::from_scene_with_execution(
            &scene,
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda {
                device: 0,
                threads: 128,
            },
        )
        .unwrap();
        let mut camera = Camera::default();
        camera
            .fit_scene(
                &scene.particles,
                &scene.planes,
                spacing / 2.0,
                1280.0 / 720.0,
            )
            .unwrap();
        for step in [0, 240, 1200] {
            while solver.snapshot().completed_steps() < step {
                let _ = solver.step().unwrap();
            }
            let particles = solver.snapshot().particles();
            let mut state = Sha256::new();
            for p in particles {
                state.update(p.identity.to_le_bytes());
                for value in [
                    p.position_m.x,
                    p.position_m.y,
                    p.position_m.z,
                    p.velocity_m_s.x,
                    p.velocity_m_s.y,
                    p.velocity_m_s.z,
                ] {
                    state.update(value.to_bits().to_le_bytes());
                }
            }
            let state_hash = hex(&state.finalize());
            for view in 0..3 {
                let mut selected = camera.clone();
                if view == 1 {
                    selected.orbit(0.6, 0.25).unwrap();
                }
                if view == 2 {
                    selected.pan(0.15, -0.1).unwrap();
                    selected.zoom(0.7).unwrap();
                }
                g.draw(
                    &target.create_view(&Default::default()),
                    &depth.create_view(&Default::default()),
                    &DrawScene {
                        particles,
                        boundaries: &boundaries,
                        radius_m: spacing / 2.0,
                        camera: &selected,
                        colours: &colours,
                        particle_mass_kg: scene.configuration.derived.particle_mass_kg,
                        width: 1280,
                        height: 720,
                    },
                    &[],
                )
                .unwrap();
                let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("matched pixel readback"),
                    size: 1280 * 720 * 4,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
                        buffer: &readback,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(1280 * 4),
                            rows_per_image: Some(720),
                        },
                    },
                    extent,
                );
                g.queue.submit([encoder.finish()]);
                let (tx, rx) = mpsc::sync_channel(1);
                readback.map_async(wgpu::MapMode::Read, .., move |result| {
                    tx.send(result).unwrap();
                });
                g.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Duration::from_secs(30)),
                    })
                    .unwrap();
                rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
                let pixels = readback.get_mapped_range(..).unwrap();
                let name = format!("{count}-{step}-{view}.rgba");
                fs::write(output.join(&name), &*pixels).unwrap();
                records.push(
                    serde_json::json!({"file":name,"step":step,"particles":count,
                    "state_sha256":state_hash,"rgba_sha256":hex(&Sha256::digest(&*pixels))}),
                );
                drop(pixels);
                readback.unmap();
                g.check().unwrap();
            }
        }
    }
    fs::write(
        output.join("images.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
}
