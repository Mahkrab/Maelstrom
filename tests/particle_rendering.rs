//! Analytic occlusion is checked against independent f64 rays. Visual sequences
//! render the same immutable accepted state through every representation.
#![cfg(feature = "viewer")]

use maelstrom::{
    math::Vec3,
    render::{BoundaryLines, Camera, DrawScene, Graphics, GraphicsSelection, ParticleRenderer},
    scene::Particle,
};
#[cfg(feature = "cuda")]
use std::{fs, path::Path};
use std::{sync::mpsc, time::Duration};

struct Target {
    colour: wgpu::Texture,
    depth: wgpu::Texture,
    readback: wgpu::Buffer,
    width: u32,
    height: u32,
    padded: u32,
}
impl Target {
    fn new(g: &Graphics, width: u32, height: u32) -> Self {
        let padded = (width * 4).div_ceil(256) * 256;
        Self {
            colour: g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("particle comparison"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }),
            depth: g.depth(width, height).unwrap(),
            readback: g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle pixels"),
                size: u64::from(padded) * u64::from(height),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            width,
            height,
            padded,
        }
    }
    fn pixels(&self, g: &Graphics, scene: &DrawScene<'_>) -> Vec<u8> {
        g.draw(
            &self.colour.create_view(&Default::default()),
            &self.depth.create_view(&Default::default()),
            scene,
            &[],
        )
        .unwrap();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.colour,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::sync_channel(1);
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        g.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        let mapped = self.readback.get_mapped_range(..).unwrap();
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in mapped.chunks_exact(self.padded as usize) {
            pixels.extend_from_slice(&row[..self.width as usize * 4]);
        }
        drop(mapped);
        self.readback.unmap();
        g.check().unwrap();
        pixels
    }
}
fn graphics(software: bool) -> Graphics {
    let instance = Graphics::instance();
    let g = pollster::block_on(Graphics::new(
        &instance,
        GraphicsSelection { index: 0, software },
        None,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
    .unwrap();
    eprintln!("PARTICLE_ADAPTER {:?}", g.adapter_info);
    g
}

#[test]
#[ignore = "requires an explicitly available hardware graphics adapter"]
fn analytic_overlap_depth_matches_independent_rays_and_near_fallback() {
    check_analytic(false);
}

#[test]
#[ignore = "requires an explicitly available software Vulkan adapter"]
fn software_analytic_overlap_depth_matches_independent_rays() {
    check_analytic(true);
}

fn check_analytic(software: bool) {
    let mut g = graphics(software);
    let target = Target::new(&g, 128, 128);
    let boundaries = BoundaryLines::new(&[], &[], 0.05).unwrap();
    let colours = serde_json::from_value(
        serde_json::json!({"colour_mode":"kinetic_energy","min_energy_j":0.0,"max_energy_j":1.0}),
    )
    .unwrap();
    let mut camera = Camera {
        eye_m: [0.0, 0.0, 5.0],
        target_m: [0.0; 3],
        up: [0.0, 1.0, 0.0],
        vertical_fov_deg: 90.0,
        near_m: 0.1,
        far_m: 10.0,
    };
    let mut particles = vec![
        Particle {
            identity: 0,
            position_m: Vec3::new(-0.4, 0.0, 0.2),
            velocity_m_s: Vec3::new(1.0, 0.0, 0.0),
        },
        Particle {
            identity: 1,
            position_m: Vec3::new(0.4, 0.0, -0.2),
            velocity_m_s: Vec3::ZERO,
        },
    ];
    fn scene<'a>(
        particles: &'a [Particle],
        camera: &'a Camera,
        boundaries: &'a BoundaryLines,
        colours: &'a maelstrom::presentation::ParticleColours,
    ) -> DrawScene<'a> {
        DrawScene {
            particles,
            boundaries,
            radius_m: 1.0,
            camera,
            colours,
            particle_mass_kg: 2.0,
            width: 128,
            height: 128,
        }
    }
    g.set_particle_renderer(ParticleRenderer::Impostor);
    let pixels = target.pixels(&g, &scene(&particles, &camera, &boundaries, &colours));
    let mut checked = 0;
    for y in 0..128 {
        for x in 0..128 {
            let d = [
                (x as f64 + 0.5) / 64.0 - 1.0,
                1.0 - (y as f64 + 0.5) / 64.0,
                -1.0,
            ];
            let a = d.iter().map(|v| v * v).sum::<f64>();
            let hit = |p: &Particle| {
                let c = [
                    f64::from(p.position_m.x),
                    0.0,
                    f64::from(p.position_m.z) - 5.0,
                ];
                let b = d.iter().zip(c).map(|(d, c)| d * c).sum::<f64>();
                let discriminant = b * b - a * (c.iter().map(|v| v * v).sum::<f64>() - 1.0);
                if discriminant > 0.01 {
                    let distance = (b - discriminant.sqrt()) / a;
                    assert!(distance > 0.0);
                    Some(distance)
                } else {
                    None
                }
            };
            if let (Some(a), Some(b)) = (hit(&particles[0]), hit(&particles[1])) {
                if (a - b).abs() < 0.001 {
                    continue;
                }
                let offset = (y * 128 + x) * 4;
                let (red, blue) = (pixels[offset], pixels[offset + 2]);
                if a < b {
                    assert!(red > 0 && blue == 0, "front red at {x},{y}: {red},{blue}");
                } else {
                    assert!(blue > 0 && red == 0, "front blue at {x},{y}: {red},{blue}");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 200);
    // Off-axis tangency needs a larger perspective rectangle than c.xy ± r.
    particles.truncate(1);
    particles[0].position_m = Vec3::new(3.0, 1.5, 0.0);
    let off_axis = target.pixels(&g, &scene(&particles, &camera, &boundaries, &colours));
    for y in 0..128 {
        for x in 0..128 {
            let d = [
                (x as f64 + 0.5) / 64.0 - 1.0,
                1.0 - (y as f64 + 0.5) / 64.0,
                -1.0,
            ];
            let c = [3.0, 1.5, -5.0];
            let a = d.iter().map(|v| v * v).sum::<f64>();
            let b = d.iter().zip(c).map(|(d, c)| d * c).sum::<f64>();
            let discriminant = b * b - a * (c.iter().map(|v| v * v).sum::<f64>() - 1.0);
            let offset = (y * 128 + x) * 4;
            let red = off_axis[offset] > 64 && off_axis[offset + 2] == 0;
            if discriminant > 0.005 {
                assert!(red, "missing off-axis interior at {x},{y}");
            }
            if discriminant < -0.005 {
                assert!(!red, "off-axis rectangle leaked at {x},{y}");
            }
        }
    }
    for near_eye in [[0.0, 0.0, 0.5], [0.0, 0.0, 1.05]] {
        camera.eye_m = near_eye;
        particles.truncate(1);
        particles[0].position_m = Vec3::ZERO;
        g.set_particle_renderer(ParticleRenderer::Indexed);
        let control = target.pixels(&g, &scene(&particles, &camera, &boundaries, &colours));
        g.set_particle_renderer(ParticleRenderer::Impostor);
        assert_eq!(
            target.pixels(&g, &scene(&particles, &camera, &boundaries, &colours)),
            control
        );
    }
}

#[cfg(feature = "cuda")]
fn png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
    let file = fs::File::create_new(path).unwrap();
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixels).unwrap();
    writer.finish().unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA, hardware graphics, MAELSTROM_PARTICLE_INPUTS and a new MAELSTROM_PARTICLE_OUTPUT directory"]
fn matched_particle_stills_and_motion() {
    use maelstrom::{
        config::{ConfigurationOverlay, NumericalMode},
        execution::ExecutionMode,
        neighbours::SearchBackend,
        scene::load_resolved_scene,
        simulation::Simulation,
    };
    use sha2::{Digest, Sha256};
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let inputs = std::path::PathBuf::from(std::env::var_os("MAELSTROM_PARTICLE_INPUTS").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("MAELSTROM_PARTICLE_OUTPUT").unwrap());
    fs::create_dir(&output).unwrap();
    let mut g = graphics(false);
    let target = Target::new(&g, 1280, 720);
    let mut records = Vec::new();
    let colours = serde_json::from_value(
        serde_json::json!({"colour_mode":"kinetic_energy","min_energy_j":0.0,"max_energy_j":1.962}),
    )
    .unwrap();
    for count in [8000, 32000, 110592] {
        let mut overlay = ConfigurationOverlay::default();
        NumericalMode::Experimental.fill_overrides(&mut overlay.numerical);
        let scene =
            load_resolved_scene(&inputs.join(format!("{count}.toml")), Some(&overlay)).unwrap();
        assert_eq!(scene.particles.len(), count);
        let spacing = f64::from(scene.configuration.numerical.particle_spacing_m);
        let boundaries = BoundaryLines::new(&scene.particles, &scene.planes, spacing).unwrap();
        let mut camera = Camera::default();
        camera
            .fit_scene(
                &scene.particles,
                &scene.planes,
                spacing / 2.0,
                1280.0 / 720.0,
            )
            .unwrap();
        let mut solver = Simulation::from_scene_with_execution(
            &scene,
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda {
                device: 0,
                threads: 128,
            },
        )
        .unwrap();
        for (name, _) in [
            ("indexed", ParticleRenderer::Indexed),
            ("low", ParticleRenderer::Low),
            ("impostor", ParticleRenderer::Impostor),
            ("adaptive", ParticleRenderer::Adaptive),
        ] {
            fs::create_dir(output.join(format!("{count}-{name}"))).unwrap();
        }
        let states = (0..=720).step_by(8).chain([1200]);
        for (frame, step) in states.enumerate() {
            while solver.snapshot().completed_steps() < step {
                let _ = solver.step().unwrap();
            }
            let particles = solver.snapshot().particles();
            let mut state = Sha256::new();
            for p in particles {
                state.update(p.identity.to_le_bytes());
                for v in [
                    p.position_m.x,
                    p.position_m.y,
                    p.position_m.z,
                    p.velocity_m_s.x,
                    p.velocity_m_s.y,
                    p.velocity_m_s.z,
                ] {
                    state.update(v.to_bits().to_le_bytes());
                }
            }
            let state_hash = hex(&state.finalize());
            let mut selected = camera.clone();
            // Hold the initial view through impact, then orbit and zoom smoothly.
            if step > 360 && step <= 720 {
                let t = (step - 360) as f64 / 360.0;
                selected.orbit(0.7 * t, 0.2 * t).unwrap();
                selected.zoom(1.0 - 0.4 * t).unwrap();
            }
            let views = if step == 0 || step == 240 || step == 720 || step == 1200 {
                3
            } else {
                1
            };
            for view in 0..views {
                let mut viewed = selected.clone();
                if view == 1 {
                    viewed = camera.clone();
                    viewed.orbit(0.6, 0.25).unwrap();
                }
                if view == 2 {
                    viewed = camera.clone();
                    viewed
                        .fit(particles, spacing / 2.0, 1280.0 / 720.0)
                        .unwrap();
                    viewed.zoom(0.65).unwrap();
                }
                for (name, mode) in [
                    ("indexed", ParticleRenderer::Indexed),
                    ("low", ParticleRenderer::Low),
                    ("impostor", ParticleRenderer::Impostor),
                    ("adaptive", ParticleRenderer::Adaptive),
                ] {
                    g.set_particle_renderer(mode);
                    let pixels = target.pixels(
                        &g,
                        &DrawScene {
                            particles,
                            boundaries: &boundaries,
                            radius_m: spacing / 2.0,
                            camera: &viewed,
                            colours: &colours,
                            particle_mass_kg: scene.configuration.derived.particle_mass_kg,
                            width: 1280,
                            height: 720,
                        },
                    );
                    let file = if view == 0 && step <= 720 {
                        format!("{count}-{name}/frame-{frame:03}.png")
                    } else {
                        format!("{count}-{name}/still-{step}-{view}.png")
                    };
                    png(&output.join(&file), 1280, 720, &pixels);
                    records.push(serde_json::json!({"file":file,"count":count,"step":step,"view":view,"renderer":name,
                        "state_sha256":state_hash,"rgba_sha256":hex(&Sha256::digest(&pixels))}));
                }
            }
        }
    }
    fs::write(
        output.join("images.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
}
