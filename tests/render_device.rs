//! Real device drawing is an explicitly selected integration check. Pixel probes
//! use an asymmetric fixture so a mirrored or vertically inverted view fails.
#![cfg(feature = "render")]
use maelstrom::{
    math::Vec3,
    render::{BoundaryLines, Camera, DrawScene, Graphics, GraphicsSelection},
    scene::Particle,
};
use std::{sync::mpsc, time::Duration};

#[test]
#[ignore = "requires an explicitly available software Vulkan adapter"]
fn software_vulkan_draws_known_geometry_depth_and_retains_input_bits() {
    check_drawing(true);
}

#[test]
#[ignore = "requires an explicitly available hardware graphics adapter"]
fn hardware_draws_known_geometry_depth_and_retains_input_bits() {
    check_drawing(false);
}

#[test]
#[ignore = "requires an explicitly available hardware graphics adapter"]
fn independent_hardware_render_threads_close_cleanly() {
    std::thread::scope(|scope| {
        let first = scope.spawn(|| check_drawing(false));
        let second = scope.spawn(|| check_drawing(false));
        first.join().unwrap();
        second.join().unwrap();
    });
}

fn check_drawing(software: bool) {
    let instance = Graphics::instance();
    let g = pollster::block_on(Graphics::new(
        &instance,
        GraphicsSelection { index: 0, software },
        None,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
    .unwrap();
    eprintln!("Graphics integration: {:?}", g.adapter_info);
    assert_eq!(
        g.adapter_info.device_type == wgpu::DeviceType::Cpu,
        software
    );
    let camera = Camera {
        eye_m: [0.0, 0.0, 2.0],
        target_m: [0.0; 3],
        up: [0.0, 1.0, 0.0],
        vertical_fov_deg: 90.0,
        near_m: 0.1,
        far_m: 10.0,
    };
    let particle = |identity, x, y, z| Particle {
        identity,
        position_m: Vec3::new(x, y, z),
        velocity_m_s: Vec3::ZERO,
    };
    let front = vec![particle(0, 0.0, 0.0, 0.5), particle(1, 0.6, 0.6, 0.0)];
    let mut both = front.clone();
    both.push(particle(2, 0.0, 0.0, -0.5));
    let before = both.clone();
    let boundary = BoundaryLines::new(&[], &[], 0.05).unwrap();
    let draw_colours =
        |particles: &[Particle], colours: &maelstrom::presentation::ParticleColours, mass: f32| {
            let target = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test target"),
                size: wgpu::Extent3d {
                    width: 128,
                    height: 128,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = g.depth(128, 128).unwrap();
            g.draw(
                &target.create_view(&Default::default()),
                &depth.create_view(&Default::default()),
                &DrawScene {
                    particles,
                    boundaries: &boundary,
                    radius_m: 0.2,
                    camera: &camera,
                    colours,
                    particle_mass_kg: mass,
                    width: 128,
                    height: 128,
                },
                &[],
            )
            .unwrap();
            let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test readback"),
                size: 128 * 128 * 4,
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
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(512),
                        rows_per_image: Some(128),
                    },
                },
                wgpu::Extent3d {
                    width: 128,
                    height: 128,
                    depth_or_array_layers: 1,
                },
            );
            g.queue.submit([encoder.finish()]);
            let (tx, rx) = mpsc::sync_channel(1);
            readback.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
            g.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(30)),
                })
                .unwrap();
            rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
            let bytes = readback.get_mapped_range(..).unwrap().to_vec();
            readback.unmap();
            g.check().unwrap();
            bytes
        };
    let defaults = maelstrom::presentation::ParticleColours::default();
    let draw = |particles: &[Particle]| draw_colours(particles, &defaults, 1.0);
    let pixels = draw(&both);
    assert_eq!(
        pixels,
        draw(&front),
        "far particle must be occluded even when drawn last"
    );
    assert_eq!(
        pixels,
        draw(&both),
        "same-context repeated drawing changed pixels"
    );
    assert_eq!(both, before);
    let at = |x: usize, y: usize| &pixels[(y * 128 + x) * 4..(y * 128 + x) * 4 + 4];
    assert_eq!(at(5, 5), [16, 24, 32, 255]);
    assert!(
        at(83, 45)[2] > at(83, 45)[0] + 30,
        "upper-right sphere missing"
    );
    assert_eq!(
        at(45, 83),
        [16, 24, 32, 255],
        "asymmetric point was mirrored or inverted"
    );
    assert!(at(64, 64)[2] > at(64, 64)[0] + 30, "central sphere missing");
    assert_eq!(
        draw(&[])
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| **p != [16, 24, 32, 255])
            .count(),
        0
    );
    assert!(g.depth(0, 128).is_err());
    assert!(g.depth(u32::MAX, 128).is_err());
    let energy = maelstrom::presentation::ParticleColours::KineticEnergy {
        min_energy_j: 0.0,
        max_energy_j: 4.0,
    };
    let mut centre = vec![particle(0, 0.0, 0.0, 0.0)];
    let probe = |pixels: Vec<u8>| pixels[(64 * 128 + 64) * 4..(64 * 128 + 64) * 4 + 4].to_vec();
    let blue = probe(draw_colours(&centre, &energy, 2.0));
    assert_eq!(&blue[..2], &[0, 0]);
    assert!(blue[2] > 100);
    centre[0].velocity_m_s = Vec3::new(2.0, 0.0, 0.0);
    let red = probe(draw_colours(&centre, &energy, 2.0));
    assert_eq!(&red[1..3], &[0, 0]);
    assert_eq!(red[0], blue[2]);
    let middle = probe(draw_colours(&centre, &energy, 1.0));
    assert_eq!(middle[0], middle[2]);
    assert!(middle[0] > 100);
    assert_eq!(middle[1], 0);
    centre[0].velocity_m_s = Vec3::new(20.0, 0.0, 0.0);
    assert_eq!(probe(draw_colours(&centre, &energy, 2.0)), red);
    let green = maelstrom::presentation::ParticleColours::Fixed {
        colour: "#00ff00".into(),
    };
    let pixel = probe(draw_colours(&centre, &green, 2.0));
    assert_eq!((pixel[0], pixel[2]), (0, 0));
    assert!(pixel[1] > 100);
    g.device.destroy();
    assert!(
        g.check().is_err(),
        "destroyed graphics resources must reject further work"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and an explicitly available software Vulkan adapter"]
fn cuda_worker_snapshots_draw_on_an_independent_graphics_device() {
    use maelstrom::{
        application::{Action, Controller, RunSpec, RunStatus},
        execution::ExecutionMode,
        neighbours::SearchBackend,
        scene::load_resolved_scene,
        simulation::Simulation,
    };
    let scene =
        load_resolved_scene(std::path::Path::new("scenes/falling-block-04.toml"), None).unwrap();
    let mass = scene.configuration.derived.particle_mass_kg;
    let colours = maelstrom::presentation::ParticleColours::default();
    let mut direct = Simulation::from_scene_with_execution(
        &scene,
        SearchBackend::UniformGrid,
        ExecutionMode::Serial,
    )
    .unwrap();
    let boundaries = BoundaryLines::new(&scene.particles, &scene.planes, 0.05).unwrap();
    let mut camera = Camera::default();
    camera.fit(&scene.particles, 0.025, 1.0).unwrap();
    let mut controller = Controller::new().unwrap();
    let (handle, start) = controller
        .start(RunSpec {
            scene,
            backend: SearchBackend::UniformGrid,
            execution: ExecutionMode::Cuda {
                device: 0,
                threads: 128,
            },
            steps: 4,
            initially_paused: true,
        })
        .unwrap();
    assert_eq!(start.wait().unwrap(), RunStatus::Paused);
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
    eprintln!(
        "CUDA/graphics transport: CUDA {:?}; graphics {:?}",
        handle.observe().info.unwrap().device,
        g.adapter_info
    );
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("CUDA observation target"),
        size: wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth = g.depth(128, 128).unwrap();
    let bits = |particles: &[Particle]| {
        particles
            .iter()
            .map(|p| {
                (
                    p.identity,
                    [
                        p.position_m.x,
                        p.position_m.y,
                        p.position_m.z,
                        p.velocity_m_s.x,
                        p.velocity_m_s.y,
                        p.velocity_m_s.z,
                    ]
                    .map(f32::to_bits),
                )
            })
            .collect::<Vec<_>>()
    };
    for step in 1..=2 {
        controller
            .control(&handle, Action::SingleStep)
            .unwrap()
            .wait()
            .unwrap();
        let _ = direct.step().unwrap();
        let observation = handle.observe();
        let snapshot = observation.snapshot.unwrap();
        assert_eq!(snapshot.completed_steps(), step);
        let before = bits(snapshot.particles());
        assert_eq!(before, bits(direct.snapshot().particles()));
        g.draw(
            &target.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
            &DrawScene {
                particles: snapshot.particles(),
                boundaries: &boundaries,
                radius_m: 0.025,
                camera: &camera,
                colours: &colours,
                particle_mass_kg: mass,
                width: 128,
                height: 128,
            },
            &[],
        )
        .unwrap();
        g.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        g.check().unwrap();
        assert_eq!(bits(snapshot.particles()), before);
        camera.orbit(0.2, 0.1).unwrap();
    }
    assert!(g.depth(u32::MAX, 128).is_err());
    g.device.destroy();
    assert!(g.check().is_err());
    handle.cancel();
    controller.join().unwrap();
    let final_state = handle.observe();
    assert_eq!(final_state.status, RunStatus::Cancelled);
    assert_eq!(final_state.snapshot.unwrap().completed_steps(), 2);
}
