//! Graphics resources are presentation-owned. The same scene pass accepts a surface
//! view or a caller-owned texture view; window creation stays outside rendering.

use super::{
    BoundaryLines, Camera, RenderError,
    camera::{dot, position, sub},
};
use crate::scene::Particle;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::util::DeviceExt;

// Mesa's D3D12 driver registers thread-local cleanup callbacks. Keep its loader
// alive until process exit so a rendering thread cannot unload those callbacks
// before its own thread-local destructors run. Devices remain independently owned.
static GRAPHICS_LOADER: OnceLock<wgpu::Instance> = OnceLock::new();

/// Explicit selection among graphics adapters of the requested device class.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphicsSelection {
    /// Zero-based index within hardware or software adapters, enumerated at launch.
    pub index: usize,
    /// Require a software/CPU adapter. Hardware selection never falls back to it.
    pub software: bool,
}

/// A scene to draw from immutable accepted data.
pub struct DrawScene<'a> {
    /// Accepted particles, in ascending identity order.
    pub particles: &'a [Particle],
    /// Frozen initial plane outlines.
    pub boundaries: &'a BoundaryLines,
    /// Display particle radius in metres.
    pub radius_m: f64,
    /// Presentation camera.
    pub camera: &'a Camera,
    /// Actual non-zero drawable width.
    pub width: u32,
    /// Actual non-zero drawable height.
    pub height: u32,
}

/// Optional graphics resources, with no solver or CUDA context ownership.
pub struct Graphics {
    /// Logical graphics device for caller-owned surface/offscreen resources.
    pub device: wgpu::Device,
    /// Graphics submission queue, separate from simulation execution.
    pub queue: wgpu::Queue,
    /// Selected graphics adapter and driver context.
    pub adapter_info: wgpu::AdapterInfo,
    /// Selected adapter, retained for surface capability queries.
    pub adapter: wgpu::Adapter,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    sphere: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    mesh: wgpu::Buffer,
    mesh_count: u32,
    failure: Arc<Mutex<Option<String>>>,
}
impl Graphics {
    /// Enables Vulkan and OpenGL ES, including WSLg's accelerated D3D12 driver.
    pub fn instance() -> wgpu::Instance {
        Self::instance_with_descriptor(wgpu::InstanceDescriptor::new_without_display_handle())
    }
    pub(crate) fn instance_with_descriptor(descriptor: wgpu::InstanceDescriptor) -> wgpu::Instance {
        let create = |descriptor| {
            wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
                ..descriptor
            })
        };
        GRAPHICS_LOADER
            .get_or_init(|| create(wgpu::InstanceDescriptor::new_without_display_handle()));
        // Each caller needs its own EGL context; sharing the retained loader's
        // context between rendering threads can fail with EGL_BAD_ACCESS.
        create(descriptor)
    }
    /// Acquires exactly the selected adapter and compiles shared scene pipelines.
    /// Software rendering is never an implicit fallback from hardware selection.
    pub async fn new(
        instance: &wgpu::Instance,
        selection: GraphicsSelection,
        surface: Option<&wgpu::Surface<'_>>,
        format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let adapter = Self::select_adapter(instance, selection, surface).await?;
        Self::from_adapter(adapter, surface, format).await
    }
    pub(crate) async fn select_adapter(
        instance: &wgpu::Instance,
        selection: GraphicsSelection,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<wgpu::Adapter, RenderError> {
        let mut adapters = instance
            .enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::GL)
            .await;
        // Prefer native Vulkan when both APIs expose hardware. A usable WSLg
        // OpenGL adapter still takes precedence over CPU Vulkan rendering.
        adapters.sort_by_key(|a| a.get_info().backend != wgpu::Backend::Vulkan);
        let adapter = adapters.into_iter()
            .filter(|a| (a.get_info().device_type == wgpu::DeviceType::Cpu) == selection.software)
            .filter(|a| surface.is_none_or(|s| a.is_surface_supported(s)))
            .nth(selection.index)
            .ok_or_else(|| {
                RenderError(format!(
                "Vulkan/OpenGL {} adapter {} unavailable; software rendering requires --software explicitly",
                if selection.software { "software" } else { "hardware" }, selection.index
                ))
            })?;
        Ok(adapter)
    }
    pub(crate) async fn from_adapter(
        adapter: wgpu::Adapter,
        surface: Option<&wgpu::Surface<'_>>,
        format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        if !format.is_srgb() {
            return Err(RenderError("drawing requires an sRGB colour target".into()));
        }
        if let Some(surface) = surface
            && !adapter.is_surface_supported(surface)
        {
            return Err(RenderError(
                "selected graphics adapter does not support this window surface".into(),
            ));
        }
        let adapter_info = adapter.get_info();
        let limits = wgpu::Limits {
            max_texture_dimension_2d: 8192.min(adapter.limits().max_texture_dimension_2d),
            // Drawing needs no compute resources. Requiring their default
            // limits would reject otherwise usable OpenGL ES 3.0 adapters.
            ..wgpu::Limits::downlevel_webgl2_defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("particle viewer"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| RenderError(format!("graphics device: {e}")))?;
        let failure = Arc::new(Mutex::new(None));
        let errors = failure.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            *errors.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(format!("graphics device error: {e}"));
        }));
        let lost = failure.clone();
        device.set_device_lost_callback(move |reason, message| {
            *lost.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!(
                "graphics device unavailable: {reason:?}: {message}"
            ));
        });
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accepted particles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particles.wgsl").into()),
        });
        let targets = [Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let vertex = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0=>Float32x3],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![1=>Float32x3],
        };
        let make_pipeline =
            |entry, fragment, topology, buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        buffers,
                    },
                    primitive: wgpu::PrimitiveState {
                        topology,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fragment),
                        compilation_options: Default::default(),
                        targets: &targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };
        let sphere = make_pipeline(
            "sphere",
            "particle_colour",
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(vertex.clone()), Some(instance_layout)],
        );
        let lines = make_pipeline(
            "line",
            "line_colour",
            wgpu::PrimitiveTopology::LineList,
            &[Some(vertex)],
        );
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diagnostic overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let overlay = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("overlay"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 20,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0=>Float32x2,1=>Float32x3],
                })],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("colour"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let mesh_data = sphere_mesh();
        let mesh_count = (mesh_data.len() / 3) as u32;
        let mesh = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unit sphere, 16 longitudes x 12 latitudes"),
            contents: &bytes(&mesh_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        if let Some(error) = scope.pop().await {
            return Err(RenderError(format!("graphics pipeline: {error}")));
        }
        Ok(Self {
            device,
            queue,
            adapter_info,
            adapter,
            uniform,
            bind,
            sphere,
            lines,
            overlay,
            mesh,
            mesh_count,
            failure,
        })
    }
    /// Returns a retained asynchronous graphics failure, if one was reported.
    pub fn check(&self) -> Result<(), RenderError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| RenderError(format!("graphics device polling: {e}")))?;
        if let Some(error) = self
            .failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            Err(RenderError(error))
        } else {
            Ok(())
        }
    }
    /// Creates a matching single-sample depth target after checking device limits.
    pub fn depth(&self, width: u32, height: u32) -> Result<wgpu::Texture, RenderError> {
        let limit = self.device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > limit || height > limit {
            return Err(RenderError(format!(
                "drawable dimensions exceed graphics limit {limit}"
            )));
        }
        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }))
    }
    /// Draws opaque spheres and clipped plane outlines into a supplied sRGB view.
    /// Overlay vertices are clip-space XY followed by linear RGB, five floats each.
    pub fn draw(
        &self,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        scene: &DrawScene<'_>,
        overlay: &[f32],
    ) -> Result<(), RenderError> {
        self.check()?;
        scene
            .camera
            .validate(f64::from(scene.width) / f64::from(scene.height))?;
        if !overlay.len().is_multiple_of(15) {
            return Err(RenderError(
                "overlay must contain complete triangles".into(),
            ));
        }
        let (right, up, forward, distance) = scene.camera.basis()?;
        let convert = |p| -> Result<[f32; 3], RenderError> {
            let d = sub(p, scene.camera.eye_m);
            let result = [
                dot(d, right) / distance,
                dot(d, up) / distance,
                dot(d, forward) / distance,
            ]
            .map(|x| x as f32);
            if !result.into_iter().all(f32::is_finite) {
                return Err(RenderError("view-relative geometry is not finite".into()));
            }
            Ok(result)
        };
        if !scene.radius_m.is_finite() || scene.radius_m <= 0.0 {
            return Err(RenderError("invalid display radius".into()));
        }
        let radius = (scene.radius_m / distance) as f32;
        if !radius.is_finite() || radius == 0.0 {
            return Err(RenderError("display radius cannot be represented".into()));
        }
        let fy = 1.0 / (scene.camera.vertical_fov_deg.to_radians() / 2.0).tan();
        let near = scene.camera.near_m / distance;
        let far = scene.camera.far_m / distance;
        let mut uniform = vec![
            (fy * f64::from(scene.height) / f64::from(scene.width)) as f32,
            fy as f32,
            (far / (far - near)) as f32,
            (-near * far / (far - near)) as f32,
        ];
        for v in [right, up, forward] {
            uniform.extend(v.map(|x| x as f32));
            uniform.push(0.0);
        }
        uniform.extend([0.0, 0.0, 0.0, radius]);
        self.queue.write_buffer(&self.uniform, 0, &bytes(&uniform));
        if scene
            .particles
            .windows(2)
            .any(|p| p[0].identity >= p[1].identity)
        {
            return Err(RenderError(
                "drawing requires ascending particle identities".into(),
            ));
        }
        let centres = scene
            .particles
            .iter()
            .map(|p| convert(position(p)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let lines = scene
            .boundaries
            .vertices_m
            .iter()
            .map(|p| convert(*p))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let make_buffer = |label, data: &[f32]| -> Result<wgpu::Buffer, RenderError> {
            let data = if data.is_empty() { &[0.0; 3][..] } else { data };
            let size = data
                .len()
                .checked_mul(4)
                .ok_or_else(|| RenderError("graphics buffer overflow".into()))?;
            if size as u64 > self.device.limits().max_buffer_size {
                return Err(RenderError("geometry exceeds graphics buffer limit".into()));
            }
            Ok(self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &bytes(data),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
        };
        let centres_buffer = make_buffer("accepted centres", &centres)?;
        let line_buffer = make_buffer("plane outlines", &lines)?;
        let overlay_buffer = make_buffer("overlay", overlay)?;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0051815,
                            g: 0.0091341,
                            b: 0.0144438,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_pipeline(&self.sphere);
            pass.set_vertex_buffer(0, self.mesh.slice(..));
            pass.set_vertex_buffer(1, centres_buffer.slice(..));
            pass.draw(
                0..self.mesh_count,
                0..u32::try_from(scene.particles.len())
                    .map_err(|_| RenderError("particle draw count exceeds u32".into()))?,
            );
            pass.set_pipeline(&self.lines);
            pass.set_vertex_buffer(0, line_buffer.slice(..));
            pass.draw(0..(lines.len() / 3) as u32, 0..1);
        }
        if !overlay.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("diagnostics"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.overlay);
            pass.set_vertex_buffer(0, overlay_buffer.slice(..));
            pass.draw(0..(overlay.len() / 5) as u32, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.check()
    }
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn sphere_mesh() -> Vec<f32> {
    let point = |lat: u32, lon: u32| {
        let t = std::f32::consts::PI * lat as f32 / 12.0;
        let p = std::f32::consts::TAU * lon as f32 / 16.0;
        [t.sin() * p.cos(), t.cos(), t.sin() * p.sin()]
    };
    let mut vertices = Vec::new();
    for lat in 0..12 {
        for lon in 0..16 {
            for (a, b) in [
                (lat, lon),
                (lat + 1, lon),
                (lat + 1, lon + 1),
                (lat, lon),
                (lat + 1, lon + 1),
                (lat, lon + 1),
            ] {
                vertices.extend(point(a, b));
            }
        }
    }
    vertices
}
