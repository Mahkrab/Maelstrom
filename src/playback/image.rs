//! One reusable texture presents saved pixels with aspect-preserving letterboxing.
//! Sampling decodes sRGB to linear light before the surface encodes it again.

use crate::render::{Graphics, RenderError};

pub(super) struct Image {
    texture: wgpu::Texture,
    bind: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    width: u32,
    height: u32,
}
impl Image {
    pub fn new(
        g: &Graphics,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        if width > g.device.limits().max_texture_dimension_2d
            || height > g.device.limits().max_texture_dimension_2d
        {
            return Err(RenderError(
                "saved image exceeds graphics texture limit".into(),
            ));
        }
        let texture = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("saved image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let layout = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("saved image"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let sampler = g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("saved image"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("saved image"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("saved image"),
            source: wgpu::ShaderSource::Wgsl(include_str!("image.wgsl").into()),
        });
        let pipeline_layout = g
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("saved image"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = g
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("saved image"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        g.check()?;
        Ok(Self {
            texture,
            bind,
            pipeline,
            width,
            height,
        })
    }
    pub fn upload(&self, g: &Graphics, rgba: &[u8]) -> Result<(), RenderError> {
        if rgba.len() != self.width as usize * self.height as usize * 4 {
            return Err(RenderError(
                "saved image byte count does not match dimensions".into(),
            ));
        }
        g.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.check()
    }
    pub fn draw(
        &self,
        g: &Graphics,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        g.check()?;
        let mut encoder = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("saved image"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            let [x, y, w, h] = viewport(self.width, self.height, width, height);
            pass.set_viewport(x, y, w, h, 0.0, 1.0);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.draw(0..3, 0..1);
        }
        g.queue.submit([encoder.finish()]);
        g.check()
    }
}
pub(super) fn viewport(iw: u32, ih: u32, width: u32, height: u32) -> [f32; 4] {
    let scale = (width as f32 / iw as f32).min(height as f32 / ih as f32);
    let w = iw as f32 * scale;
    let h = ih as f32 * scale;
    [(width as f32 - w) / 2.0, (height as f32 - h) / 2.0, w, h]
}
