//! One submitted texture is read back completely before the next observation is
//! admitted. Row padding is removed without changing the top-left image origin.

use super::{ExportError, fail};
use crate::{
    render::{BoundaryLines, DrawScene, Graphics, Settings},
    workflow::{SnapshotRecord, publish_bytes},
};
use std::{sync::mpsc, time::Duration};

pub(super) fn pixels(graphics: &Graphics, scene: &DrawScene<'_>) -> Result<Vec<u8>, ExportError> {
    graphics.check().map_err(fail)?;
    let depth = graphics.depth(scene.width, scene.height).map_err(fail)?;
    let extent = wgpu::Extent3d {
        width: scene.width,
        height: scene.height,
        depth_or_array_layers: 1,
    };
    let texture = graphics.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offline colour"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    graphics
        .draw(
            &texture.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
            scene,
            &[],
        )
        .map_err(fail)?;
    let row = scene.width * 4;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let size = u64::from(padded) * u64::from(scene.height);
    if size > graphics.device.limits().max_buffer_size {
        return Err(fail("readback exceeds graphics buffer limit"));
    }
    let buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offline readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = graphics.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(scene.height),
            },
        },
        extent,
    );
    graphics.queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::sync_channel(1);
    buffer.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    // No observation deadline abandons a live device operation. Cancellation is
    // checked between synchronous frames, as it is between solver stages.
    loop {
        match graphics.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(1)),
        }) {
            Ok(_) | Err(wgpu::PollError::Timeout) => {}
            Err(error) => return Err(fail(error)),
        }
        match rx.try_recv() {
            Ok(result) => {
                result.map_err(fail)?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) => graphics.check().map_err(fail)?,
            Err(e) => return Err(fail(e)),
        }
    }
    let mapped = buffer.get_mapped_range(..).map_err(fail)?;
    let mut rgba = Vec::with_capacity((u64::from(row) * u64::from(scene.height)) as usize);
    for bytes in mapped.chunks_exact(padded as usize) {
        rgba.extend_from_slice(&bytes[..row as usize]);
    }
    drop(mapped);
    buffer.unmap();
    graphics.check().map_err(fail)?;
    Ok(rgba)
}

pub(super) fn encode(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, ExportError> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        encoder.set_compression(png::Compression::Balanced);
        encoder.set_filter(png::Filter::Adaptive);
        let mut writer = encoder.write_header().map_err(fail)?;
        writer.write_image_data(rgba).map_err(fail)?;
        writer.finish().map_err(fail)?;
    }
    Ok(bytes)
}

type EncodePng = fn(u32, u32, &[u8]) -> Result<Vec<u8>, ExportError>;

/// The two fallible frame stages are kept separate so sequence recovery can be
/// checked with real encoder and device failures before publication.
#[derive(Clone, Copy)]
pub(super) struct FrameOperations {
    pub read_pixels: fn(&Graphics, &DrawScene<'_>) -> Result<Vec<u8>, ExportError>,
    pub encode_png: EncodePng,
}
impl Default for FrameOperations {
    fn default() -> Self {
        Self {
            read_pixels: pixels,
            encode_png: encode,
        }
    }
}

pub(super) struct FrameWriter<'a> {
    pub operations: FrameOperations,
    pub graphics: &'a Graphics,
    pub settings: &'a Settings,
    pub boundaries: &'a BoundaryLines,
    pub spacing: f64,
    pub dt: f64,
    pub destination: &'a std::path::Path,
    pub render_seconds: f64,
    pub encode_seconds: f64,
}
impl FrameWriter<'_> {
    pub fn write(
        &mut self,
        snapshot: &SnapshotRecord,
        index: usize,
    ) -> Result<super::Frame, ExportError> {
        let particles: Vec<_> = snapshot.particles().iter().map(|p| p.particle()).collect();
        let start = std::time::Instant::now();
        let rgba = (self.operations.read_pixels)(
            self.graphics,
            &DrawScene {
                particles: &particles,
                boundaries: self.boundaries,
                radius_m: self.spacing / 2.0,
                camera: &self.settings.camera,
                width: self.settings.width,
                height: self.settings.height,
            },
        )?;
        self.render_seconds += start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let bytes = (self.operations.encode_png)(self.settings.width, self.settings.height, &rgba)?;
        self.encode_seconds += start.elapsed().as_secs_f64();
        let file = format!("frame-{index:08}.png");
        publish_bytes(&self.destination.join(&file), &bytes)
            .map_err(|e| fail(format!("frame {index} publication: {}", e.kind())))?;
        Ok(super::Frame {
            index,
            step: snapshot.completed_steps().to_string(),
            simulated_seconds: snapshot.completed_steps() as f64 * self.dt,
            file,
            png_sha256: super::hash(&bytes),
            rgba_sha256: super::hash(&rgba),
            snapshot_sha256: super::hash(snapshot.to_json().map_err(fail)?.as_bytes()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_rgba_lengths_fail_encoding() {
        assert!(encode(2, 2, &[0; 15]).is_err());
        assert!(encode(2, 2, &[0; 17]).is_err());
    }
    #[test]
    #[ignore = "requires explicitly available software Vulkan"]
    fn destroyed_device_rejects_offscreen_readback() {
        let instance = Graphics::instance();
        let g = pollster::block_on(Graphics::new(
            &instance,
            crate::render::GraphicsSelection {
                index: 0,
                software: true,
            },
            None,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ))
        .unwrap();
        g.device.destroy();
        let boundary = BoundaryLines::new(&[], &[], 0.05).unwrap();
        assert!(
            pixels(
                &g,
                &DrawScene {
                    particles: &[],
                    boundaries: &boundary,
                    radius_m: 0.025,
                    camera: &crate::render::Camera::default(),
                    width: 16,
                    height: 16
                }
            )
            .is_err()
        );
    }
}
