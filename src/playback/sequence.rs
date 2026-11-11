//! The completed manifest owns frame order and timing. Each image is bounded,
//! checked against both recorded hashes and decoded only when requested.

use crate::{
    export::{Frame, frame_index::FrameIndex},
    render::RenderError,
};
use serde::{
    Deserialize, Deserializer,
    de::{self, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufReader, Cursor, Read},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    source_cache: Option<CacheSource>,
    #[serde(default)]
    cache_selection: Option<crate::export::CacheSelection>,
    #[serde(default)]
    timestep_bits: Option<u32>,
    complete: bool,
    frame_count: usize,
    accepted_steps: String,
    step_stride: String,
    #[serde(deserialize_with = "frames")]
    frames: FrameIndex,
    render: Dimensions,
}
#[derive(Deserialize)]
struct CacheSource {
    steps: String,
    stride: String,
    frames: u64,
}
#[derive(Deserialize)]
struct Dimensions {
    width: u32,
    height: u32,
    format: String,
}

fn frames<'de, D: Deserializer<'de>>(deserializer: D) -> Result<FrameIndex, D::Error> {
    struct Frames;
    impl<'de> Visitor<'de> for Frames {
        type Value = FrameIndex;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("ordered frame records")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut frames = FrameIndex::new().map_err(de::Error::custom)?;
            while let Some(frame) = seq.next_element::<Frame>()? {
                // These fields have fixed-format meanings. Bound retained text
                // before growing the index, including for malformed manifests.
                if frame.step.len() > 20
                    || frame.file.len() > 36
                    || frame.png_sha256.len() != 71
                    || frame.rgba_sha256.len() != 71
                    || frame.snapshot_sha256.len() != 71
                {
                    return Err(de::Error::custom("invalid sequence frame text length"));
                }
                frames.push(frame).map_err(de::Error::custom)?;
            }
            Ok(frames)
        }
    }
    deserializer.deserialize_seq(Frames)
}

/// A completed, ordered PNG export. Physical provenance is retained on disk,
/// not interpreted as a request to recreate its simulation backend.
pub struct Sequence {
    root: PathBuf,
    pub(super) frames: std::sync::Arc<FrameIndex>,
    pub(super) width: u32,
    pub(super) height: u32,
}

fn error(message: impl Into<String>) -> RenderError {
    RenderError(message.into())
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
fn valid_hash(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}
fn read(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, RenderError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|e| error(format!("{label}: {}", e.kind())))?;
    if !metadata.file_type().is_file() || metadata.len() > limit {
        return Err(error(format!(
            "{label}: expected a regular file within the size limit"
        )));
    }
    let file = fs::File::open(path).map_err(|e| error(format!("{label}: {}", e.kind())))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| error(format!("{label}: {}", e.kind())))?;
    if bytes.len() as u64 > limit {
        return Err(error(format!("{label}: size limit exceeded")));
    }
    Ok(bytes)
}

impl Sequence {
    /// Opens a completed export directory or `manifest.json`. Progress files,
    /// unsupported schemas, unsafe names and inconsistent schedules are rejected
    /// before a window opens. Reads the frame index from a stream, skipping
    /// numerical provenance without retaining it. The validated frame index uses
    /// temporary disk storage rather than memory proportional to the history.
    pub fn open(path: &Path) -> Result<Self, RenderError> {
        let manifest = if path.is_dir() {
            path.join("manifest.json")
        } else {
            path.to_owned()
        };
        if manifest.file_name().is_none_or(|n| n != "manifest.json") {
            return Err(error(
                "play requires an export directory or manifest.json, not a result or progress file",
            ));
        }
        let metadata = fs::symlink_metadata(&manifest)
            .map_err(|e| error(format!("sequence manifest: {}", e.kind())))?;
        if !metadata.file_type().is_file() {
            return Err(error("sequence manifest: expected a regular file"));
        }
        let file = fs::File::open(&manifest)
            .map_err(|e| error(format!("sequence manifest: {}", e.kind())))?;
        let length = file
            .metadata()
            .map_err(|e| error(format!("sequence manifest: {e}")))?
            .len();
        let mut bounded = file.take(length.saturating_add(1));
        let m: Manifest = serde_json::from_reader(BufReader::new(&mut bounded))
            .map_err(|e| error(format!("sequence manifest: {e}")))?;
        if bounded.limit() != 1 {
            return Err(error("sequence manifest changed length while reading"));
        }
        if !matches!(m.schema_version, 1 | 2) || !m.complete {
            return Err(error(
                "play requires a completed sequence with manifest schema 1 or 2",
            ));
        }
        if m.render.format != "png"
            || !(1..=8192).contains(&m.render.width)
            || !(1..=8192).contains(&m.render.height)
        {
            return Err(error("sequence requires PNG dimensions 1..=8192"));
        }
        let last: u64 = m
            .accepted_steps
            .parse()
            .map_err(|_| error("invalid accepted_steps"))?;
        let stride: u64 = m
            .step_stride
            .parse()
            .map_err(|_| error("invalid step_stride"))?;
        let count = last
            .checked_div(stride)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_add(u64::from(!last.is_multiple_of(stride))));
        let cached = if m.schema_version == 2 {
            if m.kind.as_deref() != Some("maelstrom-cache-render") {
                return Err(error("unsupported schema-2 render kind"));
            }
            let source = m
                .source_cache
                .as_ref()
                .ok_or_else(|| error("missing source cache schedule"))?;
            let selection = m
                .cache_selection
                .ok_or_else(|| error("missing cached frame selection"))?;
            let schedule = crate::cache::Schedule::new(
                source
                    .steps
                    .parse()
                    .map_err(|_| error("invalid cached steps"))?,
                source
                    .stride
                    .parse()
                    .map_err(|_| error("invalid cached stride"))?,
            )
            .map_err(|e| error(e.to_string()))?;
            if source.frames > schedule.len() {
                return Err(error("invalid cached frame availability"));
            }
            let selected = selection
                .count(source.frames)
                .map_err(|e| error(e.to_string()))?;
            let dt = f32::from_bits(
                m.timestep_bits
                    .ok_or_else(|| error("missing cached timestep"))?,
            );
            if !dt.is_finite() || dt <= 0.0 || selected != m.frames.len() as u64 {
                return Err(error("invalid cached timestep or selection count"));
            }
            Some((schedule, selection, f64::from(dt)))
        } else {
            None
        };
        if m.frames.is_empty()
            || m.frame_count != m.frames.len()
            || (cached.is_none() && count != Some(m.frames.len() as u64))
        {
            return Err(error(
                "sequence frame count or sampling schedule is inconsistent",
            ));
        }
        let mut previous_time = None;
        for i in 0..m.frames.len() {
            let f = m.frames.get(i).map_err(|e| error(e.to_string()))?;
            let expected_step = if let Some((schedule, selection, _)) = cached {
                schedule
                    .step(selection.first + i as u64 * selection.every)
                    .ok_or_else(|| error("invalid cached frame mapping"))?
            } else {
                (i as u64).checked_mul(stride).unwrap_or(last).min(last)
            };
            if f.index != i
                || f.file != format!("frame-{i:08}.png")
                || f.step != expected_step.to_string()
                || !f.simulated_seconds.is_finite()
                || f.simulated_seconds < 0.0
                || (cached.is_none() && i == 0 && f.simulated_seconds != 0.0)
                || cached.is_some_and(|(_, _, dt)| f.simulated_seconds != expected_step as f64 * dt)
                || previous_time.is_some_and(|t| f.simulated_seconds <= t)
                || !valid_hash(&f.png_sha256)
                || !valid_hash(&f.rgba_sha256)
                || !valid_hash(&f.snapshot_sha256)
            {
                return Err(error(format!(
                    "invalid sequence frame {i}: order, filename, step, timestamp or hash"
                )));
            }
            previous_time = Some(f.simulated_seconds);
        }
        let root = manifest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_owned();
        Ok(Self {
            root,
            frames: std::sync::Arc::new(m.frames),
            width: m.render.width,
            height: m.render.height,
        })
    }

    /// Validated image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Ordered simulation timestamps, independent of video frame rate.
    pub fn timestamps(&self) -> Result<Vec<f64>, RenderError> {
        let mut times = Vec::new();
        times
            .try_reserve_exact(self.frames.len())
            .map_err(|e| error(e.to_string()))?;
        for i in 0..self.frames.len() {
            times.push(self.timestamp(i)?);
        }
        Ok(times)
    }

    /// Recorded time at an existing image index, with no timestamp array.
    pub fn timestamp(&self, index: usize) -> Result<f64, RenderError> {
        Ok(self
            .frames
            .get(index)
            .map_err(|e| error(e.to_string()))?
            .simulated_seconds)
    }

    /// Number of recorded images; always at least one.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Decodes one opaque RGBA8 frame after checking PNG bytes, dimensions and
    /// decoded-pixel integrity. Memory is bounded by this frame, not the history.
    /// Missing or altered images fail explicitly; nothing is regenerated.
    pub fn decode(&self, index: usize) -> Result<Vec<u8>, RenderError> {
        let f = self.frames.get(index).map_err(|e| error(e.to_string()))?;
        let label = format!("sequence frame {index}");
        let bytes = read(&self.root.join(&f.file), 300 * 1024 * 1024, &label)?;
        if hash(&bytes) != f.png_sha256 {
            return Err(error(format!("{label}: PNG hash mismatch")));
        }
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_limits(png::Limits {
            bytes: 272 * 1024 * 1024,
        });
        let mut reader = decoder
            .read_info()
            .map_err(|e| error(format!("{label}: PNG header: {e}")))?;
        let info = reader.info();
        if info.width != self.width
            || info.height != self.height
            || info.color_type != png::ColorType::Rgba
            || info.bit_depth != png::BitDepth::Eight
            || info.animation_control.is_some()
        {
            return Err(error(format!(
                "{label}: expected one RGBA8 image matching manifest dimensions"
            )));
        }
        let expected = self.width as usize * self.height as usize * 4;
        if reader.output_buffer_size() != Some(expected) {
            return Err(error(format!("{label}: unexpected decoded size")));
        }
        let mut rgba = crate::storage::vector(expected)
            .map_err(|e| error(format!("{label}: pixel allocation: {e}")))?;
        rgba.resize(expected, 0);
        reader
            .next_frame(&mut rgba)
            .map_err(|e| error(format!("{label}: PNG data: {e}")))?;
        reader
            .finish()
            .map_err(|e| error(format!("{label}: PNG end: {e}")))?;
        if rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 255) || hash(&rgba) != f.rgba_sha256 {
            return Err(error(format!("{label}: opaque pixel hash mismatch")));
        }
        Ok(rgba)
    }
}
