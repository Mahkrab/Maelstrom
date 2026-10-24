//! The completed manifest owns frame order and timing. Each image is bounded,
//! checked against both recorded hashes and decoded only when requested.

use crate::{export::Frame, render::RenderError};
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
    complete: bool,
    frame_count: usize,
    accepted_steps: String,
    step_stride: String,
    #[serde(deserialize_with = "frames")]
    frames: Vec<Frame>,
    render: Dimensions,
}
#[derive(Deserialize)]
struct Dimensions {
    width: u32,
    height: u32,
    format: String,
}

fn frames<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Frame>, D::Error> {
    struct Frames;
    impl<'de> Visitor<'de> for Frames {
        type Value = Vec<Frame>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("at most 1000000 frame records")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut frames = Vec::new();
            while let Some(frame) = seq.next_element::<Frame>()? {
                if frames.len() == 1_000_000 {
                    return Err(de::Error::custom("sequence exceeds 1000000 frames"));
                }
                // These fields have fixed-format meanings. Bound retained text
                // before growing the index, including for malformed manifests.
                if frame.step.len() > 20
                    || frame.file.len() > 18
                    || frame.png_sha256.len() != 71
                    || frame.rgba_sha256.len() != 71
                    || frame.snapshot_sha256.len() != 71
                {
                    return Err(de::Error::custom("invalid sequence frame text length"));
                }
                frames.push(frame);
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
    pub(super) frames: Vec<Frame>,
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
    /// numerical provenance without retaining it. Up to one million frames are
    /// supported, independently of the size of the recorded particle input.
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
        let m: Manifest = serde_json::from_reader(BufReader::new(file))
            .map_err(|e| error(format!("sequence manifest: {e}")))?;
        if m.schema_version != 1 || !m.complete {
            return Err(error(
                "play requires a completed sequence with manifest schema 1",
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
        if m.frames.is_empty()
            || m.frames.len() > 1_000_000
            || m.frame_count != m.frames.len()
            || count != Some(m.frames.len() as u64)
        {
            return Err(error(
                "sequence frame count or sampling schedule is inconsistent",
            ));
        }
        for (i, f) in m.frames.iter().enumerate() {
            let expected_step = (i as u64).checked_mul(stride).unwrap_or(last).min(last);
            if f.index != i
                || f.file != format!("frame-{i:08}.png")
                || f.step != expected_step.to_string()
                || !f.simulated_seconds.is_finite()
                || !(0.0..=1e12).contains(&f.simulated_seconds)
                || (i == 0 && f.simulated_seconds != 0.0)
                || (i > 0 && f.simulated_seconds <= m.frames[i - 1].simulated_seconds)
                || !valid_hash(&f.png_sha256)
                || !valid_hash(&f.rgba_sha256)
                || !valid_hash(&f.snapshot_sha256)
            {
                return Err(error(format!(
                    "invalid sequence frame {i}: order, filename, step, timestamp or hash"
                )));
            }
        }
        let root = manifest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_owned();
        Ok(Self {
            root,
            frames: m.frames,
            width: m.render.width,
            height: m.render.height,
        })
    }

    /// Number of recorded images; always at least one.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Decodes one opaque RGBA8 frame after checking PNG bytes, dimensions and
    /// decoded-pixel integrity. Memory is bounded by this frame, not the history.
    /// Missing or altered images fail explicitly; nothing is regenerated.
    pub fn decode(&self, index: usize) -> Result<Vec<u8>, RenderError> {
        let f = self
            .frames
            .get(index)
            .ok_or_else(|| error("frame index outside sequence"))?;
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
        let mut rgba = vec![0; expected];
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
