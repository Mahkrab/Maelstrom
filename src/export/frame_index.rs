//! A derived fixed-record index retains exact frame metadata on temporary storage.
//! It never replaces a manifest's authority; only validated rows enter this index.
use super::{ExportError, Frame, fail};
use crate::storage::Scratch;
use serde::{
    Serialize, Serializer,
    ser::{Error, SerializeSeq},
};

const ROW: u64 = 120;
pub(crate) struct FrameIndex {
    file: Scratch,
    count: usize,
    previous_time: Option<f64>,
}
impl FrameIndex {
    pub(crate) fn new() -> Result<Self, ExportError> {
        Ok(Self {
            file: Scratch::new().map_err(fail)?,
            count: 0,
            previous_time: None,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.count
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub(crate) fn push(&mut self, frame: Frame) -> Result<(), ExportError> {
        if !frame.simulated_seconds.is_finite()
            || frame.simulated_seconds < 0.0
            || self
                .previous_time
                .is_some_and(|time| frame.simulated_seconds <= time)
        {
            return Err(fail(
                "frame timestamps must be finite, non-negative and strictly increasing at binary64 precision",
            ));
        }
        if frame.index != self.count || frame.file != format!("frame-{:08}.png", self.count) {
            return Err(fail("invalid indexed frame order or filename"));
        }
        let step: u64 = frame.step.parse().map_err(fail)?;
        if frame.step != step.to_string() {
            return Err(fail("non-canonical frame step"));
        }
        let mut bytes = [0; ROW as usize];
        bytes[..8].copy_from_slice(&(frame.index as u64).to_le_bytes());
        bytes[8..16].copy_from_slice(&step.to_le_bytes());
        bytes[16..24].copy_from_slice(&frame.simulated_seconds.to_le_bytes());
        for (offset, hash) in [
            (24, &frame.png_sha256),
            (56, &frame.rgba_sha256),
            (88, &frame.snapshot_sha256),
        ] {
            let hex = hash
                .strip_prefix("sha256:")
                .filter(|h| h.len() == 64)
                .ok_or_else(|| fail("invalid frame digest"))?;
            for (i, pair) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
                let digit = |b| match b {
                    b'0'..=b'9' => Ok(b - b'0'),
                    b'a'..=b'f' => Ok(b - b'a' + 10),
                    _ => Err(fail("invalid frame digest")),
                };
                bytes[offset + i] = digit(pair[0])? * 16 + digit(pair[1])?;
            }
        }
        let count = self
            .count
            .checked_add(1)
            .ok_or_else(|| fail("frame index count overflow"))?;
        (count as u64)
            .checked_mul(ROW)
            .ok_or_else(|| fail("frame index byte offset overflow"))?;
        self.file.append(&bytes).map_err(fail)?;
        self.count = count;
        self.previous_time = Some(frame.simulated_seconds);
        Ok(())
    }
    pub(crate) fn get(&self, index: usize) -> Result<Frame, ExportError> {
        if index >= self.count {
            return Err(fail("frame index outside sequence"));
        }
        let mut bytes = [0; ROW as usize];
        let offset = (index as u64)
            .checked_mul(ROW)
            .ok_or_else(|| fail("frame index offset overflow"))?;
        self.file.read_at(offset, &mut bytes).map_err(fail)?;
        let number = |offset| -> Result<u64, ExportError> {
            Ok(u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().map_err(fail)?,
            ))
        };
        let hash = |offset| -> Result<String, ExportError> {
            Ok(super::format_digest(
                bytes[offset..offset + 32].try_into().map_err(fail)?,
            ))
        };
        if number(0)? != index as u64 {
            return Err(fail("temporary frame index order mismatch"));
        }
        Ok(Frame {
            index,
            step: number(8)?.to_string(),
            simulated_seconds: f64::from_bits(number(16)?),
            file: format!("frame-{index:08}.png"),
            png_sha256: hash(24)?,
            rgba_sha256: hash(56)?,
            snapshot_sha256: hash(88)?,
        })
    }
}
impl Serialize for FrameIndex {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.count))?;
        for i in 0..self.count {
            sequence.serialize_element(&self.get(i).map_err(S::Error::custom)?)?;
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(index: usize, time: f64) -> Frame {
        Frame {
            index,
            step: index.to_string(),
            simulated_seconds: time,
            file: format!("frame-{index:08}.png"),
            png_sha256: super::super::hash(b"png"),
            rgba_sha256: super::super::hash(b"rgba"),
            snapshot_sha256: super::super::hash(b"snapshot"),
        }
    }
    #[test]
    fn large_timestamps_fail_only_when_the_recorded_precision_cannot_distinguish_frames() {
        let mut index = FrameIndex::new().unwrap();
        index.push(frame(0, 1e20)).unwrap();
        // Adjacent integers above 2^53 can collapse when converted to f64.
        assert_eq!((1u64 << 54) as f64, ((1u64 << 54) + 1) as f64);
        assert!(
            index
                .push(frame(1, 1e20))
                .unwrap_err()
                .to_string()
                .contains("binary64 precision")
        );
        assert_eq!(index.len(), 1);
        index.push(frame(1, 1e20 + 1e6)).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.push(frame(2, f64::INFINITY)).is_err());
        assert!(index.push(frame(2, -1.0)).is_err());
    }
}
