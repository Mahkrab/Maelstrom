//! Fixed little-endian arrays preserve binary32 bits and stable row identities.
//! The canonical digest is inside the checked frame, binding it to the payload.
use super::*;
use crate::{math::Vec3, scene::Particle, workflow::SnapshotRecord};

pub(super) const HEADER: usize = 160;
const MAGIC: &[u8; 8] = b"MSPCACHE";

/// Compact accepted-step schedule, including both endpoints. No frame array is allocated.
#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    last: u64,
    stride: u64,
    count: u64,
}
impl Schedule {
    /// Rejects zero stride and an observation count that cannot fit `u64`.
    pub fn new(last: u64, stride: u64) -> Result<Self, CacheError> {
        if stride == 0 {
            return Err(invalid("step stride must be positive"));
        }
        let count = (last / stride)
            .checked_add(1)
            .and_then(|n| n.checked_add(u64::from(!last.is_multiple_of(stride))))
            .ok_or_else(|| invalid("frame count overflow"))?;
        Ok(Self {
            last,
            stride,
            count,
        })
    }
    /// Number of observations, including the initial observation.
    pub fn len(self) -> u64 {
        self.count
    }
    /// Schedules always contain the initial observation.
    pub fn is_empty(self) -> bool {
        false
    }
    // The iterator owner supplies only indexes below count. All non-terminal
    // products are bounded by last; the extra terminal index needs no multiply.
    pub(crate) fn at(self, index: u64) -> u64 {
        if index == self.count - 1 {
            self.last
        } else {
            index * self.stride
        }
    }
    /// Accepted step for an output index, or None outside the schedule.
    pub fn step(self, index: u64) -> Option<u64> {
        (index < self.count).then(|| self.at(index))
    }
}

pub(super) fn size(count: usize) -> Result<usize, CacheError> {
    count
        .checked_mul(24)
        .and_then(|n| n.checked_add(HEADER))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("cache frame length overflow"))
}
fn put(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn get(bytes: &[u8], offset: usize) -> Result<u64, CacheError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| invalid("short cache header"))?
            .try_into()
            .map_err(invalid)?,
    ))
}
fn raw_digest(text: &str) -> Result<[u8; 32], CacheError> {
    if !valid_digest(text) {
        return Err(invalid("invalid cache digest"));
    }
    let mut digest = [0; 32];
    for (i, pair) in text.as_bytes()[7..].as_chunks::<2>().0.iter().enumerate() {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        digest[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(digest)
}
pub(super) fn encode(
    snapshot: &SnapshotRecord,
    index: u64,
    request: &str,
    canonical: [u8; 32],
) -> Result<Vec<u8>, CacheError> {
    let n = snapshot.particles().len();
    let len = size(n)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(invalid)?;
    bytes.resize(len, 0);
    bytes[..8].copy_from_slice(MAGIC);
    bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
    bytes[12..16].copy_from_slice(&(HEADER as u32).to_le_bytes());
    for (o, v) in [
        (16, index),
        (24, snapshot.completed_steps()),
        (32, n as u64),
        (40, (24 * n) as u64),
        (48, (24 * n) as u64),
        (56, HEADER as u64),
        (64, (HEADER + 12 * n) as u64),
    ] {
        put(&mut bytes, o, v);
    }
    bytes[80..112].copy_from_slice(&raw_digest(request)?);
    bytes[112..144].copy_from_slice(&canonical);
    for (i, r) in snapshot.particles().iter().enumerate() {
        let p = r.particle();
        for (offset, v) in [
            (HEADER + 12 * i, p.position_m),
            (HEADER + 12 * n + 12 * i, p.velocity_m_s),
        ] {
            for (j, f) in [v.x, v.y, v.z].iter().enumerate() {
                bytes[offset + 4 * j..offset + 4 * j + 4]
                    .copy_from_slice(&f.to_bits().to_le_bytes());
            }
        }
    }
    Ok(bytes)
}
pub(super) fn decode(
    bytes: &[u8],
    receipt: &Receipt,
    request: &str,
    initial: &[Particle],
) -> Result<(SnapshotRecord, [u8; 32], f64), CacheError> {
    let validation_started = std::time::Instant::now();
    let n = initial.len();
    if bytes.len() != size(n)?
        || bytes.len() as u64 != receipt.bytes
        || bytes.get(..8) != Some(MAGIC.as_slice())
    {
        return Err(invalid("cache frame length or magic mismatch"));
    }
    if bytes[8..12] != [1, 0, 0, 0]
        || bytes[12..16] != (HEADER as u32).to_le_bytes()
        || bytes[72..80] != [0; 8]
        || bytes[144..160] != [0; 16]
    {
        return Err(invalid("unsupported cache frame version, codec or flags"));
    }
    for (o, v) in [
        (16, receipt.index),
        (24, decimal(&receipt.step)?),
        (32, n as u64),
        (40, (24 * n) as u64),
        (48, (24 * n) as u64),
        (56, HEADER as u64),
        (64, (HEADER + 12 * n) as u64),
    ] {
        if get(bytes, o)? != v {
            return Err(invalid(
                "cache frame header disagrees with receipt or layout",
            ));
        }
    }
    if bytes[80..112] != raw_digest(request)?
        || bytes[112..144] != raw_digest(&receipt.snapshot_sha256)?
        || digest(bytes) != receipt.file_sha256
    {
        return Err(invalid(
            "cache frame checksum, request or canonical binding mismatch",
        ));
    }
    let validation_s = validation_started.elapsed().as_secs_f64();
    let mut particles = Vec::new();
    particles.try_reserve_exact(n).map_err(invalid)?;
    for (i, p) in initial.iter().enumerate() {
        let vector = |offset: usize| -> Result<Vec3, CacheError> {
            let f = |j: usize| -> Result<f32, CacheError> {
                Ok(f32::from_bits(u32::from_le_bytes(
                    bytes[offset + j * 4..offset + j * 4 + 4]
                        .try_into()
                        .map_err(invalid)?,
                )))
            };
            Ok(Vec3::new(f(0)?, f(1)?, f(2)?))
        };
        particles.push(Particle {
            identity: p.identity,
            position_m: vector(HEADER + 12 * i)?,
            velocity_m_s: vector(HEADER + 12 * n + 12 * i)?,
        });
    }
    Ok((
        SnapshotRecord::from_particles(particles, decimal(&receipt.step)?).map_err(invalid)?,
        raw_digest(&receipt.snapshot_sha256)?,
        validation_s,
    ))
}
