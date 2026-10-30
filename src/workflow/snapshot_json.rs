//! Snapshot provenance uses the standalone pretty JSON bytes, including field
//! order and whitespace. Fixed schema text avoids repeated generic formatting;
//! numeric tokens still use serde_json's formatter. No complete JSON buffer is kept.

use super::{RecordError, SnapshotRecord};
use serde_json::ser::{CompactFormatter, Formatter};
use sha2::{Digest, Sha256};
use std::io::{self, BufWriter, Write};

struct HashWriter(Sha256);

impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl SnapshotRecord {
    pub(crate) fn canonical_sha256(&self) -> Result<[u8; 32], RecordError> {
        let mut hash = HashWriter(Sha256::new());
        {
            // Buffer the small numeric tokens and schema fragments so hashing
            // receives sizeable blocks without materialising the whole document.
            let mut writer = BufWriter::with_capacity(64 * 1024, &mut hash);
            self.write_canonical_json(&mut writer)?;
            writer.flush().map_err(serde_json::Error::io)?;
        }
        Ok(hash.0.finalize().into())
    }

    fn write_canonical_json(&self, writer: &mut impl Write) -> Result<(), RecordError> {
        self.validate()?;
        self.write_json_body(writer)
            .map_err(serde_json::Error::io)?;
        Ok(())
    }

    fn write_json_body(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut numbers = CompactFormatter;
        writer.write_all(b"{\n  \"schema_version\": ")?;
        numbers.write_u32(writer, self.schema_version)?;
        writer.write_all(b",\n  \"completed_steps\": \"")?;
        numbers.write_u64(writer, self.completed_steps)?;
        writer.write_all(b"\",\n  \"particles\": [")?;
        for (index, particle) in self.particles.iter().enumerate() {
            writer.write_all(if index == 0 { b"\n" } else { b",\n" })?;
            writer.write_all(b"    {\n      \"identity\": \"")?;
            numbers.write_u64(writer, particle.identity)?;
            writer.write_all(b"\",\n      \"position_m\": {\n        \"x\": ")?;
            numbers.write_f32(writer, particle.position_m.x)?;
            writer.write_all(b",\n        \"y\": ")?;
            numbers.write_f32(writer, particle.position_m.y)?;
            writer.write_all(b",\n        \"z\": ")?;
            numbers.write_f32(writer, particle.position_m.z)?;
            writer.write_all(b"\n      },\n      \"velocity_m_s\": {\n        \"x\": ")?;
            numbers.write_f32(writer, particle.velocity_m_s.x)?;
            writer.write_all(b",\n        \"y\": ")?;
            numbers.write_f32(writer, particle.velocity_m_s.y)?;
            writer.write_all(b",\n        \"z\": ")?;
            numbers.write_f32(writer, particle.velocity_m_s.z)?;
            writer.write_all(b"\n      }\n    }")?;
        }
        if !self.particles.is_empty() {
            writer.write_all(b"\n  ")?;
        }
        writer.write_all(b"]\n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::Vec3, workflow::record::ParticleRecord};

    fn check(snapshot: &SnapshotRecord) {
        let expected = snapshot.to_json().unwrap();
        let mut actual = Vec::new();
        snapshot.write_canonical_json(&mut actual).unwrap();
        assert_eq!(actual, expected.as_bytes());
        assert_eq!(
            snapshot.canonical_sha256().unwrap().as_slice(),
            Sha256::digest(&actual).as_slice()
        );
    }

    #[test]
    fn canonical_bytes_preserve_empty_records_decimal_limits_and_signed_zero() {
        let mut snapshot = SnapshotRecord {
            schema_version: 1,
            completed_steps: u64::MAX,
            particles: Vec::new(),
        };
        check(&snapshot);
        snapshot.particles.push(ParticleRecord {
            identity: u64::MAX,
            position_m: Vec3::new(1.0, -0.0, 2.5),
            velocity_m_s: Vec3::ZERO,
        });
        let expected = "{\n  \"schema_version\": 1,\n  \"completed_steps\": \"18446744073709551615\",\n  \"particles\": [\n    {\n      \"identity\": \"18446744073709551615\",\n      \"position_m\": {\n        \"x\": 1.0,\n        \"y\": -0.0,\n        \"z\": 2.5\n      },\n      \"velocity_m_s\": {\n        \"x\": 0.0,\n        \"y\": 0.0,\n        \"z\": 0.0\n      }\n    }\n  ]\n}";
        let mut actual = Vec::new();
        snapshot.write_canonical_json(&mut actual).unwrap();
        assert_eq!(actual, expected.as_bytes());
        check(&snapshot);
    }

    #[test]
    fn canonical_numbers_match_derived_serialisation_across_finite_binary32_values() {
        let mut words = vec![
            0, 1, 0x007fffff, 0x00800000, 0x3dcccccd, 0x3f800000, 0x4b800001, 0x7f7fffff,
            0x80000000, 0xff7fffff,
        ];
        let mut word = 0x12345678_u32;
        for _ in 0..4096 {
            word ^= word << 13;
            word ^= word >> 17;
            word ^= word << 5;
            if f32::from_bits(word).is_finite() {
                words.push(word);
            }
        }
        let particles = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| {
                let value = f32::from_bits(word);
                ParticleRecord {
                    identity: i as u64 * 31,
                    position_m: Vec3::new(value, -value, value),
                    velocity_m_s: Vec3::new(-value, value, -value),
                }
            })
            .collect();
        check(&SnapshotRecord {
            schema_version: 1,
            completed_steps: 123456789,
            particles,
        });
    }

    #[test]
    fn invalid_records_fail_before_writing_and_sink_errors_remain_typed() {
        let mut snapshot = SnapshotRecord {
            schema_version: 2,
            completed_steps: 0,
            particles: Vec::new(),
        };
        for kind in 0..4 {
            snapshot.schema_version = if kind == 0 { 2 } else { 1 };
            snapshot.particles = match kind {
                1 => vec![ParticleRecord {
                    identity: 1,
                    position_m: Vec3::new(f32::NAN, 0.0, 0.0),
                    velocity_m_s: Vec3::ZERO,
                }],
                2 | 3 => vec![
                    ParticleRecord {
                        identity: 1,
                        position_m: Vec3::ZERO,
                        velocity_m_s: Vec3::ZERO
                    };
                    2
                ],
                _ => Vec::new(),
            };
            if kind == 3 {
                snapshot.particles[1].identity = 0;
            }
            let mut bytes = Vec::new();
            assert_eq!(
                snapshot
                    .write_canonical_json(&mut bytes)
                    .unwrap_err()
                    .to_string(),
                snapshot.to_json().unwrap_err().to_string()
            );
            assert!(bytes.is_empty());
            assert!(snapshot.canonical_sha256().is_err());
        }
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        snapshot.particles.clear();
        match snapshot.write_canonical_json(&mut Broken).unwrap_err() {
            RecordError::Json(e) => assert_eq!(e.io_error_kind(), Some(io::ErrorKind::BrokenPipe)),
            other => panic!("unexpected error: {other}"),
        }
    }
}
