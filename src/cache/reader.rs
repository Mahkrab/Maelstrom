//! Readers retain typed inputs and a compact schedule, opening only selected frames.
//! A checked binary frame carries the writer's canonical digest into presentation;
//! full validation independently recomputes that digest when requested.
use super::*;
use crate::{
    scene::ResolvedScene,
    workflow::{RecordedInput, SnapshotRecord},
};
use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
};

/// An immutable snapshot whose bytes and canonical-digest binding were checked together.
pub struct ValidatedFrame {
    snapshot: SnapshotRecord,
    canonical: [u8; 32],
    read_timings: ReadTimings,
}
/// Per-frame CPU wall costs; these intervals do not include rendering or canonical rehashing.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ReadTimings {
    /// File opens/reads and receipt validation.
    pub reading_s: f64,
    /// Header and complete binary checksum validation.
    pub binary_validation_s: f64,
    /// Conversion to finite, immutable snapshot records.
    pub decoding_s: f64,
}
impl ValidatedFrame {
    /// Exact accepted state, in stable identity order.
    pub fn snapshot(&self) -> &SnapshotRecord {
        &self.snapshot
    }
    /// Existing canonical pretty-JSON SHA-256, not the binary checksum.
    pub fn canonical_sha256(&self) -> String {
        hex(&self.canonical)
    }
    /// Diagnostic costs measured while reading this frame.
    pub fn read_timings(&self) -> ReadTimings {
        self.read_timings
    }
}

/// Validated cache metadata. Frame data are read on demand, without a solver.
pub struct Cache {
    root: PathBuf,
    request: Request,
    pub(crate) input: RecordedInput,
    scene: ResolvedScene,
    schedule: Schedule,
    frames: u64,
    complete: bool,
    request_hash: String,
    identity: String,
    receipt_hashes: crate::storage::Scratch,
}

impl Cache {
    /// Opens a complete cache, or its receipt-backed prefix when explicitly allowed.
    /// Metadata, receipt chain, identities and completed diagnostics are checked here;
    /// particle files are checked by [`Self::frame`] before consumption.
    pub fn open(root: &Path, allow_partial: bool) -> Result<Self, CacheError> {
        Self::open_with_cancellation(root, allow_partial, || false)
    }
    /// Opens metadata and receipt indexes with cooperative cancellation.
    /// Cancellation never publishes or changes source cache files.
    pub fn open_with_cancellation(
        root: &Path,
        allow_partial: bool,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<Self, CacheError> {
        if cancelled() {
            return Err(CacheError::Cancelled);
        }
        directory(root)?;
        directory(&root.join("frames"))?;
        directory(&root.join("receipts"))?;
        let bytes = read(&root.join("request.json"), 1024 * 1024)?;
        let request_hash = digest(&bytes);
        let request: Request = serde_json::from_slice(&bytes)?;
        let schedule = Schedule::new(decimal(&request.steps)?, decimal(&request.stride)?)?;
        if request.schema_version != 1
            || request.kind != "maelstrom-particle-cache"
            || !valid_digest(&request.input_sha256)
            || !valid_digest(&request.identities_sha256)
        {
            return Err(invalid("unsupported or invalid cache request"));
        }
        format::size(request.particle_count)?;
        let (input, input_hash) = input(&root.join("input.json"), &mut cancelled)?;
        if input_hash != request.input_sha256 {
            return Err(invalid("cache input checksum mismatch"));
        }
        let scene = input.resolve_observation().map_err(invalid)?;
        if scene.particles.len() != request.particle_count
            || scene.configuration.numerical.fixed_timestep_s.to_bits() != request.timestep_bits
        {
            return Err(invalid("cache input count or timestep mismatch"));
        }
        let identity_size = u64::try_from(request.particle_count)
            .map_err(invalid)?
            .checked_mul(8)
            .ok_or_else(|| invalid("cache identity length overflow"))?;
        let file = regular(&root.join("identities.bin"), identity_size)?;
        if file.metadata()?.len() != identity_size {
            return Err(invalid("cache identity table length mismatch"));
        }
        let mut reader = BufReader::new(file);
        let mut hash = Sha256::new();
        for (index, particle) in scene.particles.iter().enumerate() {
            if index.is_multiple_of(4096) && cancelled() {
                return Err(CacheError::Cancelled);
            }
            let mut bytes = [0; 8];
            reader.read_exact(&mut bytes)?;
            hash.update(bytes);
            if particle.identity.to_le_bytes() != bytes {
                return Err(invalid("cache identity table mismatch"));
            }
        }
        if !reader.fill_buf()?.is_empty()
            || hex(hash.finalize().as_slice()) != request.identities_sha256
        {
            return Err(invalid("cache identity table mismatch"));
        }
        let completion = match fs::symlink_metadata(root.join("cache.json")) {
            Ok(_) => Some(serde_json::from_slice::<Completion>(&read(
                &root.join("cache.json"),
                1024 * 1024,
            )?)?),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        if completion.is_none() && !allow_partial {
            return Err(invalid(
                "incomplete cache requires explicit --allow-partial",
            ));
        }
        let mut chain = Sha256::new();
        let mut frames = 0;
        let mut receipt_hashes = crate::storage::Scratch::new()?;
        while frames < schedule.len() {
            if cancelled() {
                return Err(CacheError::Cancelled);
            }
            let path = root.join("receipts").join(receipt_name(frames));
            if completion.is_none()
                && matches!(fs::symlink_metadata(&path),Err(e) if e.kind()==io::ErrorKind::NotFound)
            {
                break;
            }
            let bytes = read(&path, 2048)?;
            let receipt: Receipt = serde_json::from_slice(&bytes)?;
            check_receipt(&receipt, frames, schedule, request.particle_count)?;
            chain.update(&bytes);
            receipt_hashes.append(Sha256::digest(&bytes).as_slice())?;
            frames += 1;
        }
        let identity = if let Some(c) = &completion {
            if c.schema_version != 1
                || !c.complete
                || c.request_sha256 != request_hash
                || c.frames != frames
                || frames != schedule.len()
                || c.accepted_steps != request.steps
                || c.receipts_sha256 != hex(chain.finalize().as_slice())
                || !valid_digest(&c.diagnostics_sha256)
                || c.cache_id != result_id(&request_hash, &c.receipts_sha256, &c.diagnostics_sha256)
            {
                return Err(invalid(
                    "cache completion disagrees with request or receipt chain",
                ));
            }
            let (hash, warnings) = diagnostics_cancellable(
                &root.join("diagnostics.jsonl"),
                decimal(&request.steps)?,
                request.timestep_bits,
                &mut cancelled,
            )?;
            if hash != c.diagnostics_sha256 || warnings != c.warnings {
                return Err(invalid(
                    "cache diagnostic checksum or warning count mismatch",
                ));
            }
            c.cache_id.clone()
        } else {
            format!("incomplete:{request_hash}:{frames}")
        };
        Ok(Self {
            root: root.to_owned(),
            request,
            input,
            scene,
            schedule,
            frames,
            complete: completion.is_some(),
            request_hash,
            identity,
            receipt_hashes,
        })
    }
    /// Number of committed observations available, not the number of solver steps.
    pub fn frame_count(&self) -> u64 {
        self.frames
    }
    /// Whether the complete requested simulation was published successfully.
    pub fn is_complete(&self) -> bool {
        self.complete
    }
    /// Content identity, or an explicitly incomplete prefix identifier.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Frozen physical and display inputs. No mutable scene files are consulted.
    pub fn scene(&self) -> &ResolvedScene {
        &self.scene
    }
    /// Original resolved input record, retained for provenance and explicit reruns.
    pub fn recorded_input(&self) -> &RecordedInput {
        &self.input
    }
    /// Accepted step for an available frame index.
    pub fn step(&self, index: u64) -> Option<u64> {
        if index < self.frames {
            self.schedule.step(index)
        } else {
            None
        }
    }
    /// Summary suitable for inspection without loading particle frames.
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"schema_version":1,"kind":"maelstrom-particle-cache","complete":self.complete,"frames":self.frames,"expected_frames":self.schedule.len(),"steps":self.request.steps,"stride":self.request.stride,"particle_count":self.request.particle_count,"cache_id":self.identity,"request_sha256":self.request_hash,"execution":self.request.execution,"backend":self.request.backend,"metadata":self.request.metadata,"build":self.request.build})
    }
    /// Opens and validates one independently addressable frame. No fluid steps execute.
    pub fn frame(&self, index: u64) -> Result<ValidatedFrame, CacheError> {
        let reading_started = std::time::Instant::now();
        if index >= self.frames {
            return Err(invalid("frame index outside committed cache prefix"));
        }
        directory(&self.root)?;
        directory(&self.root.join("frames"))?;
        directory(&self.root.join("receipts"))?;
        let receipt_bytes = read(&self.root.join("receipts").join(receipt_name(index)), 2048)?;
        let actual: [u8; 32] = Sha256::digest(&receipt_bytes).into();
        let mut expected = [0; 32];
        self.receipt_hashes.read_at(
            index
                .checked_mul(32)
                .ok_or_else(|| invalid("receipt index offset overflow"))?,
            &mut expected,
        )?;
        if actual != expected {
            return Err(invalid("cache receipt changed after opening"));
        }
        let receipt: Receipt = serde_json::from_slice(&receipt_bytes)?;
        check_receipt(&receipt, index, self.schedule, self.request.particle_count)?;
        let bytes = read(
            &self.root.join("frames").join(frame_name(index)),
            format::size(self.request.particle_count)? as u64,
        )?;
        let reading_s = reading_started.elapsed().as_secs_f64();
        let decoding_started = std::time::Instant::now();
        let (snapshot, canonical, binary_validation_s) =
            format::decode(&bytes, &receipt, &self.request_hash, &self.scene.particles)?;
        Ok(ValidatedFrame {
            snapshot,
            canonical,
            read_timings: ReadTimings {
                reading_s,
                binary_validation_s,
                decoding_s: (decoding_started.elapsed().as_secs_f64() - binary_validation_s)
                    .max(0.0),
            },
        })
    }
    /// Checks all available particle files and independently recomputes canonical hashes.
    pub fn validate_full(&self, mut cancelled: impl FnMut() -> bool) -> Result<(), CacheError> {
        for i in 0..self.frames {
            if cancelled() {
                return Err(CacheError::Cancelled);
            }
            let frame = self.frame(i)?;
            if frame.snapshot.canonical_sha256().map_err(invalid)? != frame.canonical {
                return Err(invalid(
                    "cached canonical snapshot digest disagrees with particle bits",
                ));
            }
        }
        Ok(())
    }
}
fn check_receipt(
    r: &Receipt,
    index: u64,
    schedule: Schedule,
    count: usize,
) -> Result<(), CacheError> {
    if r.schema_version != 1
        || r.index != index
        || Some(decimal(&r.step)?) != schedule.step(index)
        || r.bytes != format::size(count)? as u64
        || !valid_digest(&r.file_sha256)
        || !valid_digest(&r.snapshot_sha256)
    {
        return Err(invalid("invalid cache frame receipt"));
    }
    Ok(())
}
fn diagnostics_cancellable(
    path: &Path,
    steps: u64,
    timestep_bits: u32,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<(String, u64), CacheError> {
    let file = regular(path, steps.saturating_mul(4096))?;
    let mut reader = BufReader::new(file);
    let mut hash = Sha256::new();
    let mut warnings = 0;
    let mut line = Vec::new();
    for step in 1..=steps {
        if cancelled() {
            return Err(CacheError::Cancelled);
        }
        line.clear();
        reader.by_ref().take(4097).read_until(b'\n', &mut line)?;
        if line.len() > 4096 || line.last() != Some(&b'\n') {
            return Err(invalid("short or oversized cache diagnostic"));
        }
        let d: serde_json::Value = serde_json::from_slice(&line)?;
        let speed = d["maximum_speed_m_s"]
            .as_f64()
            .filter(|s| s.is_finite() && *s >= 0.0)
            .ok_or_else(|| invalid("invalid cache diagnostic speed"))?;
        let _ = speed;
        let warning = d["warning"]
            .as_bool()
            .ok_or_else(|| invalid("invalid cache diagnostic warning"))?;
        let expected = match d["cfl_limit"]["kind"].as_str() {
            Some("unbounded") => false,
            Some("finite") => {
                let limit = d["cfl_limit"]["seconds"]
                    .as_f64()
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .ok_or_else(|| invalid("invalid cache CFL limit"))?;
                f32::from_bits(timestep_bits) > (limit as f32)
            }
            _ => return Err(invalid("invalid cache CFL kind")),
        };
        if d["completed_steps"].as_str().map(decimal).transpose()? != Some(step)
            || warning != expected
        {
            return Err(invalid("cache diagnostic order or CFL warning mismatch"));
        }
        warnings += u64::from(warning);
        hash.update(&line);
    }
    if !reader.fill_buf()?.is_empty() {
        return Err(invalid("extra cache diagnostics"));
    }
    Ok((hex(hash.finalize().as_slice()), warnings))
}
