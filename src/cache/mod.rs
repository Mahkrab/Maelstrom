//! Particle caches retain scheduled accepted observations independently of images.
//! Frames and receipts are immutable; only a completed manifest certifies the
//! requested sequence. Reading a cache never constructs a simulation or a device.

mod bake;
mod format;
mod reader;
mod writer;

pub use bake::{BakeRequest, bake};
pub use format::Schedule;
pub use reader::{Cache, ReadTimings, ValidatedFrame};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fmt, fs,
    io::{self, Read},
    path::Path,
};

/// An invalid cache, a failed operation, cancellation, or a lost writer.
#[derive(Debug)]
pub enum CacheError {
    /// Unsupported, inconsistent or corrupt input.
    Invalid(String),
    /// Filesystem operation failed; any committed prefix is retained.
    Io(io::Error),
    /// Cooperative cancellation before completion publication.
    Cancelled,
    /// The bounded writer stopped without returning its result.
    WriterStopped,
}
impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => f.write_str(s),
            Self::Io(e) => write!(f, "cache I/O: {e}"),
            Self::Cancelled => f.write_str("cache bake cancelled"),
            Self::WriterStopped => f.write_str("cache writer stopped before completing its frame"),
        }
    }
}
impl std::error::Error for CacheError {}
impl From<io::Error> for CacheError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for CacheError {
    fn from(e: serde_json::Error) -> Self {
        invalid(format!("cache JSON: {e}"))
    }
}
pub(crate) fn invalid(s: impl fmt::Display) -> CacheError {
    CacheError::Invalid(s.to_string())
}
pub(crate) fn digest(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes).as_slice())
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    format!(
        "sha256:{}",
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    )
}
fn valid_digest(s: &str) -> bool {
    s.strip_prefix("sha256:").is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// Build identity is recorded even in headless cache producers.
pub(crate) fn build_identity() -> serde_json::Value {
    serde_json::json!({"commit":option_env!("MAELSTROM_EXPORT_COMMIT"),"dirty":option_env!("MAELSTROM_EXPORT_DIRTY"),"source_sha256":option_env!("MAELSTROM_EXPORT_SOURCE"),"lock_sha256":option_env!("MAELSTROM_EXPORT_LOCK"),"rustc":option_env!("MAELSTROM_EXPORT_RUSTC"),"target":option_env!("MAELSTROM_EXPORT_TARGET"),"profile":option_env!("MAELSTROM_EXPORT_PROFILE"),"opt_level":option_env!("MAELSTROM_EXPORT_OPT_LEVEL"),"rustflags_sha256":option_env!("MAELSTROM_EXPORT_CARGO_ENCODED_RUSTFLAGS"),"debug_assertions":cfg!(debug_assertions),"features":{"cuda":cfg!(feature="cuda"),"render":cfg!(feature="render"),"viewer":cfg!(feature="viewer")}})
}

// Keep wire structs private: callers cannot manufacture a validated frame or digest.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    kind: String,
    steps: String,
    stride: String,
    particle_count: usize,
    timestep_bits: u32,
    input_sha256: String,
    identities_sha256: String,
    execution: crate::execution::ExecutionMode,
    backend: String,
    metadata: serde_json::Value,
    build: serde_json::Value,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u32,
    index: u64,
    step: String,
    bytes: u64,
    file_sha256: String,
    snapshot_sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema_version: u32,
    complete: bool,
    request_sha256: String,
    frames: u64,
    accepted_steps: String,
    warnings: u64,
    receipts_sha256: String,
    diagnostics_sha256: String,
    cache_id: String,
    timings: serde_json::Value,
}

fn decimal(s: &str) -> Result<u64, CacheError> {
    let n: u64 = s.parse().map_err(invalid)?;
    if n.to_string() != s {
        return Err(invalid("non-canonical cache step"));
    }
    Ok(n)
}

// Refuse symlinks and verify the opened object too. Linux O_NOFOLLOW closes the
// final-component race; all intermediate cache directories are checked at open/use.
fn regular(path: &Path, limit: u64) -> Result<fs::File, CacheError> {
    let m = fs::symlink_metadata(path)?;
    if !m.is_file() || m.len() > limit {
        return Err(invalid("expected a bounded regular cache file"));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || opened.len() > limit {
        return Err(invalid("opened cache file exceeds its limit"));
    }
    Ok(file)
}
fn directory(path: &Path) -> Result<(), CacheError> {
    if !fs::symlink_metadata(path)?.is_dir() {
        return Err(invalid("expected a cache directory, not a symlink"));
    }
    Ok(())
}
fn read(path: &Path, limit: u64) -> Result<Vec<u8>, CacheError> {
    let file = regular(path, limit)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(file.metadata()?.len()).map_err(invalid)?)
        .map_err(invalid)?;
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(invalid("cache file grew beyond its limit"));
    }
    Ok(bytes)
}
fn sync_directory(path: &Path) -> Result<(), CacheError> {
    #[cfg(unix)]
    fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}
fn publish(path: &Path, bytes: &[u8]) -> Result<(), CacheError> {
    crate::workflow::publish_bytes(path, bytes)?;
    Ok(())
}
fn receipt_name(index: u64) -> String {
    format!("frame-{index:08}.json")
}
fn frame_name(index: u64) -> String {
    format!("frame-{index:08}.mpc")
}
fn result_id(request: &str, receipts: &str, diagnostics: &str) -> String {
    digest(format!("maelstrom-cache-v1|{request}|{receipts}|{diagnostics}").as_bytes())
}

#[cfg(test)]
mod tests;
