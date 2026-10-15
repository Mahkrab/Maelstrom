//! Presentation reads the same immutable observations as machine output.
//! Publication stages a complete file beside its destination and links it into
//! place only after writing and syncing succeed; existing files are never replaced.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{CompletionStatus, RecordedCflLimit, RunResult};

/// Presentation selected after all numerical and observation work has stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Versioned, replayable JSON result document.
    Json,
    /// Human-readable completion, comparison and diagnostic summary.
    Text,
}

/// Encoding or file publication failed. No destination was published by this
/// attempt, and the in-memory result records an output failure.
#[derive(Debug)]
pub enum OutputError {
    /// The result could not be encoded as JSON.
    Encoding(serde_json::Error),
    /// Creating, writing, syncing or publishing the staged file failed.
    Io(io::Error),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoding(error) => write!(f, "result encoding failed: {error}"),
            Self::Io(error) => write!(f, "result publication failed: {error}"),
        }
    }
}

impl std::error::Error for OutputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encoding(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}

impl RunResult {
    /// Publishes the last accepted state as a standalone JSON snapshot using the
    /// same non-replacing publication as results. A failure is retained in this
    /// result so subsequent result output reports the whole request accurately.
    ///
    /// # Errors
    /// Returns [`OutputError`] if no accepted state exists, encoding fails or the
    /// destination cannot be published. Existing files are never replaced.
    pub fn publish_snapshot(&mut self, destination: &Path) -> Result<(), OutputError> {
        let publication = (|| {
            let snapshot = self.final_snapshot().ok_or_else(|| {
                OutputError::Io(io::Error::other("no accepted snapshot is available"))
            })?;
            let encoded = snapshot
                .to_json()
                .map_err(|error| OutputError::Io(io::Error::other(error)))?;
            publish_bytes(destination, encoded.as_bytes()).map_err(OutputError::Io)
        })();
        if let Err(error) = &publication {
            self.fail(
                CompletionStatus::OutputFailure,
                error.to_string(),
                self.completed_steps(),
            );
        }
        publication
    }

    /// Summarises completion, comparison identities, all-step CFL warnings and
    /// each declared checkpoint. It uses the same observations as [`Self::to_json`].
    #[must_use]
    pub fn summary(&self) -> String {
        let mut text = format!(
            "Maelstrom result schema {}: {:?}\nSteps: {}/{} accepted; CFL warning steps: {}\n",
            self.schema_version,
            self.status,
            self.completed_steps(),
            self.request.steps,
            self.warning_count()
        );
        text.push_str(&format!(
            "Application: {}; backend: {}\n",
            self.application_version, self.backend
        ));
        if let Some(input) = &self.input {
            let (scene, configuration) = input.display_names();
            text.push_str(&format!(
                "Scene: {scene:?}; configuration: {configuration:?}\n"
            ));
        }
        if let Some(metadata) = &self.metadata {
            text.push_str(&format!("Run: {}\nInput: {}\nConfiguration: {}\nTarget: {}/{}; debug assertions: {}; build revision: {:?}\n",
                metadata.run_id, metadata.input_id, metadata.configuration_id,
                metadata.target_os, metadata.target_arch, metadata.debug_assertions,
                metadata.build_revision.as_deref().unwrap_or("unavailable")));
        } else {
            text.push_str("Validated input identity and accepted state: unavailable\n");
        }
        if self.steps.is_empty() {
            text.push_str(
                "Post-acceleration maximum speed and CFL limit: unavailable (no accepted steps)\n",
            );
        } else {
            let maximum = self
                .steps
                .iter()
                .map(|s| s.maximum_speed_m_s)
                .fold(0.0_f32, f32::max);
            let minimum_limit = self
                .steps
                .iter()
                .filter_map(|s| match s.cfl_limit {
                    RecordedCflLimit::Finite(seconds) => Some(seconds),
                    RecordedCflLimit::Unbounded => None,
                })
                .reduce(f32::min);
            text.push_str(&format!(
                "Post-acceleration maximum speed: {maximum} m/s; tightest CFL limit: {}\n",
                minimum_limit.map_or_else(|| "unbounded".to_owned(), |s| format!("{s} s"))
            ));
        }
        text.push_str(&format!(
            "Checkpoints observed: {}/{}\n",
            self.checkpoints.len(),
            self.request.checkpoints.len()
        ));
        for checkpoint in &self.checkpoints {
            text.push_str(&format!(
                "Checkpoint {}: ",
                checkpoint.snapshot().completed_steps()
            ));
            if let Some(density) = checkpoint.density() {
                text.push_str(&format!(
                    "{} samples; mean relative density error {}; maximum {}\n",
                    density.sample_count,
                    density
                        .mean_error
                        .map_or_else(|| "unavailable".to_owned(), |n| n.to_string()),
                    density
                        .maximum_error
                        .map_or_else(|| "unavailable".to_owned(), |n| n.to_string())
                ));
            } else {
                text.push_str("density audit failed\n");
            }
        }
        for failure in &self.failures {
            // Debug string formatting escapes controls in externally sourced text.
            text.push_str(&format!(
                "{:?} after step {}: {:?}\n",
                failure.status, failure.completed_steps, failure.message
            ));
        }
        text
    }

    /// Publishes one complete result without overwriting an existing destination.
    /// Encoding, writing and syncing happen before atomic hard-link publication.
    /// On any failure, records [`CompletionStatus::OutputFailure`] while retaining
    /// accepted state, diagnostics and earlier failures. A later retry records that
    /// history rather than turning the original request into a clean completion.
    ///
    /// The destination filesystem must support hard links. This guarantees atomic
    /// visibility, not crash durability of the directory entry. No partial stream
    /// is exposed as a result; temporary-file cleanup is best effort on failure.
    pub fn publish(&mut self, destination: &Path, format: OutputFormat) -> Result<(), OutputError> {
        let encoded = match format {
            OutputFormat::Json => self.to_json().map_err(OutputError::Encoding),
            OutputFormat::Text => Ok(self.summary()),
        };
        let publication = encoded
            .and_then(|text| publish_bytes(destination, text.as_bytes()).map_err(OutputError::Io));
        if let Err(error) = &publication {
            self.fail(
                CompletionStatus::OutputFailure,
                error.to_string(),
                self.completed_steps(),
            );
        }
        publication
    }
}

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

struct StagedFile {
    path: PathBuf,
    file: File,
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn publish_bytes(destination: &Path, bytes: &[u8]) -> io::Result<()> {
    publish_with(destination, |file| file.write_all(bytes))
}

fn publish_with(
    destination: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let directory = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut staged = None;
    for _ in 0..1024 {
        let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            ".maelstrom-output-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                staged = Some(StagedFile { path, file });
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let mut staged = staged.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "no unused temporary output name",
        )
    })?;
    write(&mut staged.file)?;
    staged.file.sync_all()?;
    fs::hard_link(&staged.path, destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partial_write_failure_never_publishes_a_destination() {
        let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "maelstrom-partial-write-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let destination = directory.join("result.json");
        let error = publish_with(&destination, |file| {
            file.write_all(b"{\"status\":\"completed\"")?;
            Err(io::Error::other("injected write failure"))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(!destination.exists());
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
        fs::remove_dir(&directory).unwrap();
    }
}
