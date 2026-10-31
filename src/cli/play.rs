//! Saved image playback does not pass through numerical input resolution.

use super::{CliError, arguments::Play};
use maelstrom::{playback, render::GraphicsSelection};

pub(super) fn launch(play: &Play, storage: &super::storage::Storage) -> Result<u8, CliError> {
    let path = if play.last {
        let source =
            super::storage::read(&storage.runs.join("last.json"), 1024 * 1024).map_err(|e| {
                CliError::input(format!("no last compiled run available: {}", e.message))
            })?;
        serde_json::from_str::<std::path::PathBuf>(&source)
            .map_err(|e| CliError::input(format!("last compiled run: {e}")))?
    } else {
        storage.run(
            play.sequence
                .as_deref()
                .ok_or_else(|| CliError::usage("play requires RUN or --last".into()))?,
        )?
    };
    let sequence = playback::Sequence::open(&path).map_err(|e| CliError::input(e.to_string()))?;
    playback::run(
        sequence,
        play.speed,
        play.paused,
        play.repeat,
        GraphicsSelection {
            index: play.graphics_adapter,
            software: play.software,
        },
    )
    .map_err(|e| CliError::input(e.to_string()))?;
    Ok(0)
}
