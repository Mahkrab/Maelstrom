//! Saved image playback does not pass through numerical input resolution.

use super::{CliError, arguments::Play};
use maelstrom::{playback, render::GraphicsSelection};

pub(super) fn launch(play: &Play) -> Result<u8, CliError> {
    let sequence =
        playback::Sequence::open(&play.sequence).map_err(|e| CliError::input(e.to_string()))?;
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
