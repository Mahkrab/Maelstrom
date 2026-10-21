//! Presentation time follows recorded simulated timestamps, never solver work.
//! Seeking pauses, and late redraws select an existing image without interpolation.

use crate::render::RenderError;

pub(super) struct Clock {
    times: Vec<f64>,
    pub position: f64,
    pub speed: f64,
    pub playing: bool,
    repeat: bool,
}
impl Clock {
    pub fn new(
        times: Vec<f64>,
        speed: f64,
        paused: bool,
        repeat: bool,
    ) -> Result<Self, RenderError> {
        if !speed.is_finite() || !(0.01..=100.0).contains(&speed) {
            return Err(RenderError(
                "playback speed must be finite and within 0.01..=100".into(),
            ));
        }
        let playing = !paused && times.len() > 1;
        Ok(Self {
            times,
            position: 0.0,
            speed,
            playing,
            repeat,
        })
    }
    pub fn frame(&self) -> usize {
        self.times
            .partition_point(|t| *t <= self.position)
            .saturating_sub(1)
    }
    pub fn seek(&mut self, index: usize) {
        self.position = self.times[index.min(self.times.len() - 1)];
        self.playing = false;
    }
    pub fn toggle(&mut self) {
        if self.times.len() == 1 {
            return;
        }
        if !self.playing && self.frame() == self.times.len() - 1 {
            self.position = 0.0;
        }
        self.playing = !self.playing;
    }
    pub fn restart(&mut self) {
        self.position = 0.0;
        self.playing = self.times.len() > 1;
    }
    pub fn advance(&mut self, seconds: f64) {
        if !self.playing {
            return;
        }
        self.position += seconds * self.speed;
        let last = self.times[self.times.len() - 1];
        if self.repeat {
            let end = last + last - self.times[self.times.len() - 2];
            self.position %= end;
        } else if self.position >= last {
            self.position = last;
            self.playing = false;
        }
    }
}
