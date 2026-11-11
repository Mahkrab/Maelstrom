//! Presentation time follows recorded simulated timestamps, never solver work.
//! Seeking pauses, and late redraws select an existing image without interpolation.

use crate::render::RenderError;

pub(super) struct Clock {
    times: Times,
    origin: f64,
    pub position: f64,
    pub speed: f64,
    pub playing: bool,
    repeat: bool,
}
enum Times {
    #[cfg(test)]
    Memory(Vec<f64>),
    #[cfg(feature = "viewer")]
    Index(std::sync::Arc<crate::export::frame_index::FrameIndex>),
}
impl Times {
    fn len(&self) -> usize {
        match self {
            #[cfg(test)]
            Self::Memory(times) => times.len(),
            #[cfg(feature = "viewer")]
            Self::Index(index) => index.len(),
        }
    }
    fn get(&self, index: usize) -> Result<f64, RenderError> {
        match self {
            #[cfg(test)]
            Self::Memory(times) => times
                .get(index)
                .copied()
                .ok_or_else(|| RenderError("timestamp index outside sequence".into())),
            #[cfg(feature = "viewer")]
            Self::Index(rows) => rows
                .get(index)
                .map(|frame| frame.simulated_seconds)
                .map_err(|e| RenderError(e.to_string())),
        }
    }
}
impl Clock {
    #[cfg(test)]
    pub fn new(
        times: Vec<f64>,
        speed: f64,
        paused: bool,
        repeat: bool,
    ) -> Result<Self, RenderError> {
        Self::using(Times::Memory(times), speed, paused, repeat)
    }
    #[cfg(feature = "viewer")]
    pub fn indexed(
        rows: std::sync::Arc<crate::export::frame_index::FrameIndex>,
        speed: f64,
        paused: bool,
        repeat: bool,
    ) -> Result<Self, RenderError> {
        Self::using(Times::Index(rows), speed, paused, repeat)
    }
    fn using(times: Times, speed: f64, paused: bool, repeat: bool) -> Result<Self, RenderError> {
        if !speed.is_finite() || !(0.01..=100.0).contains(&speed) {
            return Err(RenderError(
                "playback speed must be finite and within 0.01..=100".into(),
            ));
        }
        let origin = times.get(0)?;
        let playing = !paused && times.len() > 1;
        Ok(Self {
            times,
            origin,
            position: 0.0,
            speed,
            playing,
            repeat,
        })
    }
    fn time(&self, index: usize) -> Result<f64, RenderError> {
        Ok(self.times.get(index)? - self.origin)
    }
    pub fn frame(&self) -> Result<usize, RenderError> {
        let (mut first, mut end) = (0, self.times.len());
        while first < end {
            let middle = first + (end - first) / 2;
            if self.time(middle)? <= self.position {
                first = middle + 1;
            } else {
                end = middle;
            }
        }
        Ok(first.saturating_sub(1))
    }
    pub fn seek(&mut self, index: usize) -> Result<(), RenderError> {
        self.position = self.time(index.min(self.times.len() - 1))?;
        self.playing = false;
        Ok(())
    }
    pub fn toggle(&mut self) -> Result<(), RenderError> {
        if self.times.len() == 1 {
            return Ok(());
        }
        if !self.playing && self.frame()? == self.times.len() - 1 {
            self.position = 0.0;
        }
        self.playing = !self.playing;
        Ok(())
    }
    pub fn restart(&mut self) {
        self.position = 0.0;
        self.playing = self.times.len() > 1;
    }
    pub fn advance(&mut self, seconds: f64) -> Result<(), RenderError> {
        if !self.playing {
            return Ok(());
        }
        self.position += seconds * self.speed;
        let last = self.time(self.times.len() - 1)?;
        if self.repeat {
            let end = last + last - self.time(self.times.len() - 2)?;
            self.position %= end;
        } else if self.position >= last {
            self.position = last;
            self.playing = false;
        }
        Ok(())
    }
}
