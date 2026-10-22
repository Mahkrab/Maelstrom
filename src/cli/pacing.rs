//! Wall-clock pacing follows acceptance; it never controls numerical step size.
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
pub(super) struct Pacer {
    start: Instant,
    timestep: f64,
    enabled: bool,
    lag: bool,
}
impl Pacer {
    pub fn new(timestep: f32, enabled: bool) -> Self {
        Self {
            start: Instant::now(),
            timestep: f64::from(timestep),
            enabled,
            lag: false,
        }
    }
    pub fn wait(&mut self, step: u64, cancelled: &AtomicBool) {
        if !self.enabled {
            return;
        }
        let Ok(offset) = Duration::try_from_secs_f64(step as f64 * self.timestep) else {
            self.lag = true;
            return;
        };
        let Some(target) = self.start.checked_add(offset) else {
            self.lag = true;
            return;
        };
        let now = Instant::now();
        if now > target && now.duration_since(target).as_secs_f64() > self.timestep {
            self.lag = true;
        }
        while !cancelled.load(Ordering::Relaxed) {
            let remaining = target.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(10)));
        }
    }
    pub fn lagged(&self) -> bool {
        self.lag
    }
}
