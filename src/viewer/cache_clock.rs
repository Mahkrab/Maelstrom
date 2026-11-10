//! Recorded timestamps select exact observations. Wall time only decides which
//! observation to request; it never creates or interpolates particle data.
use crate::render::RenderError;

/// How cached playback responds when presentation cannot keep up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePlayback {
    /// Follow monotonic elapsed time, skipping intermediate presentations.
    #[default]
    RealTime,
    /// Present each observation, allowing playback to run slower than requested.
    FrameAccurate,
}
impl std::str::FromStr for CachePlayback {
    type Err = RenderError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "real-time" => Ok(Self::RealTime),
            "frame-accurate" => Ok(Self::FrameAccurate),
            _ => Err(RenderError(
                "cache playback must be real-time or frame-accurate".into(),
            )),
        }
    }
}

/// Advances an already started clock. The terminal state gets one final recorded
/// interval before looping or pausing; a one-state cache uses a 1/60 s hold.
pub(super) fn advance(
    position: &mut f64,
    elapsed: f64,
    speed: f64,
    looping: bool,
    terminal: f64,
    hold: f64,
) -> bool {
    *position += elapsed * speed;
    let cycle = terminal + hold;
    if *position >= cycle {
        if looping {
            *position %= cycle;
        } else {
            *position = terminal;
            return true;
        }
    }
    false
}

/// Finds the most recent observation at or before the requested simulation time,
/// including an irregular terminal gap, without allocating a timestamp array.
pub(super) fn target(count: u64, position: f64, time: impl Fn(u64) -> f64) -> u64 {
    let (mut first, mut end) = (0, count);
    while first < end {
        let middle = first + (end - first) / 2;
        if time(middle) <= position {
            first = middle + 1;
        } else {
            end = middle;
        }
    }
    first.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slow_redraws_and_disk_stalls_skip_presentations_without_stopping_time() {
        let times = [0., 0.02, 0.04, 0.06, 0.065];
        let mut position = 0.;
        assert!(!advance(&mut position, 0.015, 4., false, 0.065, 0.005));
        assert_eq!(target(5, position, |i| times[i as usize]), 3);
        // No frame availability is an input to the real-time clock.
        assert!(advance(&mut position, 0.1, 1., false, 0.065, 0.005));
        assert_eq!(target(5, position, |i| times[i as usize]), 4);
    }
    #[test]
    fn terminal_hold_loop_and_long_elapsed_use_recorded_final_gap() {
        let mut position = 0.065;
        assert!(!advance(&mut position, 0.004, 1., true, 0.065, 0.005));
        assert_eq!(
            target(5, position, |i| [0., 0.02, 0.04, 0.06, 0.065][i as usize]),
            4
        );
        assert!(!advance(&mut position, 0.142, 1., true, 0.065, 0.005));
        assert!((position - 0.001).abs() < 1e-12);
        assert_eq!(target(1, 123., |_| 0.), 0);
    }
    #[test]
    fn exact_boundaries_and_speed_change_preserve_fractional_time() {
        let mut position = 0.;
        advance(&mut position, 0.01, 1., false, 1., 0.02);
        advance(&mut position, 0.01, 2., false, 1., 0.02);
        assert!((position - 0.03).abs() < 1e-12);
        assert_eq!(target(4, 0.02, |i| i as f64 * 0.02), 1);
        assert_eq!(target(4, 0.019, |i| i as f64 * 0.02), 0);
    }
}
