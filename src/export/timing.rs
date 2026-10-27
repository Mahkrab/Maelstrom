//! Export intervals account for synchronous work before manifest construction.
//! Final manifest publication, caller output and teardown belong to command wall
//! time; the manifest cannot contain a measurement of its own completed write.

use super::{ExportError, fail};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Clone, Copy, Default)]
pub(super) struct FrameTimings {
    pub particle_copy: Duration,
    pub graphics_readback: Duration,
    pub png_encoding: Duration,
    pub publication: Duration,
    pub hashing: Duration,
}

#[derive(Default)]
pub(super) struct ExportTimings {
    pub setup: Duration,
    pub solver_steps: Duration,
    pub snapshot_capture: Duration,
    pub comparison: Duration,
    pub progress: Duration,
    pub frames: FrameTimings,
}

impl ExportTimings {
    pub fn profile(&self, elapsed: Duration) -> Result<Value, ExportError> {
        let components = [
            ("setup", self.setup),
            ("solver_steps", self.solver_steps),
            ("snapshot_capture", self.snapshot_capture),
            ("comparison", self.comparison),
            ("frame_particle_copy", self.frames.particle_copy),
            ("graphics_readback", self.frames.graphics_readback),
            ("png_encoding", self.frames.png_encoding),
            ("frame_publication", self.frames.publication),
            ("frame_hashing", self.frames.hashing),
            ("progress", self.progress),
        ];
        let accounted: Duration = components.iter().map(|(_, time)| *time).sum();
        let other = elapsed
            .checked_sub(accounted)
            .ok_or_else(|| fail("export timing intervals exceed their enclosing duration"))?;
        let mut seconds: serde_json::Map<_, _> = components
            .into_iter()
            .map(|(name, duration)| (name.to_owned(), json!(duration.as_secs_f64())))
            .collect();
        seconds.insert("other".into(), json!(other.as_secs_f64()));
        Ok(json!({
            "schema_version": 1,
            "boundary": "export_entry_to_before_manifest",
            "total_s": elapsed.as_secs_f64(),
            "components_s": seconds,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_accounts_for_frame_work_and_rejects_overlapping_intervals() {
        let timings = ExportTimings {
            setup: Duration::from_secs(2),
            solver_steps: Duration::from_secs(3),
            frames: FrameTimings {
                publication: Duration::from_secs(5),
                hashing: Duration::from_secs(7),
                ..Default::default()
            },
            progress: Duration::from_secs(11),
            ..Default::default()
        };
        let profile = timings.profile(Duration::from_secs(31)).unwrap();
        assert_eq!(profile["schema_version"], 1);
        assert_eq!(profile["boundary"], "export_entry_to_before_manifest");
        assert_eq!(profile["total_s"], 31.0);
        assert_eq!(profile["components_s"]["frame_publication"], 5.0);
        assert_eq!(profile["components_s"]["frame_hashing"], 7.0);
        assert_eq!(profile["components_s"]["progress"], 11.0);
        assert_eq!(profile["components_s"]["other"], 3.0);
        assert!(timings.profile(Duration::from_secs(27)).is_err());
    }
}
