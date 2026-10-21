//! Saved observations are validated before a destination is created. Initial
//! inputs describe a request; only stored snapshots establish accepted history.

use super::{ExportError, FrameSchedule, fail};
use crate::{
    execution::ExecutionMode,
    scene::{Particle, ResolvedScene},
    workflow::{ExecutionOptions, SnapshotRecord, replay_options},
};
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) struct Recorded {
    pub scene: ResolvedScene,
    pub steps: u64,
    pub completed: u64,
    pub options: ExecutionOptions,
    pub snapshots: BTreeMap<u64, SnapshotRecord>,
    pub provenance: Value,
    pub complete: bool,
}
impl Recorded {
    pub fn parse(json: &str) -> Result<Self, ExportError> {
        let (scene, request, options) = replay_options(json).map_err(fail)?;
        let value: Value = serde_json::from_str(json).map_err(fail)?;
        let completed = decimal(&value["completed_steps"])?;
        if completed > request.steps {
            return Err(fail("completed steps exceed request"));
        }
        let status = value["status"]
            .as_str()
            .ok_or_else(|| fail("missing result status"))?;
        let complete = match status {
            "completed" | "completed_with_warnings" => true,
            "input_validation_failure"
            | "simulation_failure"
            | "cancelled"
            | "audit_failure"
            | "output_failure" => false,
            _ => return Err(fail("unknown result status")),
        };
        if complete && completed != request.steps {
            return Err(fail("complete result has incomplete steps"));
        }
        let failures = value["failures"]
            .as_array()
            .ok_or_else(|| fail("missing result failures"))?;
        if complete && !failures.is_empty() {
            return Err(fail("complete result contains failures"));
        }
        for f in failures {
            if decimal(&f["completed_steps"])? > completed
                || !f["message"].is_string()
                || !matches!(
                    f["status"].as_str(),
                    Some(
                        "input_validation_failure"
                            | "simulation_failure"
                            | "cancelled"
                            | "audit_failure"
                            | "output_failure"
                    )
                )
            {
                return Err(fail("invalid recorded failure"));
            }
        }
        if !complete && failures.is_empty() {
            return Err(fail("incomplete result has no failure"));
        }
        let diagnostics = value["steps"]
            .as_array()
            .ok_or_else(|| fail("missing accepted diagnostics"))?;
        if u64::try_from(diagnostics.len()).ok() != Some(completed) {
            return Err(fail("accepted diagnostics do not cover completed steps"));
        }
        for (i, d) in diagnostics.iter().enumerate() {
            if decimal(&d["completed_steps"])? != i as u64 + 1
                || !d["warning"].is_boolean()
                || !d["maximum_speed_m_s"]
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && v >= 0.0)
            {
                return Err(fail("invalid accepted diagnostic"));
            }
            let limit = &d["cfl_limit"];
            if !(limit["kind"] == "unbounded"
                || (limit["kind"] == "finite"
                    && limit["seconds"]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && v > 0.0)))
            {
                return Err(fail("invalid CFL limit"));
            }
        }
        let mut snapshots = BTreeMap::new();
        let checkpoints = value["checkpoints"]
            .as_array()
            .ok_or_else(|| fail("missing checkpoints"))?;
        let mut previous = None;
        for checkpoint in checkpoints {
            let snapshot =
                SnapshotRecord::from_json(&checkpoint["snapshot"].to_string()).map_err(fail)?;
            let step = snapshot.completed_steps();
            if previous.is_some_and(|p| p >= step)
                || request.checkpoints.binary_search(&step).is_err()
            {
                return Err(fail(
                    "checkpoint indices must be requested, ascending and unique",
                ));
            }
            previous = Some(step);
            insert(&mut snapshots, snapshot, completed, &scene)?;
        }
        let final_snapshot =
            SnapshotRecord::from_json(&value["final_snapshot"].to_string()).map_err(fail)?;
        if final_snapshot.completed_steps() != completed {
            return Err(fail("final snapshot disagrees with completed count"));
        }
        insert(&mut snapshots, final_snapshot, completed, &scene)?;
        if complete
            && request
                .checkpoints
                .iter()
                .any(|s| !snapshots.contains_key(s))
        {
            return Err(fail("complete result lacks requested checkpoints"));
        }
        Ok(Self {
            scene,
            steps: request.steps,
            completed,
            options,
            snapshots,
            complete,
            provenance: serde_json::json!({"request": value["request"], "completed_steps": value["completed_steps"], "status": status, "failures": failures, "metadata": value["metadata"], "steps": diagnostics, "source_result_sha256": super::hash(json.as_bytes())}),
        })
    }
    pub fn require_schedule(&self, schedule: &FrameSchedule) -> Result<(), ExportError> {
        let missing: Vec<_> = schedule
            .steps
            .iter()
            .filter(|s| !self.snapshots.contains_key(s))
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(fail(format!(
                "saved observations missing scheduled indices: {missing:?}"
            )));
        }
        Ok(())
    }
}
fn decimal(value: &Value) -> Result<u64, ExportError> {
    let s = value
        .as_str()
        .ok_or_else(|| fail("expected canonical decimal step string"))?;
    let n: u64 = s.parse().map_err(fail)?;
    if n.to_string() != s {
        return Err(fail("non-canonical step string"));
    }
    Ok(n)
}
fn insert(
    map: &mut BTreeMap<u64, SnapshotRecord>,
    snapshot: SnapshotRecord,
    completed: u64,
    scene: &ResolvedScene,
) -> Result<(), ExportError> {
    if snapshot.completed_steps() > completed
        || snapshot.particles().len() != scene.particles.len()
        || snapshot
            .particles()
            .iter()
            .zip(&scene.particles)
            .any(|(a, b)| a.particle().identity != b.identity)
    {
        return Err(fail("snapshot step or identities disagree with inputs"));
    }
    if let Some(old) = map.get(&snapshot.completed_steps()) {
        compare(old, &snapshot, ExecutionMode::Serial)?;
    } else {
        map.insert(snapshot.completed_steps(), snapshot);
    }
    Ok(())
}

pub(super) fn compare(
    expected: &SnapshotRecord,
    actual: &SnapshotRecord,
    mode: ExecutionMode,
) -> Result<(), ExportError> {
    if expected.completed_steps() != actual.completed_steps()
        || expected.particles().len() != actual.particles().len()
    {
        return Err(fail("snapshot structure mismatch"));
    }
    for (a, b) in expected.particles().iter().zip(actual.particles()) {
        let (a, b) = (a.particle(), b.particle());
        if a.identity != b.identity {
            return Err(fail("snapshot identity mismatch"));
        }
        for (i, (x, y)) in components(&a).into_iter().zip(components(&b)).enumerate() {
            let equal = match mode {
                ExecutionMode::Serial => x.to_bits() == y.to_bits(),
                _ => (f64::from(x) - f64::from(y)).abs() <= if i < 3 { 1e-5 } else { 0.005 },
            };
            if !equal {
                return Err(fail(format!(
                    "recorded checkpoint mismatch at step {}, particle {}, component {i}: expected {x:?}, actual {y:?}",
                    actual.completed_steps(),
                    a.identity
                )));
            }
        }
    }
    Ok(())
}
fn components(p: &Particle) -> [f32; 6] {
    [
        p.position_m.x,
        p.position_m.y,
        p.position_m.z,
        p.velocity_m_s.x,
        p.velocity_m_s.y,
        p.velocity_m_s.z,
    ]
}
