//! Opt-in frame observations stay in memory during viewing. Writing the trace
//! after worker shutdown keeps filesystem latency outside the measured frames.

use crate::{
    application::Observation,
    render::{DrawTimings, RenderError},
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

pub(super) struct Profile {
    output: File,
    start: Instant,
    rows: Vec<String>,
}
impl Profile {
    pub(super) fn from_environment() -> Result<Option<Self>, RenderError> {
        std::env::var_os("MAELSTROM_VIEW_PROFILE")
            .map(|path| {
                let output = File::create_new(path)
                    .map_err(|e| RenderError(format!("viewer profile output: {e}")))?;
                Ok(Self {
                    output,
                    start: Instant::now(),
                    rows: Vec::new(),
                })
            })
            .transpose()
    }
    pub(super) fn record(
        &mut self,
        observation: Option<&Observation>,
        draw: DrawTimings,
        acquire_s: f64,
        present_s: f64,
        frame_s: f64,
    ) {
        // Bound opt-in memory even when a viewer is left open overnight.
        if self.rows.len() >= 1_000_000 {
            return;
        }
        let (generation, step, simulated, solver, snapshot, status) =
            observation.map_or((0, 0, 0.0, 0.0, 0.0, "idle".to_owned()), |o| {
                (
                    o.generation,
                    o.snapshot.as_ref().map_or(0, |s| s.completed_steps()),
                    o.simulated_time_s().unwrap_or(0.0),
                    o.timings.solver_s,
                    o.timings.snapshot_s,
                    format!("{:?}", o.status),
                )
            });
        self.rows.push(format!("{:.9},{generation},{step},{simulated:.9},{solver:.9},{snapshot:.9},{status},{acquire_s:.9},{:.9},{:.9},{:.9},{present_s:.9},{frame_s:.9}",
            self.start.elapsed().as_secs_f64(), draw.prepare_s, draw.upload_s, draw.submit_s));
    }
    pub(super) fn finish(self) -> Result<(), RenderError> {
        let mut out = BufWriter::new(self.output);
        let write = || -> std::io::Result<()> {
            writeln!(
                out,
                "wall_s,generation,step,simulated_s,solver_s,snapshot_s,status,acquire_s,prepare_s,upload_s,submit_s,present_s,redraw_s"
            )?;
            for row in self.rows {
                writeln!(out, "{row}")?;
            }
            out.flush()
        };
        write().map_err(|e| RenderError(format!("viewer profile write: {e}")))
    }
}
