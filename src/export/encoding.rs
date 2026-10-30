//! One owned frame at a time crosses to a CPU worker. Graphics and exclusive
//! publication stay on the export thread; dropping the worker always joins it.

use super::{ExportError, Frame, fail, offscreen::EncodePng, timing::FrameTimings};
use crate::workflow::SnapshotRecord;
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
    time::Instant,
};

pub(super) struct Job {
    pub snapshot: SnapshotRecord,
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub index: usize,
    pub dt: f64,
    pub encode: EncodePng,
}

pub(super) struct Encoded {
    pub frame: Frame,
    pub png: Vec<u8>,
    pub timings: FrameTimings,
}

fn encode(job: Job) -> Result<Encoded, ExportError> {
    let mut timings = FrameTimings::default();
    let start = Instant::now();
    let png = (job.encode)(job.width, job.height, &job.rgba)?;
    timings.png_encoding = start.elapsed();
    let start = Instant::now();
    let frame = Frame {
        index: job.index,
        step: job.snapshot.completed_steps().to_string(),
        simulated_seconds: job.snapshot.completed_steps() as f64 * job.dt,
        file: format!("frame-{:08}.png", job.index),
        png_sha256: super::hash(&png),
        rgba_sha256: super::hash(&job.rgba),
        snapshot_sha256: super::format_digest(job.snapshot.canonical_sha256().map_err(fail)?),
    };
    timings.hashing = start.elapsed();
    Ok(Encoded {
        frame,
        png,
        timings,
    })
}

pub(super) struct Worker {
    requests: Option<SyncSender<Job>>,
    results: Receiver<Result<Encoded, ExportError>>,
    thread: Option<JoinHandle<()>>,
    pending: bool,
}

impl Worker {
    pub fn new() -> Result<Self, ExportError> {
        let (requests, jobs) = mpsc::sync_channel(1);
        let (completed, results) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("frame-encoding".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    if completed.send(encode(job)).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| fail(format!("start frame encoder: {e}")))?;
        Ok(Self {
            requests: Some(requests),
            results,
            thread: Some(thread),
            pending: false,
        })
    }

    pub fn submit(&mut self, job: Job) -> Result<(), ExportError> {
        if self.pending {
            return Err(fail("previous frame has not been collected"));
        }
        self.requests
            .as_ref()
            .ok_or_else(|| fail("frame encoder is closed"))?
            .send(job)
            .map_err(|_| fail("frame encoder stopped before accepting a frame"))?;
        self.pending = true;
        Ok(())
    }

    pub fn collect(&mut self) -> Result<Option<Encoded>, ExportError> {
        if !self.pending {
            return Ok(None);
        }
        self.pending = false;
        self.results
            .recv()
            .map_err(|_| fail("frame encoder stopped before completing a frame"))?
            .map(Some)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // There is at most one outstanding result, so the capacity-one result
        // channel cannot block shutdown, even on cancellation or publication error.
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job() -> Job {
        Job {
            snapshot: SnapshotRecord::from_json(
                r#"{"schema_version":1,"completed_steps":"7","particles":[]}"#,
            )
            .unwrap(),
            rgba: vec![16, 24, 32, 255],
            width: 1,
            height: 1,
            index: 2,
            dt: 0.125,
            encode: super::super::offscreen::encode,
        }
    }

    #[test]
    fn worker_preserves_bytes_and_hashes_with_one_outstanding_frame() {
        let request = job();
        let expected_png = (request.encode)(1, 1, &request.rgba).unwrap();
        let expected_snapshot_hash =
            super::super::hash(request.snapshot.to_json().unwrap().as_bytes());
        let mut worker = Worker::new().unwrap();
        assert!(worker.collect().unwrap().is_none());
        worker.submit(request).unwrap();
        assert!(worker.submit(job()).is_err());
        let result = worker.collect().unwrap().unwrap();
        assert_eq!(result.png, expected_png);
        assert_eq!(result.frame.png_sha256, super::super::hash(&expected_png));
        assert_eq!(result.frame.snapshot_sha256, expected_snapshot_hash);
        assert_eq!(result.frame.step, "7");
        assert_eq!(result.frame.simulated_seconds, 0.875);
        assert_eq!(result.frame.index, 2);
        assert_eq!(result.frame.file, "frame-00000002.png");
        assert!(worker.collect().unwrap().is_none());
        worker.submit(job()).unwrap();
        // Cancellation can drop a live job without collecting its result.
        drop(worker);
    }

    #[test]
    fn worker_reports_encoding_failure_and_unexpected_unwind_without_deadlock() {
        let mut worker = Worker::new().unwrap();
        let mut invalid = job();
        invalid.rgba.clear();
        worker.submit(invalid).unwrap();
        assert!(worker.collect().is_err());
        worker.submit(job()).unwrap();
        assert!(worker.collect().unwrap().is_some());
        fn unwind(_: u32, _: u32, _: &[u8]) -> Result<Vec<u8>, ExportError> {
            panic!("injected encoder unwind")
        }
        let mut invalid = job();
        invalid.encode = unwind;
        worker.submit(invalid).unwrap();
        assert!(
            worker
                .collect()
                .err()
                .unwrap()
                .to_string()
                .contains("stopped")
        );
        assert!(worker.submit(job()).is_err());
    }
}
