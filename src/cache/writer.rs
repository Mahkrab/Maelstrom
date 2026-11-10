//! A single owned particle frame crosses to the writer. Hashing, packing and
//! synced file preparation overlap later solver work; receipts commit on collection.
use super::*;
use crate::workflow::SnapshotRecord;
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
    time::Instant,
};

#[derive(Default, Serialize)]
pub(super) struct Timings {
    pub canonical_s: f64,
    pub packing_s: f64,
    pub binary_hash_s: f64,
    pub writing_s: f64,
    pub receipt_publication_s: f64,
    pub writer_wait_s: f64,
}
struct Job {
    snapshot: SnapshotRecord,
    index: u64,
}
struct Prepared {
    receipt: Receipt,
    timings: Timings,
}
pub(super) struct Writer {
    sender: Option<SyncSender<Job>>,
    receiver: Receiver<Result<Prepared, CacheError>>,
    thread: Option<JoinHandle<()>>,
    pending: bool,
    root: PathBuf,
    pub count: u64,
    pub timings: Timings,
    chain: Sha256,
}
impl Writer {
    pub fn new(root: &Path, request: String) -> Result<Self, CacheError> {
        let (tx, jobs) = mpsc::sync_channel::<Job>(1);
        let (done, receiver) = mpsc::sync_channel(1);
        let destination = root.to_owned();
        let thread = thread::Builder::new()
            .name("particle-cache-writer".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    let result = (|| {
                        let mut t = Timings::default();
                        let start = Instant::now();
                        let canonical = job.snapshot.canonical_sha256().map_err(invalid)?;
                        t.canonical_s = start.elapsed().as_secs_f64();
                        let start = Instant::now();
                        let bytes = format::encode(&job.snapshot, job.index, &request, canonical)?;
                        t.packing_s = start.elapsed().as_secs_f64();
                        let start = Instant::now();
                        let file_sha256 = digest(&bytes);
                        t.binary_hash_s = start.elapsed().as_secs_f64();
                        let start = Instant::now();
                        publish(
                            &destination.join("frames").join(frame_name(job.index)),
                            &bytes,
                        )?;
                        sync_directory(&destination.join("frames"))?;
                        t.writing_s = start.elapsed().as_secs_f64();
                        Ok(Prepared {
                            receipt: Receipt {
                                schema_version: 1,
                                index: job.index,
                                step: job.snapshot.completed_steps().to_string(),
                                bytes: bytes.len() as u64,
                                file_sha256,
                                snapshot_sha256: hex(&canonical),
                            },
                            timings: t,
                        })
                    })();
                    if done.send(result).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            sender: Some(tx),
            receiver,
            thread: Some(thread),
            pending: false,
            root: root.to_owned(),
            count: 0,
            timings: Timings::default(),
            chain: Sha256::new(),
        })
    }
    pub fn submit(&mut self, snapshot: SnapshotRecord, index: u64) -> Result<(), CacheError> {
        if self.pending || index != self.count {
            return Err(invalid("previous cache frame has not been collected"));
        }
        self.sender
            .as_ref()
            .ok_or(CacheError::WriterStopped)?
            .send(Job { snapshot, index })
            .map_err(|_| CacheError::WriterStopped)?;
        self.pending = true;
        Ok(())
    }
    pub fn collect(&mut self, cancelled: &mut impl FnMut() -> bool) -> Result<(), CacheError> {
        if !self.pending {
            return Ok(());
        }
        let start = Instant::now();
        // Poll only the CPU result, never abandon a resource-owning thread.
        let prepared = loop {
            if cancelled() {
                return Err(CacheError::Cancelled);
            }
            match self
                .receiver
                .recv_timeout(std::time::Duration::from_millis(10))
            {
                Ok(r) => break r?,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => return Err(CacheError::WriterStopped),
            }
        };
        self.pending = false;
        self.timings.writer_wait_s += start.elapsed().as_secs_f64();
        if cancelled() {
            return Err(CacheError::Cancelled);
        }
        let start = Instant::now();
        let bytes = serde_json::to_vec(&prepared.receipt)?;
        publish(
            &self.root.join("receipts").join(receipt_name(self.count)),
            &bytes,
        )?;
        sync_directory(&self.root.join("receipts"))?;
        self.timings.receipt_publication_s += start.elapsed().as_secs_f64();
        self.chain.update(&bytes);
        self.count += 1;
        self.timings.canonical_s += prepared.timings.canonical_s;
        self.timings.packing_s += prepared.timings.packing_s;
        self.timings.binary_hash_s += prepared.timings.binary_hash_s;
        self.timings.writing_s += prepared.timings.writing_s;
        Ok(())
    }
    pub fn chain(&self) -> String {
        hex(self.chain.clone().finalize().as_slice())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        // At most one outstanding result fits the result channel. Closing the
        // input and joining cannot deadlock, including cancellation or an unwind.
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
