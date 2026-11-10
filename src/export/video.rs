//! Raw RGBA frames cross a bounded pipe into FFmpeg. Only a successfully closed
//! MP4 and its provenance are published; teardown kills and reaps failed children.
use super::*;
use std::{
    io::{Read, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc,
    thread,
    time::Duration,
};

/// Constant-frame-rate policy over an explicit ordered selection.
#[derive(Clone, Copy, Serialize)]
pub struct VideoTiming {
    /// Video frames per second, independent of recorded timestamps.
    pub fps: f64,
    /// libx264 constant rate factor, 0..=51.
    pub crf: u8,
    /// Omit the last selected observation only when explicitly requested.
    pub omit_terminal: bool,
}
impl VideoTiming {
    /// Validates the selection and endpoint policy before starting an encoder.
    pub fn count(self, selection: CacheSelection, available: u64) -> Result<u64, ExportError> {
        if !self.fps.is_finite() || self.fps <= 0.0 || self.fps > 1000.0 || self.crf > 51 {
            return Err(fail("video requires finite FPS in (0,1000] and CRF 0..=51"));
        }
        let count = selection
            .count(available)?
            .saturating_sub(u64::from(self.omit_terminal));
        if count == 0 {
            return Err(fail("terminal omission would leave no video frames"));
        }
        Ok(count)
    }
}

struct Encoder {
    child: Child,
    input: Option<mpsc::SyncSender<Vec<u8>>>,
    writer: Option<thread::JoinHandle<Result<(), std::io::Error>>>,
    errors: Option<thread::JoinHandle<String>>,
    stage: PathBuf,
    bytes: usize,
}
impl Encoder {
    fn new(
        executable: &Path,
        output: &Path,
        input: (u32, u32),
        size: (u32, u32),
        timing: VideoTiming,
    ) -> Result<(Self, String), ExportError> {
        for (w, h) in [input, size] {
            if !(1..=8192).contains(&w) || !(1..=8192).contains(&h) {
                return Err(fail("video dimensions must be 1..=8192"));
            }
        }
        if !size.0.is_multiple_of(2) || !size.1.is_multiple_of(2) {
            return Err(fail("H.264 yuv420p requires even output width and height"));
        }
        if fs::symlink_metadata(output).is_ok() || fs::symlink_metadata(sidecar(output)).is_ok() {
            return Err(fail("video or provenance destination already exists"));
        }
        let version = Command::new(executable)
            .arg("-version")
            .output()
            .map_err(|e| {
                fail(format!(
                    "cannot execute FFmpeg: {e}; install FFmpeg with libx264 or pass --ffmpeg PATH"
                ))
            })?;
        if !version.status.success() {
            return Err(fail("FFmpeg -version failed; check --ffmpeg PATH"));
        }
        let version = String::from_utf8_lossy(&version.stdout)
            .lines()
            .next()
            .unwrap_or("FFmpeg (unknown version)")
            .to_owned();
        let encoders = Command::new(executable)
            .args(["-hide_banner", "-encoders"])
            .output()
            .map_err(fail)?;
        if !encoders.status.success()
            || !String::from_utf8_lossy(&encoders.stdout)
                .split_whitespace()
                .any(|s| s == "libx264")
        {
            return Err(fail(
                "FFmpeg has no libx264 H.264 encoder; install a build with libx264",
            ));
        }
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut attempt = 0u64;
        let stage = loop {
            let path = parent.join(format!(".maelstrom-video-{}-{attempt}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => attempt += 1,
                Err(e) => return Err(fail(e)),
            }
        };
        let started = Command::new(executable)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
            ])
            .arg(format!("{}x{}", input.0, input.1))
            .arg("-framerate")
            .arg(timing.fps.to_string())
            .args(["-i", "pipe:0", "-an", "-vf"])
            .arg(format!("scale={}:{}", size.0, size.1))
            .args(["-c:v", "libx264", "-crf"])
            .arg(timing.crf.to_string())
            .args([
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
            ])
            .arg(stage.join("video.mp4"))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match started {
            Ok(c) => c,
            Err(e) => {
                let _ = fs::remove_dir_all(&stage);
                return Err(fail(e));
            }
        };
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| fail("FFmpeg stdin unavailable"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| fail("FFmpeg stderr unavailable"))?;
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let writer = thread::spawn(move || {
            let mut stdin = stdin;
            while let Ok(bytes) = rx.recv() {
                stdin.write_all(&bytes)?;
            }
            Ok(())
        });
        let errors = thread::spawn(move || {
            let mut stderr = stderr;
            let mut retained = Vec::new();
            let mut buf = [0u8; 4096];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
                retained.extend_from_slice(&buf[..n]);
                if retained.len() > 65536 {
                    retained.drain(..retained.len() - 65536);
                }
            }
            String::from_utf8_lossy(&retained).into_owned()
        });
        Ok((
            Self {
                child,
                input: Some(tx),
                writer: Some(writer),
                errors: Some(errors),
                stage,
                bytes: input.0 as usize * input.1 as usize * 4,
            },
            version,
        ))
    }
    fn send(
        &mut self,
        mut bytes: Vec<u8>,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<(), ExportError> {
        if bytes.len() != self.bytes {
            return Err(fail("video RGBA frame size mismatch"));
        }
        loop {
            if cancelled() {
                return Err(fail("video export cancelled"));
            }
            match self
                .input
                .as_ref()
                .ok_or_else(|| fail("encoder closed"))?
                .try_send(bytes)
            {
                Ok(()) => return Ok(()),
                Err(mpsc::TrySendError::Full(b)) => {
                    bytes = b;
                    if let Some(status) = self.child.try_wait().map_err(fail)? {
                        return self
                            .finish(cancelled)
                            .and(Err(fail(format!("FFmpeg exited during input: {status}"))));
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(fail(
                        "FFmpeg closed its input; check encoder availability and output storage",
                    ));
                }
            }
        }
    }
    fn finish(&mut self, cancelled: &mut impl FnMut() -> bool) -> Result<(), ExportError> {
        self.input.take();
        let status = loop {
            if cancelled() {
                return Err(fail("video export cancelled"));
            }
            if let Some(s) = self.child.try_wait().map_err(fail)? {
                break s;
            }
            thread::sleep(Duration::from_millis(10));
        };
        let written = self
            .writer
            .take()
            .ok_or_else(|| fail("encoder writer unavailable"))?
            .join()
            .map_err(|_| fail("encoder writer panicked"))?;
        let errors = self
            .errors
            .take()
            .ok_or_else(|| fail("encoder diagnostics unavailable"))?
            .join()
            .map_err(|_| fail("encoder diagnostic reader panicked"))?;
        if !status.success() {
            return Err(fail(format!("FFmpeg failed ({status}): {errors}")));
        }
        written.map_err(fail)?;
        if fs::metadata(self.stage.join("video.mp4"))
            .map_err(fail)?
            .len()
            == 0
        {
            return Err(fail("FFmpeg produced an empty MP4"));
        }
        Ok(())
    }
    fn publish(&self, output: &Path, provenance: &Value) -> Result<(), ExportError> {
        fs::File::open(self.stage.join("video.mp4"))
            .and_then(|f| f.sync_all())
            .map_err(fail)?;
        let bytes = serde_json::to_vec_pretty(provenance).map_err(fail)?;
        publish_bytes(&self.stage.join("provenance.json"), &bytes).map_err(fail)?;
        fs::hard_link(self.stage.join("provenance.json"), sidecar(output)).map_err(fail)?;
        if let Err(e) = fs::hard_link(self.stage.join("video.mp4"), output) {
            let _ = fs::remove_file(sidecar(output));
            return Err(fail(e));
        }
        Ok(())
    }
}
impl Drop for Encoder {
    fn drop(&mut self) {
        self.input.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        if let Some(e) = self.errors.take() {
            let _ = e.join();
        }
        let _ = fs::remove_dir_all(&self.stage);
    }
}
fn sidecar(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".json");
    PathBuf::from(name)
}

/// Streams one validated frame at a time. The callback must return top-left RGBA8;
/// its provenance should identify the presentation and original observation source.
#[allow(clippy::too_many_arguments)]
pub fn encode_frames(
    output: &Path,
    executable: &Path,
    input_size: (u32, u32),
    output_size: (u32, u32),
    timing: VideoTiming,
    selection: CacheSelection,
    timestamps: &[f64],
    source: Value,
    mut frame: impl FnMut(u64) -> Result<Vec<u8>, ExportError>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Value, ExportError> {
    let count = timing.count(selection, timestamps.len() as u64)?;
    if timestamps.iter().any(|t| !t.is_finite() || *t < 0.0)
        || timestamps.windows(2).any(|t| t[1] <= t[0])
    {
        return Err(fail(
            "video source timestamps must be finite, non-negative and strictly increasing",
        ));
    }

    if cancelled() {
        return Err(fail("video export cancelled"));
    }
    let (mut encoder, version) = Encoder::new(executable, output, input_size, output_size, timing)?;
    let started = Instant::now();
    let mut selected = Vec::with_capacity(count as usize);
    for n in 0..count {
        if cancelled() {
            return Err(fail("video export cancelled"));
        }
        let index = selection.first + n * selection.every;
        encoder.send(frame(index)?, &mut cancelled)?;
        selected.push(json!({"source_index":index,"simulated_seconds":timestamps[index as usize],"video_seconds":n as f64/timing.fps}));
    }
    encoder.finish(&mut cancelled)?;
    let provenance = json!({"schema_version":1,"kind":"maelstrom-video","complete":true,"source":source,
        "encoder":{"version":version,"executable":executable,"codec":"libx264","container":"mp4","pixel_format":"yuv420p","crf":timing.crf},
        "timing":timing,"selection":selection,"frames":selected,"frame_count":count,"duration_seconds":count as f64/timing.fps,
        "input_dimensions":input_size,"output_dimensions":output_size,"elapsed_seconds":started.elapsed().as_secs_f64()});
    if cancelled() {
        return Err(fail("video export cancelled before publication"));
    }
    encoder.publish(output, &provenance)?;
    Ok(provenance)
}

/// Draws exact observations using the same indexed graphics and readback as PNG export.
pub fn cached_pixels(
    cache: &crate::cache::Cache,
    settings: &Settings,
    graphics: &Graphics,
    boundaries: &BoundaryLines,
    index: u64,
) -> Result<Vec<u8>, ExportError> {
    let frame = cache.frame(index).map_err(fail)?;
    let particles: Vec<_> = frame
        .snapshot()
        .particles()
        .iter()
        .map(|p| p.particle())
        .collect();
    let scene = cache.scene();
    super::offscreen::pixels(
        graphics,
        &crate::render::DrawScene {
            particles: &particles,
            boundaries,
            radius_m: f64::from(scene.configuration.numerical.particle_spacing_m) / 2.0,
            camera: &settings.camera,
            colours: &settings.particles,
            particle_mass_kg: scene.configuration.derived.particle_mass_kg,
            width: settings.width,
            height: settings.height,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_policy_is_explicit() {
        let s = CacheSelection {
            first: 0,
            last: 300,
            every: 1,
        };
        let mut t = VideoTiming {
            fps: 60.0,
            crf: 18,
            omit_terminal: false,
        };
        assert_eq!(t.count(s, 301).unwrap(), 301);
        t.omit_terminal = true;
        assert_eq!(t.count(s, 301).unwrap(), 300);
        assert!(
            t.count(
                CacheSelection {
                    first: 0,
                    last: 0,
                    every: 1
                },
                1
            )
            .is_err()
        );
        assert!(t.count(CacheSelection { every: 0, ..s }, 301).is_err());
        t.fps = f64::NAN;
        assert!(t.count(s, 301).is_err());
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn root() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "maelstrom-video-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    #[ignore = "requires FFmpeg with libx264 and ffprobe"]
    fn real_encoder_order_duration_cleanup_and_no_overwrite() {
        let root = root();
        let output = root.join("movie.mp4");
        let timing = VideoTiming {
            fps: 60.0,
            crf: 18,
            omit_terminal: false,
        };
        let selection = CacheSelection {
            first: 0,
            last: 2,
            every: 1,
        };
        let mut seen = vec![];
        let result = encode_frames(
            &output,
            Path::new("ffmpeg"),
            (16, 16),
            (32, 24),
            timing,
            selection,
            &[0.0, 0.1, 0.2],
            json!({}),
            |i| {
                seen.push(i);
                Ok([i as u8 * 100, 0, 0, 255].repeat(256))
            },
            || false,
        )
        .unwrap();
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(result["duration_seconds"], 0.05);
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,nb_read_frames",
                "-of",
                "json",
            ])
            .arg(&output)
            .output()
            .unwrap();
        assert!(probe.status.success());
        let v: Value = serde_json::from_slice(&probe.stdout).unwrap();
        assert_eq!(v["streams"][0]["codec_name"], "h264");
        assert_eq!(v["streams"][0]["nb_read_frames"], "3");
        assert_eq!(v["streams"][0]["height"], 24);
        assert!(
            encode_frames(
                &output,
                Path::new("ffmpeg"),
                (16, 16),
                (16, 16),
                timing,
                selection,
                &[0.0, 0.1, 0.2],
                json!({}),
                |_| panic!("existing destination must reject before rendering"),
                || false
            )
            .is_err()
        );
        let failed = root.join("failed.mp4");
        assert!(
            encode_frames(
                &failed,
                Path::new("ffmpeg"),
                (16, 16),
                (16, 16),
                timing,
                selection,
                &[0.0, 0.1, 0.2],
                json!({}),
                |_| Err(fail("injected read failure")),
                || false
            )
            .is_err()
        );
        assert!(!failed.exists());
        assert!(!sidecar(&failed).exists());
        let cancelled = root.join("cancelled.mp4");
        let mut checks = 0;
        assert!(
            encode_frames(
                &cancelled,
                Path::new("ffmpeg"),
                (16, 16),
                (16, 16),
                timing,
                selection,
                &[0.0, 0.1, 0.2],
                json!({}),
                |_| Ok([0, 0, 0, 255].repeat(256)),
                || {
                    checks += 1;
                    checks > 3
                }
            )
            .is_err()
        );
        assert!(!cancelled.exists());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn missing_encoder_and_invalid_dimensions_do_not_create_output() {
        let root = root();
        let output = root.join("movie.mp4");
        let t = VideoTiming {
            fps: 60.0,
            crf: 18,
            omit_terminal: false,
        };
        let s = CacheSelection {
            first: 0,
            last: 0,
            every: 1,
        };
        let e = encode_frames(
            &output,
            &root.join("missing-ffmpeg"),
            (16, 16),
            (16, 16),
            t,
            s,
            &[0.0],
            json!({}),
            |_| panic!(),
            || false,
        )
        .unwrap_err();
        assert!(e.to_string().contains("install FFmpeg"));
        assert!(
            encode_frames(
                &output,
                Path::new("ffmpeg"),
                (16, 16),
                (17, 16),
                t,
                s,
                &[0.0],
                json!({}),
                |_| panic!(),
                || false
            )
            .is_err()
        );
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir(root).unwrap();
    }
}

#[cfg(all(test, unix))]
mod subprocess_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    #[test]
    fn encoder_exit_and_missing_codec_are_actionable_and_leave_no_output() {
        let root =
            std::env::temp_dir().join(format!("maelstrom-encoder-failure-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let executable = root.join("encoder");
        fs::write(&executable,"#!/bin/sh\ncase \"$1\" in\n-version) echo 'test encoder'; exit 0;;\n-hide_banner) if [ \"$2\" = '-encoders' ]; then echo ' V libx264'; exit 0; fi;;\nesac\necho 'injected encoder failure' >&2\nexit 23\n").unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        let output = root.join("failed.mp4");
        let t = VideoTiming {
            fps: 60.0,
            crf: 18,
            omit_terminal: false,
        };
        let s = CacheSelection {
            first: 0,
            last: 0,
            every: 1,
        };
        let e = encode_frames(
            &output,
            &executable,
            (16, 16),
            (16, 16),
            t,
            s,
            &[0.0],
            json!({}),
            |_| Ok([0, 0, 0, 255].repeat(256)),
            || false,
        )
        .unwrap_err();
        assert!(e.to_string().contains("FFmpeg"));
        assert!(!output.exists());
        assert!(!sidecar(&output).exists());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::write(&executable, "#!/bin/sh\necho 'test encoder without H264'\n").unwrap();
        let e = encode_frames(
            &output,
            &executable,
            (16, 16),
            (16, 16),
            t,
            s,
            &[0.0],
            json!({}),
            |_| panic!(),
            || false,
        )
        .unwrap_err();
        assert!(e.to_string().contains("no libx264"));
        fs::remove_dir_all(root).unwrap();
    }
}
