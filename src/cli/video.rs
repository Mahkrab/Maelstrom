//! Video commands consume validated observations or images, never numerical requests.
use super::{
    CliError,
    arguments::{Cli, Video},
};
#[cfg(not(feature = "render"))]
pub(super) fn launch(_: &Cli, _: &Video) -> Result<u8, CliError> {
    Err(CliError::input(
        "video requires --features render or viewer".into(),
    ))
}
#[cfg(feature = "render")]
pub(super) fn launch(cli: &Cli, args: &Video) -> Result<u8, CliError> {
    use super::arguments::VideoSource;
    use maelstrom::{
        export::{
            CacheSelection,
            video::{VideoTiming, cached_pixels, encode_observations},
        },
        presentation::{CameraMode, Presentation},
        render::{BoundaryLines, Graphics, GraphicsSelection},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let output = args
        .output
        .as_ref()
        .ok_or_else(|| CliError::usage("video requires --output FILE".into()))?;
    let flag = Arc::new(AtomicBool::new(false));
    let signal = flag.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|e| CliError::input(e.to_string()))?;
    let cancel = || flag.load(Ordering::Relaxed);
    let timing = VideoTiming {
        fps: args.fps,
        crf: args.crf,
        omit_terminal: args.omit_terminal,
    };
    let range = |count: u64| -> Result<CacheSelection, CliError> {
        Ok(CacheSelection {
            first: args.first_frame.unwrap_or(0),
            last: args.last_frame.unwrap_or(
                count
                    .checked_sub(1)
                    .ok_or_else(|| CliError::input("source has no frames".into()))?,
            ),
            every: args.every,
        })
    };
    let result = match &args.source {
        VideoSource::Images { sequence } => {
            let opened = maelstrom::playback::Sequence::open(sequence)
                .map_err(|e| CliError::input(e.to_string()))?;
            let dims = opened.dimensions();
            let manifest = if sequence.is_dir() {
                sequence.join("manifest.json")
            } else {
                sequence.clone()
            };
            #[derive(serde::Deserialize)]
            struct PresentationOnly {
                render: maelstrom::presentation::Settings,
            }
            let source_file =
                std::fs::File::open(&manifest).map_err(|e| CliError::input(e.to_string()))?;
            let saved: PresentationOnly =
                serde_json::from_reader(std::io::BufReader::new(source_file))
                    .map_err(|e| CliError::input(e.to_string()))?;
            saved
                .render
                .validate()
                .map_err(|e| CliError::input(e.to_string()))?;
            use sha2::{Digest, Sha256};
            use std::io::Read;
            let mut file =
                std::fs::File::open(&manifest).map_err(|e| CliError::input(e.to_string()))?;
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let n = file
                    .read(&mut buffer)
                    .map_err(|e| CliError::input(e.to_string()))?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            let source = serde_json::json!({"mode":"validated_png_sequence", "manifest_sha256":hash.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>(),"render":saved.render});
            encode_observations(
                output,
                &args.ffmpeg,
                dims,
                (args.width.unwrap_or(dims.0), args.height.unwrap_or(dims.1)),
                timing,
                range(opened.frame_count() as u64)?,
                opened.frame_count() as u64,
                |i| {
                    opened
                        .timestamp(i as usize)
                        .map_err(|e| maelstrom::export::ExportError(e.to_string()))
                },
                source,
                |i| {
                    opened
                        .decode(i as usize)
                        .map_err(|e| maelstrom::export::ExportError(e.to_string()))
                },
                cancel,
            )
        }
        VideoSource::Cache {
            cache,
            render_config,
            graphics_adapter,
            software,
        } => {
            let opened = maelstrom::cache::Cache::open(cache, false)
                .map_err(|e| CliError::input(e.to_string()))?;
            let text = std::fs::read_to_string(render_config)
                .map_err(|e| CliError::input(e.to_string()))?;
            let presentation =
                Presentation::parse(&text).map_err(|e| CliError::input(e.to_string()))?;
            if presentation.camera_mode != CameraMode::Fixed {
                return Err(CliError::usage(
                    "video cache requires an explicit saved fixed camera".into(),
                ));
            }
            let mut settings = presentation.settings;
            if let Some(w) = args.width {
                settings.width = w;
            }
            if let Some(h) = args.height {
                settings.height = h;
            }
            settings
                .validate()
                .map_err(|e| CliError::input(e.to_string()))?;
            let selection = range(opened.frame_count())?;
            timing
                .count(selection, opened.frame_count())
                .map_err(|e| CliError::usage(e.to_string()))?;
            let instance = Graphics::instance();
            let graphics = pollster::block_on(Graphics::new(
                &instance,
                GraphicsSelection {
                    index: *graphics_adapter,
                    software: *software,
                },
                None,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ))
            .map_err(|e| CliError::input(e.to_string()))?;
            let scene = opened.scene();
            let boundaries = BoundaryLines::new(
                &scene.particles,
                &scene.planes,
                f64::from(scene.configuration.numerical.particle_spacing_m),
            )
            .map_err(|e| CliError::input(e.to_string()))?;
            let dt = f64::from(scene.configuration.numerical.fixed_timestep_s);
            let source = serde_json::json!({"cache":opened.info(),"render":settings,"renderer":"opaque-spheres-lines-v2 indexed","solver_steps":0,"timestep_bits":scene.configuration.numerical.fixed_timestep_s.to_bits(),"application_version":env!("CARGO_PKG_VERSION")});
            encode_observations(
                output,
                &args.ffmpeg,
                (settings.width, settings.height),
                (settings.width, settings.height),
                timing,
                selection,
                opened.frame_count(),
                |i| {
                    opened.step(i).map(|step| step as f64 * dt).ok_or_else(|| {
                        maelstrom::export::ExportError(
                            "cache timestamp index outside sequence".into(),
                        )
                    })
                },
                source,
                |i| cached_pixels(&opened, &settings, &graphics, &boundaries, i),
                cancel,
            )
        }
    };
    let value = result.map_err(|e| CliError {
        code: if cancel() { 130 } else { 5 },
        message: e.to_string(),
    })?;
    if matches!(cli.format, super::arguments::Format::Json) {
        use std::io::Write;
        // Preserve the complete JSON response without rebuilding its frame array.
        let mut name = output.as_os_str().to_os_string();
        name.push(".json");
        let mut file = std::fs::File::open(std::path::PathBuf::from(name))
            .map_err(|e| CliError::output(e.to_string()))?;
        let mut stdout = std::io::stdout().lock();
        std::io::copy(&mut file, &mut stdout)
            .and_then(|_| stdout.write_all(b"\n"))
            .and_then(|_| stdout.flush())
            .map_err(|e| CliError::output(format!("stdout: {e}")))?;
    } else {
        super::print_document(
            cli.format,
            format!(
                "Encoded {} frames to {}; no simulation performed.",
                value["frame_count"],
                output.display()
            ),
            value,
        )?;
    }
    Ok(0)
}
