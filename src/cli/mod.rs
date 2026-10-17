//! The CLI resolves declared inputs before execution and presents immutable
//! observations afterwards. Stdout carries requested data; stderr carries
//! diagnostics. Presentation flags never enter the numerical request.

mod arguments;
mod inputs;

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use clap::Parser;
use maelstrom::workflow::{
    CompletionStatus, ExecutionOptions, MeasurementMode, OutputFormat, RunRequest, replay_options,
    run_with_options,
};
use serde_json::{Value, json};

use arguments::{Cli, Command, Format, Output};
use inputs::{ResolvedInputs, SCENES};

struct CliError {
    code: u8,
    message: String,
}

impl CliError {
    fn usage(message: String) -> Self {
        Self { code: 2, message }
    }
    fn input(message: String) -> Self {
        Self { code: 3, message }
    }
    fn output(message: String) -> Self {
        Self { code: 5, message }
    }
}

pub(super) fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = if error.use_stderr() { 2 } else { 0 };
            return match error.print() {
                Ok(()) => ExitCode::from(code),
                Err(_) => ExitCode::from(5),
            };
        }
    };
    match execute(&cli) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            let _ = writeln!(io::stderr().lock(), "maelstrom: {}", error.message);
            ExitCode::from(error.code)
        }
    }
}

fn execute(cli: &Cli) -> Result<u8, CliError> {
    match &cli.command {
        Command::Scenes => {
            let mut scenes = Vec::new();
            let mut text =
                String::from("Bundled scenes (configuration: defaults-v4; no randomness):\n");
            for (name, source) in SCENES {
                let document = maelstrom::scene::parse_scene(source)
                    .map_err(|error| CliError::input(error.to_string()))?;
                text.push_str(&format!("  {name}: {}\n", document.description));
                scenes.push(json!({"name": name, "description": document.description}));
            }
            print_document(
                cli.format,
                text,
                json!({"schema_version": 1, "application_version": env!("CARGO_PKG_VERSION"), "scenes": scenes}),
            )?;
            Ok(0)
        }
        Command::Validate(inputs) | Command::Inspect(inputs) => {
            let resolved = inputs::resolve(inputs)?;
            diagnostic(cli, &resolved)?;
            let inspect = matches!(cli.command, Command::Inspect(_));
            let text = if inspect {
                let mut text = resolved.configuration.canonical_toml();
                if let Some(scene) = &resolved.scene {
                    text.push_str(&format!("\nScene: {:?}\nDescription: {:?}\nParticles: {}; planes: {}; randomness: none\nScene schema: {}\n",
                        scene.name, scene.description, scene.particles.len(), scene.planes.len(), scene.schema_version));
                }
                text
            } else {
                format!(
                    "Valid configuration {:?}{}; no simulation performed.\n",
                    resolved.configuration.name,
                    resolved
                        .scene
                        .as_ref()
                        .map_or_else(String::new, |scene| format!(
                            " and scene {:?} ({} particles, {} planes)",
                            scene.name,
                            scene.particles.len(),
                            scene.planes.len()
                        ))
                )
            };
            let mut document = inspection(&resolved);
            document["operation"] = json!(if inspect { "inspect" } else { "validate" });
            print_document(cli.format, text, document)?;
            Ok(0)
        }
        Command::Run(run) => {
            let execution = match run.execution {
                arguments::CpuMode::Serial => {
                    if run.threads.is_some() {
                        return Err(CliError::usage(
                            "--threads requires --execution threaded".into(),
                        ));
                    }
                    maelstrom::execution::ExecutionMode::Serial
                }
                arguments::CpuMode::Threaded => {
                    if run.backend != arguments::Backend::UniformGrid {
                        return Err(CliError::usage(
                            "--execution threaded requires --backend uniform-grid".into(),
                        ));
                    }
                    let mode = match run.threads {
                        Some(workers) => maelstrom::execution::ExecutionMode::Threaded { workers },
                        None => maelstrom::execution::ExecutionMode::threaded_default()
                            .map_err(|e| CliError::input(e.to_string()))?,
                    };
                    mode.validate()
                        .map_err(|e| CliError::usage(e.to_string()))?;
                    mode
                }
            };
            if run.inputs.scene.is_none() {
                return Err(CliError::usage(
                    "run requires SCENE; use 'maelstrom scenes' or 'maelstrom run --help'".into(),
                ));
            }
            let resolved = inputs::resolve(&run.inputs)?;
            diagnostic(cli, &resolved)?;
            let steps = run_limit(
                run.steps,
                run.duration,
                resolved.configuration.numerical.fixed_timestep_s,
            )?;
            if run.checkpoints.iter().any(|&step| step > steps)
                || run.checkpoints.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(CliError::usage("--checkpoints must be unique, ascending and no greater than the resolved step limit; for example 0,6,12 with --steps 12".into()));
            }
            let scene = resolved
                .scene
                .ok_or_else(|| CliError::usage("run requires SCENE".into()))?;
            execute_run(
                cli,
                &scene,
                RunRequest {
                    steps,
                    checkpoints: run.checkpoints.clone(),
                },
                &run.output,
                ExecutionOptions {
                    backend: run.backend.into(),
                    execution,
                    measurement: run.measure.then_some(if run.stage_profile {
                        MeasurementMode::Stages
                    } else {
                        MeasurementMode::Total
                    }),
                },
            )
        }
        Command::Replay(replay) => {
            let source = fs::read_to_string(&replay.file)
                .map_err(|error| CliError::input(format!("result {:?}: {error}", replay.file)))?;
            let (scene, request, options) = replay_options(&source)
                .map_err(|error| CliError::input(format!("result {:?}: {error}", replay.file)))?;
            diagnostic(
                cli,
                &ResolvedInputs {
                    configuration: scene.configuration.clone(),
                    scene: Some(scene.clone()),
                },
            )?;
            execute_run(cli, &scene, request, &replay.output, options)
        }
    }
}

fn diagnostic(cli: &Cli, inputs: &ResolvedInputs) -> Result<(), CliError> {
    if cli.verbose {
        writeln!(
            io::stderr().lock(),
            "maelstrom: resolved configuration {:?}; scene {:?}; {} particles; fixed timestep {} s",
            inputs.configuration.name,
            inputs.scene.as_ref().map(|s| &s.name),
            inputs.scene.as_ref().map_or(0, |s| s.particles.len()),
            inputs.configuration.numerical.fixed_timestep_s
        )
        .map_err(|error| CliError::output(format!("stderr: {error}")))?;
    }
    Ok(())
}

fn inspection(inputs: &ResolvedInputs) -> Value {
    let configuration = &inputs.configuration;
    json!({
        "schema_version": 1, "application_version": env!("CARGO_PKG_VERSION"), "valid": true,
        "configuration": {"schema_version": configuration.schema_version, "name": configuration.name,
            "description": configuration.description, "numerical": configuration.numerical,
            "derived": configuration.derived, "provenance": configuration.sources,
            "comparison_key": configuration.comparison_key()},
        "scene": inputs.scene.as_ref().map(|scene| json!({"schema_version": scene.schema_version,
            "name": scene.name, "description": scene.description, "particle_count": scene.particles.len(),
            "plane_count": scene.planes.len(), "randomness": scene.randomness,
            "planes": scene.planes.iter().map(|plane| json!({"normal": plane.normal(), "offset_m": plane.offset()})).collect::<Vec<_>>() }))
    })
}

fn run_limit(steps: Option<u64>, duration: Option<f64>, timestep: f32) -> Result<u64, CliError> {
    if let Some(steps) = steps {
        return Ok(steps);
    }
    let duration =
        duration.ok_or_else(|| CliError::usage("run requires --steps or --duration".into()))?;
    let steps = (duration / f64::from(timestep)).ceil();
    if !duration.is_finite()
        || duration < 0.0
        || !steps.is_finite()
        || steps >= 18_446_744_073_709_551_616.0
    {
        return Err(CliError::usage("--duration must be finite, non-negative and representable as a u64 count of fixed timesteps; use --steps for an exact count".into()));
    }
    Ok(steps as u64)
}

fn execute_run(
    cli: &Cli,
    scene: &maelstrom::scene::ResolvedScene,
    request: RunRequest,
    output: &Output,
    options: ExecutionOptions,
) -> Result<u8, CliError> {
    check_destinations(output)?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&cancelled);
    ctrlc::set_handler(move || signal_flag.store(true, Ordering::Relaxed)).map_err(|error| {
        CliError {
            code: 1,
            message: format!("could not install Ctrl-C handler: {error}"),
        }
    })?;
    if cli.verbose {
        writeln!(io::stderr().lock(), "maelstrom: running {} steps; checkpoints {:?}; Ctrl-C preserves the last accepted state", request.steps, request.checkpoints)
            .map_err(|error| CliError::output(format!("stderr: {error}")))?;
    }
    let is_cancelled = || cancelled.load(Ordering::Relaxed);
    let mut result = run_with_options(scene, request, options, is_cancelled);
    if let Some(path) = &output.snapshot
        && let Err(error) = result.publish_snapshot(path)
    {
        writeln!(io::stderr().lock(), "maelstrom: snapshot {path:?}: {error}")
            .map_err(|error| CliError::output(format!("stderr: {error}")))?;
    }
    let format = match cli.format {
        Format::Text => OutputFormat::Text,
        Format::Json => OutputFormat::Json,
    };
    if let Some(path) = &output.output {
        result.publish(path, format).map_err(|error| {
            CliError::output(format!(
                "output {path:?}: {error}; choose a new destination in an existing directory"
            ))
        })?;
    } else {
        let encoded = match cli.format {
            Format::Text => result.summary(),
            Format::Json => result
                .to_json()
                .map_err(|error| CliError::output(error.to_string()))?,
        };
        print_stdout(&encoded)?;
    }
    if !cli.quiet && result.warning_count() > 0 {
        writeln!(io::stderr().lock(), "maelstrom: warning: {} accepted steps exceeded the CFL limit; the fixed timestep was retained", result.warning_count())
            .map_err(|error| CliError::output(format!("stderr: {error}")))?;
    }
    for failure in result.failures() {
        writeln!(
            io::stderr().lock(),
            "maelstrom: {:?} after accepted step {}: {:?}",
            failure.status,
            failure.completed_steps,
            failure.message
        )
        .map_err(|error| CliError::output(format!("stderr: {error}")))?;
    }
    Ok(match result.status() {
        CompletionStatus::Completed | CompletionStatus::CompletedWithWarnings => 0,
        CompletionStatus::InputValidationFailure => 3,
        CompletionStatus::SimulationFailure | CompletionStatus::AuditFailure => 4,
        CompletionStatus::OutputFailure => 5,
        CompletionStatus::Cancelled => 130,
    })
}

fn check_destinations(output: &Output) -> Result<(), CliError> {
    let mut seen = Vec::new();
    for path in [&output.output, &output.snapshot].into_iter().flatten() {
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(CliError::output(format!(
                    "output {path:?} already exists; choose a new path (existing files are never replaced)"
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(CliError::output(format!("output {path:?}: {error}"))),
        }
        let directory = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let directory = fs::canonicalize(directory).map_err(|error| {
            CliError::output(format!(
                "output directory for {path:?}: {error}; create the directory first"
            ))
        })?;
        let name = path
            .file_name()
            .ok_or_else(|| CliError::output(format!("output {path:?} requires a file name")))?;
        let destination: PathBuf = directory.join(name);
        if seen.contains(&destination) {
            return Err(CliError::output(
                "--output and --snapshot must name different new files".into(),
            ));
        }
        seen.push(destination);
    }
    Ok(())
}

fn print_document(format: Format, text: String, json: Value) -> Result<(), CliError> {
    let encoded = match format {
        Format::Text => text,
        Format::Json => serde_json::to_string_pretty(&json)
            .map_err(|error| CliError::output(error.to_string()))?,
    };
    print_stdout(&encoded)
}

fn print_stdout(text: &str) -> Result<(), CliError> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(text.as_bytes())
        .and_then(|()| {
            if text.ends_with('\n') {
                Ok(())
            } else {
                stdout.write_all(b"\n")
            }
        })
        .and_then(|()| stdout.flush())
        .map_err(|error| CliError::output(format!("stdout: {error}")))
}
