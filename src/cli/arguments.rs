//! Commands expose input resolution separately from execution. Every run names
//! its scene and limit; formatting and diagnostic flags are process concerns.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

const EXAMPLES: &str = "Examples:\n  maelstrom scenes\n  maelstrom inspect falling-block-04\n  maelstrom validate ./scene.toml --config ./config.toml\n  maelstrom run falling-block-04 --steps 12 --checkpoints 0,6,12\n  maelstrom run ./scene.toml --duration 0.1 --format json --output result.json\n  maelstrom replay result.json --format json\n\nPrecedence: built-in defaults < selected configuration file < scene values < --set.\nNo environment variables select numerical inputs. Use '<command> --help' for details.";

#[derive(Parser)]
#[command(name = "maelstrom", version, about = "Deterministic headless Position-Based Fluids simulation", after_help = EXAMPLES)]
pub(super) struct Cli {
    /// Select result presentation; JSON uses versioned documents
    #[arg(long, global = true, value_enum, default_value = "text")]
    pub format: Format,
    /// Suppress optional stderr diagnostics; requested stdout data is retained
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
    /// Report resolved input and execution context to stderr
    #[arg(short, long, global = true)]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
pub(super) enum Format {
    Text,
    Json,
}

#[derive(Subcommand)]
pub(super) enum Command {
    /// List the bundled, deterministic scenes
    Scenes,
    /// Check configuration and optional scene without advancing state
    #[command(
        after_help = "Examples:\n  maelstrom validate\n  maelstrom validate --config ./config.toml\n  maelstrom validate falling-block-04 --set cfl_factor=0.5"
    )]
    Validate(Inputs),
    /// Show all effective configuration values, provenance and scene summary
    #[command(
        after_help = "Examples:\n  maelstrom inspect --format json\n  maelstrom inspect falling-block-04 --config ./config.toml\n  maelstrom inspect ./scene.toml --set 'external_acceleration_m_s2=[0.0, -9.81, 0.0]'"
    )]
    Inspect(Inputs),
    /// Run a named scene or external TOML file with an explicit limit
    #[command(
        after_help = "Examples:\n  maelstrom run falling-block-04 --steps 12 --checkpoints 0,6,12\n  maelstrom run ./scene.toml --config ./config.toml --steps 12 --format json -o result.json\n  maelstrom run falling-block-04 --duration 0.1 --snapshot final.json\n\nDuration is simulated seconds, rounded up to whole fixed timesteps; it is not a wall-clock timeout.\nCheckpoints must be unique, ascending and within the run limit. No checkpoints are selected by default.\nOutput and snapshot paths must be new files in existing directories; replacement is never implicit."
    )]
    Run(Run),
    /// Repeat the recorded inputs and request from a result JSON file
    #[command(
        after_help = "Example: maelstrom replay result.json --format json -o repeated.json\nReplay validates recorded inputs and backend. It reruns the original request; it does not certify the original outcome."
    )]
    Replay(Replay),
}

#[derive(Args)]
pub(super) struct Inputs {
    /// Bundled name from 'scenes', or path to an external scene TOML file
    #[arg(value_name = "SCENE")]
    pub scene: Option<String>,
    /// Select a base configuration TOML file instead of the scene's selection
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Override a primary numerical field using a TOML literal; repeat for fields
    #[arg(long = "set", value_name = "FIELD=VALUE")]
    pub overrides: Vec<String>,
    /// Divide the effective timestep by N and the complete Jacobi correction
    /// scale by N² after --set overrides
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=16))]
    pub refine_timestep: Option<u32>,
}

#[derive(Args)]
pub(super) struct Run {
    #[command(flatten)]
    pub inputs: Inputs,
    /// Number of fixed timesteps; zero observes the initial state
    #[arg(
        long,
        required_unless_present = "duration",
        conflicts_with = "duration"
    )]
    pub steps: Option<u64>,
    /// Simulated duration in seconds, rounded up to whole fixed timesteps
    #[arg(long, value_name = "SECONDS")]
    pub duration: Option<f64>,
    /// Accepted step boundaries to snapshot and audit; comma-separated
    #[arg(long, value_delimiter = ',', value_name = "STEPS")]
    pub checkpoints: Vec<u64>,
    /// Record monotonic elapsed seconds around each complete solver step
    #[arg(long)]
    pub measure: bool,
    /// Also profile internal stages; adds clock overhead, so report separately
    #[arg(long, requires = "measure")]
    pub stage_profile: bool,
    #[command(flatten)]
    pub output: Output,
}

#[derive(Args)]
pub(super) struct Replay {
    /// Versioned JSON result containing complete inputs and a run request
    pub file: PathBuf,
    #[command(flatten)]
    pub output: Output,
}

#[derive(Args)]
pub(super) struct Output {
    /// Publish the result to a new file instead of stdout
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
    /// Publish the last accepted state to a separate new JSON file
    #[arg(long, value_name = "FILE")]
    pub snapshot: Option<PathBuf>,
}
