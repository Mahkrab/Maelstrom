//! Commands distinguish fresh execution, numerical reruns and saved playback.
//! Old spellings remain parseable without crowding primary help.
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "maelstrom",
    version,
    about = "Particle-fluid simulation, compilation and playback"
)]
pub(super) struct Cli {
    /// Result presentation; progress and errors use stderr
    #[arg(long, global = true, value_enum, default_value = "text")]
    pub format: Format,
    /// Suppress optional diagnostics, not errors or requested data
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
    /// Include resolved input context in diagnostics
    #[arg(short, long, global = true)]
    pub verbose: bool,
    /// User scene-library root, overriding MAELSTROM_SCENES_DIR
    #[arg(long, global = true)]
    pub scenes_dir: Option<PathBuf>,
    /// Durable run-library root, overriding MAELSTROM_RUNS_DIR
    #[arg(long, global = true)]
    pub runs_dir: Option<PathBuf>,
    /// Diagnostic colour policy; machine data is never coloured
    #[arg(long, global = true, value_enum, default_value = "auto")]
    pub color: Color,
    #[command(subcommand)]
    pub command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
pub(super) enum Format {
    Text,
    Json,
}
#[derive(Clone, Copy, ValueEnum)]
pub(super) enum Color {
    Auto,
    Always,
    Never,
}
#[derive(Subcommand)]
pub(super) enum Command {
    /// Discover or create editable scenes
    Scenes {
        #[command(subcommand)]
        command: Option<Scenes>,
    },
    /// Diagnose installed simulation and graphics capabilities
    Doctor {
        #[command(subcommand)]
        command: Option<Doctor>,
    },
    /// Show effective application preferences
    Config {
        #[command(subcommand)]
        command: Config,
    },
    /// Show resolved storage locations
    Paths,
    /// Print shell completions without changing shell configuration
    Completions {
        #[arg(value_parser = ["bash", "zsh", "fish"])]
        shell: String,
    },
    /// Inspect or validate compiled output
    Runs {
        #[command(subcommand)]
        command: Runs,
    },
    /// Validate inputs without simulation
    Validate(Inputs),
    /// Show resolved inputs and execution selection
    Inspect {
        #[command(flatten)]
        inputs: Inputs,
        #[command(flatten)]
        selection: Selection,
    },
    /// Execute headlessly at 1x simulated time, or use --unpaced
    Run(Run),
    /// Recompute recorded inputs from zero; not playback or resumption
    Rerun(Replay),
    /// Simulate once and publish portable images for later playback
    Compile(Compile),
    /// Start a fresh simulation with live visualisation
    View(View),
    /// Play existing images with no simulation or particle rendering
    Play(Play),
    #[cfg(feature = "render")]
    /// Render saved observations without recomputation
    Export(Export),
    #[command(hide = true)]
    Replay(Replay),
    #[command(hide = true)]
    CudaCheck {
        #[arg(long, default_value_t = 0)]
        device: u32,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
    },
}
#[derive(Subcommand)]
pub(super) enum Scenes {
    /// List bundled and user scenes
    List,
    /// Create a new editable package without replacing existing files
    Create {
        name: String,
        #[arg(long, default_value = "builtin:falling-block-04")]
        from: String,
    },
}
#[derive(Subcommand)]
pub(super) enum Doctor {
    /// Check independent CUDA resource lifetimes
    Cuda {
        #[arg(long, default_value_t = 0)]
        device: u32,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
    },
    /// Check windowless graphics independently of simulation
    Graphics {
        #[arg(long, default_value_t = 0)]
        graphics_adapter: usize,
        /// Require CPU graphics; independent of the simulation backend
        #[arg(long)]
        software: bool,
    },
}
#[derive(Subcommand)]
pub(super) enum Config {
    Show,
}
#[derive(Subcommand)]
pub(super) enum Runs {
    List,
    Show { run: String },
    Validate { run: String },
}
#[derive(Args, Clone)]
pub(super) struct Inputs {
    /// Named scene, namespace-qualified name or explicit TOML path
    #[arg(value_name = "SCENE")]
    pub scene: Option<String>,
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Primary numerical override expressed as a TOML literal
    #[arg(long = "set", value_name = "FIELD=VALUE")]
    pub overrides: Vec<String>,
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=16))]
    pub refine_timestep: Option<u32>,
}
#[derive(Args, Clone, Default)]
pub(super) struct Selection {
    /// auto (default), brute-force, serial-grid, threaded-grid or cuda
    #[arg(long, alias = "mode", value_name = "BACKEND")]
    pub backend: Option<String>,
    /// Exact CPU pool size; constrains auto to threaded-grid
    #[arg(long, alias = "workers")]
    pub threads: Option<usize>,
    /// CUDA ordinal; constrains auto to CUDA
    #[arg(long)]
    pub device: Option<u32>,
    #[arg(long)]
    pub block_size: Option<u32>,
    #[arg(long, value_enum, hide = true)]
    pub execution: Option<ExecutionKind>,
}
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(super) enum ExecutionKind {
    Serial,
    Threaded,
    Cuda,
}
#[derive(Args)]
pub(super) struct Run {
    #[command(flatten)]
    pub inputs: Inputs,
    #[command(flatten)]
    pub selection: Selection,
    #[command(flatten)]
    pub limit: Limit,
    /// Run without wall-clock pacing
    #[arg(long)]
    pub unpaced: bool,
    #[arg(long, value_delimiter = ',', value_name = "STEPS")]
    pub checkpoints: Vec<u64>,
    #[arg(long)]
    pub measure: bool,
    #[arg(long, requires = "measure")]
    pub stage_profile: bool,
    #[command(flatten)]
    pub output: Output,
}
#[derive(Args)]
pub(super) struct Limit {
    /// Exact fixed timestep count; zero observes initial state
    #[arg(
        long,
        required_unless_present = "duration",
        conflicts_with = "duration"
    )]
    pub steps: Option<u64>,
    /// Simulated seconds, rounded up to complete fixed timesteps
    #[arg(long, value_name = "SECONDS")]
    pub duration: Option<f64>,
}
#[derive(Args)]
pub(super) struct Replay {
    pub file: PathBuf,
    #[command(flatten)]
    pub output: Output,
}
#[derive(Args)]
pub(super) struct Output {
    /// Publish a result to a new file; never replace an existing path
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
    /// Publish the last accepted numerical snapshot to a separate new file
    #[arg(long, value_name = "FILE")]
    pub snapshot: Option<PathBuf>,
}
#[derive(Args)]
pub(super) struct Compile {
    #[command(flatten)]
    pub inputs: Inputs,
    #[command(flatten)]
    pub selection: Selection,
    #[command(flatten)]
    pub limit: Limit,
    /// New bundle directory; omitted selects a new ID in the run library
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Human-readable catalogue label, not a lookup ID or overwrite permission
    #[arg(long)]
    pub name: Option<String>,
    /// Presentation-only TOML settings
    #[arg(long)]
    pub render_config: Option<PathBuf>,
    #[arg(long)]
    pub width: Option<u32>,
    #[arg(long)]
    pub height: Option<u32>,
    /// Sample accepted steps, always retaining both endpoints
    #[arg(long)]
    pub frame_stride: Option<u64>,
    #[arg(long, default_value_t = 0)]
    pub graphics_adapter: usize,
    /// Require CPU graphics; independent of the simulation backend
    #[arg(long)]
    pub software: bool,
    #[arg(long)]
    pub fit_camera: bool,
}
#[derive(Args)]
pub(super) struct Play {
    /// Run ID, completed export directory or manifest.json
    pub sequence: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    pub speed: f64,
    #[arg(long)]
    pub paused: bool,
    #[arg(long)]
    pub repeat: bool,
    #[arg(long, default_value_t = 0)]
    pub graphics_adapter: usize,
    /// Require CPU graphics; independent of the simulation backend
    #[arg(long)]
    pub software: bool,
}
#[derive(Args)]
pub(super) struct View {
    #[command(flatten)]
    pub inputs: Inputs,
    #[command(flatten)]
    pub selection: Selection,
    #[command(flatten)]
    pub limit: Limit,
    #[arg(long)]
    pub paused: bool,
    #[arg(long)]
    pub render_config: Option<PathBuf>,
    #[arg(long)]
    pub width: Option<u32>,
    #[arg(long)]
    pub height: Option<u32>,
    #[arg(long, default_value_t = 0)]
    pub graphics_adapter: usize,
    /// Require CPU graphics; independent of the simulation backend
    #[arg(long)]
    pub software: bool,
    #[arg(long)]
    pub fit_camera: bool,
}
#[cfg(feature = "render")]
#[derive(Args)]
pub(super) struct Export {
    #[command(subcommand)]
    pub source: ExportSource,
    #[arg(short, long, global = true)]
    pub output: Option<PathBuf>,
    #[arg(long, global = true)]
    pub render_config: Option<PathBuf>,
    #[arg(long, global = true)]
    pub width: Option<u32>,
    #[arg(long, global = true)]
    pub height: Option<u32>,
    #[arg(long = "frame-stride", alias = "step-stride", global = true)]
    pub step_stride: Option<u64>,
    #[arg(long, global = true, default_value_t = 0)]
    pub graphics_adapter: usize,
    /// Require CPU graphics; independent of the simulation backend
    #[arg(long, global = true)]
    pub software: bool,
    #[arg(long, global = true)]
    pub fit_camera: bool,
}
#[cfg(feature = "render")]
#[derive(Subcommand)]
pub(super) enum ExportSource {
    /// Render stored observations; missing steps never trigger a rerun
    Saved {
        file: PathBuf,
        #[arg(long)]
        last_step: Option<u64>,
    },
    #[command(hide = true)]
    Rerun { file: PathBuf },
    #[command(hide = true)]
    Direct {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long)]
        steps: u64,
        #[command(flatten)]
        selection: Selection,
    },
}
