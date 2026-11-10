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
        #[command(flatten)]
        presentation: PresentationOptions,
        #[command(flatten)]
        limit: Limit,
    },
    /// Execute headlessly at 1x simulated time, or use --unpaced
    Run(Run),
    /// Recompute recorded inputs from zero; not playback or resumption
    Rerun(Replay),
    /// Simulate once and publish portable images for later playback
    Compile(Compile),
    /// Bake, inspect or validate reusable particle observations
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Render particle caches without executing fluid timesteps
    Render(CacheRender),
    /// Encode saved images or cached observations as H.264 MP4
    Video(Video),
    /// View a fresh simulation, or open saved observations with --cache
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
    /// Numerical preset (local scenes use saved settings by default). Explicit --set values win
    #[arg(long, value_parser = ["reference", "fast", "experimental", "paper-baseline", "scene"])]
    pub numerical_mode: Option<String>,
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
#[derive(Args, Default)]
pub(super) struct Limit {
    /// Exact fixed timestep count; zero observes initial state
    #[arg(long, conflicts_with = "duration")]
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
    /// Bake a new reusable cache here, then render it; omitted keeps direct compilation
    #[arg(long)]
    pub cache_output: Option<PathBuf>,
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
    #[arg(long, conflicts_with = "fit_camera")]
    pub no_fit_camera: bool,
}
#[derive(Subcommand)]
pub(super) enum CacheCommand {
    /// Start a new simulation and cache scheduled accepted states, with no graphics
    Bake(Box<CacheBake>),
    /// Inspect metadata and the committed prefix; never continue a solver
    Inspect {
        cache: PathBuf,
        #[arg(long)]
        allow_partial: bool,
    },
    /// Validate binary frames; --full independently recomputes canonical hashes
    Validate {
        cache: PathBuf,
        #[arg(long)]
        allow_partial: bool,
        #[arg(long)]
        full: bool,
    },
}
#[derive(Args)]
pub(super) struct CacheBake {
    #[command(flatten)]
    pub inputs: Inputs,
    #[command(flatten)]
    pub selection: Selection,
    #[command(flatten)]
    pub limit: Limit,
    /// Sample accepted steps, retaining both endpoints
    #[arg(long, default_value_t = 4)]
    pub frame_stride: u64,
    /// New cache directory; never overwrite existing output
    #[arg(short, long)]
    pub output: PathBuf,
}
#[derive(Args)]
pub(super) struct CacheRender {
    pub cache: PathBuf,
    /// New image sequence directory
    #[arg(short, long)]
    pub output: PathBuf,
    #[command(flatten)]
    pub presentation: PresentationOptions,
    /// Single cache-frame index; conflicts with range selection
    #[arg(long, conflicts_with_all=["first_frame","last_frame","every"])]
    pub frame: Option<u64>,
    #[arg(long)]
    pub first_frame: Option<u64>,
    #[arg(long)]
    pub last_frame: Option<u64>,
    /// Increment in cached-frame indices, not fluid timesteps
    #[arg(long)]
    pub every: Option<u64>,
    /// Render only an explicitly selected receipt-backed incomplete prefix
    #[arg(long)]
    pub allow_partial: bool,
    #[arg(long, default_value_t = 0)]
    pub graphics_adapter: usize,
    #[arg(long)]
    pub software: bool,
}
#[derive(Args)]
pub(super) struct Play {
    /// Run ID, completed export directory or manifest.json
    #[arg(required_unless_present = "last", conflicts_with = "last")]
    pub sequence: Option<PathBuf>,
    /// Play the most recently completed compile, including explicit output paths
    #[arg(long)]
    pub last: bool,
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
    /// Open saved observations without creating a solver
    #[arg(long, conflicts_with_all=["scene", "config", "numerical_mode", "overrides", "refine_timestep", "backend", "threads", "device", "block_size", "execution", "steps", "duration"])]
    pub cache: Option<PathBuf>,
    /// Particle read-ahead budget in MiB (includes decode workspace)
    #[arg(long, default_value_t = 128)]
    pub cache_memory_mb: usize,
    /// Follow recorded time or present every observation (cached viewing only)
    #[arg(long, requires="cache", value_parser=["real-time", "frame-accurate"])]
    pub cache_playback: Option<String>,
    /// Explicitly open an incomplete cache's committed prefix
    #[arg(long, requires = "cache")]
    pub allow_partial: bool,
    /// Cached preview geometry; indexed matches offline exports
    #[arg(long, default_value="indexed", value_parser=["indexed", "impostor"])]
    pub preview_renderer: String,
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
    #[arg(long, conflicts_with = "fit_camera")]
    pub no_fit_camera: bool,
    /// Destination for the explicit save-camera action; defaults to the scene render.toml
    #[arg(long)]
    pub camera_output: Option<PathBuf>,
}

#[derive(Args, Default)]
pub(super) struct PresentationOptions {
    #[arg(long)]
    pub render_config: Option<PathBuf>,
    #[arg(long)]
    pub width: Option<u32>,
    #[arg(long)]
    pub height: Option<u32>,
    #[arg(long)]
    pub frame_stride: Option<u64>,
    #[arg(long)]
    pub fit_camera: bool,
    #[arg(long, conflicts_with = "fit_camera")]
    pub no_fit_camera: bool,
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
    #[arg(long, global = true, conflicts_with = "fit_camera")]
    pub no_fit_camera: bool,
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

#[derive(Args)]
pub(super) struct Video {
    #[command(subcommand)]
    pub source: VideoSource,
    #[arg(short, long, global = true)]
    pub output: Option<PathBuf>,
    /// Constant video frame rate, independent of simulation frequency
    #[arg(long, global = true, default_value_t = 60.0)]
    pub fps: f64,
    /// H.264 quality: 0 highest quality, 18 high quality, 51 lowest quality
    #[arg(long, global=true, default_value_t=18, value_parser=clap::value_parser!(u8).range(0..=51))]
    pub crf: u8,
    #[arg(long, global = true, default_value = "ffmpeg")]
    pub ffmpeg: PathBuf,
    #[arg(long, global = true)]
    pub width: Option<u32>,
    #[arg(long, global = true)]
    pub height: Option<u32>,
    #[arg(long, global = true)]
    pub first_frame: Option<u64>,
    #[arg(long, global = true)]
    pub last_frame: Option<u64>,
    #[arg(long, global = true, default_value_t = 1)]
    pub every: u64,
    /// Explicitly omit the final selected observation from the video
    #[arg(long, global = true)]
    pub omit_terminal: bool,
}
#[derive(Subcommand)]
pub(super) enum VideoSource {
    /// Encode a validated PNG bundle; no cache needed
    Images { sequence: PathBuf },
    /// Stream cached rendering directly to FFmpeg
    Cache {
        cache: PathBuf,
        #[arg(long)]
        render_config: PathBuf,
        #[arg(long, default_value_t = 0)]
        graphics_adapter: usize,
        #[arg(long)]
        software: bool,
    },
}
