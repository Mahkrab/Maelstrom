//! Libraries are durable user data, not working-directory searches or caches.
//! Discovery never creates directories; writes exclusively create new packages.
use super::{CliError, arguments::Cli};
use serde::Deserialize;
use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub(super) struct Storage {
    pub scenes: PathBuf,
    pub runs: PathBuf,
    pub preferences: PathBuf,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Preferences {
    pub schema_version: u32,
    pub backend: Option<String>,
    pub numerical_mode: Option<String>,
    pub threads: Option<usize>,
    pub device: Option<u32>,
    pub block_size: Option<u32>,
}
impl Storage {
    pub fn for_cli(cli: &Cli) -> Result<Self, CliError> {
        Self::new(cli.scenes_dir.as_deref(), cli.runs_dir.as_deref())
    }
    pub fn new(scenes: Option<&Path>, runs: Option<&Path>) -> Result<Self, CliError> {
        let data = base("XDG_DATA_HOME", ".local/share")?;
        Ok(Self {
            scenes: location(
                scenes,
                "MAELSTROM_SCENES_DIR",
                data.join("maelstrom/scenes"),
            )?,
            runs: location(runs, "MAELSTROM_RUNS_DIR", data.join("maelstrom/runs"))?,
            preferences: base("XDG_CONFIG_HOME", ".config")?.join("maelstrom/config.toml"),
        })
    }
    pub fn preferences(&self) -> Result<Preferences, CliError> {
        let source = match fs::File::open(&self.preferences) {
            Ok(file) => read_file(file, 1024 * 1024)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Preferences::default()),
            Err(e) => return Err(CliError::input(format!("application preferences: {e}"))),
        };
        let p: Preferences = toml::from_str(&source)
            .map_err(|e| CliError::input(format!("application preferences: {e}")))?;
        if p.schema_version != 1 {
            return Err(CliError::input(
                "application preferences require schema_version = 1".into(),
            ));
        }
        if p.backend.as_deref().is_some_and(|v| {
            ![
                "auto",
                "brute-force",
                "serial-grid",
                "threaded-grid",
                "cuda",
            ]
            .contains(&v)
        }) {
            return Err(CliError::input("application preferences: backend must be auto, brute-force, serial-grid, threaded-grid or cuda".into()));
        }
        if p.numerical_mode.as_deref().is_some_and(|v| {
            ![
                "reference",
                "fast",
                "experimental",
                "paper-baseline",
                "scene",
            ]
            .contains(&v)
        }) {
            return Err(CliError::input(
                "application preferences: numerical_mode must be reference, fast, experimental, paper-baseline or scene".into(),
            ));
        }
        if p.threads.is_some_and(|n| !(1..=256).contains(&n)) {
            return Err(CliError::input(
                "application preferences: threads must be in 1..=256".into(),
            ));
        }
        if p.device.is_some_and(|n| n > i32::MAX as u32) {
            return Err(CliError::input(
                "application preferences: device must be a representable CUDA ordinal".into(),
            ));
        }
        if p.block_size
            .is_some_and(|n| ![32, 64, 128, 256].contains(&n))
        {
            return Err(CliError::input(
                "application preferences: block_size must be 32, 64, 128 or 256".into(),
            ));
        }
        let cuda = p.device.is_some() || p.block_size.is_some();
        let mode = p.backend.as_deref().unwrap_or("auto");
        if (p.threads.is_some() && (cuda || !["auto", "threaded-grid"].contains(&mode)))
            || (cuda && !["auto", "cuda"].contains(&mode))
        {
            return Err(CliError::input(
                "application preferences: backend and CPU/CUDA resources contradict each other"
                    .into(),
            ));
        }
        Ok(p)
    }
    pub fn scene(&self, selection: &str) -> Result<Option<PathBuf>, CliError> {
        let (namespace, name) = selection
            .split_once(':')
            .map_or(("", selection), |(a, b)| (a, b));
        if selection.contains('/') || selection.contains('\\') || selection.ends_with(".toml") {
            return Ok(Some(PathBuf::from(selection)));
        }
        safe_name(name)?;
        if !["", "user", "builtin"].contains(&namespace) {
            return Err(CliError::input("unknown scene namespace".into()));
        }
        let path = self.scenes.join(name).join("scene.toml");
        let user = path
            .try_exists()
            .map_err(|e| CliError::input(format!("scene lookup: {e}")))?;
        let builtin = super::bundled::scene(name).is_some();
        match namespace {
            "builtin" if builtin => Ok(None),
            "user" if user => Ok(Some(path)),
            "" if user && builtin => Err(CliError::input(format!(
                "ambiguous scene {name:?}; select user:{name} or builtin:{name}"
            ))),
            "" if user => Ok(Some(path)),
            "" if builtin => Ok(None),
            _ => Err(CliError::input(format!(
                "unknown scene {selection:?}; use 'maelstrom scenes list' or an explicit path"
            ))),
        }
    }
    pub fn run(&self, selection: &Path) -> Result<PathBuf, CliError> {
        let text = selection
            .to_str()
            .ok_or_else(|| CliError::input("run selector is not UTF-8".into()))?;
        if selection.is_absolute()
            || selection.components().count() > 1
            || text.ends_with(".json")
            || selection.is_dir()
        {
            return Ok(selection.to_owned());
        }
        safe_name(text)?;
        Ok(self.runs.join(text))
    }
}
fn base(variable: &str, suffix: &str) -> Result<PathBuf, CliError> {
    if let Some(p) = env::var_os(variable)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
    {
        return Ok(p);
    }
    env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .map(|p| p.join(suffix))
        .ok_or_else(|| CliError::input(format!("set {variable} or HOME to an absolute directory")))
}
fn location(flag: Option<&Path>, variable: &str, default: PathBuf) -> Result<PathBuf, CliError> {
    if let Some(p) = flag {
        return if p.is_absolute() {
            Ok(p.to_owned())
        } else {
            env::current_dir()
                .map(|d| d.join(p))
                .map_err(|e| CliError::input(e.to_string()))
        };
    }
    if let Some(value) = env::var_os(variable).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(value);
        if !p.is_absolute() {
            return Err(CliError::input(format!("{variable} must be absolute")));
        }
        return Ok(p);
    }
    Ok(default)
}
pub(super) fn safe_name(name: &str) -> Result<(), CliError> {
    if name.is_empty()
        || name.len() > 120
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(CliError::input(
            "names must contain 1..=120 ASCII letters, digits, '-' or '_'".into(),
        ));
    }
    Ok(())
}
pub(super) fn read(path: &Path, limit: u64) -> Result<String, CliError> {
    read_file(
        fs::File::open(path)
            .map_err(|e| CliError::input(format!("read selected document: {e}")))?,
        limit,
    )
}
pub(super) fn read_file(file: fs::File, limit: u64) -> Result<String, CliError> {
    let mut text = String::new();
    file.take(limit + 1)
        .read_to_string(&mut text)
        .map_err(|e| CliError::input(format!("read document: {e}")))?;
    if text.len() as u64 > limit {
        return Err(CliError::input(
            "selected document exceeds size limit".into(),
        ));
    }
    Ok(text)
}
