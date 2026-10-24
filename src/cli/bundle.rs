//! A compiled run reuses the versioned image manifest and embedded resolved input.
//! Run IDs are exact selectors; labels never authorise overwriting existing data.
use super::{
    CliError,
    arguments::{Cli, Compile, Runs},
    storage::{self, Storage},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, io::BufReader, path::Path};

fn open(storage: &Storage, selection: &Path) -> Result<BufReader<fs::File>, CliError> {
    let path = storage.run(selection)?;
    let path = if path.is_dir() {
        path.join("manifest.json")
    } else {
        path
    };
    let file = fs::File::open(path)
        .map_err(|e| CliError::input(format!("selected run document: {}", e.kind())))?;
    if !file
        .metadata()
        .map_err(|e| CliError::input(e.to_string()))?
        .is_file()
    {
        return Err(CliError::input(
            "selected run document must be a regular file".into(),
        ));
    }
    Ok(BufReader::new(file))
}

pub(super) fn source(storage: &Storage, selection: &Path) -> Result<String, CliError> {
    use std::io::Read;
    // Numerical reruns need the complete recorded input; unlike playback this
    // route intentionally materialises it. Scene-document limits do not apply.
    let mut text = String::new();
    open(storage, selection)?
        .read_to_string(&mut text)
        .map_err(|e| CliError::input(format!("selected run document: {}", e.kind())))?;
    Ok(text)
}
fn read(storage: &Storage, selection: &str) -> Result<Value, CliError> {
    let v: Value = serde_json::from_reader(open(storage, Path::new(selection))?)
        .map_err(|e| CliError::input(format!("run manifest: {e}")))?;
    if v["schema_version"] != 1 || v["complete"] != true {
        return Err(CliError::input(
            "expected a completed schema-1 image manifest".into(),
        ));
    }
    Ok(v)
}
pub(super) fn manage(cli: &Cli, storage: &Storage, command: &Runs) -> Result<u8, CliError> {
    match command {
        Runs::List => {
            let mut paths = Vec::new();
            match fs::read_dir(&storage.runs) {
                Ok(items) => {
                    for item in items {
                        let item = item.map_err(|e| CliError::input(e.to_string()))?;
                        if item
                            .file_type()
                            .map_err(|e| CliError::input(e.to_string()))?
                            .is_dir()
                        {
                            paths.push(item.path());
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(CliError::input(format!("run library: {e}"))),
            }
            paths.sort();
            let mut rows = Vec::new();
            for path in paths {
                let (complete, status) = if path
                    .join("manifest.json")
                    .try_exists()
                    .map_err(|e| CliError::input(e.to_string()))?
                {
                    #[derive(Deserialize)]
                    struct Completion {
                        schema_version: u32,
                        complete: bool,
                    }
                    match open(storage, &path).and_then(|reader| {
                        serde_json::from_reader::<_, Completion>(reader)
                            .map_err(|e| CliError::input(e.to_string()))
                    }) {
                        Ok(v) if v.schema_version == 1 && v.complete => {
                            (true, "completed; frame integrity not checked")
                        }
                        _ => (false, "invalid manifest"),
                    }
                } else {
                    (false, "incomplete")
                };
                let label = if path
                    .join("run.json")
                    .try_exists()
                    .map_err(|e| CliError::input(e.to_string()))?
                {
                    let meta: Value =
                        serde_json::from_str(&storage::read(&path.join("run.json"), 1024 * 1024)?)
                            .map_err(|e| CliError::input(e.to_string()))?;
                    meta["label"].clone()
                } else {
                    Value::Null
                };
                rows.push(json!({"id":path.file_name().map(|s|s.to_string_lossy()),"path":path,"complete":complete,
                    "status":status,"label":label}));
            }
            let text = rows
                .iter()
                .map(|r| {
                    format!(
                        "{}: {}\n",
                        r["id"].as_str().unwrap_or("?"),
                        r["status"].as_str().unwrap_or("?")
                    )
                })
                .collect();
            super::print_document(cli.format, text, json!({"schema_version":1,"runs":rows}))?;
        }
        Runs::Show { run } => {
            let mut value = read(storage, run)?;
            let selected = storage.run(Path::new(run))?;
            let root = if selected.is_dir() {
                selected.as_path()
            } else {
                selected.parent().unwrap_or(Path::new("."))
            };
            if root
                .join("run.json")
                .try_exists()
                .map_err(|e| CliError::input(e.to_string()))?
            {
                value["catalogue"] =
                    serde_json::from_str(&storage::read(&root.join("run.json"), 1024 * 1024)?)
                        .map_err(|e| CliError::input(e.to_string()))?;
            }
            super::print_document(
                cli.format,
                serde_json::to_string_pretty(&value)
                    .map_err(|e| CliError::output(e.to_string()))?,
                value,
            )?;
        }
        Runs::Validate { run } => {
            validate(storage, run)?;
            super::print_document(
                cli.format,
                "Compiled run integrity verified; no simulation performed.".into(),
                json!({"schema_version":1,"valid":true,"run":run}),
            )?;
        }
    }
    Ok(0)
}
#[cfg(feature = "render")]
fn validate(storage: &Storage, run: &str) -> Result<(), CliError> {
    maelstrom::workflow::replay_options_from_reader(open(storage, Path::new(run))?)
        .map_err(|e| CliError::input(e.to_string()))?;
    let sequence = maelstrom::playback::Sequence::open(&storage.run(Path::new(run))?)
        .map_err(|e| CliError::input(e.to_string()))?;
    for index in 0..sequence.frame_count() {
        sequence
            .decode(index)
            .map_err(|e| CliError::input(e.to_string()))?;
    }
    Ok(())
}
#[cfg(not(feature = "render"))]
fn validate(_: &Storage, _: &str) -> Result<(), CliError> {
    Err(CliError::input(
        "image validation requires render support; reinstall with --features viewer or render"
            .into(),
    ))
}
#[cfg(not(feature = "render"))]
pub(super) fn compile(_: &Cli, _: &Storage, _: &Compile) -> Result<u8, CliError> {
    Err(CliError::input("compilation requires render support; reinstall with --features viewer for compilation and playback".into()))
}
#[cfg(feature = "render")]
pub(super) fn compile(cli: &Cli, storage: &Storage, args: &Compile) -> Result<u8, CliError> {
    use maelstrom::{
        export::{ExportRequest, Source},
        render::{GraphicsSelection, Settings},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let resolved = super::inputs::resolve_in(&args.inputs, storage)?;
    let scene = resolved
        .scene
        .ok_or_else(|| CliError::usage("compile requires SCENE".into()))?;
    let steps = super::run_limit(
        args.limit.steps,
        args.limit.duration,
        scene.configuration.numerical.fixed_timestep_s,
    )?;
    let options = super::selection::resolve(&args.selection, &storage.preferences()?)?;
    if let Some(name) = &args.name
        && (name.is_empty() || name.len() > 256 || name.chars().any(char::is_control))
    {
        return Err(CliError::input(
            "run label must contain 1..=256 non-control characters".into(),
        ));
    }
    let mut settings = match &args.render_config {
        Some(path) => Settings::parse(&storage::read(path, 1024 * 1024)?)
            .map_err(|e| CliError::input(e.to_string()))?,
        None => Settings::default(),
    };
    if let Some(v) = args.width {
        settings.width = v;
    }
    if let Some(v) = args.height {
        settings.height = v;
    }
    if let Some(stride) = args.frame_stride {
        settings.step_stride = stride;
    }
    settings
        .validate()
        .map_err(|e| CliError::input(e.to_string()))?;
    let destination = match &args.output {
        Some(path) => path.clone(),
        None => {
            fs::create_dir_all(&storage.runs)
                .map_err(|e| CliError::output(format!("create run library: {e}")))?;
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| CliError::output(e.to_string()))?;
            storage
                .runs
                .join(format!("run-{}-{}", time.as_nanos(), std::process::id()))
        }
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancel);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|e| CliError::input(e.to_string()))?;
    let exported = maelstrom::export::export(
        ExportRequest {
            source: Source::Direct {
                scene: &scene,
                steps,
                options,
            },
            settings,
            graphics: GraphicsSelection {
                index: args.graphics_adapter,
                software: args.software,
            },
            fit_camera: args.fit_camera,
            destination: &destination,
        },
        || cancel.load(Ordering::Relaxed),
    );
    let mut manifest = match exported {
        Ok(v) => v,
        Err(e) => {
            return Err(CliError {
                code: if cancel.load(Ordering::Relaxed) {
                    130
                } else {
                    5
                },
                message: format!(
                    "compile: {e}; any published partial output is retained at {}",
                    destination.display()
                ),
            });
        }
    };
    // Keep the existing immutable image manifest intact. Catalogue metadata is
    // additive; playback and numerical reruns need only manifest.json.
    maelstrom::playback::Sequence::open(&destination)
        .map_err(|e| CliError::output(format!("compiled manifest is not playable: {e}")))?;
    let label = json!({"schema_version":1,"label":args.name,"id":destination.file_name().map(|s|s.to_string_lossy())});
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join("run.json"))
        .map_err(|e| CliError::output(e.to_string()))?;
    file.write_all(
        &serde_json::to_vec_pretty(&label).map_err(|e| CliError::output(e.to_string()))?,
    )
    .and_then(|_| file.sync_all())
    .map_err(|e| CliError::output(e.to_string()))?;
    manifest["bundle"] = label;
    super::print_document(
        cli.format,
        format!(
            "Compiled {} frames to {}\nPlay with: maelstrom play {}",
            manifest["frame_count"],
            destination.display(),
            destination.display()
        ),
        json!({"schema_version":1,"path":destination,"id":destination.file_name().map(|s|s.to_string_lossy()),"complete":true,"frame_count":manifest["frame_count"],"label":args.name,"execution":manifest["execution"]}),
    )?;
    Ok(0)
}
