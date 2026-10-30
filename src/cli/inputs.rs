//! Resolve source bytes once, retaining portable provenance alongside accepted inputs.

use super::{
    CliError,
    arguments::{Inputs, Limit},
    bundled,
    storage::Storage,
};
use maelstrom::{
    config::{
        ConfigurationOverlay, ResolvedConfiguration, parse_configuration,
        parse_configuration_overlay, resolve_effective_configuration,
    },
    scene::{
        ConfigurationSource, ResolvedScene, SceneDocument, parse_scene, resolve_scene,
        resolve_scene_with_configuration,
    },
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize)]
pub(super) struct DocumentIdentity {
    pub domain: &'static str,
    pub label: String,
    pub sha256: String,
}
pub(super) fn identity(domain: &'static str, label: String, source: &str) -> DocumentIdentity {
    DocumentIdentity {
        domain,
        label,
        sha256: Sha256::digest(source.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    }
}

pub(super) struct ResolvedInputs {
    pub configuration: ResolvedConfiguration,
    pub scene: Option<ResolvedScene>,
    pub document: Option<SceneDocument>,
    pub directory: Option<PathBuf>,
    pub documents: Vec<DocumentIdentity>,
    pub selected_configuration: Option<maelstrom::config::ConfigurationDocument>,
}
impl ResolvedInputs {
    pub fn steps(&self, limit: &Limit) -> Result<u64, CliError> {
        let saved = self.document.as_ref().and_then(|doc| doc.run.as_ref());
        let (steps, duration) = if limit.steps.is_some() || limit.duration.is_some() {
            (limit.steps, limit.duration)
        } else {
            (
                saved.and_then(|r| r.steps),
                saved.and_then(|r| r.duration_s),
            )
        };
        super::run_limit(
            steps,
            duration,
            self.configuration.numerical.fixed_timestep_s,
        )
    }
}

pub(super) fn resolve_in(inputs: &Inputs, storage: &Storage) -> Result<ResolvedInputs, CliError> {
    let mut overlay = overrides(&inputs.overrides)?;
    if let Some(factor) = inputs.refine_timestep {
        if overlay.numerical.refine_timestep.is_some() {
            return Err(CliError::usage(
                "select refinement once, using --refine-timestep or --set refine_timestep".into(),
            ));
        }
        overlay.numerical.refine_timestep = Some(factor);
    }
    let mut documents = Vec::new();
    let preferences = storage.preferences()?;
    {
        let name = inputs
            .numerical_mode
            .as_deref()
            .or(preferences.numerical_mode.as_deref())
            .unwrap_or("fast");
        let mode = name
            .parse::<maelstrom::config::NumericalMode>()
            .map_err(|e| CliError::usage(e.to_string()))?;
        mode.fill_overrides(&mut overlay.numerical);
        // The preset is provenance, not a replay instruction. Resolved primary
        // values remain the sole numerical inputs when a record is rerun.
        if mode != maelstrom::config::NumericalMode::Scene {
            documents.push(identity("numerical_mode", name.into(), name));
        }
    }
    let explicit = inputs
        .config
        .as_ref()
        .map(|path| {
            let source = read(path, "configuration")?;
            documents.push(identity("configuration", label(path), &source));
            parse_configuration(&source).map_err(|e| CliError::input(format!("configuration: {e}")))
        })
        .transpose()?;
    let Some(selection) = &inputs.scene else {
        let configuration =
            resolve_effective_configuration(explicit.as_ref(), None, Some(&overlay))
                .map_err(|e| CliError::input(e.to_string()))?;
        return Ok(ResolvedInputs {
            configuration,
            scene: None,
            document: None,
            directory: None,
            documents,
            selected_configuration: explicit,
        });
    };
    let path = storage.scene(selection)?;
    let (source, directory, scene_label) = if let Some(path) = &path {
        let path =
            fs::canonicalize(path).map_err(|e| CliError::input(format!("scene: {}", e.kind())))?;
        (
            read(&path, "scene")?,
            path.parent().map(Path::to_path_buf),
            label(&path),
        )
    } else {
        let name = selection.strip_prefix("builtin:").unwrap_or(selection);
        (
            bundled::scene(name)
                .ok_or_else(|| CliError::input("unknown bundled scene".into()))?
                .into(),
            None,
            format!("builtin:{name}"),
        )
    };
    documents.push(identity("scene", scene_label, &source));
    let document = parse_scene(&source).map_err(|e| CliError::input(format!("scene: {e}")))?;
    let selected = if let Some(explicit) = explicit {
        Some(explicit)
    } else if let Some(directory) = &directory {
        if document.configuration.source == ConfigurationSource::File {
            let reference =
                document.configuration.path.as_deref().ok_or_else(|| {
                    CliError::input("configuration file source requires path".into())
                })?;
            if reference.trim().is_empty() || Path::new(reference).is_absolute() {
                return Err(CliError::input(
                    "configuration path must be relative to the scene document".into(),
                ));
            }
            let scene_label = documents
                .iter()
                .find(|d| d.domain == "scene")
                .map_or("selected scene", |d| d.label.as_str());
            let source = read(
                &directory.join(reference),
                &format!("scene {scene_label} configuration"),
            )?;
            documents.push(identity("configuration", reference.into(), &source));
            Some(
                parse_configuration(&source)
                    .map_err(|e| CliError::input(format!("scene configuration: {e}")))?,
            )
        } else {
            None
        }
    } else {
        let selected = bundled::configuration(&document)?;
        if selected.is_some() {
            // The embedded catalogue owns the exact original source bytes.
            let reference = document.configuration.path.as_deref().unwrap_or_default();
            if let Some(source) = bundled::configuration_source(&document)? {
                documents.push(identity(
                    "configuration",
                    format!("builtin:{reference}"),
                    source,
                ));
            }
        }
        selected
    };
    let scene = if inputs.config.is_some() {
        resolve_scene_with_configuration(
            &document,
            selected
                .as_ref()
                .ok_or_else(|| CliError::input("missing explicit configuration".into()))?,
            Some(&overlay),
        )
    } else {
        resolve_scene(&document, selected.as_ref(), Some(&overlay))
    }
    .map_err(|e| CliError::input(format!("scene: {e}")))?;
    Ok(ResolvedInputs {
        configuration: scene.configuration.clone(),
        scene: Some(scene),
        document: Some(document),
        directory,
        documents,
        selected_configuration: selected,
    })
}

fn label(path: &Path) -> String {
    path.file_name().map_or_else(
        || "selected document".into(),
        |name| name.to_string_lossy().into_owned(),
    )
}
fn read(path: &Path, kind: &str) -> Result<String, CliError> {
    let file = fs::File::open(path).map_err(|e| {
        CliError::input(format!(
            "could not read {kind} {:?}: {}",
            label(path),
            e.kind()
        ))
    })?;
    super::storage::read_file(
        file,
        if kind == "scene" {
            64 * 1024 * 1024
        } else {
            1024 * 1024
        },
    )
}

fn overrides(values: &[String]) -> Result<ConfigurationOverlay, CliError> {
    let mut source = String::from("[numerical]\n");
    for assignment in values {
        let (field, value) = assignment.split_once('=').ok_or_else(|| {
            CliError::usage(format!(
                "--set {assignment:?} requires FIELD=VALUE, for example cfl_factor=0.5"
            ))
        })?;
        if field.is_empty()
            || !field
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || value.is_empty()
            || value.chars().any(char::is_control)
        {
            return Err(CliError::usage(format!(
                "--set {assignment:?} requires a field name and a single-line TOML literal"
            )));
        }
        source.push_str(&format!("{field} = {value}\n"));
    }
    parse_configuration_overlay(&source).map_err(|error| {
        CliError::usage(format!(
            "--set: {error}; use primary numerical fields shown by 'maelstrom inspect'"
        ))
    })
}
