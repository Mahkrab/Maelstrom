//! Scene-local presentation selection is shared by inspection and drawing commands.
//! An explicit replacement file is never merged with the scene's render document.

use super::{
    CliError,
    arguments::PresentationOptions,
    inputs::{DocumentIdentity, ResolvedInputs, identity},
};
use maelstrom::presentation::{CameraMode, Presentation};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Serialize)]
pub(super) struct ResolvedPresentation {
    pub effective: Presentation,
    pub provenance: BTreeMap<String, String>,
    pub document: Option<DocumentIdentity>,
    pub drawing_supported: bool,
}

pub(super) fn resolve(
    inputs: Option<&ResolvedInputs>,
    options: &PresentationOptions,
) -> Result<ResolvedPresentation, CliError> {
    let directory = inputs.and_then(|i| i.directory.as_deref());
    let embedded = if directory.is_none() {
        inputs
            .and_then(|i| i.document.as_ref())
            .and_then(|d| super::bundled::render(&d.name))
    } else {
        None
    };
    resolve_selected(directory, options, embedded)
}

#[cfg(feature = "viewer")]
pub(super) fn resolve_directory(
    directory: Option<&Path>,
    options: &PresentationOptions,
) -> Result<ResolvedPresentation, CliError> {
    resolve_selected(directory, options, None)
}

#[cfg(feature = "viewer")]
pub(super) fn resolve_bundled(
    name: &str,
    options: &PresentationOptions,
) -> Result<ResolvedPresentation, CliError> {
    resolve_selected(None, options, super::bundled::render(name))
}

fn resolve_selected(
    directory: Option<&Path>,
    options: &PresentationOptions,
    embedded: Option<&str>,
) -> Result<ResolvedPresentation, CliError> {
    let implicit = directory.map(|dir| dir.join("render.toml"));
    let selected = options.render_config.as_deref().or(implicit.as_deref());
    let mut effective = Presentation::default();
    let mut document = None;
    if let Some(path) = selected {
        match fs::File::open(path) {
            Ok(file) => {
                let source = super::storage::read_file(file, 1024 * 1024)?;
                effective =
                    Presentation::parse(&source).map_err(|e| CliError::input(e.to_string()))?;
                let label = if options.render_config.is_none() {
                    "render.toml".into()
                } else {
                    file_label(path)
                };
                document = Some(identity("render", label, &source));
            }
            Err(e)
                if options.render_config.is_none()
                    && e.kind() == std::io::ErrorKind::NotFound
                    && matches!(fs::symlink_metadata(path),Err(ref metadata) if metadata.kind()==std::io::ErrorKind::NotFound) =>
                {}
            Err(e) => {
                return Err(CliError::input(format!(
                    "selected render document: {}",
                    e.kind()
                )));
            }
        }
    }
    if selected.is_none()
        && let Some(source) = embedded
    {
        effective = Presentation::parse(source).map_err(|e| CliError::input(e.to_string()))?;
        document = Some(identity("render", "render.toml".into(), source));
    }
    let mut provenance = BTreeMap::new();
    for field in [
        "width",
        "height",
        "format",
        "step_stride",
        "camera",
        "camera_mode",
        "particles",
        "camera.eye_m",
        "camera.target_m",
        "camera.up",
        "camera.vertical_fov_deg",
        "camera.near_m",
        "camera.far_m",
        "particles.colour_mode",
        "particles.colour",
        "particles.min_energy_j",
        "particles.max_energy_j",
    ] {
        let supplied = effective.supplied_fields.iter().any(|f| f == field)
            || (field == "camera_mode"
                && document.is_some()
                && (effective.schema_version == 1
                    || effective.supplied_fields.iter().any(|f| f == "camera")));
        provenance.insert(
            field.into(),
            if supplied {
                document
                    .as_ref()
                    .map_or("built_in_default", |d| d.label.as_str())
            } else {
                "built_in_default"
            }
            .into(),
        );
    }
    for (field, value) in [("width", options.width), ("height", options.height)] {
        if let Some(value) = value {
            if field == "width" {
                effective.settings.width = value;
            } else {
                effective.settings.height = value;
            }
            provenance.insert(field.into(), "command_line".into());
        }
    }
    if let Some(stride) = options.frame_stride {
        effective.settings.step_stride = stride;
        provenance.insert("step_stride".into(), "command_line".into());
    }
    if options.fit_camera && options.no_fit_camera {
        return Err(CliError::usage(
            "--fit-camera conflicts with --no-fit-camera".into(),
        ));
    }
    if options.fit_camera || options.no_fit_camera {
        effective.camera_mode = if options.fit_camera {
            CameraMode::Fit
        } else {
            CameraMode::Fixed
        };
        provenance.insert("camera_mode".into(), "command_line".into());
    }
    effective
        .settings
        .validate()
        .map_err(|e| CliError::input(e.to_string()))?;
    let drawing_supported = effective.drawing_supported();
    Ok(ResolvedPresentation {
        effective,
        provenance,
        document,
        drawing_supported,
    })
}

fn file_label(path: &Path) -> String {
    path.file_name().map_or_else(
        || "selected render document".into(),
        |p| p.to_string_lossy().into_owned(),
    )
}
