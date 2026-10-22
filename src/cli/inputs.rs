//! Named scenes embed the same TOML inputs shipped with the application. External
//! paths use the library loader, including its scene-relative file resolution.

use std::{fs, path::Path};

use maelstrom::{
    config::{
        ConfigurationOverlay, ResolvedConfiguration, load_configuration,
        parse_configuration_overlay, resolve_effective_configuration,
    },
    scene::{
        ResolvedScene, load_resolved_scene, parse_scene, resolve_scene,
        resolve_scene_with_configuration,
    },
};

use super::{CliError, arguments::Inputs, bundled};

pub(super) struct ResolvedInputs {
    pub configuration: ResolvedConfiguration,
    pub scene: Option<ResolvedScene>,
}

pub(super) fn resolve_in(
    inputs: &Inputs,
    storage: &super::storage::Storage,
) -> Result<ResolvedInputs, CliError> {
    let mut inputs = inputs.clone();
    if let Some(selection) = &inputs.scene {
        inputs.scene = Some(match storage.scene(selection)? {
            Some(path) => path.to_string_lossy().into_owned(),
            None => selection
                .strip_prefix("builtin:")
                .unwrap_or(selection)
                .to_owned(),
        });
    }
    resolve_document(&inputs)
}

fn resolve_document(inputs: &Inputs) -> Result<ResolvedInputs, CliError> {
    let mut overlay = overrides(&inputs.overrides)?;
    if let Some(factor) = inputs.refine_timestep {
        if overlay.numerical.refine_timestep.is_some() {
            return Err(CliError::usage(
                "select refinement once, using --refine-timestep or --set refine_timestep"
                    .to_owned(),
            ));
        }
        overlay.numerical.refine_timestep = Some(factor);
    }
    let configuration = inputs
        .config
        .as_ref()
        .map(|path| {
            load_configuration(path)
                .map_err(|error| CliError::input(format!("configuration {path:?}: {error}")))
        })
        .transpose()?;
    let Some(selection) = &inputs.scene else {
        let configuration =
            resolve_effective_configuration(configuration.as_ref(), None, Some(&overlay))
                .map_err(|error| CliError::input(format!("configuration: {error}")))?;
        return Ok(ResolvedInputs {
            configuration,
            scene: None,
        });
    };
    let scene = if let Some(source) = bundled::scene(selection) {
        let document = parse_scene(source)
            .map_err(|error| CliError::input(format!("bundled scene {selection:?}: {error}")))?;
        let resolved = if let Some(configuration) = &configuration {
            resolve_scene_with_configuration(&document, configuration, Some(&overlay))
        } else {
            let selected = bundled::configuration(&document)?;
            resolve_scene(&document, selected.as_ref(), Some(&overlay))
        };
        resolved.map_err(|error| CliError::input(format!("scene {selection:?}: {error}")))?
    } else if let Some(configuration) = &configuration {
        let source =
            fs::read_to_string(selection).map_err(|error| scene_read_error(selection, error))?;
        let document = parse_scene(&source)
            .map_err(|error| CliError::input(format!("scene {selection:?}: {error}")))?;
        resolve_scene_with_configuration(&document, configuration, Some(&overlay))
            .map_err(|error| CliError::input(format!("scene {selection:?}: {error}")))?
    } else {
        load_resolved_scene(Path::new(selection), Some(&overlay))
            .map_err(|error| CliError::input(format!("scene {selection:?}: {error}; use 'maelstrom scenes' for bundled names or supply a readable TOML path")))?
    };
    Ok(ResolvedInputs {
        configuration: scene.configuration.clone(),
        scene: Some(scene),
    })
}

fn scene_read_error(selection: &str, error: std::io::Error) -> CliError {
    CliError::input(format!(
        "scene {selection:?}: {error}; use 'maelstrom scenes' for bundled names or supply a readable TOML path"
    ))
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
