//! Bundled inputs use a build-generated catalogue, never the source filesystem.
//! Configuration references retain the same scene-relative meaning as file inputs.

use std::path::{Component, Path, PathBuf};

use maelstrom::{
    config::{ConfigurationDocument, parse_configuration},
    scene::{ConfigurationSource, SceneDocument},
};

use super::CliError;

include!(concat!(env!("OUT_DIR"), "/bundled_inputs.rs"));

pub(super) fn scene(name: &str) -> Option<&'static str> {
    SCENES
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, source)| *source)
}

pub(super) fn configuration(
    document: &SceneDocument,
) -> Result<Option<ConfigurationDocument>, CliError> {
    configuration_source(document)?
        .map(|source| {
            parse_configuration(source)
                .map_err(|error| CliError::input(format!("bundled configuration: {error}")))
        })
        .transpose()
}

pub(super) fn configuration_source(
    document: &SceneDocument,
) -> Result<Option<&'static str>, CliError> {
    if document.configuration.source != ConfigurationSource::File {
        return Ok(None);
    }
    let reference = document.configuration.path.as_deref().ok_or_else(|| {
        CliError::input("bundled scene configuration: file source requires path".into())
    })?;
    let mut path = PathBuf::from("scenes");
    for component in Path::new(reference).components() {
        match component {
            Component::Normal(name) => path.push(name),
            Component::CurDir => {}
            Component::ParentDir if path.pop() => {}
            _ => {
                return Err(CliError::input(format!(
                    "bundled configuration path must stay within the bundle: {reference:?}"
                )));
            }
        }
    }
    let source = CONFIGURATIONS
        .iter()
        .find(|(key, _)| Path::new(key) == path)
        .map(|(_, source)| *source)
        .ok_or_else(|| CliError::input(format!("configuration {reference:?} is not bundled")))?;
    Ok(Some(source))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> SceneDocument {
        maelstrom::scene::parse_scene(SCENES[0].1).unwrap()
    }

    #[test]
    fn references_resolve_without_access_to_the_source_tree() {
        let mut scene = document();
        scene.configuration.path = Some("./../config/./defaults-v4.toml".into());
        let config = configuration(&scene)
            .unwrap_or_else(|e| panic!("{}", e.message))
            .unwrap();
        assert_eq!(config.name, "maelstrom-defaults-v4");
        for path in [
            "../../config/defaults-v4.toml",
            "/config/defaults-v4.toml",
            "../config/missing.toml",
            "",
        ] {
            scene.configuration.path = Some(path.into());
            assert!(configuration(&scene).is_err(), "{path}");
        }
        scene.configuration.path = None;
        assert!(configuration(&scene).is_err());
    }

    #[test]
    fn built_in_defaults_do_not_supply_a_configuration_document() {
        let mut scene = document();
        scene.configuration.source = ConfigurationSource::BuiltInDefaults;
        scene.configuration.path = None;
        assert!(
            configuration(&scene)
                .unwrap_or_else(|e| panic!("{}", e.message))
                .is_none()
        );
    }
}
