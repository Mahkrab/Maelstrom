//! Viewer input selection reuses the headless resolver and its existing precedence.
//! Paths are input selectors; public feedback uses labels rather than personal paths.

use super::{
    CliError,
    arguments::{Inputs, View},
    inputs,
};
use maelstrom::{
    application::RunSpec,
    execution::ExecutionMode,
    neighbours::SearchBackend,
    render::{GraphicsSelection, Settings},
    viewer::Draft,
};
use std::fs;

pub(super) fn launch(view: &View, storage: &super::storage::Storage) -> Result<u8, CliError> {
    let resolved = inputs::resolve_in(&view.inputs, storage)?;
    let options = super::selection::resolve(&view.selection, &storage.preferences()?)?;
    let steps = super::run_limit(
        view.limit.steps,
        view.limit.duration,
        resolved.configuration.numerical.fixed_timestep_s,
    )?;
    let (mode, workers, device, block_size) = match options.execution {
        ExecutionMode::Cuda { device, threads } => ("cuda-grid", 4, device, threads),
        ExecutionMode::Threaded { workers } => ("threaded-grid", workers, 0, 128),
        ExecutionMode::Serial => (
            if options.backend == SearchBackend::BruteForce {
                "brute-force"
            } else {
                "serial-grid"
            },
            4,
            0,
            128,
        ),
    };
    let mut settings = match &view.render_config {
        Some(path) => Settings::parse(&fs::read_to_string(path).map_err(|e| {
            CliError::input(format!("could not read selected render file: {}", e.kind()))
        })?)
        .map_err(|e| CliError::input(e.to_string()))?,
        None => Settings::default(),
    };
    if let Some(width) = view.width {
        settings.width = width;
    }
    if let Some(height) = view.height {
        settings.height = height;
    }
    settings
        .validate()
        .map_err(|e| CliError::input(e.to_string()))?;
    let draft =
        Draft {
            scene: view.inputs.scene.clone().ok_or_else(|| {
                CliError::usage("view requires SCENE; use 'maelstrom scenes'".into())
            })?,
            config: view
                .inputs
                .config
                .as_ref()
                .map_or_else(String::new, |p| p.to_string_lossy().into()),
            mode: mode.into(),
            workers: workers.to_string(),
            device: device.to_string(),
            block_size: block_size.to_string(),
            steps: steps.to_string(),
        };
    let overrides = view.inputs.overrides.clone();
    let refinement = view.inputs.refine_timestep;
    let storage = storage.clone();
    let paused = view.paused;
    maelstrom::viewer::run(
        draft,
        settings,
        GraphicsSelection {
            index: view.graphics_adapter,
            software: view.software,
        },
        view.fit_camera,
        Box::new(move |draft| {
            let mut spec = resolve_in(draft, &overrides, refinement, &storage)?;
            spec.initially_paused = paused;
            Ok(spec)
        }),
    )
    .map_err(|e| CliError::input(e.to_string()))?;
    Ok(0)
}
#[cfg(test)]
fn resolve(
    draft: &Draft,
    overrides: &[String],
    refinement: Option<u32>,
) -> Result<RunSpec, String> {
    let storage = super::storage::Storage::new(None, None).map_err(|e| e.message)?;
    resolve_in(draft, overrides, refinement, &storage)
}
fn resolve_in(
    draft: &Draft,
    overrides: &[String],
    refinement: Option<u32>,
    storage: &super::storage::Storage,
) -> Result<RunSpec, String> {
    let (backend, execution) = match draft.mode.as_str() {
        "brute-force" => (SearchBackend::BruteForce, ExecutionMode::Serial),
        "serial-grid" => (SearchBackend::UniformGrid, ExecutionMode::Serial),
        "threaded-grid" => (
            SearchBackend::UniformGrid,
            ExecutionMode::Threaded {
                workers: draft
                    .workers
                    .parse()
                    .map_err(|_| "workers must be an integer".to_owned())?,
            },
        ),
        "cuda-grid" => (
            SearchBackend::UniformGrid,
            ExecutionMode::Cuda {
                device: draft
                    .device
                    .parse()
                    .map_err(|_| "CUDA device must be a u32 ordinal".to_owned())?,
                threads: draft
                    .block_size
                    .parse()
                    .map_err(|_| "CUDA block must be 32, 64, 128 or 256".to_owned())?,
            },
        ),
        _ => {
            return Err("mode must be brute-force, serial-grid, threaded-grid or cuda-grid".into());
        }
    };
    execution.validate().map_err(|e| e.to_string())?;
    let steps = draft
        .steps
        .parse()
        .map_err(|_| "steps must be a finite u64 count".to_owned())?;
    let selected = Inputs {
        scene: Some(draft.scene.clone()),
        config: (!draft.config.is_empty()).then(|| draft.config.clone().into()),
        overrides: overrides.to_vec(),
        refine_timestep: refinement,
    };
    let resolved = inputs::resolve_in(&selected, storage).map_err(|e| redact(e.message, draft))?;
    let scene = resolved.scene.ok_or_else(|| "select a scene".to_owned())?;
    Ok(RunSpec {
        scene,
        backend,
        execution,
        steps,
        initially_paused: true,
    })
}
fn redact(mut message: String, draft: &Draft) -> String {
    for path in [&draft.scene, &draft.config] {
        if !path.is_empty() {
            message = message.replace(path, "<selected input>");
            // A scene-selected configuration can fail under the scene's own
            // absolute directory, outside both the invocation directory and home.
            if let Ok(selected) = fs::canonicalize(path)
                && let Some(directory) = selected.parent().filter(|p| p.parent().is_some())
            {
                message = message.replace(
                    &directory.to_string_lossy().to_string(),
                    "<input directory>",
                );
                let quoted = format!("{directory:?}");
                message = message.replace(quoted.trim_matches('"'), "<input directory>");
            }
        }
    }
    if let Ok(directory) = std::env::current_dir() {
        message = message.replace(
            &directory.to_string_lossy().to_string(),
            "<invocation directory>",
        );
    }
    if let Some(directory) = std::env::var_os("HOME").filter(|p| !p.is_empty()) {
        message = message.replace(&directory.to_string_lossy().to_string(), "<home>");
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    fn draft() -> Draft {
        Draft {
            scene: "falling-block-04".into(),
            config: String::new(),
            mode: "serial-grid".into(),
            workers: "2".into(),
            device: "0".into(),
            block_size: "128".into(),
            steps: "3".into(),
        }
    }
    #[test]
    fn external_and_bundled_selection_share_precedence_and_identity() {
        let overrides = vec!["cfl_factor=0.1".into()];
        let mut d = draft();
        let bundled = resolve(&d, &overrides, None).unwrap();
        d.scene = "scenes/falling-block-04.toml".into();
        let external = resolve(&d, &overrides, None).unwrap();
        assert_eq!(external.scene, bundled.scene);
        assert_eq!(external.scene.configuration.numerical.cfl_factor, 0.1);
        d.config = "config/defaults-v4.toml".into();
        let selected = resolve(&d, &overrides, None).unwrap();
        assert_eq!(
            selected.scene.configuration.comparison_key(),
            bundled.scene.configuration.comparison_key()
        );
    }
    #[test]
    fn invalid_settings_and_selected_paths_have_explicit_redacted_feedback() {
        let mut d = draft();
        d.workers = "0".into();
        d.mode = "threaded-grid".into();
        assert!(resolve(&d, &[], None).is_err());
        d.mode = "cuda-grid".into();
        d.block_size = "31".into();
        assert!(resolve(&d, &[], None).is_err());
        d.mode = "serial-grid".into();
        d.steps = "-1".into();
        assert!(resolve(&d, &[], None).is_err());
        d.steps = "3".into();
        d.scene = "/unusable/private/selected-scene.toml".into();
        let error = resolve(&d, &[], None).unwrap_err();
        assert!(!error.contains("/unusable/private"));
        assert!(error.contains("<selected input>"));
    }
    #[test]
    fn scene_selected_configuration_errors_hide_the_external_input_directory() {
        let directory = std::env::temp_dir().join(format!(
            "maelstrom-viewer-private-input-{}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("scene.toml");
        let source = include_str!("../../scenes/falling-block-04.toml")
            .replace("../config/defaults-v4.toml", "missing-config.toml");
        fs::write(&path, source).unwrap();
        let mut d = draft();
        d.scene = path.to_string_lossy().into_owned();
        let message = resolve(&d, &[], None).unwrap_err();
        fs::remove_file(path).unwrap();
        fs::remove_dir(&directory).unwrap();
        assert!(!message.contains(&directory.to_string_lossy().to_string()));
        assert!(message.contains("<input directory>"), "{message}");
    }
}
