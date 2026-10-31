//! Viewer input selection reuses the headless resolver and its existing precedence.
//! Paths are input selectors; public feedback uses labels rather than personal paths.

use super::{
    CliError,
    arguments::{Inputs, View},
    inputs,
};
use maelstrom::{
    application::RunSpec, execution::ExecutionMode, neighbours::SearchBackend,
    render::GraphicsSelection, viewer::Draft,
};
use std::fs;

pub(super) fn launch(view: &View, storage: &super::storage::Storage) -> Result<u8, CliError> {
    let resolved = inputs::resolve_in(&view.inputs, storage)?;
    let options = super::selection::resolve(&view.selection, &storage.preferences()?)?;
    let steps = resolved.steps(&view.limit)?;
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
    let presentation_options = super::arguments::PresentationOptions {
        render_config: view.render_config.clone(),
        width: view.width,
        height: view.height,
        fit_camera: view.fit_camera,
        no_fit_camera: view.no_fit_camera,
        ..Default::default()
    };
    let selected = super::presentation::resolve(Some(&resolved), &presentation_options)?;
    let fit_camera = selected.effective.camera_mode == maelstrom::presentation::CameraMode::Fit;
    let settings = selected.effective.settings;
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
            numerical_mode: view.inputs.numerical_mode.clone().unwrap_or_default(),
            workers: workers.to_string(),
            device: device.to_string(),
            block_size: block_size.to_string(),
            steps: steps.to_string(),
        };
    let overrides = view.inputs.overrides.clone();
    let refinement = view.inputs.refine_timestep;
    let visual_storage = storage.clone();
    let camera_output = view.camera_output.clone();
    let presentation_resolver: maelstrom::viewer::PresentationResolver = Box::new(move |draft| {
        resolve_presentation(
            draft,
            &visual_storage,
            &presentation_options,
            camera_output.as_deref(),
        )
    });
    let storage = storage.clone();
    let paused = view.paused;
    maelstrom::viewer::run_with_presentation(
        draft,
        settings,
        GraphicsSelection {
            index: view.graphics_adapter,
            software: view.software,
        },
        fit_camera,
        Box::new(move |draft| {
            let mut spec = resolve_in(draft, &overrides, refinement, &storage)?;
            spec.initially_paused = paused;
            Ok(spec)
        }),
        presentation_resolver,
    )
    .map_err(|e| CliError::input(e.to_string()))?;
    Ok(0)
}
fn resolve_presentation(
    draft: &Draft,
    storage: &super::storage::Storage,
    options: &super::arguments::PresentationOptions,
    camera_output: Option<&std::path::Path>,
) -> Result<maelstrom::viewer::ViewPresentation, String> {
    let selected = storage
        .scene(&draft.scene)
        .map_err(|e| redact(e.message, draft))?;
    let path = selected
        .map(fs::canonicalize)
        .transpose()
        .map_err(|e| format!("selected scene directory: {}", e.kind()))?;
    let directory = path.as_deref().and_then(std::path::Path::parent);
    let presentation = match directory {
        Some(directory) => super::presentation::resolve_directory(Some(directory), options),
        None => super::presentation::resolve_bundled(
            draft.scene.strip_prefix("builtin:").unwrap_or(&draft.scene),
            options,
        ),
    }
    .map_err(|e| redact(e.message, draft))?
    .effective;
    let camera_save = directory
        .map(|directory| {
            let default = directory.join("render.toml");
            maelstrom::presentation::CameraSaveTarget::new(camera_output.unwrap_or(&default))
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    Ok(maelstrom::viewer::ViewPresentation {
        presentation,
        camera_save,
    })
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
        numerical_mode: (!draft.numerical_mode.is_empty()).then(|| draft.numerical_mode.clone()),
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
    #[test]
    fn presentation_saves_to_scene_unless_an_output_is_explicitly_selected() {
        let directory = std::env::temp_dir().join(format!(
            "maelstrom-viewer-camera-selection-{}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let scene = directory.join("scene.toml");
        fs::write(&scene, "").unwrap();
        let replacement = directory.join("replacement.toml");
        let source = "schema_version=2\nwidth=123\n";
        fs::write(&replacement, source).unwrap();
        let storage = super::super::storage::Storage::new(None, None)
            .unwrap_or_else(|e| panic!("{}", e.message));
        let mut d = draft();
        d.scene = scene.to_string_lossy().into_owned();
        let options = super::super::arguments::PresentationOptions {
            render_config: Some(replacement.clone()),
            ..Default::default()
        };
        let selected = resolve_presentation(&d, &storage, &options, None).unwrap();
        assert_eq!(selected.presentation.settings.width, 123);
        let local = directory.join("render.toml");
        assert!(!local.exists());
        selected
            .camera_save
            .unwrap()
            .save(&Default::default(), 1.)
            .unwrap();
        assert!(local.is_file());
        assert_eq!(fs::read_to_string(&replacement).unwrap(), source);
        let output = directory.join("chosen.toml");
        let selected = resolve_presentation(&d, &storage, &options, Some(&output)).unwrap();
        assert!(!output.exists());
        selected
            .camera_save
            .unwrap()
            .save(&Default::default(), 1.)
            .unwrap();
        assert!(output.is_file());
        d.scene = "builtin:falling-block-04".into();
        assert!(
            resolve_presentation(&d, &storage, &options, Some(&output))
                .unwrap()
                .camera_save
                .is_none()
        );
        fs::remove_dir_all(directory).unwrap();
    }
    fn draft() -> Draft {
        Draft {
            scene: "falling-block-04".into(),
            config: String::new(),
            mode: "serial-grid".into(),
            numerical_mode: String::new(),
            workers: "2".into(),
            device: "0".into(),
            block_size: "128".into(),
            steps: "3".into(),
        }
    }
    #[test]
    fn draft_numerical_mode_is_validated_and_refines_once() {
        let mut d = draft();
        d.numerical_mode = "fast".into();
        let fast = resolve(&d, &[], None).unwrap();
        assert_eq!(
            fast.scene.configuration.numerical.fixed_timestep_s,
            1.0 / 240.0
        );
        assert_eq!(fast.scene.configuration.numerical.correction_scale, 0.5);
        assert_eq!(fast.scene.configuration.numerical.relaxation_scale, 2.0);
        let refined = resolve(&d, &[], Some(2)).unwrap();
        assert_eq!(
            refined.scene.configuration.numerical.correction_scale,
            0.125
        );
        assert_eq!(
            refined.scene.configuration.numerical.fixed_timestep_s,
            1.0 / 480.0
        );
        d.numerical_mode = "reference".into();
        let reference = resolve(&d, &[], None).unwrap();
        assert_eq!(
            reference.scene.configuration.numerical.correction_scale,
            0.0625
        );
        assert_eq!(
            reference.scene.configuration.numerical.relaxation_scale,
            20.0
        );
        d.numerical_mode = "experimental".into();
        let experimental = resolve(&d, &[], None).unwrap();
        assert_eq!(
            experimental.scene.configuration.numerical.solver_iterations,
            8
        );
        assert_eq!(
            experimental.scene.configuration.numerical.relaxation_scale,
            20.0
        );
        assert_eq!(
            experimental.scene.configuration.numerical.correction_scale,
            0.25
        );
        d.numerical_mode = "unknown".into();
        assert!(
            resolve(&d, &[], None)
                .unwrap_err()
                .contains("must be reference, fast")
        );
    }
    #[test]
    fn external_and_bundled_selection_share_precedence_and_identity() {
        let overrides = vec!["cfl_factor=0.1".into()];
        let mut d = draft();
        // Saved local scenes keep their numerics by default. Select the same
        // explicit policy before comparing full provenance across both routes.
        d.numerical_mode = "reference".into();
        let bundled = resolve(&d, &overrides, None).unwrap();
        d.scene = "scenes/falling-block-04/scene.toml".into();
        let external = resolve(&d, &overrides, None).unwrap();
        assert_eq!(external.scene, bundled.scene);
        assert_eq!(external.scene.configuration.numerical.cfl_factor, 0.1);
        d.config = "scenes/falling-block-04/config.toml".into();
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
        assert!(error.contains("scene") && error.contains("not found"));
    }
    #[test]
    fn scene_selected_configuration_errors_hide_the_external_input_directory() {
        let directory = std::env::temp_dir().join(format!(
            "maelstrom-viewer-private-input-{}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("scene.toml");
        let source = include_str!("../../reference/scenes/falling-block-04.toml")
            .replace("../config/defaults-v4.toml", "missing-config.toml");
        fs::write(&path, source).unwrap();
        let mut d = draft();
        d.scene = path.to_string_lossy().into_owned();
        let message = resolve(&d, &[], None).unwrap_err();
        fs::remove_file(path).unwrap();
        fs::remove_dir(&directory).unwrap();
        assert!(!message.contains(&directory.to_string_lossy().to_string()));
        assert!(
            message.contains("scene.toml") && message.contains("missing-config.toml"),
            "{message}"
        );
    }
}
