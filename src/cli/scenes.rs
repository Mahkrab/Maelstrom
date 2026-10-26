//! Editable packages retain scene-local configuration and the original geometry.
use super::{
    CliError,
    arguments::{Cli, Inputs},
    bundled, inputs,
    storage::{self, Storage},
};
use serde_json::json;
use std::fs;

pub(super) fn list(cli: &Cli, storage: &Storage) -> Result<u8, CliError> {
    let mut entries = Vec::new();
    for (name, _) in bundled::SCENES {
        entries.push((name.to_string(), "builtin", None));
    }
    match fs::read_dir(&storage.scenes) {
        Ok(items) => {
            for item in items {
                let item = item.map_err(|e| CliError::input(e.to_string()))?;
                if item
                    .file_type()
                    .map_err(|e| CliError::input(e.to_string()))?
                    .is_dir()
                    && item.path().join("scene.toml").is_file()
                {
                    let name = item
                        .file_name()
                        .into_string()
                        .map_err(|_| CliError::input("scene name must be UTF-8".into()))?;
                    storage::safe_name(&name)?;
                    entries.push((name, "user", Some(item.path().join("scene.toml"))));
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(CliError::input(format!("scene library: {e}"))),
    }
    entries.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    let mut rows = Vec::new();
    let mut text = String::new();
    for (name, origin, path) in entries {
        let selection = format!("{origin}:{name}");
        let resolved = inputs::resolve_in(
            &Inputs {
                scene: Some(selection.clone()),
                config: None,
                overrides: vec![],
                refine_timestep: None,
            },
            storage,
        )?;
        let scene = resolved
            .scene
            .ok_or_else(|| CliError::input("missing listed scene".into()))?;
        text.push_str(&format!(
            "{selection}: {} particles — {}\n",
            scene.particles.len(),
            scene.description
        ));
        rows.push(json!({"name":name,"origin":origin,"selector":selection,"path":path,"particle_count":scene.particles.len(),"description":scene.description}));
    }
    super::print_document(cli.format, text, json!({"schema_version":1,"scenes":rows}))?;
    Ok(0)
}
pub(super) fn create(cli: &Cli, storage: &Storage, name: &str, from: &str) -> Result<u8, CliError> {
    storage::safe_name(name)?;
    let selection = Inputs {
        scene: Some(from.into()),
        config: None,
        overrides: vec![],
        refine_timestep: None,
    };
    let resolved = inputs::resolve_in(&selection, storage)?;
    let presentation = super::presentation::resolve(Some(&resolved), &Default::default())?;
    let render_text = presentation
        .effective
        .to_toml()
        .map_err(|e| CliError::input(e.to_string()))?;
    let mut document = resolved
        .document
        .clone()
        .ok_or_else(|| CliError::input("missing source scene".into()))?;
    document.schema_version = 2;
    if document.run.is_none() {
        document.run = Some(maelstrom::scene::RunDefaults {
            steps: None,
            duration_s: Some(1.0),
        });
    }
    inherit_spacing(&mut document.particles);
    document.name = name.into();
    document.configuration.source = maelstrom::scene::ConfigurationSource::File;
    document.configuration.path = Some("config.toml".into());
    document.configuration.overrides = None;
    let scene_text =
        toml::to_string_pretty(&document).map_err(|e| CliError::input(e.to_string()))?;
    // Inspection includes provenance and derived diagnostics, not an input
    // document. Write primary values only, with the input schema's vector shape.
    let acceleration = resolved.configuration.numerical.external_acceleration_m_s2;
    let mut configuration = json!({
        "schema_version":2,
        "name":resolved.configuration.name,
        "description":resolved.configuration.description,
        "numerical":resolved.configuration.numerical,
    });
    configuration["numerical"]["external_acceleration_m_s2"] =
        json!([acceleration.x, acceleration.y, acceleration.z]);
    let mut assertions = serde_json::to_value(resolved.configuration.derived)
        .map_err(|e| CliError::input(e.to_string()))?;
    if let Some(fields) = assertions.as_object_mut() {
        fields.remove("lattice_offset_count");
    }
    let mut assertion_replacements = Vec::new();
    let mut asserted_fields = std::collections::BTreeSet::new();
    for (label, original) in [
        (
            "configuration",
            resolved
                .selected_configuration
                .as_ref()
                .map(|c| &c.overlay.derived),
        ),
        (
            "scene",
            resolved
                .document
                .as_ref()
                .and_then(|s| s.configuration.overrides.as_ref())
                .map(|o| &o.derived),
        ),
    ] {
        if let Some(original) = original {
            let value =
                serde_json::to_value(original).map_err(|e| CliError::input(e.to_string()))?;
            if let Some(fields) = value.as_object() {
                for (field, old) in fields {
                    if !old.is_null() {
                        asserted_fields.insert(field.clone());
                    }
                    if !old.is_null() && *old != assertions[field] {
                        assertion_replacements.push(json!({"layer":label,"field":field,"previous":old,"effective":assertions[field]}));
                    }
                }
            }
        }
    }
    if let Some(fields) = assertions.as_object_mut() {
        fields.retain(|key, _| asserted_fields.contains(key));
    }
    if !asserted_fields.is_empty() {
        configuration["derived"] = assertions;
    }
    let config_text =
        toml::to_string_pretty(&configuration).map_err(|e| CliError::input(e.to_string()))?;
    fs::create_dir_all(&storage.scenes)
        .map_err(|e| CliError::output(format!("create scene library: {e}")))?;
    let target = storage.scenes.join(name);
    fs::create_dir(&target)
        .map_err(|e| CliError::output(format!("create new scene package: {e}")))?;
    // The directory was exclusively created. On I/O failure retain the partial
    // package for inspection rather than deleting potentially valuable data.
    fs::write(target.join("config.toml"), config_text)
        .map_err(|e| CliError::output(e.to_string()))?;
    fs::write(target.join("scene.toml"), scene_text)
        .map_err(|e| CliError::output(e.to_string()))?;
    fs::write(target.join("render.toml"), render_text)
        .map_err(|e| CliError::output(e.to_string()))?;
    if !assertion_replacements.is_empty() && !cli.quiet {
        use std::io::Write;
        writeln!(
            std::io::stderr().lock(),
            "migration replaced source-layer assertions with final effective assertions: {}",
            json!(assertion_replacements)
        )
        .map_err(|e| CliError::output(format!("stderr: {e}")))?;
    }
    super::print_document(
        cli.format,
        format!("Created user:{name}\n{}", target.display()),
        json!({"schema_version":1,"name":name,"path":target,"source":from,"assertion_replacements":assertion_replacements}),
    )?;
    Ok(0)
}

fn inherit_spacing(source: &mut maelstrom::scene::ParticleSource) {
    match source {
        maelstrom::scene::ParticleSource::Lattice { spacing_m, .. } => *spacing_m = None,
        maelstrom::scene::ParticleSource::Groups { groups } => {
            for group in groups {
                inherit_spacing(&mut group.source);
            }
        }
        maelstrom::scene::ParticleSource::Explicit { .. } => {}
    }
}
