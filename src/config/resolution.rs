//! New input documents select refinement across layers before applying it once.
//! The legacy resolver remains responsible for historical per-layer behaviour.

use super::{
    ConfigError, ConfigurationDocument, ConfigurationOverlay, FieldSource, ResolvedConfiguration,
    resolved::{apply_overrides, default_model, default_sources},
    validation::{derive_values, validate_assertions, validate_model},
};

/// The selected transformation and the primary values on which it operated.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Refinement {
    /// Selected integer refinement, including one when explicitly reset.
    pub factor: u32,
    /// Origin of the factor, independently of the primary values' origins.
    pub source: FieldSource,
    /// Timestep before applying the selected factor, in seconds.
    pub unrefined_timestep_s: f32,
    /// Complete correction scale before applying the selected factor.
    pub unrefined_correction_scale: f32,
    /// Origin of the primary timestep before refinement.
    pub timestep_source: FieldSource,
    /// Origin of the primary correction scale before refinement.
    pub correction_scale_source: FieldSource,
}

/// Resolves the input schema explicitly, including built-in-only scenes.
///
/// # Errors
/// Rejects unsupported or mixed schemas, invalid supplied primary values and
/// assertions that disagree with the final derived values.
pub fn resolve_configuration_for_schema(
    schema: u32,
    file: Option<&ConfigurationDocument>,
    scene: Option<&ConfigurationOverlay>,
    command_line: Option<&ConfigurationOverlay>,
) -> Result<ResolvedConfiguration, ConfigError> {
    super::document::validate_schema(schema, "configuration")?;
    if file.is_some_and(|file| file.schema_version != schema) {
        return Err(ConfigError::InvalidField {
            field: "schema_version",
            reason:
                "scene and selected configuration versions must match; migrate with scenes create"
                    .into(),
        });
    }
    if schema == 1 {
        return super::resolve_effective_configuration(file, scene, command_line);
    }
    let mut numerical = default_model();
    let mut sources = default_sources();
    let mut factor = 1;
    let mut factor_source = FieldSource::BuiltInDefault;
    let layers = [
        (
            "configuration file",
            FieldSource::ConfigurationFile,
            file.map(|file| &file.overlay),
        ),
        ("scene", FieldSource::Scene, scene),
        (
            "command-line override",
            FieldSource::CommandLine,
            command_line,
        ),
    ];
    for (_, source, layer) in layers {
        if let Some(layer) = layer {
            // Validate every supplied field independently, so a higher layer
            // cannot hide invalid data in an otherwise unused lower layer.
            validate_supplied(layer)?;
            apply_overrides(&layer.numerical, source, &mut numerical, &mut sources);
            if let Some(selected) = layer.numerical.refine_timestep {
                if !(1..=16).contains(&selected) {
                    return Err(ConfigError::InvalidField {
                        field: "refine_timestep",
                        reason: "must be in 1..=16".into(),
                    });
                }
                factor = selected;
                factor_source = source;
            }
        }
    }
    let refinement = Refinement {
        factor,
        source: factor_source,
        unrefined_timestep_s: numerical.fixed_timestep_s,
        unrefined_correction_scale: numerical.correction_scale,
        timestep_source: sources
            .get("fixed_timestep_s")
            .copied()
            .unwrap_or(FieldSource::BuiltInDefault),
        correction_scale_source: sources
            .get("correction_scale")
            .copied()
            .unwrap_or(FieldSource::BuiltInDefault),
    };
    if factor != 1 {
        let n = factor as f32;
        numerical.fixed_timestep_s /= n;
        numerical.correction_scale /= n * n;
        sources.insert("fixed_timestep_s", factor_source);
        sources.insert("correction_scale", factor_source);
    }
    validate_model(&numerical)?;
    let derived = derive_values(&numerical)?;
    for (label, _, layer) in layers {
        if let Some(layer) = layer {
            validate_assertions(label, &layer.derived, derived)?;
        }
    }
    Ok(ResolvedConfiguration {
        schema_version: schema,
        name: file.map_or_else(|| "maelstrom-defaults-v4".into(), |f| f.name.clone()),
        description: file.map_or_else(
            || "Built-in binary32 Position-Based Fluids defaults".into(),
            |f| f.description.clone(),
        ),
        numerical,
        derived,
        sources,
        refinement: Some(refinement),
    })
}

fn validate_supplied(layer: &ConfigurationOverlay) -> Result<(), ConfigError> {
    let mut candidate = default_model();
    let mut sources = default_sources();
    apply_overrides(
        &layer.numerical,
        FieldSource::BuiltInDefault,
        &mut candidate,
        &mut sources,
    );
    validate_model(&candidate)
}
