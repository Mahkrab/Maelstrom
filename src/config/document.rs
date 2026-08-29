//! # Configuration documents
//!
//! TOML configuration documents are parsed strictly before
//! numerical resolution validation.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use super::{CONFIG_SCHEMA_VERSION, ConfigError};

/// Numerical values that a configration layer may replace.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalOverrides {
    // Lots of optional
    /// Optional scalar representation for all numerical inputs and derived values.
    pub scalar_precision: Option<String>,
    /// Optional coordinate and unit convention for the  configuration and scenes.
    pub coordinate_system: Option<String>,
    /// Optional timestep-selection policy.
    pub timestep_policy: Option<String>,
    /// Optional densitykernel identifier.
    pub density_kernel: Option<String>,
    /// Optional correction-direction kernel identifier.
    pub correction_kernel: Option<String>,
    /// Optional solver-update ordering policy.
    pub solver_update_policy: Option<String>,
    /// Optional external acceleration vector in ms^-2. 
    /// 
    /// <sub>come back when LaTeX is avaliable in inline rustdoc? I know there is a workarund...<sub> 
    pub external_acceleration_m_s2: Option<[f32; 3]>,
    /// Optional fixed timestep in seconds.
    pub fixed_timestep_s: Option<f32>,
    /// Optional dimensionless CFL factor.
    pub cfl_factor: Option<f32>,
    /// Optional centre-to-centre particle spacing in metres.
    pub particle_spacing_m: Option<f32>,
    /// Optional multiverse of madness ratio of support radius to particle spacing.
    pub support_radius_ratio: Option<f32>,
    /// Optional physical rest density in kilograms per cubic metre.
    pub physical_rest_density_kg_m3: Option<f32>,
    /// Optional number of solver iterations performed for each timestep.
    pub solver_iterations: Option<u32>,
    /// Optional dimensionless scale used to derive relaxation.
    pub relaxation_scale: Option<f32>,
    /// Optional switch for the artificial-pressure correction.
    pub artificial_pressure_enabled: Option<bool>,
    /// Optional dimensionless scale used to derive artificial-pressure strength.
    pub artificial_pressure_strength_scale: Option<f32>,
    /// Optional ratio of artificial-pressure reference separation to support radius.
    pub artificial_pressure_reference_separation_ratio: Option<f32>,
    /// Optional exponent applied by the artificial-pressure correction.
    pub artificial_pressure_exponent: Option<f32>,
    /// Optional kernel-support comparison policy.
    pub support_policy: Option<String>,
    /// Optional boundary model.
    pub boundary_model: Option<String>,
    /// Optional hello neighbour-set rebuild policy.
    pub neighbour_rebuild_policy: Option<String>,
    /// Optional serial particle-ordering policy.
    pub particle_order_policy: Option<String>,
    /// Optional switch for XSPH viscosity, currently only `false` is supported.
    pub xsph_viscosity_enabled: Option<bool>,
    /// Optional switch for vorticity confinement, currently only `false` is supported.
    pub vorticity_confinement_enabled: Option<bool>,
}

/// Optional exact values used to verify derived configuration values.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedAssertions {
    /// Optional expected particle diameter in metres.
    pub particle_diameter_m: Option<f32>,
    /// Optional expected kernel support radius in metres.
    pub support_radius_m: Option<f32>,
    /// Optional expected mass-divided rest density in inverse cubic metres.
    pub normalised_rest_density_per_m3: Option<f32>,
    /// Optional expected particle mass in kilograms.
    pub particle_mass_kg: Option<f32>,
    /// Optional expected relaxation value in inverse square metres.
    pub relaxation_per_m2: Option<f32>,
    /// Optional expected artificial-pressure strength in square metres.
    pub artificial_pressure_strength_m2: Option<f32>,
    /// Optional expected artificial-pressure reference separation in metres.
    pub artificial_pressure_reference_separation_m: Option<f32>,
}

/// A numerical overlay added by a scene or supplied by a later caler.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationOverlay {
    /// Primary numerical values that replace values from lower-precedence layers.
    #[serde(default)]
    pub numerical: NumericalOverrides,
    /// Exact checks on values derived after this overlay is applied.
    #[serde(default)]
    pub derived: DerivedAssertions,
}

/// Configuration document schema.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationDocument {
    /// Version of the configuration document schema.
    pub schema_version: u32,
    /// Human-readable configuration name.
    pub name: String,
    /// Human-readable description of the configuration.
    pub description: String,
    /// Numerical values and derived assertions contained in this document.
    #[serde(flatten)]
    pub overlay: ConfigurationOverlay,
}

/// Parses and validates the structural part of a configuration document.
///
/// ### Errors
///
/// Returns [`ConfigError`] for invalid TOML, unknown fields, unsupported schema
/// versions, or missing identifying texts.
pub fn parse_configuration(input: &str) -> Result<ConfigurationDocument, ConfigError> {
    let document: ConfigurationDocument =
        toml::from_str(input).map_err(|error| ConfigError::Parse {
            document: "configuration",
            message: error.message().to_owned(),
        })?;
    validate_schema(document.schema_version, "configuration")?;
    validate_text("name", &document.name)?;
    validate_text("description", &document.description)?;
    Ok(document)
}

/// Loads and parses a configuration document from a specified path.
///
/// ### Errors
///
/// Returns [`ConfigError`] when the file cannot be read or its contents do not
/// satisfy the configuration schema.
pub fn load_configuration(path: &Path) -> Result<ConfigurationDocument, ConfigError> {
    let input = fs::read_to_string(path).map_err(|error| ConfigError::Io {
        document: "configuration document",
        operation: "read",
        kind: error.kind(),
    })?;
    parse_configuration(&input)
}

/// Parses a standalone configuration overlay using the same strict field set.
///
/// ### Errors
///
/// Returns [`ConfigError`] for invalid TOML or unknown fields.
pub fn parse_configuration_overlay(input: &str) -> Result<ConfigurationOverlay, ConfigError> {
    toml::from_str(input).map_err(|error| ConfigError::Parse {
        document: "configuration overlay",
        message: error.message().to_owned(),
    })
}

pub(crate) fn validate_schema(found: u32, document: &'static str) -> Result<(), ConfigError> {
    if found == CONFIG_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(ConfigError::UnsupportedSchema {
            document,
            supported: CONFIG_SCHEMA_VERSION,
            found,
        })
    }
}

pub(crate) fn validate_text(field: &'static str, value: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        return Err(ConfigError::InvalidField {
            field,
            reason: "must not be empty".to_owned(),
        });
    }
    if value.chars().any(char::is_control) {
        return Err(ConfigError::InvalidField {
            field,
            reason: "must not contain control characters".to_owned(),
        });
    }
    Ok(())
}
