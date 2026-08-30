//! # Resolved configurations
//!
//! Configuration layers are applied in precedence order to
//! produce validated numerical values.

use std::{collections::BTreeMap, fmt, io};

use crate::{NumericalError, fluid::PbfParameters, math::Vec3};

use super::{
    ConfigurationDocument, ConfigurationOverlay, NumericalOverrides,
    validation::{derive_values, validate_assertions, validate_model},
};

/// Supported configuration schema version.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;
/// Scalar representation used by every numerical input and derived value.
pub const SCALAR_PRECISION: &str = "binary32";
/// Coordinate and unit convention used by configuration and scenes.
pub const COORDINATE_SYSTEM: &str = "right_handed_y_up_metres_seconds_kilograms";
/// Supported timestep selection policy.
pub const TIMESTEP_POLICY: &str = "fixed";
/// Density kernel used by the numerical contract.
pub const DENSITY_KERNEL: &str = "poly6";
/// Correction-direction kernel used by the numerical contract.
pub const CORRECTION_KERNEL: &str = "spiky";
/// Supported solver update ordering.
pub const SOLVER_UPDATE_POLICY: &str = "jacobi_immutable_snapshot";
/// Kernel support comparison used by discovery and kernel evaluation.
pub const SUPPORT_POLICY: &str = "strict_less_than";
/// Supported boundary model.
pub const BOUNDARY_MODEL: &str = "static_plane_projection";
/// Supported neighbour-set lifetime.
pub const NEIGHBOUR_REBUILD_POLICY: &str = "once_after_prediction";
/// Serial ordering policy.
pub const PARTICLE_ORDER_POLICY: &str = "ascending_particle_identity";
/// Largest supported support-to-spacing ratio for bounded lattice derivation.
pub const MAX_SUPPORT_RADIUS_RATIO: f32 = 8.0;
/// Largest supported fixed solver iteration count.
pub const MAX_SOLVER_ITERATIONS: u32 = 64;

const DEFAULT_NAME: &str = "maelstrom-defaults-v1";
const DEFAULT_DESCRIPTION: &str = "Built-in binary32 Position-Based Fluids defaults";

/// Origin of one effective numerical value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldSource {
    /// The built-in default supplied by the simulator.
    BuiltInDefault,
    /// A value supplied by a configuration document.
    ConfigurationFile,
    /// A value supplied by a scene overlay.
    Scene,
    /// A value supplied by a command-line overlay.
    CommandLine,
}

/// Deterministic field-by-field provenance for a resolved numerical model. <sub>*fancy*</sub>
pub type ValueSources = BTreeMap<&'static str, FieldSource>;

/// Primary values and fixed policies used to derive simulator parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericalModel {
    /// Scalar representation used by every numerical value.
    pub scalar_precision: String,
    /// Coordinate and unit convention used by configuration and scenes.
    pub coordinate_system: String,
    /// Policy used to select the timestep.
    pub timestep_policy: String,
    /// Kernel used for density estimation.
    pub density_kernel: String,
    /// Kernel used to determine correction directions.
    pub correction_kernel: String,
    /// Ordering used to update the solver state.
    pub solver_update_policy: String,
    /// External acceleration in metres per second squared.
    ///
    /// TODO: Maybe implement LateX parsing in inline rustdoc.
    pub external_acceleration_m_s2: Vec3,
    /// Fixed timestep in seconds.
    pub fixed_timestep_s: f32,
    /// Dimensionless CFL factor.
    pub cfl_factor: f32,
    /// Centre-to-centre particle spacing in metres.
    pub particle_spacing_m: f32,
    /// Dimensionless ratio of support radius to particle spacing.
    pub support_radius_ratio: f32,
    /// Physical rest density in kilograms per cubic metre.
    pub physical_rest_density_kg_m3: f32,
    /// Number of solver iterations performed for each timestep.
    pub solver_iterations: u32,
    /// Dimensionless scale used to derive relaxation.
    pub relaxation_scale: f32,
    /// Whether to apply the artificial-pressure correction.
    pub artificial_pressure_enabled: bool,
    /// Dimensionless scale used to derive artificial-pressure strength.
    pub artificial_pressure_strength_scale: f32,
    /// Ratio of artificial-pressure reference separation to support radius.
    pub artificial_pressure_reference_separation_ratio: f32,
    /// Exponent applied by the artificial-pressure correction.
    pub artificial_pressure_exponent: f32,
    /// Comparison used to decide whether a particle is within kernel support.
    pub support_policy: String,
    /// Boundary model used to constrain particles.
    pub boundary_model: String,
    /// Policy determining when neighbour sets are rebuilt.
    pub neighbour_rebuild_policy: String,
    /// Policy determining the serial particle-processing order.
    pub particle_order_policy: String,
    /// Whether XSPH viscosity is enabled.
    pub xsph_viscosity_enabled: bool,
    /// Whether vorticity confinement is enabled.
    pub vorticity_confinement_enabled: bool,
}

/// Values calculated from the primary numerical model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedDerivedValues {
    /// Particle diameter, equal to particle spacing, in metres.
    pub particle_diameter_m: f32,
    /// Kernel support radius in metres.
    pub support_radius_m: f32,
    /// Mass-divided rest density in inverse cubic metres.
    pub normalised_rest_density_per_m3: f32,
    /// Particle mass in kilograms.
    pub particle_mass_kg: f32,
    /// Relaxation term in inverse square metres.
    pub relaxation_per_m2: f32,
    /// Artificial-pressure strength in square metres.
    pub artificial_pressure_strength_m2: f32,
    /// Artificial-pressure reference separation in metres.
    pub artificial_pressure_reference_separation_m: f32,
    /// Number of lattice offsets included in the equilibrium density sum.
    pub lattice_offset_count: usize,
}

/// Fully resolved, validated and inspectable numerical confguration.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedConfiguration {
    /// Version of the resolved configuration schema.
    pub schema_version: u32,
    /// Human-readable configuration name.
    pub name: String,
    /// Human-readable configuration description.
    pub description: String,
    /// Validated primary numerical values and policies.
    pub numerical: NumericalModel,
    /// Values derived authoritatively from [`Self::numerical`].
    pub derived: ResolvedDerivedValues,
    /// Source layer for each resolved numerical field.
    pub sources: ValueSources,
}

impl ResolvedConfiguration {
    /// Builds the fluid parameter object consumed by the numerical kernels.
    ///
    /// ### Errors
    ///
    /// Returns [`ConfigError`] if a resolved value cannot be represented by
    /// the numerical parameter type.
    pub fn pbf_parameters(&self) -> Result<PbfParameters, ConfigError> {
        PbfParameters::new(
            self.derived.support_radius_m,
            self.derived.normalised_rest_density_per_m3,
            self.derived.relaxation_per_m2,
            self.derived.artificial_pressure_strength_m2,
            self.derived.artificial_pressure_reference_separation_m,
            self.numerical.artificial_pressure_exponent,
        )
        .map_err(ConfigError::Numerical)
    }
}

/// Configuration parsing, validation, derivation or inspection failure.
#[derive(Debug)]
pub enum ConfigError {
    /// A configuration document could not be read or written.
    Io {
        /// Kind of document involved in the operation.
        document: &'static str,
        /// Filesystem operation that failed.
        operation: &'static str,
        /// Operating-system error kind.
        kind: io::ErrorKind,
    },
    /// TOML input could not be parsed.
    Parse {
        /// Kind of document that could not be parsed.
        document: &'static str,
        /// Parser diagnostic explaining the failure.
        message: String,
    },
    /// A document declares an unsupported schema version.
    UnsupportedSchema {
        /// Kind of document declaring the unsupported version.
        document: &'static str,
        /// Schema version supported by this build.
        supported: u32,
        /// Schema version declared by the document.
        found: u32,
    },
    /// A named field violates its supported domain or policy.
    InvalidField {
        /// Name of the invalid field.
        field: &'static str,
        /// Explanation of the failed validation rule.
        reason: String,
    },
    /// A supplied derived assertion differs from the authoritative value.
    DerivedMismatch {
        /// Overlay layer that supplied the assertion.
        layer: &'static str,
        /// Name of the derived field.
        field: &'static str,
        /// Authoritative derived binary32 value.
        expected: f32,
        /// Derived assertion supplied by the overlay.
        provided: f32,
    },
    /// A numerical operation required while resolving configuration failed.
    Numerical(NumericalError),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                document,
                operation,
                kind,
            } => write!(formatter, "could not {operation} {document}: {kind}"),
            Self::Parse { document, message } => {
                write!(formatter, "invalid {document} TOML: {message}")
            }
            Self::UnsupportedSchema {
                document,
                supported,
                found,
            } => write!(
                formatter,
                "unsupported {document} schema version {found}; supported version is {supported}"
            ),
            Self::InvalidField { field, reason } => write!(formatter, "invalid {field}: {reason}"),
            Self::DerivedMismatch {
                layer,
                field,
                expected,
                provided,
            } => write!(
                formatter,
                "{layer} derived assertion {field} must equal the authoritative binary32 value {expected}, got {provided}"
            ),
            Self::Numerical(error) => write!(formatter, "invalid numerical configuration: {error}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numerical(error) => Some(error),
            _ => None,
        }
    }
}

/// Resolves all configuration layers in their fixed precedence order.
///
/// ### Errors
///
/// Returns [`ConfigError`] for invalid domains, unsupported fixed policies,
/// non-finite derived values, or contradictory derived assertions.
pub fn resolve_effective_configuration(
    file: Option<&ConfigurationDocument>,
    scene: Option<&ConfigurationOverlay>,
    command_line: Option<&ConfigurationOverlay>,
) -> Result<ResolvedConfiguration, ConfigError> {
    let mut numerical = default_model();
    let mut name = DEFAULT_NAME.to_owned();
    let mut description = DEFAULT_DESCRIPTION.to_owned();
    let mut sources = default_sources();

    if let Some(document) = file {
        name = document.name.clone();
        description = document.description.clone();
        apply_layer(
            "configuration file",
            FieldSource::ConfigurationFile,
            &document.overlay,
            &mut numerical,
            &mut sources,
        )?;
    }
    if let Some(overlay) = scene {
        apply_layer(
            "scene",
            FieldSource::Scene,
            overlay,
            &mut numerical,
            &mut sources,
        )?;
    }
    if let Some(overlay) = command_line {
        apply_layer(
            "command-line override",
            FieldSource::CommandLine,
            overlay,
            &mut numerical,
            &mut sources,
        )?;
    }

    validate_model(&numerical)?;
    let derived = derive_values(&numerical)?;
    Ok(ResolvedConfiguration {
        schema_version: CONFIG_SCHEMA_VERSION,
        name,
        description,
        numerical,
        derived,
        sources,
    })
}

fn apply_layer(
    layer: &'static str,
    source: FieldSource,
    overlay: &ConfigurationOverlay,
    numerical: &mut NumericalModel,
    sources: &mut ValueSources,
) -> Result<(), ConfigError> {
    apply_overrides(&overlay.numerical, source, numerical, sources);
    validate_model(numerical)?;
    let derived = derive_values(numerical)?;
    validate_assertions(layer, &overlay.derived, derived)
}

fn apply_overrides(
    overrides: &NumericalOverrides,
    source: FieldSource,
    model: &mut NumericalModel,
    sources: &mut ValueSources,
) {
    macro_rules! apply {
        ($field:ident) => {
            if let Some(value) = overrides.$field.clone() {
                model.$field = value;
                sources.insert(stringify!($field), source);
            }
        };
    }
    apply!(scalar_precision);
    apply!(coordinate_system);
    apply!(timestep_policy);
    apply!(density_kernel);
    apply!(correction_kernel);
    apply!(solver_update_policy);
    if let Some(value) = overrides.external_acceleration_m_s2 {
        model.external_acceleration_m_s2 = Vec3::new(value[0], value[1], value[2]);
        sources.insert("external_acceleration_m_s2", source);
    }
    apply!(fixed_timestep_s);
    apply!(cfl_factor);
    apply!(particle_spacing_m);
    apply!(support_radius_ratio);
    apply!(physical_rest_density_kg_m3);
    apply!(solver_iterations);
    apply!(relaxation_scale);
    apply!(artificial_pressure_enabled);
    apply!(artificial_pressure_strength_scale);
    apply!(artificial_pressure_reference_separation_ratio);
    apply!(artificial_pressure_exponent);
    apply!(support_policy);
    apply!(boundary_model);
    apply!(neighbour_rebuild_policy);
    apply!(particle_order_policy);
    apply!(xsph_viscosity_enabled);
    apply!(vorticity_confinement_enabled);
}

fn default_model() -> NumericalModel {
    NumericalModel {
        scalar_precision: SCALAR_PRECISION.to_owned(),
        coordinate_system: COORDINATE_SYSTEM.to_owned(),
        timestep_policy: TIMESTEP_POLICY.to_owned(),
        density_kernel: DENSITY_KERNEL.to_owned(),
        correction_kernel: CORRECTION_KERNEL.to_owned(),
        solver_update_policy: SOLVER_UPDATE_POLICY.to_owned(),
        external_acceleration_m_s2: Vec3::new(0.0, -9.81, 0.0),
        fixed_timestep_s: 1.0_f32 / 120.0_f32,
        cfl_factor: 0.4,
        particle_spacing_m: 0.05,
        support_radius_ratio: 2.0,
        physical_rest_density_kg_m3: 1000.0,
        solver_iterations: 4,
        relaxation_scale: 1.0e-6,
        artificial_pressure_enabled: true,
        artificial_pressure_strength_scale: 0.1,
        artificial_pressure_reference_separation_ratio: 0.2,
        artificial_pressure_exponent: 4.0,
        support_policy: SUPPORT_POLICY.to_owned(),
        boundary_model: BOUNDARY_MODEL.to_owned(),
        neighbour_rebuild_policy: NEIGHBOUR_REBUILD_POLICY.to_owned(),
        particle_order_policy: PARTICLE_ORDER_POLICY.to_owned(),
        xsph_viscosity_enabled: false,
        vorticity_confinement_enabled: false,
    }
}

fn default_sources() -> ValueSources {
    [
        "scalar_precision",
        "coordinate_system",
        "timestep_policy",
        "density_kernel",
        "correction_kernel",
        "solver_update_policy",
        "external_acceleration_m_s2",
        "fixed_timestep_s",
        "cfl_factor",
        "particle_spacing_m",
        "support_radius_ratio",
        "physical_rest_density_kg_m3",
        "solver_iterations",
        "relaxation_scale",
        "artificial_pressure_enabled",
        "artificial_pressure_strength_scale",
        "artificial_pressure_reference_separation_ratio",
        "artificial_pressure_exponent",
        "support_policy",
        "boundary_model",
        "neighbour_rebuild_policy",
        "particle_order_policy",
        "xsph_viscosity_enabled",
        "vorticity_confinement_enabled",
    ]
    .into_iter()
    .map(|field| (field, FieldSource::BuiltInDefault))
    .collect()
}
