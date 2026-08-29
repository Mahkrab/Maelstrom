//! TODO: Add top level document documentation. have fun future me!

use serde::{Deserialize, Serialize};

use crate::config::{CONFIG_SCHEMA_VERSION, ConfigError, ConfigurationOverlay};

/// Chooses the numerical model that a scene is resolved against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigurationSource {
    /// Start from the simulator's built-in numerical defaults.
    BuiltInDefaults,
    /// Start from the configuration document nameth [`ConfigurationSelection::path`].
    File,
}

/// Connects a scene to a base configuration and its scene-specific .changes. 
///
/// A file selection needs a relative path. a scene overlay is applied after
/// that base configuration and before any command-line override.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationSelection {
    /// Source from which resolution obtains the base numerical configuration.
    pub source: ConfigurationSource,
    /// Relative path required when [`Self::source`] is [`ConfigurationSource::File`].
    #[serde(default)]
    pub path: Option<String>,
    /// Values that this scene changes without altering the selected base document.
    #[serde(default)]
    pub overrides: Option<ConfigurationOverlay>,
}

/// Defines which endpoint a lattice construction includes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeEndpointRule {
    /// The origin is included and, `origin + count * spacing` is excluded.
    CountedHalfOpen,
}

/// Defines the stable order used to assign lattice identities and positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeOrder {
    /// `x` is outermost, then `y` ad `z` varies fastest.
    XThenYThenZ,
}

/// States whether building the scene may introduce random values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomnessPolicy { // Bit random eh
    /// No random generation is performed.
    None,
}

/// One particle written directly into a scene document.
///
/// Explicit particles retain this identity and document order, so their
/// identities must be strictly ascending during resolution.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParticleDocument {
    /// Stable particle identity.
    pub identity: u64,
    /// Initial position in metres.
    pub position_m: [f32; 3],
    /// Initial velocity in metres per second.
    pub velocity_m_s: [f32; 3],
}

/// HGow a scene obtains its initial particles without simulation work.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "construction", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParticleSource {
    /// Particles listed directly in document order.
    Explicit {
        /// Particle definitions whichs identities must be ascending.
        #[serde(default)]
        items: Vec<ParticleDocument>,
    },
    /// Particles generated from a regular lattice in the declared order. <sub>mmm lettuce</sub>
    Lattice {
        /// Identity assigned to the first generated particle; later identities increase by one.
        first_identity: u64,
        /// Position of the lattice origin in metres.
        origin_m: [f32; 3],
        /// Number of generated positions along the `x`, `y` and `z` axes.
        counts: [u32; 3],
        /// Centre-to-centre particle spacing in metres.
        spacing_m: f32,
        /// Rule that determines which lattice endpoints are included.
        endpoint_rule: LatticeEndpointRule,
        /// Stable order used to generate identities and positions.
        order: LatticeOrder,
        /// Velocity assigned to every generated particle in metres per second. TODO
        initial_velocity_m_s: [f32; 3],
    },
}

/// One static half-space constraint, in document order.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlaneDocument {
    /// Unit normal pointing into the permitted half-space.
    pub normal: [f32; 3],
    /// Offset in metres: permitted positions satisfy `normal · position >= offset_m`.
    pub offset_m: f32,
}

/// Parsed scene data before configuration, geometry and particle checks are resolved.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    /// Version of the scene document schema.
    pub schema_version: u32,
    /// Display name
    pub name: String,
    /// Display description
    pub description: String,
    /// Randomness policy carried into the resolved scene.
    pub randomness: RandomnessPolicy,
    /// Permits explicitly listed particles at exactly the same position, near overlaps still fail.
    #[serde(default)]
    pub allow_coincident_particles: bool,
    /// Numerical configuration selected before particle construction.
    pub configuration: ConfigurationSelection,
    /// Deterministic source of the initial particles.
    pub particles: ParticleSource,
    /// Static half-space constraints in document order.
    #[serde(default)]
    pub planes: Vec<PlaneDocument>,
}

/// Parses a scene document and checks its schema and identifying text.
///
/// Numerical configuration, geometry and particle construction are checked by
/// [`crate::scene::resolve_scene`].
///
/// ### Errors
///
/// Returns [`ConfigError`] for invalid TOML, unknown fields, an unsupported
/// schema version, or invalid identifying text.
pub fn parse_scene(input: &str) -> Result<SceneDocument, ConfigError> {
    let document: SceneDocument = toml::from_str(input).map_err(|error| ConfigError::Parse {
        document: "scene",
        message: error.message().to_owned(),
    })?;
    if document.schema_version != CONFIG_SCHEMA_VERSION {
        return Err(ConfigError::UnsupportedSchema {
            document: "scene",
            supported: CONFIG_SCHEMA_VERSION,
            found: document.schema_version,
        });
    }
    validate_scene_text("scene name", &document.name)?;
    validate_scene_text("scene description", &document.description)?;
    Ok(document)
}

fn validate_scene_text(field: &'static str, value: &str) -> Result<(), ConfigError> {
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
