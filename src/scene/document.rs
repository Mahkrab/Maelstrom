//! Scene documents describe authoring inputs before geometry and numerical
//! resolution. Version guards retain the meaning of existing single-source scenes.

use serde::{Deserialize, Serialize};

use crate::config::{CURRENT_INPUT_SCHEMA_VERSION, ConfigError, ConfigurationOverlay};

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
pub enum RandomnessPolicy {
    // Bit random eh
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
        /// First identity; omission starts at zero or follows the preceding group.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_identity: Option<u64>,
        /// Position of the lattice origin in metres.
        origin_m: [f32; 3],
        /// Number of generated positions along the `x`, `y` and `z` axes.
        counts: [u32; 3],
        /// Optional spacing assertion in metres; omission inherits numerical spacing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spacing_m: Option<f32>,
        /// Rule that determines which lattice endpoints are included.
        endpoint_rule: LatticeEndpointRule,
        /// Stable order used to generate identities and positions.
        order: LatticeOrder,
        /// Velocity assigned to every generated particle in metres per second. TODO
        initial_velocity_m_s: [f32; 3],
    },
    /// Named initial sources flattened in document order into one fluid model.
    Groups {
        /// Explicit or lattice sources; nested groups are rejected.
        groups: Vec<ParticleGroup>,
    },
}

/// A named initial source sharing the scene's numerical model.
// The tagged source rejects unknown construction fields. Applying that rule to
// the outer table would also reject its flattened construction tag.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ParticleGroup {
    /// Case-sensitive display name, unique within this scene.
    pub name: String,
    /// Construction fields written alongside the name in the group table.
    #[serde(flatten)]
    pub source: ParticleSource,
}

/// An enclosing axis-aligned box. Select bounds or origin and dimensions, in metres.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerBox {
    /// Lower particle-centre bounds, paired with `max_m`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_m: Option<[f32; 3]>,
    /// Upper particle-centre bounds, paired with `min_m`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_m: Option<[f32; 3]>,
    /// Lower corner, paired with `dimensions_m` instead of bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_m: Option<[f32; 3]>,
    /// Three positive extents from the origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions_m: Option<[f32; 3]>,
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
    /// Permits exactly coincident initial particles; other overlaps still fail.
    #[serde(default)]
    pub allow_coincident_particles: bool,
    /// Numerical configuration selected before particle construction.
    pub configuration: ConfigurationSelection,
    /// Deterministic source of the initial particles.
    pub particles: ParticleSource,
    /// Reusable finite run limit, independent of numerical timestep selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunDefaults>,
    /// Enclosing box, expanded before additional explicit planes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerBox>,
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
    if !matches!(document.schema_version, 1 | 2) {
        return Err(ConfigError::UnsupportedSchema {
            document: "scene",
            supported: CURRENT_INPUT_SCHEMA_VERSION,
            found: document.schema_version,
        });
    }
    validate_scene_text("scene name", &document.name)?;
    validate_scene_text("scene description", &document.description)?;
    document.validate_schema_fields()?;
    if let Some(run) = &document.run {
        run.validate()?;
    }
    Ok(document)
}

impl SceneDocument {
    pub(super) fn validate_schema_fields(&self) -> Result<(), ConfigError> {
        if self.schema_version != 1 {
            return Ok(());
        }
        let invalid = if self.run.is_some() {
            Some(("run", "requires scene schema 2"))
        } else if self.container.is_some()
            || matches!(self.particles, ParticleSource::Groups { .. })
        {
            Some((
                "scene",
                "container and particle groups require scene schema 2",
            ))
        } else if matches!(
            self.particles,
            ParticleSource::Lattice {
                spacing_m: None,
                ..
            }
        ) {
            Some(("spacing_m", "is required by scene schema 1"))
        } else if matches!(
            self.particles,
            ParticleSource::Lattice {
                first_identity: None,
                ..
            }
        ) {
            Some(("first_identity", "is required by scene schema 1"))
        } else {
            None
        };
        if let Some((field, reason)) = invalid {
            return Err(ConfigError::InvalidField {
                field,
                reason: reason.into(),
            });
        }
        Ok(())
    }
}

/// A scene's optional run-length selection; higher CLI limits replace it whole.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunDefaults {
    /// Exact number of fixed timesteps, including zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u64>,
    /// Simulated seconds, rounded up using the effective timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
}

impl RunDefaults {
    /// Converts a finite run length to whole effective fixed timesteps.
    ///
    /// # Errors
    /// Rejects invalid timesteps, ambiguous limits and unrepresentable counts.
    pub fn resolved_steps(&self, timestep_s: f32) -> Result<u64, ConfigError> {
        self.validate()?;
        if !timestep_s.is_finite() || timestep_s <= 0.0 {
            return Err(ConfigError::InvalidField {
                field: "fixed_timestep_s",
                reason: "must be positive and finite".into(),
            });
        }
        if let Some(steps) = self.steps {
            return Ok(steps);
        }
        let duration = self.duration_s.ok_or_else(|| ConfigError::InvalidField {
            field: "run.duration_s",
            reason: "missing duration".into(),
        })?;
        let steps = (duration / f64::from(timestep_s)).ceil();
        if !steps.is_finite() || steps >= 18_446_744_073_709_551_616.0 {
            return Err(ConfigError::InvalidField { field:"run.duration_s",reason:"duration must fit a u64 count of fixed timesteps; select steps for an exact count".into() });
        }
        Ok(steps as u64)
    }

    /// Checks that the selection is finite and unambiguous.
    ///
    /// # Errors
    /// Rejects missing or conflicting limits and invalid durations.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.steps.is_some() == self.duration_s.is_some()
            || self.duration_s.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(ConfigError::InvalidField {
                field: "run",
                reason: "select exactly one non-negative steps or finite duration_s".into(),
            });
        }
        Ok(())
    }
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
