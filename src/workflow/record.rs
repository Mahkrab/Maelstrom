//! Versioned observation records carry actual inputs, not paths to mutable files.
//! JSON numbers round-trip finite binary32 values; identities and step counters
//! use decimal strings so readers need not represent 64-bit integers as doubles.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

#[cfg(any(feature = "render", test))]
#[path = "snapshot_json.rs"]
mod snapshot_json;

use crate::{
    boundary::Plane,
    config::{
        FieldSource, NumericalModel, ResolvedConfiguration, ResolvedDerivedValues,
        resolve_effective_configuration,
    },
    math::Vec3,
    scene::{PLANE_UNIT_LENGTH_SQUARED_TOLERANCE, Particle, RandomnessPolicy, ResolvedScene},
    simulation::{SerialSimulation, Snapshot, StepError},
};

/// Schema version for result, input and snapshot documents produced here.
pub const SCHEMA_VERSION: u32 = 1;

/// A malformed, unsupported or numerically inconsistent recorded input.
#[derive(Debug)]
pub enum RecordError {
    /// JSON does not match the record structure.
    Json(serde_json::Error),
    /// A version, identity, policy or cross-field invariant is invalid.
    Invalid(String),
    /// Reconstructing the recorded solver inputs failed validation.
    Simulation(StepError),
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid record JSON: {error}"),
            Self::Invalid(message) => write!(f, "invalid record: {message}"),
            Self::Simulation(error) => write!(f, "invalid recorded inputs: {error}"),
        }
    }
}

impl std::error::Error for RecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Simulation(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<serde_json::Error> for RecordError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<StepError> for RecordError {
    fn from(error: StepError) -> Self {
        Self::Simulation(error)
    }
}

/// One particle in a recorded initial state or accepted snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticleRecord {
    #[serde(with = "decimal")]
    identity: u64,
    position_m: Vec3,
    velocity_m_s: Vec3,
}

impl ParticleRecord {
    /// Returns an independent particle value; changing it cannot alter the record.
    #[must_use]
    pub fn particle(&self) -> Particle {
        Particle {
            identity: self.identity,
            position_m: self.position_m,
            velocity_m_s: self.velocity_m_s,
        }
    }
}

impl From<&Particle> for ParticleRecord {
    fn from(p: &Particle) -> Self {
        Self {
            identity: p.identity,
            position_m: p.position_m,
            velocity_m_s: p.velocity_m_s,
        }
    }
}

/// A portable, immutable copy of accepted state. Coordinates are right-handed,
/// positive Y upward; positions use metres and velocities metres per second.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRecord {
    schema_version: u32,
    #[serde(with = "decimal")]
    completed_steps: u64,
    particles: Vec<ParticleRecord>,
}

impl SnapshotRecord {
    /// Copies a solver snapshot into a portable record. Call between steps so
    /// observation remains outside the primary solver timing boundary.
    #[must_use]
    pub fn capture(snapshot: &Snapshot) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            completed_steps: snapshot.completed_steps(),
            particles: snapshot
                .particles()
                .iter()
                .map(ParticleRecord::from)
                .collect(),
        }
    }

    /// Number of fully accepted timesteps represented by this snapshot.
    #[must_use]
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// Particles in strictly ascending stable identity order.
    ///
    /// ```compile_fail
    /// use maelstrom::workflow::SnapshotRecord;
    /// fn replace(snapshot: &SnapshotRecord) {
    ///     snapshot.particles()[0] = snapshot.particles()[1].clone();
    /// }
    /// ```
    #[must_use]
    pub fn particles(&self) -> &[ParticleRecord] {
        &self.particles
    }

    /// Encodes a complete standalone snapshot, rejecting invalid decoded records.
    pub fn to_json(&self) -> Result<String, RecordError> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Decodes and validates a snapshot without installing it into a solver.
    pub fn from_json(json: &str) -> Result<Self, RecordError> {
        let value: Self = serde_json::from_str(json)?;
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), RecordError> {
        version(self.schema_version)?;
        validate_particles(&self.particles)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigurationRecord {
    schema_version: u32,
    name: String,
    description: String,
    numerical: NumericalModel,
    derived: ResolvedDerivedValues,
    provenance: BTreeMap<String, FieldSource>,
    comparison_key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlaneRecord {
    normal: Vec3,
    offset_m: f32,
}

/// Full resolved physical input, with display metadata and provenance kept apart
/// from exact comparison keys. No external file is needed to reproduce it.
/// Deserialisation alone is structural; [`Self::resolve`] checks its semantics.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedInput {
    schema_version: u32,
    scene_schema_version: u32,
    name: String,
    description: String,
    randomness: RandomnessPolicy,
    configuration: ConfigurationRecord,
    planes: Vec<PlaneRecord>,
    particles: Vec<ParticleRecord>,
    comparison_key: String,
}

impl RecordedInput {
    pub(super) fn solver_iterations(&self) -> u32 {
        self.configuration.numerical.solver_iterations
    }
    pub(super) fn display_names(&self) -> (&str, &str) {
        (&self.name, &self.configuration.name)
    }

    pub(crate) fn capture(scene: &ResolvedScene, solver: &SerialSimulation) -> Self {
        let config = solver.configuration();
        let mut ordered = scene.clone();
        ordered.particles = solver.snapshot().particles().to_vec();
        Self {
            schema_version: SCHEMA_VERSION,
            scene_schema_version: scene.schema_version,
            name: scene.name.clone(),
            description: scene.description.clone(),
            randomness: scene.randomness,
            configuration: ConfigurationRecord {
                schema_version: config.schema_version,
                name: config.name.clone(),
                description: config.description.clone(),
                numerical: config.numerical.clone(),
                derived: config.derived,
                provenance: config
                    .sources
                    .iter()
                    .filter(|(key, source)| {
                        **key != "xsph_viscosity_coefficient"
                            || config.numerical.xsph_viscosity_enabled
                            || config.numerical.xsph_viscosity_coefficient.to_bits()
                                != crate::config::DEFAULT_XSPH_VISCOSITY_COEFFICIENT.to_bits()
                            || **source != FieldSource::BuiltInDefault
                    })
                    .filter(|(key, source)| {
                        **key != "vorticity_confinement_strength_m_s"
                            || config.numerical.vorticity_confinement_enabled
                            || config
                                .numerical
                                .vorticity_confinement_strength_m_s
                                .to_bits()
                                != crate::config::DEFAULT_VORTICITY_CONFINEMENT_STRENGTH_M_S
                                    .to_bits()
                            || **source != FieldSource::BuiltInDefault
                    })
                    .map(|(&key, &value)| (key.to_owned(), value))
                    .collect(),
                comparison_key: config.comparison_key(),
            },
            planes: scene
                .planes
                .iter()
                .map(|plane| PlaneRecord {
                    normal: plane.normal(),
                    offset_m: plane.offset(),
                })
                .collect(),
            particles: ordered.particles.iter().map(ParticleRecord::from).collect(),
            comparison_key: ordered.comparison_key(),
        }
    }

    /// Exact physical input key, excluding names, descriptions and provenance.
    #[must_use]
    pub fn comparison_key(&self) -> &str {
        &self.comparison_key
    }

    /// Exact resolved numerical key, including policies and derived binary32 bits.
    #[must_use]
    pub fn configuration_key(&self) -> &str {
        &self.configuration.comparison_key
    }

    /// Reconstructs and validates the recorded values without applying defaults
    /// or re-resolving external files. Derived values must still match exactly.
    pub fn resolve(&self) -> Result<ResolvedScene, RecordError> {
        version(self.schema_version)?;
        scene_version(self.scene_schema_version)?;
        if self.scene_schema_version != self.configuration.schema_version {
            return Err(RecordError::Invalid(
                "scene and configuration versions must match".into(),
            ));
        }
        validate_particles(&self.particles)?;
        // The registry supplies only static field names; every value and its
        // original source come from the record, not today's default settings.
        let mut registry = resolve_effective_configuration(None, None, None)
            .map_err(StepError::from)?
            .sources;
        registry.remove("vorticity_confinement_strength_m_s");
        if self.configuration.numerical.vorticity_confinement_enabled
            || self
                .configuration
                .numerical
                .vorticity_confinement_strength_m_s
                .to_bits()
                != crate::config::DEFAULT_VORTICITY_CONFINEMENT_STRENGTH_M_S.to_bits()
            || self
                .configuration
                .provenance
                .contains_key("vorticity_confinement_strength_m_s")
        {
            registry.insert(
                "vorticity_confinement_strength_m_s",
                FieldSource::BuiltInDefault,
            );
        }
        registry.remove("correction_scale");
        registry.remove("xsph_viscosity_coefficient");
        if self.configuration.numerical.xsph_viscosity_enabled
            || self
                .configuration
                .numerical
                .xsph_viscosity_coefficient
                .to_bits()
                != crate::config::DEFAULT_XSPH_VISCOSITY_COEFFICIENT.to_bits()
            || self
                .configuration
                .provenance
                .contains_key("xsph_viscosity_coefficient")
        {
            registry.insert("xsph_viscosity_coefficient", FieldSource::BuiltInDefault);
        }
        if self.configuration.numerical.correction_scale != 1.0
            || self
                .configuration
                .provenance
                .contains_key("correction_scale")
        {
            registry.insert("correction_scale", FieldSource::BuiltInDefault);
        }
        if registry.len() != self.configuration.provenance.len() {
            return Err(RecordError::Invalid(
                "incomplete configuration provenance".into(),
            ));
        }
        let mut sources = BTreeMap::new();
        for key in registry.keys() {
            let source = self
                .configuration
                .provenance
                .get(*key)
                .ok_or_else(|| RecordError::Invalid(format!("missing provenance for {key}")))?;
            sources.insert(*key, *source);
        }
        let config = &self.configuration;
        // Historical disabled records omit this inactive setting. Expose its
        // built-in origin in today's resolved model without changing the record.
        sources
            .entry("xsph_viscosity_coefficient")
            .or_insert(FieldSource::BuiltInDefault);
        sources
            .entry("vorticity_confinement_strength_m_s")
            .or_insert(FieldSource::BuiltInDefault);
        let configuration = ResolvedConfiguration {
            schema_version: config.schema_version,
            name: config.name.clone(),
            description: config.description.clone(),
            numerical: config.numerical.clone(),
            derived: config.derived,
            sources,
            refinement: None,
        };
        configuration.validate().map_err(StepError::from)?;
        if configuration.comparison_key() != config.comparison_key {
            return Err(RecordError::Invalid(
                "configuration comparison key disagrees with values".into(),
            ));
        }
        let planes = self
            .planes
            .iter()
            .map(|p| {
                Plane::new(p.normal, p.offset_m, PLANE_UNIT_LENGTH_SQUARED_TOLERANCE)
                    .map_err(StepError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scene = ResolvedScene {
            schema_version: self.scene_schema_version,
            name: self.name.clone(),
            description: self.description.clone(),
            randomness: self.randomness,
            configuration,
            planes,
            particles: self
                .particles
                .iter()
                .map(ParticleRecord::particle)
                .collect(),
        };
        SerialSimulation::from_scene(&scene)?;
        if scene.comparison_key() != self.comparison_key {
            return Err(RecordError::Invalid(
                "scene comparison key disagrees with values".into(),
            ));
        }
        Ok(scene)
    }
}

pub(super) fn version(value: u32) -> Result<(), RecordError> {
    if value == SCHEMA_VERSION {
        Ok(())
    } else {
        Err(RecordError::Invalid(format!(
            "unsupported schema version {value}"
        )))
    }
}

pub(super) fn scene_version(value: u32) -> Result<(), RecordError> {
    if matches!(value, 1 | 2) {
        Ok(())
    } else {
        Err(RecordError::Invalid(format!(
            "unsupported scene schema version {value}"
        )))
    }
}

fn validate_particles(particles: &[ParticleRecord]) -> Result<(), RecordError> {
    let mut previous = None;
    for particle in particles {
        if !particle.position_m.is_finite() || !particle.velocity_m_s.is_finite() {
            return Err(RecordError::Invalid("non-finite particle state".into()));
        }
        if previous.is_some_and(|id| id >= particle.identity) {
            return Err(RecordError::Invalid(
                "particle identities must be unique and ascending".into(),
            ));
        }
        previous = Some(particle.identity);
    }
    Ok(())
}

pub(super) mod decimal {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let text = String::deserialize(deserializer)?;
        let value: u64 = text.parse().map_err(D::Error::custom)?;
        if value.to_string() != text {
            return Err(D::Error::custom("expected canonical unsigned decimal"));
        }
        Ok(value)
    }
}

pub(super) mod decimal_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    #[derive(Deserialize)]
    struct Number(#[serde(with = "super::decimal")] u64);

    pub fn serialize<S: Serializer>(values: &[u64], serializer: S) -> Result<S::Ok, S::Error> {
        values
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u64>, D::Error> {
        Ok(Vec::<Number>::deserialize(deserializer)?
            .into_iter()
            .map(|n| n.0)
            .collect())
    }
}
