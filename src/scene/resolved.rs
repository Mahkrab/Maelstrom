use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use crate::{
    NumericalError,
    boundary::Plane,
    config::{
        ConfigError, ConfigurationDocument, ConfigurationOverlay, ResolvedConfiguration,
        load_configuration, resolve_effective_configuration,
    },
    math::Vec3,
};

use super::{
    ConfigurationSource, RandomnessPolicy, SceneDocument,
    geometry::{resolve_planes, validate_bounds, validate_plane_compatibility},
    parse_scene,
    particles::resolve_particles,
};

/// Absolute tolerance applied to squared plane-normal length.
pub const PLANE_UNIT_LENGTH_SQUARED_TOLERANCE: f32 = 1.0e-6;
/// Relative allowance for binary32 rounding in minimum-separation checks.
pub const PARTICLE_SEPARATION_RELATIVE_TOLERANCE: f32 = 8.0 * f32::EPSILON;
/// Maximum number of planes in one scene.
pub const MAX_PLANES: usize = 32;
/// Maximum explicit particle count before pairwise overlap validation is impractical.
pub const MAX_EXPLICIT_PARTICLES: usize = 4_096;
/// Maximum deterministic lattice particle count accepted by scene construction.
pub const MAX_LATTICE_PARTICLES: usize = 1_000_000;

/// Accepted initial state for one particle.
///
/// Resolved scenes keep particles in ascending identity order, so callers can
/// compare or replay the same input without reconstructing that order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// Stable particle identity.
    pub identity: u64, // Who are you?? Who am I??
    /// Initial particle position in metres.
    pub position_m: Vec3,
    /// Initial particle velocity in metres per second.
    pub velocity_m_s: Vec3,
}

/// A complete scene input that has passed configuration, geometry and particle validation.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedScene {
    /// Schema version carried from the source document.
    pub schema_version: u32,
    /// Display name, retained for inspection but excluded from comparison identity.
    pub name: String,
    /// Display description, retained for inspection but excluded from comparison identity.
    pub description: String,
    /// Randomness policy used while constructing this input.
    pub randomness: RandomnessPolicy,
    /// Numerical configuration against which the particle and plane data were checked.
    pub configuration: ResolvedConfiguration,
    /// Initial particles in stable identity order.
    pub particles: Vec<Particle>,
    /// Static half-space constraints in document order.
    pub planes: Vec<Plane>,
}

/// Scene loading, construction or validation failure.
#[derive(Debug)]
pub enum SceneError {
    /// Resolving the selected numerical configuration failed.
    Configuration(ConfigError),
    /// A numerical operation required while resolving the scene failed.
    Numerical(NumericalError),
    /// A scene or configuration document could not be read.
    Io {
        /// Kind of document involved in the operation.
        document: &'static str,
        /// Filesystem operation that failed.
        operation: &'static str,
        /// Operating-system error kind.
        kind: io::ErrorKind,
    },
    /// The selected configuration source and supplied values are inconsistent.
    InvalidConfigurationSelection {
        /// Explanation of the inconsistency.
        reason: String,
    },
    /// A configuration file was selected but not supplied for resolution.
    MissingConfigurationDocument,
    /// The scene contains more planes than the supported limit.
    TooManyPlanes {
        /// Largest supported plane count.
        maximum: usize,
        /// Plane count requested by the scene.
        actual: usize,
    },
    /// Two planes define mutually incompatible permitted half-spaces.
    ContradictoryPlanes {
        /// Index of the first conflicting plane.
        first: usize,
        /// Index of the second conflicting plane.
        second: usize,
    },
    /// The scene planes have no point in common within their permitted half-spaces.
    NoFeasiblePlaneIntersection,
    /// A lattice axis has no generated positions.
    EmptyLatticeAxis {
        /// Name of the empty lattice axis.
        axis: &'static str,
    },
    /// A particle construction requests more particles than the supported limit.
    ParticleCountTooLarge {
        /// Kind of particle construction making the request.
        construction: &'static str,
        /// Largest supported particle count.
        maximum: usize,
        /// Number of particles requested by the construction.
        requested: u64,
    },
    /// Generated particle identities cannot be represented by `u64`.
    ParticleIdentityOverflow,
    /// Explicit particle identities are not in strict ascending order.
    ParticleIdentityOrder {
        /// Previously observed particle identity.
        previous: u64,
        /// Identity that failed to exceed the previous identity.
        next: u64,
    },
    /// Adjacent lattice indices collapse to the same binary32 coordinate.
    LatticeCoordinateCollapse {
        /// Axis on which coordinate collapse occurred.
        axis: &'static str,
        /// Earlier lattice index.
        previous_index: u32,
        /// Later lattice index that produced the same coordinate.
        next_index: u32,
    },
    /// Lattice spacing differs from the resolved configuration's particle spacing.
    ParticleSpacingMismatch {
        /// Spacing specified by the scene in metres.
        scene: f32,
        /// Spacing resolved from the numerical configuration in metres.
        configuration: f32,
    },
    /// Two particles are closer than the permitted minimum separation.
    OverlappingParticles {
        /// Identity of the first overlapping particle.
        first: u64,
        /// Identity of the second overlapping particle.
        second: u64,
    },
    /// A particle lies outside the half-space permitted by a plane.
    ParticleOutsidePlane {
        /// Identity of the out-of-bounds particle.
        particle: u64,
        /// Index of the plane that excludes the particle.
        plane: usize,
    },
}

impl fmt::Display for SceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => write!(formatter, "invalid scene configuration: {error}"),
            Self::Numerical(error) => write!(formatter, "invalid scene number: {error}"),
            Self::Io {
                document,
                operation,
                kind,
            } => write!(formatter, "could not {operation} {document}: {kind}"),
            Self::InvalidConfigurationSelection { reason } => {
                write!(formatter, "invalid scene configuration selection: {reason}")
            }
            Self::MissingConfigurationDocument => write!(
                formatter,
                "scene selects a configuration file but none was supplied"
            ),
            Self::TooManyPlanes { maximum, actual } => {
                write!(
                    formatter,
                    "scene supports at most {maximum} planes, got {actual}"
                )
            }
            Self::ContradictoryPlanes { first, second } => write!(
                formatter,
                "scene planes {first} and {second} define contradictory halfspaces"
            ),
            Self::NoFeasiblePlaneIntersection => write!(
                formatter,
                "scene planes have no shared point in their permitted halfspaces"
            ),
            Self::EmptyLatticeAxis { axis } => {
                write!(
                    formatter,
                    "lattice count on the {axis} axis must be positive"
                )
            }
            Self::ParticleCountTooLarge {
                construction,
                maximum,
                requested,
            } => write!(
                formatter,
                "{construction} scene supports at most {maximum} particles, requested {requested}"
            ),
            Self::ParticleIdentityOverflow => {
                write!(formatter, "particle identity range exceeds u64")
            }
            Self::ParticleIdentityOrder { previous, next } => write!(
                formatter,
                "explicit particle identities must be unique and ascending, got {previous} then {next}"
            ),
            Self::LatticeCoordinateCollapse {
                axis,
                previous_index,
                next_index,
            } => write!(
                formatter,
                "lattice {axis} coordinates at indices {previous_index} and {next_index} are not strictly increasing in binary32"
            ),
            Self::ParticleSpacingMismatch {
                scene,
                configuration,
            } => write!(
                formatter,
                "lattice spacing {scene} m must equal effective configuration spacing {configuration} m"
            ),
            Self::OverlappingParticles { first, second } => write!(
                formatter,
                "particles {first} and {second} are closer than the configured particle diameter"
            ),
            Self::ParticleOutsidePlane { particle, plane } => write!(
                formatter,
                "particle {particle} is outside scene plane {plane}"
            ),
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::Numerical(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ConfigError> for SceneError {
    fn from(error: ConfigError) -> Self {
        Self::Configuration(error)
    }
}

impl From<NumericalError> for SceneError {
    fn from(error: NumericalError) -> Self {
        Self::Numerical(error)
    }
}

/// Resolves a parsed scene into the immutableinput eaten by the simulator.
///
/// `configuration_file` must be supplied exactly when the scene selects a
/// file. The optional command-line overlay is applied last; parsing command
/// lines remains outside this module.
///
/// ### Errors
///
/// TODO: Convert this to a list, am too lazy for now, HA.
/// Returns [`SceneError`] for an inconsistent selection, invalid numerical
/// configuration, invalid geometry, unstable identities, overlap, bounds
/// violations or impractical construction sizes.
pub fn resolve_scene(
    document: &SceneDocument,
    configuration_file: Option<&ConfigurationDocument>,
    command_line: Option<&ConfigurationOverlay>,
) -> Result<ResolvedScene, SceneError> {
    let selected_file = match document.configuration.source {
        ConfigurationSource::BuiltInDefaults => {
            if document.configuration.path.is_some() {
                return Err(SceneError::InvalidConfigurationSelection {
                    reason: "built_in_defaults must not include path".to_owned(),
                });
            }
            if configuration_file.is_some() {
                return Err(SceneError::InvalidConfigurationSelection {
                    reason: "a configuration document was supplied for built_in_defaults"
                        .to_owned(),
                });
            }
            None
        }
        ConfigurationSource::File => {
            let path = document.configuration.path.as_deref().ok_or_else(|| {
                SceneError::InvalidConfigurationSelection {
                    reason: "file source requires path".to_owned(),
                }
            })?;
            validate_relative_reference(path)?;
            Some(configuration_file.ok_or(SceneError::MissingConfigurationDocument)?)
        }
    };

    let configuration = resolve_effective_configuration(
        selected_file,
        document.configuration.overrides.as_ref(),
        command_line,
    )?;
    let planes = resolve_planes(&document.planes)?;
    validate_plane_compatibility(&planes)?;
    let particles = resolve_particles(
        &document.particles,
        &configuration,
        document.allow_coincident_particles,
    )?;
    validate_bounds(&particles, &planes)?;

    Ok(ResolvedScene {
        schema_version: document.schema_version,
        name: document.name.clone(),
        description: document.description.clone(),
        randomness: document.randomness,
        configuration,
        particles,
        planes,
    })
}

/// Loads a scene and any relative configuration reference from the filesystem.
///
/// The selected configuration path is resolved from the canonical scene-file
/// directory, never from the process current directory. Absolute references
/// are rejected so recorded scenes do not embed machine-specific paths.
///
/// ### Errors
///
/// TODO: convert to list.
/// Returns [`SceneError`] for filesystem, parsing, resolution or validation
/// failures.
pub fn load_resolved_scene(
    scene_path: &Path,
    command_line: Option<&ConfigurationOverlay>,
) -> Result<ResolvedScene, SceneError> {
    let absolute_scene = fs::canonicalize(scene_path)
        .map_err(|error| io_error("scene document", "locate", error))?;
    let scene_input = fs::read_to_string(&absolute_scene)
        .map_err(|error| io_error("scene document", "read", error))?;
    let scene = parse_scene(&scene_input)?;

    let configuration = if scene.configuration.source == ConfigurationSource::File {
        let reference = scene.configuration.path.as_deref().ok_or_else(|| {
            SceneError::InvalidConfigurationSelection {
                reason: "file source requires path".to_owned(),
            }
        })?;
        validate_relative_reference(reference)?;
        let parent =
            absolute_scene
                .parent()
                .ok_or_else(|| SceneError::InvalidConfigurationSelection {
                    reason: "scene document has no containing directory".to_owned(),
                })?;
        let configuration_path = parent.join(reference);
        Some(load_configuration(&configuration_path)?)
    } else {
        None
    };

    resolve_scene(&scene, configuration.as_ref(), command_line)
}

fn validate_relative_reference(reference: &str) -> Result<(), SceneError> {
    if reference.trim().is_empty() {
        return Err(SceneError::InvalidConfigurationSelection {
            reason: "configuration path must not be empty".to_owned(),
        });
    }
    if PathBuf::from(reference).is_absolute() {
        return Err(SceneError::InvalidConfigurationSelection {
            reason: "configuration path must be relative to the scene document".to_owned(),
        });
    }
    Ok(())
}

fn io_error(document: &'static str, operation: &'static str, error: io::Error) -> SceneError {
    SceneError::Io {
        document,
        operation,
        kind: error.kind(),
    }
}
