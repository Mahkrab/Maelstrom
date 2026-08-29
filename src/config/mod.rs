//! # Versioned numerical configuration
//!
//! TOML documents are parsed into explicit layers before their numerical
//! values are validated and resolved. 
//! 
//! Always follows the same order:
//!  
//! - built-in default values
//! - a configuration document
//! - scene-owned values
//! - then command-line-owned values.

mod document;
mod inspection;
mod resolved;
mod validation;

pub use document::{
    ConfigurationDocument, ConfigurationOverlay, DerivedAssertions, NumericalOverrides,
    load_configuration, parse_configuration, parse_configuration_overlay,
};
pub use resolved::{
    BOUNDARY_MODEL, CONFIG_SCHEMA_VERSION, COORDINATE_SYSTEM, CORRECTION_KERNEL, ConfigError,
    DENSITY_KERNEL, FieldSource, MAX_SOLVER_ITERATIONS, MAX_SUPPORT_RADIUS_RATIO,
    NEIGHBOUR_REBUILD_POLICY, NumericalModel, PARTICLE_ORDER_POLICY, ResolvedConfiguration,
    ResolvedDerivedValues, SCALAR_PRECISION, SOLVER_UPDATE_POLICY, SUPPORT_POLICY, TIMESTEP_POLICY,
    ValueSources, resolve_effective_configuration,
};
