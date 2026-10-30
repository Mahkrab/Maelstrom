//! Named scheduling choices fill the command-line layer without replacing a
//! scene's physical inputs. Recorded runs retain the resulting numerical values.

use super::{ConfigError, NumericalOverrides};

/// Numerical presets selected by the application; fast is its default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalMode {
    /// Use the selected scene/configuration without applying a preset.
    Scene,
    /// Established 480 Hz, eight-iteration comparison and fallback policy.
    Reference,
    /// Reduced-work 240 Hz, eight-iteration policy.
    Fast,
}

impl std::str::FromStr for NumericalMode {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "scene" => Ok(Self::Scene),
            "reference" => Ok(Self::Reference),
            "fast" => Ok(Self::Fast),
            _ => Err(ConfigError::InvalidField {
                field: "numerical_mode",
                reason: "must be reference or fast, or scene to use saved settings".into(),
            }),
        }
    }
}

impl NumericalMode {
    /// Fills unspecified scheduling fields, preserving explicit overrides and
    /// every unrelated physical setting. Refinement defaults to one so a scene's
    /// refinement cannot silently refine the preset a second time. An explicit
    /// refinement still divides timestep by N and correction scale by N².
    pub fn fill_overrides(self, values: &mut NumericalOverrides) {
        let (timestep, correction, relaxation) = match self {
            Self::Scene => return,
            Self::Reference => (1.0 / 480.0, 0.0625, 20.0),
            Self::Fast => (1.0 / 240.0, 0.5, 2.0),
        };
        values.fixed_timestep_s.get_or_insert(timestep);
        values.solver_iterations.get_or_insert(8);
        values.correction_scale.get_or_insert(correction);
        values.relaxation_scale.get_or_insert(relaxation);
        values.refine_timestep.get_or_insert(1);
    }
}
