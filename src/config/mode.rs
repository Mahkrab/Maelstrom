//! Named scheduling choices fill the command-line layer without replacing a
//! scene's physical inputs. Recorded runs retain the resulting numerical values.

use super::{ConfigError, NumericalOverrides};

/// Equations and mass convention retained in resolved inputs and replay records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PbfFormulation {
    /// Historical lattice-calibrated mass and empirical correction refinement.
    #[default]
    Empirical,
    /// Full published Jacobi update, nominal particle volume and pure timestep refinement.
    Paper2013,
}

/// Numerical presets selected by the application; reference is its default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalMode {
    /// Use the selected scene/configuration without applying a preset.
    Scene,
    /// Established 480 Hz, eight-iteration comparison and fallback policy.
    Reference,
    /// Reduced-work 240 Hz, eight-iteration policy.
    Fast,
    /// Experimental reference-response calibration at half the step rate.
    Experimental,
    /// Published PBF equations with independently selected numerical constants.
    PaperBaseline,
}

impl std::str::FromStr for NumericalMode {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "scene" => Ok(Self::Scene),
            "reference" => Ok(Self::Reference),
            "fast" => Ok(Self::Fast),
            "experimental" => Ok(Self::Experimental),
            "paper-baseline" => Ok(Self::PaperBaseline),
            _ => Err(ConfigError::InvalidField {
                field: "numerical_mode",
                reason: "must be reference, fast, experimental, paper-baseline, or scene".into(),
            }),
        }
    }
}

impl NumericalMode {
    /// Fills unspecified scheduling fields, preserving explicit overrides and
    /// every unrelated physical setting. Refinement defaults to one so a scene's
    /// refinement cannot silently refine the preset a second time. An explicit
    /// refinement divides timestep by N. Only empirical formulations also
    /// divide the correction scale by N².
    pub fn fill_overrides(self, values: &mut NumericalOverrides) {
        let (timestep, iterations, correction, relaxation) = match self {
            Self::Scene => return,
            Self::Reference => (1.0 / 480.0, 8, 0.0625, 20.0),
            Self::Fast => (1.0 / 240.0, 8, 0.5, 2.0),
            // Match I * scale / dt² to reference while retaining its
            // regularisation and artificial pressure. This is an empirical
            // response calibration, not an incompressibility/convergence law.
            Self::Experimental => (1.0 / 240.0, 8, 0.25, 20.0),
            Self::PaperBaseline => (1.0 / 240.0, 3, 1.0, 20.0),
        };
        values
            .pbf_formulation
            .get_or_insert(if self == Self::PaperBaseline {
                PbfFormulation::Paper2013
            } else {
                PbfFormulation::Empirical
            });
        if self == Self::PaperBaseline {
            values.artificial_pressure_enabled.get_or_insert(true);
            values.artificial_pressure_strength_scale.get_or_insert(0.1);
            values
                .artificial_pressure_reference_separation_ratio
                .get_or_insert(0.2);
            values.artificial_pressure_exponent.get_or_insert(4.0);
            values.xsph_viscosity_enabled.get_or_insert(true);
            values.xsph_viscosity_coefficient.get_or_insert(0.01);
            values.vorticity_confinement_enabled.get_or_insert(false);
        }
        values.fixed_timestep_s.get_or_insert(timestep);
        values.solver_iterations.get_or_insert(iterations);
        values.correction_scale.get_or_insert(correction);
        values.relaxation_scale.get_or_insert(relaxation);
        values.refine_timestep.get_or_insert(1);
    }
}
