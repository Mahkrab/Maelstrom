//! # Configuration validation and derivation
//!
//! Primary numerical values are validated against the supported simulation
//! contract, then used to derive the authoritative dependent values.

use crate::{reference::derive_lattice_equilibrium, validation};

use super::{
    BOUNDARY_MODEL, COORDINATE_SYSTEM, CORRECTION_KERNEL, ConfigError, DENSITY_KERNEL,
    DerivedAssertions, MAX_SOLVER_ITERATIONS, MAX_SUPPORT_RADIUS_RATIO, NEIGHBOUR_REBUILD_POLICY,
    NumericalModel, PARTICLE_ORDER_POLICY, ResolvedDerivedValues, SCALAR_PRECISION,
    SOLVER_UPDATE_POLICY, SUPPORT_POLICY, TIMESTEP_POLICY,
};

pub(super) fn validate_model(model: &NumericalModel) -> Result<(), ConfigError> {
    require_policy(
        "scalar_precision",
        &model.scalar_precision,
        SCALAR_PRECISION,
    )?;
    require_policy(
        "coordinate_system",
        &model.coordinate_system,
        COORDINATE_SYSTEM,
    )?;
    require_policy("timestep_policy", &model.timestep_policy, TIMESTEP_POLICY)?;
    require_policy("density_kernel", &model.density_kernel, DENSITY_KERNEL)?;
    require_policy(
        "correction_kernel",
        &model.correction_kernel,
        CORRECTION_KERNEL,
    )?;
    require_policy(
        "solver_update_policy",
        &model.solver_update_policy,
        SOLVER_UPDATE_POLICY,
    )?;
    require_policy("support_policy", &model.support_policy, SUPPORT_POLICY)?;
    require_policy("boundary_model", &model.boundary_model, BOUNDARY_MODEL)?;
    require_policy(
        "neighbour_rebuild_policy",
        &model.neighbour_rebuild_policy,
        NEIGHBOUR_REBUILD_POLICY,
    )?;
    require_policy(
        "particle_order_policy",
        &model.particle_order_policy,
        PARTICLE_ORDER_POLICY,
    )?;
    validation::finite_vector(
        "external_acceleration_m_s2",
        model.external_acceleration_m_s2,
    )
    .map_err(ConfigError::Numerical)?;
    positive("fixed_timestep_s", model.fixed_timestep_s)?;
    positive("cfl_factor", model.cfl_factor)?;
    positive("particle_spacing_m", model.particle_spacing_m)?;
    positive("support_radius_ratio", model.support_radius_ratio)?;
    if model.support_radius_ratio > MAX_SUPPORT_RADIUS_RATIO {
        return Err(ConfigError::InvalidField {
            field: "support_radius_ratio",
            reason: format!("must be at most {MAX_SUPPORT_RADIUS_RATIO}"),
        });
    }
    positive(
        "physical_rest_density_kg_m3",
        model.physical_rest_density_kg_m3,
    )?;
    if !(1..=MAX_SOLVER_ITERATIONS).contains(&model.solver_iterations) {
        return Err(ConfigError::InvalidField {
            field: "solver_iterations",
            reason: format!("must be in 1..={MAX_SOLVER_ITERATIONS}"),
        });
    }
    positive("relaxation_scale", model.relaxation_scale)?;
    non_negative(
        "artificial_pressure_strength_scale",
        model.artificial_pressure_strength_scale,
    )?;
    let separation = model.artificial_pressure_reference_separation_ratio;
    if !separation.is_finite() || separation <= 0.0 || separation >= 1.0 {
        return Err(ConfigError::InvalidField {
            field: "artificial_pressure_reference_separation_ratio",
            reason: "must be finite and strictly between 0 and 1".to_owned(),
        });
    }
    positive(
        "artificial_pressure_exponent",
        model.artificial_pressure_exponent,
    )?;
    if model.xsph_viscosity_enabled {
        return Err(ConfigError::InvalidField {
            field: "xsph_viscosity_enabled",
            reason: "true is not supported by the current numerical contract".to_owned(),
        });
    }
    if model.vorticity_confinement_enabled {
        return Err(ConfigError::InvalidField {
            field: "vorticity_confinement_enabled",
            reason: "true is not supported by the current numerical contract".to_owned(),
        });
    }
    Ok(())
}

pub(super) fn derive_values(model: &NumericalModel) -> Result<ResolvedDerivedValues, ConfigError> {
    let support_radius_m = validation::finite_result(
        "configuration support radius",
        model.particle_spacing_m * model.support_radius_ratio,
    )
    .map_err(ConfigError::Numerical)?;
    let equilibrium = derive_lattice_equilibrium(
        model.particle_spacing_m,
        support_radius_m,
        model.physical_rest_density_kg_m3,
        MAX_SUPPORT_RADIUS_RATIO,
    )
    .map_err(ConfigError::Numerical)?;
    let support_squared = validation::finite_result(
        "configuration squared support radius",
        support_radius_m * support_radius_m,
    )
    .map_err(ConfigError::Numerical)?;
    let relaxation_per_m2 = validation::finite_result(
        "configuration relaxation",
        model.relaxation_scale / support_squared,
    )
    .map_err(ConfigError::Numerical)?;
    let artificial_pressure_strength_m2 = if model.artificial_pressure_enabled {
        validation::finite_result(
            "configuration artificial-pressure strength",
            model.artificial_pressure_strength_scale * support_squared,
        )
        .map_err(ConfigError::Numerical)?
    } else {
        0.0
    };
    let artificial_pressure_reference_separation_m = validation::finite_result(
        "configuration artificial-pressure reference separation",
        model.artificial_pressure_reference_separation_ratio * support_radius_m,
    )
    .map_err(ConfigError::Numerical)?;

    Ok(ResolvedDerivedValues {
        particle_diameter_m: model.particle_spacing_m,
        support_radius_m,
        normalised_rest_density_per_m3: equilibrium.normalised_rest_density,
        particle_mass_kg: equilibrium.particle_mass,
        relaxation_per_m2,
        artificial_pressure_strength_m2,
        artificial_pressure_reference_separation_m,
        lattice_offset_count: equilibrium.offset_count,
    })
}

pub(super) fn validate_assertions(
    layer: &'static str,
    assertions: &DerivedAssertions,
    derived: ResolvedDerivedValues,
) -> Result<(), ConfigError> {
    macro_rules! assert_exact {
        ($field:ident) => {
            if let Some(provided) = assertions.$field {
                if !provided.is_finite() || provided.to_bits() != derived.$field.to_bits() {
                    return Err(ConfigError::DerivedMismatch {
                        layer,
                        field: stringify!($field),
                        expected: derived.$field,
                        provided,
                    });
                }
            }
        };
    }
    assert_exact!(particle_diameter_m);
    assert_exact!(support_radius_m);
    assert_exact!(normalised_rest_density_per_m3);
    assert_exact!(particle_mass_kg);
    assert_exact!(relaxation_per_m2);
    assert_exact!(artificial_pressure_strength_m2);
    assert_exact!(artificial_pressure_reference_separation_m);
    Ok(())
}

fn require_policy(
    field: &'static str,
    value: &str,
    supported: &'static str,
) -> Result<(), ConfigError> {
    if value == supported {
        Ok(())
    } else {
        Err(ConfigError::InvalidField {
            field,
            reason: format!("supported value is {supported:?}, got {value:?}"),
        })
    }
}

fn positive(field: &'static str, value: f32) -> Result<(), ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ConfigError::InvalidField {
            field,
            reason: format!("must be positive and finite, got {value}"),
        })
    }
}

fn non_negative(field: &'static str, value: f32) -> Result<(), ConfigError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ConfigError::InvalidField {
            field,
            reason: format!("must be non-negative and finite, got {value}"),
        })
    }
}
