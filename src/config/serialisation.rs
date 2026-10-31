//! Numerical records retain historical field order and omit inactive defaults.
//! Enabled velocity effects always record their strengths, including defaults,
//! so a future default cannot reinterpret a saved active model.

use super::{DEFAULT_XSPH_VISCOSITY_COEFFICIENT, NumericalModel};
use serde::{Serialize, Serializer, ser::SerializeStruct};

impl Serialize for NumericalModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let correction = self.correction_scale != 1.0;
        let paper = self.pbf_formulation == super::PbfFormulation::Paper2013;
        let smoothing = self.xsph_viscosity_enabled
            || self.xsph_viscosity_coefficient.to_bits()
                != DEFAULT_XSPH_VISCOSITY_COEFFICIENT.to_bits();
        let confinement = self.vorticity_confinement_enabled
            || self.vorticity_confinement_strength_m_s.to_bits()
                != super::DEFAULT_VORTICITY_CONFINEMENT_STRENGTH_M_S.to_bits();
        let mut record = serializer.serialize_struct(
            "NumericalModel",
            24 + usize::from(paper)
                + usize::from(correction)
                + usize::from(smoothing)
                + usize::from(confinement),
        )?;
        macro_rules! field {
            ($name:ident) => {
                record.serialize_field(stringify!($name), &self.$name)?;
            };
        }
        field!(scalar_precision);
        if paper {
            field!(pbf_formulation);
        }
        field!(coordinate_system);
        field!(timestep_policy);
        field!(density_kernel);
        field!(correction_kernel);
        field!(solver_update_policy);
        field!(external_acceleration_m_s2);
        field!(fixed_timestep_s);
        field!(cfl_factor);
        field!(particle_spacing_m);
        field!(support_radius_ratio);
        field!(physical_rest_density_kg_m3);
        field!(solver_iterations);
        field!(relaxation_scale);
        if correction {
            field!(correction_scale);
        }
        field!(artificial_pressure_enabled);
        field!(artificial_pressure_strength_scale);
        field!(artificial_pressure_reference_separation_ratio);
        field!(artificial_pressure_exponent);
        field!(support_policy);
        field!(boundary_model);
        field!(neighbour_rebuild_policy);
        field!(particle_order_policy);
        field!(xsph_viscosity_enabled);
        if smoothing {
            field!(xsph_viscosity_coefficient);
        }
        field!(vorticity_confinement_enabled);
        if confinement {
            field!(vorticity_confinement_strength_m_s);
        }
        record.end()
    }
}
