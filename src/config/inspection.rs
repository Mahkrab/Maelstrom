//! # Configuration inspection.

use crate::math::Vec3;

use super::{FieldSource, ResolvedConfiguration};

impl ResolvedConfiguration {
    /// Returns TOML data that shows values and their provedence.
    #[must_use]
    pub fn canonical_toml(&self) -> String {
        // I hate this.
        let mut output = String::new();
        push_line(
            &mut output,
            "schema_version",
            &self.schema_version.to_string(),
        );
        push_string(&mut output, "name", &self.name);
        push_string(&mut output, "description", &self.description);
        output.push_str("\n[numerical]\n");
        let numerical = &self.numerical;
        push_string(&mut output, "scalar_precision", &numerical.scalar_precision);
        push_string(
            &mut output,
            "coordinate_system",
            &numerical.coordinate_system,
        );
        push_string(&mut output, "timestep_policy", &numerical.timestep_policy);
        push_string(&mut output, "density_kernel", &numerical.density_kernel);
        push_string(
            &mut output,
            "correction_kernel",
            &numerical.correction_kernel,
        );
        push_string(
            &mut output,
            "solver_update_policy",
            &numerical.solver_update_policy,
        );
        push_vec3(
            &mut output,
            "external_acceleration_m_s2",
            numerical.external_acceleration_m_s2,
        );
        push_float(&mut output, "fixed_timestep_s", numerical.fixed_timestep_s);
        push_float(&mut output, "cfl_factor", numerical.cfl_factor);
        push_float(
            &mut output,
            "particle_spacing_m",
            numerical.particle_spacing_m,
        );
        push_float(
            &mut output,
            "support_radius_ratio",
            numerical.support_radius_ratio,
        );
        push_float(
            &mut output,
            "physical_rest_density_kg_m3",
            numerical.physical_rest_density_kg_m3,
        );
        push_line(
            &mut output,
            "solver_iterations",
            &numerical.solver_iterations.to_string(),
        );
        push_float(&mut output, "relaxation_scale", numerical.relaxation_scale);
        push_line(
            &mut output,
            "artificial_pressure_enabled",
            &numerical.artificial_pressure_enabled.to_string(),
        );
        push_float(
            &mut output,
            "artificial_pressure_strength_scale",
            numerical.artificial_pressure_strength_scale,
        );
        push_float(
            &mut output,
            "artificial_pressure_reference_separation_ratio",
            numerical.artificial_pressure_reference_separation_ratio,
        );
        push_float(
            &mut output,
            "artificial_pressure_exponent",
            numerical.artificial_pressure_exponent,
        );
        push_string(&mut output, "support_policy", &numerical.support_policy);
        push_string(&mut output, "boundary_model", &numerical.boundary_model);
        push_string(
            &mut output,
            "neighbour_rebuild_policy",
            &numerical.neighbour_rebuild_policy,
        );
        push_string(
            &mut output,
            "particle_order_policy",
            &numerical.particle_order_policy,
        );
        push_line(
            &mut output,
            "xsph_viscosity_enabled",
            &numerical.xsph_viscosity_enabled.to_string(),
        );
        push_line(
            &mut output,
            "vorticity_confinement_enabled",
            &numerical.vorticity_confinement_enabled.to_string(),
        );

        output.push_str("\n[derived]\n");
        let derived = self.derived;
        push_float(
            &mut output,
            "particle_diameter_m",
            derived.particle_diameter_m,
        );
        push_float(&mut output, "support_radius_m", derived.support_radius_m);
        push_float(
            &mut output,
            "normalised_rest_density_per_m3",
            derived.normalised_rest_density_per_m3,
        );
        push_float(&mut output, "particle_mass_kg", derived.particle_mass_kg);
        push_float(&mut output, "relaxation_per_m2", derived.relaxation_per_m2);
        push_float(
            &mut output,
            "artificial_pressure_strength_m2",
            derived.artificial_pressure_strength_m2,
        );
        push_float(
            &mut output,
            "artificial_pressure_reference_separation_m",
            derived.artificial_pressure_reference_separation_m,
        );
        push_line(
            &mut output,
            "lattice_offset_count",
            &derived.lattice_offset_count.to_string(),
        );

        output.push_str("\n[provenance]\n");
        for (field, source) in &self.sources {
            push_string(&mut output, field, source.as_str());
        }
        output
    }

    /// Returns the numerical metadata needed to compare two resolved runs.
    #[must_use]
    pub fn comparison_key(&self) -> String {
        let n = &self.numerical;
        let d = self.derived;
        format!(
            // Wow this is ugly, thanks fmt...
            "configuration-comparison-v1|{}|{}|{}|{}|{}|{}|{:08x},{:08x},{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{}|{:08x}|{}|{:08x}|{:08x}|{:08x}|{}|{}|{}|{}|{}|{}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{}",
            n.scalar_precision,
            n.coordinate_system,
            n.timestep_policy,
            n.density_kernel,
            n.correction_kernel,
            n.solver_update_policy,
            n.external_acceleration_m_s2.x.to_bits(),
            n.external_acceleration_m_s2.y.to_bits(),
            n.external_acceleration_m_s2.z.to_bits(),
            n.fixed_timestep_s.to_bits(),
            n.cfl_factor.to_bits(),
            n.particle_spacing_m.to_bits(),
            n.support_radius_ratio.to_bits(),
            n.physical_rest_density_kg_m3.to_bits(),
            n.solver_iterations,
            n.relaxation_scale.to_bits(),
            n.artificial_pressure_enabled,
            n.artificial_pressure_strength_scale.to_bits(),
            n.artificial_pressure_reference_separation_ratio.to_bits(),
            n.artificial_pressure_exponent.to_bits(),
            n.support_policy,
            n.boundary_model,
            n.neighbour_rebuild_policy,
            n.particle_order_policy,
            n.xsph_viscosity_enabled,
            n.vorticity_confinement_enabled,
            d.particle_diameter_m.to_bits(),
            d.support_radius_m.to_bits(),
            d.normalised_rest_density_per_m3.to_bits(),
            d.particle_mass_kg.to_bits(),
            d.relaxation_per_m2.to_bits(),
            d.artificial_pressure_strength_m2.to_bits(),
            d.artificial_pressure_reference_separation_m.to_bits(),
            d.lattice_offset_count,
        )
    }
}

impl FieldSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::BuiltInDefault => "built_in_default",
            Self::ConfigurationFile => "configuration_file",
            Self::Scene => "scene",
            Self::CommandLine => "command_line",
        }
    }
}

fn push_line(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push_str(" = ");
    output.push_str(value);
    output.push('\n');
}

fn push_string(output: &mut String, key: &str, value: &str) {
    push_line(output, key, &quoted(value));
}

fn push_float(output: &mut String, key: &str, value: f32) {
    push_line(output, key, &format!("{value:?}"));
}

fn push_vec3(output: &mut String, key: &str, value: Vec3) {
    push_line(
        output,
        key,
        &format!("[{:?}, {:?}, {:?}]", value.x, value.y, value.z),
    );
}

fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}
