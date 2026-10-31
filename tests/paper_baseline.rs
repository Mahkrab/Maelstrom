//! The published update is checked in physical units against a binary64 pair
//! calculation, independently of production kernels and lattice calibration.
use maelstrom::{
    config::{
        ConfigurationOverlay, NumericalMode, NumericalOverrides, PbfFormulation,
        parse_configuration, resolve_configuration_for_schema, resolve_effective_configuration,
    },
    fluid::{InteractionSets, compute_jacobi_stage},
    math::Vec3,
};

fn paper(schema: u32, refinement: u32) -> maelstrom::config::ResolvedConfiguration {
    let mut numerical = NumericalOverrides {
        refine_timestep: Some(refinement),
        ..Default::default()
    };
    NumericalMode::PaperBaseline.fill_overrides(&mut numerical);
    resolve_configuration_for_schema(
        schema,
        None,
        None,
        Some(&ConfigurationOverlay {
            numerical,
            ..Default::default()
        }),
    )
    .unwrap()
}

#[test]
fn published_refinement_changes_time_without_attenuating_corrections() {
    for schema in [1, 2] {
        let base = paper(schema, 1);
        let refined = paper(schema, 4);
        assert_eq!(base.numerical.pbf_formulation, PbfFormulation::Paper2013);
        assert_eq!(base.numerical.solver_iterations, 3);
        assert_eq!(
            refined.numerical.fixed_timestep_s,
            base.numerical.fixed_timestep_s / 4.0
        );
        assert_eq!(refined.numerical.correction_scale, 1.0);
        assert_eq!(base.derived, refined.derived);
        assert!((base.derived.particle_mass_kg - 0.125).abs() < 2e-8);
        assert!((base.derived.normalised_rest_density_per_m3 - 8000.0).abs() < 0.001);
        let saved = base.canonical_toml();
        // Inspection also includes provenance and an informational lattice count;
        // only its numerical document is an input configuration.
        let doc = parse_configuration(saved.split("\n[derived]").next().unwrap()).unwrap();
        let roundtrip = resolve_effective_configuration(Some(&doc), None, None).unwrap();
        assert_eq!(roundtrip.numerical, base.numerical);
        assert_eq!(roundtrip.derived, base.derived);
        let encoded = serde_json::to_string(&base.numerical).unwrap();
        assert!(encoded.contains("paper2013"));
        assert_eq!(
            serde_json::from_str::<maelstrom::config::NumericalModel>(&encoded).unwrap(),
            base.numerical
        );
    }
}

#[test]
fn published_formulation_rejects_empirical_attenuation() {
    let mut numerical = NumericalOverrides {
        correction_scale: Some(0.25),
        ..Default::default()
    };
    NumericalMode::PaperBaseline.fill_overrides(&mut numerical);
    for schema in [1, 2] {
        let error = resolve_configuration_for_schema(
            schema,
            None,
            None,
            Some(&ConfigurationOverlay {
                numerical: numerical.clone(),
                ..Default::default()
            }),
        )
        .unwrap_err();
        assert!(error.to_string().contains("full published position update"));
    }
}

#[test]
fn pair_update_matches_published_density_multiplier_and_pressure_equations() {
    let config = paper(2, 1);
    let p = &config.numerical;
    let h = f64::from(config.derived.support_radius_m);
    let rest = 1.0 / f64::from(p.particle_spacing_m).powi(3);
    let r = f64::from(0.04_f32);
    let w = |d: f64| 315.0 / (64.0 * std::f64::consts::PI * h.powi(9)) * (h * h - d * d).powi(3);
    let gradient = 45.0 / (std::f64::consts::PI * h.powi(6)) * (h - r).powi(2);
    let density = w(0.0) + w(r);
    let constraint = density / rest - 1.0;
    let lambda =
        -constraint / (2.0 * (gradient / rest).powi(2) + f64::from(p.relaxation_scale) / h.powi(2));
    let pressure = -f64::from(p.artificial_pressure_strength_scale)
        * h
        * h
        * (w(r) / w(f64::from(p.artificial_pressure_reference_separation_ratio) * h))
            .powf(f64::from(p.artificial_pressure_exponent));
    let correction = (2.0 * lambda + pressure) * gradient / rest;
    let stage = compute_jacobi_stage(
        &[Vec3::ZERO, Vec3::new(r as f32, 0.0, 0.0)],
        &InteractionSets::new(vec![vec![1], vec![0]]).unwrap(),
        config.pbf_parameters().unwrap(),
    )
    .unwrap();
    // Binary32 kernels accumulate several products and divisions; 4 ppm covers
    // their rounding while distinguishing mass, sign and missing-neighbour terms.
    assert!((f64::from(stage.corrections[0].x) - correction).abs() < correction.abs() * 4e-6);
    assert_eq!(stage.corrections[0], -stage.corrections[1]);
}
