//! Layered inputs use independently chosen binary fractions so refinement order
//! and legacy compounding can be distinguished without rounding tolerances.

use maelstrom::{
    config::{
        ConfigError, FieldSource, parse_configuration, parse_configuration_overlay,
        resolve_configuration_for_schema, resolve_effective_configuration,
    },
    scene::{parse_scene, resolve_scene},
};

#[test]
fn refinement_is_selected_once_and_can_be_reset_without_changing_legacy_resolution() {
    let file = "schema_version=2\nname='test'\ndescription='layered inputs'\n[numerical]\nfixed_timestep_s=0.125\ncorrection_scale=0.5\nrefine_timestep=4";
    let document = parse_configuration(file).unwrap();
    let scene =
        parse_configuration_overlay("[numerical]\nfixed_timestep_s=0.25\nrefine_timestep=8")
            .unwrap();
    let cli = parse_configuration_overlay("[numerical]\nrefine_timestep=2").unwrap();
    let resolved =
        resolve_effective_configuration(Some(&document), Some(&scene), Some(&cli)).unwrap();
    assert_eq!(resolved.numerical.fixed_timestep_s, 0.125);
    assert_eq!(resolved.numerical.correction_scale, 0.125);
    let refinement = resolved.refinement.unwrap();
    assert_eq!(refinement.factor, 2);
    assert_eq!(refinement.source, FieldSource::CommandLine);
    assert_eq!(refinement.unrefined_timestep_s, 0.25);
    assert_eq!(refinement.unrefined_correction_scale, 0.5);
    let reset = parse_configuration_overlay("[numerical]\nrefine_timestep=1").unwrap();
    let resolved =
        resolve_effective_configuration(Some(&document), Some(&scene), Some(&reset)).unwrap();
    assert_eq!(resolved.numerical.fixed_timestep_s, 0.25);
    assert_eq!(resolved.numerical.correction_scale, 0.5);
    let old = parse_configuration(&file.replace("schema_version=2", "schema_version=1")).unwrap();
    let legacy = resolve_effective_configuration(Some(&old), Some(&scene), Some(&cli)).unwrap();
    assert_eq!(legacy.numerical.fixed_timestep_s, 0.015625);
    assert_eq!(legacy.numerical.correction_scale, 1.0 / 8192.0);
    assert!(legacy.refinement.is_none());
}

#[test]
fn invalid_shadowed_values_and_final_assertions_cannot_be_hidden() {
    let good = parse_configuration_overlay(
        "[numerical]\nfixed_timestep_s=0.125\nparticle_spacing_m=0.1\nrefine_timestep=1",
    )
    .unwrap();
    for invalid in [
        "fixed_timestep_s=nan",
        "fixed_timestep_s=inf",
        "fixed_timestep_s=0.0",
        "correction_scale=1.1",
        "particle_spacing_m=-0.1",
        "support_radius_ratio=9.0",
        "refine_timestep=0",
        "refine_timestep=17",
        "external_acceleration_m_s2=[0.0,nan,0.0]",
    ] {
        let file=parse_configuration(&format!("schema_version=2\nname='bad'\ndescription='invalid supplied value'\n[numerical]\n{invalid}")).unwrap();
        assert!(
            resolve_effective_configuration(Some(&file), None, Some(&good)).is_err(),
            "{invalid}"
        );
    }
    let source = "schema_version=2\nname='assertions'\ndescription='final assertions'\n[numerical]\nparticle_spacing_m=0.05\n[derived]\nsupport_radius_m=0.1";
    let file = parse_configuration(source).unwrap();
    assert!(resolve_effective_configuration(Some(&file), None, Some(&good)).is_err());
    let legacy =
        parse_configuration(&source.replace("schema_version=2", "schema_version=1")).unwrap();
    assert!(resolve_effective_configuration(Some(&legacy), None, Some(&good)).is_ok());
    for schema in [0, 3, u32::MAX] {
        assert!(matches!(
            resolve_configuration_for_schema(schema, None, None, None),
            Err(ConfigError::UnsupportedSchema { .. })
        ));
    }
    assert!(resolve_configuration_for_schema(1, Some(&file), None, None).is_err());
    let invalid = parse_configuration("schema_version=2\nname='x'\ndescription='x'\nunknown=true");
    assert!(invalid.is_err());
}

#[test]
fn inherited_spacing_has_independent_positions_and_explicit_assertions_stay_strict() {
    let source = "schema_version=2\nname='lattice'\ndescription='inherited spacing'\nrandomness='none'\n[configuration]\nsource='built_in_defaults'\n[configuration.overrides.numerical]\nparticle_spacing_m=0.125\n[particles]\nconstruction='lattice'\nfirst_identity=7\norigin_m=[1.0,2.0,3.0]\ncounts=[1,1,3]\nendpoint_rule='counted_half_open'\norder='x_then_y_then_z'\ninitial_velocity_m_s=[0.0,0.0,0.0]";
    let doc = parse_scene(source).unwrap();
    let scene = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(scene.particles.len(), 3);
    for (index, z) in [3.0, 3.125, 3.25].into_iter().enumerate() {
        assert_eq!(scene.particles[index].identity, 7 + index as u64);
        assert_eq!(
            scene.particles[index].position_m,
            maelstrom::math::Vec3::new(1.0, 2.0, z)
        );
    }
    let mismatch =
        parse_scene(&source.replace("counts=[1,1,3]", "counts=[1,1,3]\nspacing_m=0.05")).unwrap();
    assert!(resolve_scene(&mismatch, None, None).is_err());
    assert!(parse_scene(&source.replace("schema_version=2", "schema_version=1")).is_err());
    let builtin = resolve_configuration_for_schema(2, None, None, None).unwrap();
    assert_eq!(builtin.schema_version, 2);
    assert_eq!(
        builtin.numerical,
        resolve_configuration_for_schema(1, None, None, None)
            .unwrap()
            .numerical
    );
}
