use std::path::Path;

use maelstrom::config::{
    BOUNDARY_MODEL, CONFIG_SCHEMA_VERSION, COORDINATE_SYSTEM, CORRECTION_KERNEL, ConfigError,
    ConfigurationDocument, ConfigurationOverlay, DENSITY_KERNEL, DerivedAssertions, FieldSource,
    MAX_SOLVER_ITERATIONS, MAX_SUPPORT_RADIUS_RATIO, NEIGHBOUR_REBUILD_POLICY, NumericalOverrides,
    PARTICLE_ORDER_POLICY, SCALAR_PRECISION, SOLVER_UPDATE_POLICY, SUPPORT_POLICY, TIMESTEP_POLICY,
    load_configuration, parse_configuration, resolve_effective_configuration,
};

const DEFAULTS_TOML: &str = include_str!("../config/defaults-v4.toml");

fn defaults_document() -> ConfigurationDocument {
    load_configuration(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join("defaults-v4.toml"),
    )
    .expect("checked-in default configuration must load")
}

fn overlay(numerical: NumericalOverrides) -> ConfigurationOverlay {
    ConfigurationOverlay {
        numerical,
        derived: DerivedAssertions::default(),
    }
}

fn resolve_overlay(overrides: NumericalOverrides) -> Result<(), ConfigError> {
    resolve_effective_configuration(None, Some(&overlay(overrides)), None).map(|_| ())
}

#[test]
fn checked_in_configuration_resolves_to_the_selected_numerical_values() {
    let document = defaults_document();
    let resolved = resolve_effective_configuration(Some(&document), None, None)
        .expect("selected values should resolve");

    assert_eq!(document.schema_version, CONFIG_SCHEMA_VERSION);
    assert_eq!(resolved.name, "maelstrom-defaults-v4");
    assert_eq!(resolved.numerical.scalar_precision, SCALAR_PRECISION);
    assert_eq!(resolved.numerical.coordinate_system, COORDINATE_SYSTEM);
    assert_eq!(resolved.numerical.timestep_policy, TIMESTEP_POLICY);
    assert_eq!(resolved.numerical.density_kernel, DENSITY_KERNEL);
    assert_eq!(resolved.numerical.correction_kernel, CORRECTION_KERNEL);
    assert_eq!(
        resolved.numerical.solver_update_policy,
        SOLVER_UPDATE_POLICY
    );
    assert_eq!(
        resolved.numerical.fixed_timestep_s.to_bits(),
        (1.0_f32 / 120.0).to_bits()
    );
    assert_eq!(resolved.numerical.solver_iterations, 8);
    assert_eq!(resolved.numerical.support_policy, SUPPORT_POLICY);
    assert_eq!(resolved.numerical.boundary_model, BOUNDARY_MODEL);
    assert_eq!(
        resolved.numerical.neighbour_rebuild_policy,
        NEIGHBOUR_REBUILD_POLICY
    );
    assert_eq!(
        resolved.numerical.particle_order_policy,
        PARTICLE_ORDER_POLICY
    );
    assert_eq!(
        resolved.derived.particle_diameter_m.to_bits(),
        0.05_f32.to_bits()
    );
    assert_eq!(
        resolved.derived.support_radius_m.to_bits(),
        0.1_f32.to_bits()
    );
    assert_eq!(resolved.derived.lattice_offset_count, 27);
    assert!(
        resolved
            .sources
            .values()
            .all(|source| *source == FieldSource::ConfigurationFile)
    );
    resolved
        .pbf_parameters()
        .expect("resolved values should feed the numerical API");
}

#[test]
fn ordinary_valid_configurations_need_no_research_classification() {
    let input = r#"
schema_version = 1
name = "lower-gravity"
description = "A supported numerical configuration"

[numerical]
external_acceleration_m_s2 = [0.0, -1.62, 0.0]
fixed_timestep_s = 0.01
solver_iterations = 6
"#;
    let document = parse_configuration(input).expect("ordinary configuration should parse");
    let resolved = resolve_effective_configuration(Some(&document), None, None)
        .expect("supported numerical changes should resolve without a label");

    assert_eq!(resolved.name, "lower-gravity");
    assert_eq!(resolved.numerical.external_acceleration_m_s2.y, -1.62);
    assert_eq!(resolved.numerical.fixed_timestep_s, 0.01);
    assert_eq!(resolved.numerical.solver_iterations, 6);
    assert_eq!(
        resolved.sources.get("external_acceleration_m_s2"),
        Some(&FieldSource::ConfigurationFile)
    );
}

#[test]
fn schema_unknown_fields_and_missing_required_fields_are_rejected() {
    let unsupported = DEFAULTS_TOML.replacen("schema_version = 1", "schema_version = 2", 1);
    assert!(matches!(
        parse_configuration(&unsupported),
        Err(ConfigError::UnsupportedSchema { found: 2, .. })
    ));
    let old = DEFAULTS_TOML.replacen("schema_version = 1", "schema_version = 0", 1);
    assert!(matches!(
        parse_configuration(&old),
        Err(ConfigError::UnsupportedSchema { found: 0, .. })
    ));
    assert!(matches!(
        parse_configuration(include_str!("fixtures/invalid/config-unknown-field.toml")),
        Err(ConfigError::Parse { .. })
    ));
    let legacy_classification = DEFAULTS_TOML.replacen(
        "description = \"Position-Based Fluids numerical defaults\"",
        "description = \"Position-Based Fluids numerical defaults\"\nkind = \"reference\"",
        1,
    );
    assert!(matches!(
        parse_configuration(&legacy_classification),
        Err(ConfigError::Parse { .. })
    ));
    let private_path = "schema_version = 1\nname = \"unterminated /home/person/private.toml\n";
    let error = parse_configuration(private_path).expect_err("invalid string must fail");
    assert!(!error.to_string().contains("/home/person/private.toml"));
    let missing = "schema_version = 1\nname = \"missing-description\"\n";
    assert!(matches!(
        parse_configuration(missing),
        Err(ConfigError::Parse { .. })
    ));
    let empty_name =
        DEFAULTS_TOML.replacen("name = \"maelstrom-defaults-v4\"", "name = \"   \"", 1);
    assert!(matches!(
        parse_configuration(&empty_name),
        Err(ConfigError::InvalidField { field: "name", .. })
    ));

    let mismatched = parse_configuration(include_str!(
        "fixtures/invalid/config-derived-mismatch.toml"
    ))
    .expect("a contradictory derived assertion is structurally valid");
    assert!(matches!(
        resolve_effective_configuration(Some(&mismatched), None, None),
        Err(ConfigError::DerivedMismatch {
            field: "support_radius_m",
            ..
        })
    ));
}

#[test]
fn every_primary_numeric_domain_rejects_invalid_or_non_finite_values() {
    let invalid = [
        NumericalOverrides {
            external_acceleration_m_s2: Some([0.0, f32::NAN, 0.0]),
            ..Default::default()
        },
        NumericalOverrides {
            fixed_timestep_s: Some(0.0),
            ..Default::default()
        },
        NumericalOverrides {
            fixed_timestep_s: Some(f32::INFINITY),
            ..Default::default()
        },
        NumericalOverrides {
            cfl_factor: Some(-0.1),
            ..Default::default()
        },
        NumericalOverrides {
            cfl_factor: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            particle_spacing_m: Some(0.0),
            ..Default::default()
        },
        NumericalOverrides {
            support_radius_ratio: Some(0.0),
            ..Default::default()
        },
        NumericalOverrides {
            support_radius_ratio: Some(MAX_SUPPORT_RADIUS_RATIO + 0.5),
            ..Default::default()
        },
        NumericalOverrides {
            physical_rest_density_kg_m3: Some(-1.0),
            ..Default::default()
        },
        NumericalOverrides {
            solver_iterations: Some(0),
            ..Default::default()
        },
        NumericalOverrides {
            solver_iterations: Some(MAX_SOLVER_ITERATIONS + 1),
            ..Default::default()
        },
        NumericalOverrides {
            relaxation_scale: Some(0.0),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_strength_scale: Some(-0.01),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_reference_separation_ratio: Some(0.0),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_reference_separation_ratio: Some(1.0),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_reference_separation_ratio: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_exponent: Some(0.0),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_exponent: Some(f32::INFINITY),
            ..Default::default()
        },
    ];

    for overrides in invalid {
        assert!(resolve_overlay(overrides).is_err());
    }
}

#[test]
fn every_floating_primary_field_rejects_a_non_finite_value() {
    let invalid = [
        NumericalOverrides {
            external_acceleration_m_s2: Some([f32::NAN, 0.0, 0.0]),
            ..Default::default()
        },
        NumericalOverrides {
            fixed_timestep_s: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            cfl_factor: Some(f32::INFINITY),
            ..Default::default()
        },
        NumericalOverrides {
            particle_spacing_m: Some(f32::NEG_INFINITY),
            ..Default::default()
        },
        NumericalOverrides {
            support_radius_ratio: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            physical_rest_density_kg_m3: Some(f32::INFINITY),
            ..Default::default()
        },
        NumericalOverrides {
            relaxation_scale: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            correction_scale: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_strength_scale: Some(f32::INFINITY),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_reference_separation_ratio: Some(f32::NAN),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_exponent: Some(f32::NEG_INFINITY),
            ..Default::default()
        },
    ];

    for overrides in invalid {
        assert!(resolve_overlay(overrides).is_err());
    }
}

#[test]
fn different_supported_values_resolve_and_fixed_policy_alternatives_fail() {
    let boundaries = NumericalOverrides {
        external_acceleration_m_s2: Some([f32::MAX, -f32::MAX, 0.0]),
        fixed_timestep_s: Some(0.001),
        cfl_factor: Some(0.001),
        particle_spacing_m: Some(0.01),
        support_radius_ratio: Some(MAX_SUPPORT_RADIUS_RATIO),
        physical_rest_density_kg_m3: Some(1.0),
        solver_iterations: Some(MAX_SOLVER_ITERATIONS),
        relaxation_scale: Some(0.001),
        artificial_pressure_enabled: Some(false),
        artificial_pressure_strength_scale: Some(0.0),
        artificial_pressure_reference_separation_ratio: Some(0.999),
        artificial_pressure_exponent: Some(0.001),
        ..Default::default()
    };
    resolve_overlay(boundaries).expect("inclusive supported boundaries should resolve");

    let unit_support = resolve_effective_configuration(
        None,
        Some(&overlay(NumericalOverrides {
            support_radius_ratio: Some(1.0),
            ..Default::default()
        })),
        None,
    )
    .expect("a different supported ratio is an ordinary configuration");
    assert_eq!(unit_support.derived.lattice_offset_count, 1);

    let invalid_policies = [
        NumericalOverrides {
            scalar_precision: Some("binary64".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            coordinate_system: Some("left_handed_y_up".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            timestep_policy: Some("adaptive".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            density_kernel: Some("cubic_spline".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            correction_kernel: Some("wendland".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            solver_update_policy: Some("gauss_seidel".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            support_policy: Some("less_than_or_equal".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            boundary_model: Some("boundary_particles".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            neighbour_rebuild_policy: Some("every_iteration".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            particle_order_policy: Some("file_order".to_owned()),
            ..Default::default()
        },
        NumericalOverrides {
            xsph_viscosity_enabled: Some(true),
            ..Default::default()
        },
        NumericalOverrides {
            vorticity_confinement_enabled: Some(true),
            ..Default::default()
        },
    ];
    for overrides in invalid_policies {
        assert!(resolve_overlay(overrides).is_err());
    }
}

#[test]
fn invalid_numeric_errors_identify_the_field_to_correct() {
    let cases = [
        (
            "fixed_timestep_s",
            NumericalOverrides {
                fixed_timestep_s: Some(0.0),
                ..Default::default()
            },
        ),
        (
            "cfl_factor",
            NumericalOverrides {
                cfl_factor: Some(f32::NAN),
                ..Default::default()
            },
        ),
        (
            "particle_spacing_m",
            NumericalOverrides {
                particle_spacing_m: Some(-1.0),
                ..Default::default()
            },
        ),
        (
            "support_radius_ratio",
            NumericalOverrides {
                support_radius_ratio: Some(9.0),
                ..Default::default()
            },
        ),
        (
            "solver_iterations",
            NumericalOverrides {
                solver_iterations: Some(0),
                ..Default::default()
            },
        ),
        (
            "artificial_pressure_reference_separation_ratio",
            NumericalOverrides {
                artificial_pressure_reference_separation_ratio: Some(1.0),
                ..Default::default()
            },
        ),
    ];
    for (field, overrides) in cases {
        let error = resolve_effective_configuration(None, Some(&overlay(overrides)), None)
            .expect_err("invalid field must fail");
        assert!(error.to_string().contains(field));
    }
}

#[test]
fn derived_assertions_are_exact_deterministic_checks_not_tuning_inputs() {
    let exact = ConfigurationOverlay {
        numerical: NumericalOverrides::default(),
        derived: DerivedAssertions {
            particle_diameter_m: Some(0.05),
            support_radius_m: Some(0.1),
            ..Default::default()
        },
    };
    resolve_effective_configuration(None, Some(&exact), None)
        .expect("independently specified exact structural values should match");

    let mismatches = [
        DerivedAssertions {
            particle_diameter_m: Some(0.06),
            ..Default::default()
        },
        DerivedAssertions {
            support_radius_m: Some(0.2),
            ..Default::default()
        },
        DerivedAssertions {
            normalised_rest_density_per_m3: Some(1.0),
            ..Default::default()
        },
        DerivedAssertions {
            particle_mass_kg: Some(1.0),
            ..Default::default()
        },
        DerivedAssertions {
            relaxation_per_m2: Some(1.0),
            ..Default::default()
        },
        DerivedAssertions {
            artificial_pressure_strength_m2: Some(1.0),
            ..Default::default()
        },
        DerivedAssertions {
            artificial_pressure_reference_separation_m: Some(1.0),
            ..Default::default()
        },
        DerivedAssertions {
            support_radius_m: Some(f32::NAN),
            ..Default::default()
        },
    ];
    for assertions in mismatches {
        let candidate = ConfigurationOverlay {
            numerical: NumericalOverrides::default(),
            derived: assertions,
        };
        assert!(matches!(
            resolve_effective_configuration(None, Some(&candidate), None),
            Err(ConfigError::DerivedMismatch { .. })
        ));
    }
}

#[test]
fn precedence_and_field_provenance_are_explicit() {
    let file = parse_configuration(
        r#"
schema_version = 1
name = "sparse-file"
description = "A sparse configuration layer"

[numerical]
fixed_timestep_s = 0.01
particle_spacing_m = 0.04
"#,
    )
    .expect("sparse file");
    let scene = overlay(NumericalOverrides {
        fixed_timestep_s: Some(0.02),
        cfl_factor: Some(0.3),
        ..Default::default()
    });
    let command_line = overlay(NumericalOverrides {
        fixed_timestep_s: Some(0.03),
        ..Default::default()
    });

    let resolved = resolve_effective_configuration(Some(&file), Some(&scene), Some(&command_line))
        .expect("valid layered configuration");
    assert_eq!(
        resolved.numerical.fixed_timestep_s.to_bits(),
        0.03_f32.to_bits()
    );
    assert_eq!(
        resolved.sources.get("fixed_timestep_s"),
        Some(&FieldSource::CommandLine)
    );
    assert_eq!(
        resolved.sources.get("cfl_factor"),
        Some(&FieldSource::Scene)
    );
    assert_eq!(
        resolved.sources.get("particle_spacing_m"),
        Some(&FieldSource::ConfigurationFile)
    );
    assert_eq!(
        resolved.sources.get("solver_iterations"),
        Some(&FieldSource::BuiltInDefault)
    );
}

#[test]
fn comparison_identity_uses_resolved_values_not_text_or_provenance() {
    let built_in = resolve_effective_configuration(None, None, None).expect("built-in defaults");
    let from_file = resolve_effective_configuration(Some(&defaults_document()), None, None)
        .expect("configuration file");
    assert_eq!(built_in.comparison_key(), from_file.comparison_key());

    let first = parse_configuration(
        "schema_version = 1\nname = \"first\"\ndescription = \"first prose\"\n",
    )
    .expect("first metadata");
    let second = parse_configuration(
        "schema_version = 1\nname = \"second\"\ndescription = \"different prose\"\n",
    )
    .expect("second metadata");
    let first = resolve_effective_configuration(Some(&first), None, None).expect("first");
    let second = resolve_effective_configuration(Some(&second), None, None).expect("second");
    assert_eq!(first.comparison_key(), second.comparison_key());
    assert_ne!(first.canonical_toml(), second.canonical_toml());
}

#[test]
fn historical_defaults_remain_explicit_and_have_distinct_comparison_identities() {
    let current = resolve_effective_configuration(None, None, None).unwrap();
    assert_eq!(current.name, "maelstrom-defaults-v4");
    assert_eq!(
        current.numerical.relaxation_scale.to_bits(),
        20.0_f32.to_bits()
    );
    let mut keys = vec![current.comparison_key()];
    for (source, name, scale) in [
        (
            include_str!("../config/defaults-v1.toml"),
            "maelstrom-defaults-v1",
            1.0e-6_f32,
        ),
        (
            include_str!("../config/defaults-v2.toml"),
            "maelstrom-defaults-v2",
            1.0_f32,
        ),
        (
            include_str!("../config/defaults-v3.toml"),
            "maelstrom-defaults-v3",
            10.0_f32,
        ),
    ] {
        let document = parse_configuration(source).expect("historical configuration");
        let historical = resolve_effective_configuration(Some(&document), None, None).unwrap();
        assert_eq!(historical.name, name);
        assert_eq!(
            historical.numerical.relaxation_scale.to_bits(),
            scale.to_bits()
        );
        let key = historical.comparison_key();
        assert!(!keys.contains(&key));
        keys.push(key);
        assert_eq!(
            historical.derived.particle_mass_kg,
            current.derived.particle_mass_kg
        );
        assert_eq!(
            historical.derived.normalised_rest_density_per_m3,
            current.derived.normalised_rest_density_per_m3
        );
    }
}

#[test]
fn timestep_refinement_records_scaled_values_and_preserves_other_inputs() {
    let original = resolve_effective_configuration(None, None, None).unwrap();
    for factor in [1, 2, 4, 16] {
        let layer = maelstrom::config::parse_configuration_overlay(&format!(
            "[numerical]\nrefine_timestep = {factor}"
        ))
        .unwrap();
        let refined = resolve_effective_configuration(None, None, Some(&layer)).unwrap();
        let ratio = factor as f32;
        assert_eq!(
            refined.numerical.fixed_timestep_s,
            original.numerical.fixed_timestep_s / ratio
        );
        assert_eq!(
            refined.numerical.relaxation_scale,
            original.numerical.relaxation_scale
        );
        assert_eq!(
            refined.numerical.artificial_pressure_strength_scale,
            original.numerical.artificial_pressure_strength_scale
        );
        assert_eq!(
            refined.derived.support_radius_m,
            original.derived.support_radius_m
        );
        assert_eq!(
            refined.derived.particle_mass_kg,
            original.derived.particle_mass_kg
        );
        assert_eq!(refined.numerical.solver_iterations, 8);
        assert_eq!(
            refined.numerical.correction_scale,
            original.numerical.correction_scale / (ratio * ratio)
        );
        // Resolved records contain values, so reading them cannot apply the
        // refinement recipe a second time. Inspection TOML also omits the recipe.
        let values = serde_json::to_value(&refined.numerical).unwrap();
        assert!(values.get("refine_timestep").is_none());
        assert!(!refined.canonical_toml().contains("refine_timestep"));
        if factor == 1 {
            assert_eq!(original, refined);
        } else {
            assert_ne!(original.comparison_key(), refined.comparison_key());
        }
    }
}

#[test]
fn timestep_refinement_follows_overrides_and_rejects_invalid_or_overflowing_inputs() {
    use maelstrom::config::parse_configuration_overlay;
    let layer = parse_configuration_overlay("[numerical]\nrefine_timestep=2\nfixed_timestep_s=0.125\nrelaxation_scale=2.0\nartificial_pressure_strength_scale=0.25").unwrap();
    let resolved = resolve_effective_configuration(None, None, Some(&layer)).unwrap();
    assert_eq!(resolved.numerical.fixed_timestep_s, 0.0625);
    assert_eq!(resolved.numerical.relaxation_scale, 2.0);
    assert_eq!(resolved.numerical.correction_scale, 0.25);
    assert_eq!(resolved.numerical.artificial_pressure_strength_scale, 0.25);
    for input in [
        "refine_timestep=0",
        "refine_timestep=17",
        "correction_scale=1.1",
        "correction_scale=0.0",
        "refine_timestep=16\nfixed_timestep_s=1e-45",
    ] {
        let layer = parse_configuration_overlay(&format!("[numerical]\n{input}")).unwrap();
        assert!(
            resolve_effective_configuration(None, None, Some(&layer)).is_err(),
            "{input}"
        );
    }
    // A later layer composes with the earlier layer's actual resolved values.
    let once = parse_configuration_overlay("[numerical]\nrefine_timestep=2").unwrap();
    let twice = resolve_effective_configuration(None, Some(&once), Some(&once)).unwrap();
    assert_eq!(twice.numerical.relaxation_scale, 20.0);
    assert_eq!(twice.numerical.correction_scale, 0.0625);
}

#[test]
fn every_supported_simulation_value_change_changes_comparison_identity() {
    let baseline = resolve_effective_configuration(None, None, None)
        .expect("built-in defaults")
        .comparison_key();
    let changes = [
        NumericalOverrides {
            external_acceleration_m_s2: Some([1.0, -9.81, 0.0]),
            ..Default::default()
        },
        NumericalOverrides {
            fixed_timestep_s: Some(0.01),
            ..Default::default()
        },
        NumericalOverrides {
            cfl_factor: Some(0.5),
            ..Default::default()
        },
        NumericalOverrides {
            particle_spacing_m: Some(0.04),
            ..Default::default()
        },
        NumericalOverrides {
            support_radius_ratio: Some(1.5),
            ..Default::default()
        },
        NumericalOverrides {
            physical_rest_density_kg_m3: Some(900.0),
            ..Default::default()
        },
        NumericalOverrides {
            solver_iterations: Some(5),
            ..Default::default()
        },
        NumericalOverrides {
            relaxation_scale: Some(2.0e-6),
            ..Default::default()
        },
        NumericalOverrides {
            correction_scale: Some(0.25),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_enabled: Some(false),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_strength_scale: Some(0.2),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_reference_separation_ratio: Some(0.3),
            ..Default::default()
        },
        NumericalOverrides {
            artificial_pressure_exponent: Some(5.0),
            ..Default::default()
        },
    ];

    for change in changes {
        let changed = resolve_effective_configuration(None, Some(&overlay(change)), None)
            .expect("supported change");
        assert_ne!(changed.comparison_key(), baseline);
    }
}

#[test]
fn canonical_inspection_and_resolution_are_exactly_repeatable() {
    let document = defaults_document();
    let first =
        resolve_effective_configuration(Some(&document), None, None).expect("first resolution");
    let second =
        resolve_effective_configuration(Some(&document), None, None).expect("second resolution");

    assert_eq!(first, second);
    assert_eq!(first.canonical_toml(), second.canonical_toml());
    assert_eq!(first.comparison_key(), second.comparison_key());
    assert!(first.canonical_toml().contains("[provenance]"));
    assert!(!first.canonical_toml().contains("experiment"));
}

#[test]
fn failed_resolution_cannot_mutate_a_previously_resolved_value() {
    let accepted = resolve_effective_configuration(None, None, None).expect("built-in defaults");
    let snapshot = accepted.clone();
    let invalid = overlay(NumericalOverrides {
        particle_spacing_m: Some(f32::NAN),
        ..Default::default()
    });
    assert!(resolve_effective_configuration(None, Some(&invalid), None).is_err());
    assert_eq!(accepted, snapshot);
}
