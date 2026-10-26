//! Independently listed positions and planes distinguish authoring shortcuts
//! from changes to the solver's existing flattened input contract.

use maelstrom::{
    math::Vec3,
    scene::{
        ContainerBox, MAX_PARTICLE_GROUPS, MAX_PLANES, ParticleDocument, ParticleGroup,
        ParticleSource, SceneError, parse_scene, resolve_scene,
    },
};

fn prefix() -> &'static str {
    "schema_version=2\nname='groups'\ndescription='two sources'\nrandomness='none'\n[configuration]\nsource='built_in_defaults'\n[configuration.overrides.numerical]\nparticle_spacing_m=0.125\n[particles]\nconstruction='groups'\n"
}

fn explicit(name: &str, items: &[(u64, [f32; 3])]) -> ParticleGroup {
    ParticleGroup {
        name: name.into(),
        source: ParticleSource::Explicit {
            items: items
                .iter()
                .map(|&(identity, position_m)| ParticleDocument {
                    identity,
                    position_m,
                    velocity_m_s: [0.0; 3],
                })
                .collect(),
        },
    }
}

fn empty_document() -> maelstrom::scene::SceneDocument {
    parse_scene(&format!("{}groups=[]", prefix())).unwrap()
}

#[test]
fn mixed_sources_preserve_explicit_ids_and_automatically_continue_lattice_ranges() {
    let source = format!(
        "{}{}{}",
        prefix(),
        lattice("first", "[0.0,0.0,0.0]", "[1,1,1]"),
        lattice("last", "[0.5,0.0,0.0]", "[1,1,2]")
    );
    let mut doc = parse_scene(&source).unwrap();
    let ParticleSource::Groups { groups } = &mut doc.particles else {
        panic!()
    };
    groups.insert(1, explicit("listed", &[(7, [0.25, 0.0, 0.0])]));
    let resolved = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(
        resolved
            .particles
            .iter()
            .map(|p| p.identity)
            .collect::<Vec<_>>(),
        [0, 7, 8, 9]
    );
    let ParticleSource::Groups { groups } = &mut doc.particles else {
        panic!()
    };
    groups[1] = explicit("listed", &[(u64::MAX, [0.25, 0.0, 0.0])]);
    assert!(resolve_scene(&doc, None, None).is_err());
}

#[test]
fn names_ranges_and_external_group_fields_are_checked() {
    for (left, right) in [("same", "same"), ("", "right"), ("   ", "right")] {
        let source = format!(
            "{}{}{}",
            prefix(),
            lattice(left, "[0.0,0.0,0.0]", "[1,1,1]"),
            lattice(right, "[0.5,0.0,0.0]", "[1,1,1]")
        );
        assert!(resolve_scene(&parse_scene(&source).unwrap(), None, None).is_err());
    }
    let mut doc = empty_document();
    doc.particles = ParticleSource::Groups {
        groups: vec![
            explicit("first", &[(10, [0.0, 0.0, 0.0])]),
            explicit("second", &[(10, [0.5, 0.0, 0.0])]),
        ],
    };
    assert!(resolve_scene(&doc, None, None).is_err());
    doc.particles = ParticleSource::Groups {
        groups: vec![ParticleGroup {
            name: "nested".into(),
            source: ParticleSource::Groups { groups: vec![] },
        }],
    };
    assert!(resolve_scene(&doc, None, None).is_err());
    doc.particles = ParticleSource::Groups {
        groups: (0..=MAX_PARTICLE_GROUPS)
            .map(|i| explicit(&format!("group-{i}"), &[]))
            .collect(),
    };
    assert!(resolve_scene(&doc, None, None).is_err());
    let source = format!(
        "{}{}",
        prefix(),
        lattice("source", "[0.0,0.0,0.0]", "[1,1,1]")
    );
    for field in [
        "unknown=1",
        "density=1000.0",
        "first_identity=-1",
        "initial_velocity_m_s=[nan,0.0,0.0]",
    ] {
        let invalid = source.replace("counts=[1,1,1]", &format!("counts=[1,1,1]\n{field}"));
        let rejected = match parse_scene(&invalid) {
            Ok(doc) => resolve_scene(&doc, None, None).is_err(),
            Err(_) => true,
        };
        assert!(rejected, "{field}");
    }
}

#[test]
fn spatial_validation_matches_independent_all_pairs_at_faces_edges_and_corners() {
    let mut doc = empty_document();
    for origin in [-0.15625f32, -0.03125, 0.09375] {
        for dx in [0.0f32, 0.0625, 0.125, 0.1875] {
            for dy in [0.0f32, 0.0625, 0.125, 0.1875] {
                for dz in [0.0f32, 0.0625, 0.125, 0.1875] {
                    let positions = [
                        [origin; 3],
                        [origin + dx, origin + dy, origin + dz],
                        [4.0, 4.0, 4.0],
                    ];
                    for coincident in [false, true] {
                        doc.allow_coincident_particles = coincident;
                        doc.particles = ParticleSource::Groups {
                            groups: positions
                                .iter()
                                .enumerate()
                                .map(|(i, &p)| explicit(&format!("g-{i}"), &[(i as u64, p)]))
                                .collect(),
                        };
                        let mut expected = true;
                        for i in 0..positions.len() {
                            for j in i + 1..positions.len() {
                                let distance = (0..3)
                                    .map(|a| {
                                        (f64::from(positions[i][a]) - f64::from(positions[j][a]))
                                            .powi(2)
                                    })
                                    .sum::<f64>()
                                    .sqrt();
                                if distance + f64::from(0.125f32 * 8.0 * f32::EPSILON) < 0.125
                                    && !(coincident && positions[i] == positions[j])
                                {
                                    expected = false;
                                }
                            }
                        }
                        assert_eq!(
                            resolve_scene(&doc, None, None).is_ok(),
                            expected,
                            "{positions:?}; coincident {coincident}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn overlap_errors_are_deterministic_and_cover_both_within_and_between_sources() {
    let mut doc = empty_document();
    doc.particles = ParticleSource::Groups {
        groups: vec![
            explicit("one", &[(0, [-0.0625, 0.0, 0.0]), (1, [0.0625, 0.0, 0.0])]),
            explicit("two", &[(2, [0.0; 3])]),
        ],
    };
    for _ in 0..12 {
        assert!(matches!(
            resolve_scene(&doc, None, None),
            Err(SceneError::OverlappingParticles {
                first: 0,
                second: 2
            })
        ));
    }
    doc.particles = ParticleSource::Groups {
        groups: vec![explicit("one", &[(0, [0.0; 3]), (1, [0.0625, 0.0, 0.0])])],
    };
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::OverlappingParticles {
            first: 0,
            second: 1
        })
    ));
    doc.allow_coincident_particles = true;
    doc.particles = ParticleSource::Groups {
        groups: vec![
            explicit("one", &[(0, [0.0; 3]), (1, [0.0; 3])]),
            explicit("two", &[(2, [0.125, 0.0, 0.0])]),
        ],
    };
    assert!(resolve_scene(&doc, None, None).is_ok());
    let ParticleSource::Groups { groups } = &mut doc.particles else {
        panic!()
    };
    let ParticleSource::Explicit { items } = &mut groups[0].source else {
        panic!()
    };
    items[1].position_m = [1e-30, 0.0, 0.0];
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::OverlappingParticles {
            first: 0,
            second: 1
        })
    ));
}

#[test]
fn box_forms_expand_to_exact_planes_and_match_an_explicit_legacy_scene() {
    let mut doc = empty_document();
    doc.particles = ParticleSource::Explicit { items: vec![] };
    doc.container = Some(ContainerBox {
        min_m: Some([-1.0, -2.0, -3.0]),
        max_m: Some([1.0, 2.0, 3.0]),
        ..Default::default()
    });
    let bounds = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(bounds.planes.len(), 6);
    let expected = [
        ([1.0, 0.0, 0.0], -1.0),
        ([-1.0, 0.0, 0.0], -1.0),
        ([0.0, 1.0, 0.0], -2.0),
        ([0.0, -1.0, 0.0], -2.0),
        ([0.0, 0.0, 1.0], -3.0),
        ([0.0, 0.0, -1.0], -3.0),
    ];
    for (plane, (normal, offset)) in bounds.planes.iter().zip(expected) {
        assert_eq!(plane.normal(), Vec3::new(normal[0], normal[1], normal[2]));
        assert_eq!(plane.offset(), offset);
    }
    doc.container = Some(ContainerBox {
        origin_m: Some([-1.0, -2.0, -3.0]),
        dimensions_m: Some([2.0, 4.0, 6.0]),
        ..Default::default()
    });
    let dimensions = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(dimensions.planes, bounds.planes);
    doc.container = None;
    doc.schema_version = 1;
    doc.planes = expected
        .into_iter()
        .map(|(normal, offset_m)| maelstrom::scene::PlaneDocument { normal, offset_m })
        .collect();
    let explicit = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(bounds.comparison_key(), explicit.comparison_key());
}

#[test]
fn invalid_boxes_additional_planes_and_out_of_bounds_particles_are_rejected() {
    let mut doc = empty_document();
    for container in [
        ContainerBox::default(),
        ContainerBox {
            min_m: Some([0.0; 3]),
            ..Default::default()
        },
        ContainerBox {
            min_m: Some([0.0; 3]),
            max_m: Some([0.0, 1.0, 1.0]),
            ..Default::default()
        },
        ContainerBox {
            min_m: Some([f32::NAN, 0.0, 0.0]),
            max_m: Some([1.0; 3]),
            ..Default::default()
        },
        ContainerBox {
            origin_m: Some([1e20, 0.0, 0.0]),
            dimensions_m: Some([1.0; 3]),
            ..Default::default()
        },
        ContainerBox {
            origin_m: Some([f32::MAX, 0.0, 0.0]),
            dimensions_m: Some([f32::MAX, 1.0, 1.0]),
            ..Default::default()
        },
        ContainerBox {
            origin_m: Some([0.0; 3]),
            dimensions_m: Some([-1.0, 1.0, 1.0]),
            ..Default::default()
        },
        ContainerBox {
            min_m: Some([0.0; 3]),
            max_m: Some([1.0; 3]),
            origin_m: Some([0.0; 3]),
            dimensions_m: None,
        },
    ] {
        doc.container = Some(container);
        assert!(resolve_scene(&doc, None, None).is_err());
    }
    doc.container = Some(ContainerBox {
        min_m: Some([0.0; 3]),
        max_m: Some([1.0; 3]),
        ..Default::default()
    });
    doc.planes = vec![maelstrom::scene::PlaneDocument {
        normal: [0.0, 1.0, 0.0],
        offset_m: 0.5,
    }];
    let scene = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(scene.planes.len(), 7);
    assert_eq!(scene.planes[6].offset(), 0.5);
    doc.planes[0].offset_m = 2.0;
    assert!(resolve_scene(&doc, None, None).is_err());
    doc.planes = vec![
        maelstrom::scene::PlaneDocument {
            normal: [1.0, 0.0, 0.0],
            offset_m: 0.0
        };
        MAX_PLANES - 5
    ];
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::TooManyPlanes { .. })
    ));
    doc.planes.clear();
    doc.particles = ParticleSource::Groups {
        groups: vec![explicit("outside", &[(0, [1.125, 0.0, 0.0])])],
    };
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::ParticleOutsidePlane { .. })
    ));
}

#[test]
fn grouped_count_limits_fail_before_generation_and_large_regular_scenes_construct() {
    let too_many = format!(
        "{}{}{}",
        prefix(),
        lattice("one", "[0.0,0.0,0.0]", "[500001,1,1]"),
        lattice("two", "[0.0,0.0,0.0]", "[500001,1,1]")
    );
    let doc = parse_scene(&too_many).unwrap();
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::ParticleCountTooLarge {
            construction: "groups",
            ..
        })
    ));
    let source = format!(
        "{}{}{}",
        prefix(),
        lattice("one", "[0.0,0.0,0.0]", "[50,50,20]"),
        lattice("two", "[16.0,0.0,0.0]", "[50,50,20]")
    );
    let scene = resolve_scene(&parse_scene(&source).unwrap(), None, None).unwrap();
    assert_eq!(scene.particles.len(), 100000);
    assert_eq!(scene.particles[49999].identity, 49999);
    assert_eq!(scene.particles[50000].position_m, Vec3::new(16.0, 0.0, 0.0));
    assert_eq!(scene.particles[99999].identity, 99999);
}

#[test]
fn grouped_explicit_sources_use_the_total_limit_and_empty_sources_remain_valid() {
    let mut doc = empty_document();
    assert!(
        resolve_scene(&doc, None, None)
            .unwrap()
            .particles
            .is_empty()
    );
    let items: Vec<_> = (0..5000)
        .map(|i| (i as u64, [i as f32 * 0.125, 0.0, 0.0]))
        .collect();
    let group = explicit("generated", &items);
    doc.particles = ParticleSource::Groups {
        groups: vec![group.clone()],
    };
    assert_eq!(
        resolve_scene(&doc, None, None).unwrap().particles.len(),
        5000
    );
    doc.particles = group.source;
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::ParticleCountTooLarge { maximum: 4096, .. })
    ));
}

#[test]
fn legacy_documents_reject_new_shapes_and_grouped_sources_reject_invalid_coordinates() {
    let source = format!("{}{}", prefix(), lattice("one", "[0.0,0.0,0.0]", "[1,1,1]"));
    assert!(parse_scene(&source.replace("schema_version=2", "schema_version=1")).is_err());
    let mut doc = empty_document();
    doc.particles = ParticleSource::Groups {
        groups: vec![explicit("far", &[(0, [1e20, 0.0, 0.0])])],
    };
    assert!(resolve_scene(&doc, None, None).is_err());
    doc.particles = ParticleSource::Groups {
        groups: vec![explicit("nonfinite", &[(0, [f32::INFINITY, 0.0, 0.0])])],
    };
    assert!(resolve_scene(&doc, None, None).is_err());
    doc.particles = ParticleSource::Explicit { items: vec![] };
    doc.schema_version = 1;
    doc.container = Some(ContainerBox {
        min_m: Some([0.0; 3]),
        max_m: Some([1.0; 3]),
        ..Default::default()
    });
    assert!(resolve_scene(&doc, None, None).is_err());
}

#[test]
fn two_block_fixture_matches_independently_written_legacy_inputs_and_interacts() {
    use maelstrom::{
        config::parse_configuration,
        neighbours::SearchBackend,
        scene::{PlaneDocument, load_resolved_scene},
        simulation::Simulation,
    };
    let scene = load_resolved_scene(
        std::path::Path::new("tests/fixtures/two-blocks/scene.toml"),
        None,
    )
    .unwrap();
    assert_eq!(scene.particles.len(), 128);
    assert_eq!(scene.particles[0].velocity_m_s, Vec3::new(0.25, 0.0, 0.0));
    assert_eq!(scene.particles[64].velocity_m_s, Vec3::new(-0.25, 0.0, 0.0));
    let mut legacy = parse_scene(include_str!("fixtures/two-blocks/scene.toml")).unwrap();
    legacy.schema_version = 1;
    legacy.run = None;
    legacy.container = None;
    legacy.particles = ParticleSource::Explicit {
        items: (0u64..128)
            .map(|identity| {
                let local = identity % 64;
                let origin = if identity < 64 { 0.1f32 } else { 0.55 };
                ParticleDocument {
                    identity,
                    position_m: [
                        origin + 0.05 * (local / 16) as f32,
                        0.25 + 0.05 * ((local / 4) % 4) as f32,
                        0.1 + 0.05 * (local % 4) as f32,
                    ],
                    velocity_m_s: [if identity < 64 { 0.25 } else { -0.25 }, 0.0, 0.0],
                }
            })
            .collect(),
    };
    legacy.planes = vec![
        PlaneDocument {
            normal: [1.0, 0.0, 0.0],
            offset_m: 0.0,
        },
        PlaneDocument {
            normal: [-1.0, 0.0, 0.0],
            offset_m: -0.8,
        },
        PlaneDocument {
            normal: [0.0, 1.0, 0.0],
            offset_m: 0.0,
        },
        PlaneDocument {
            normal: [0.0, -1.0, 0.0],
            offset_m: -0.7,
        },
        PlaneDocument {
            normal: [0.0, 0.0, 1.0],
            offset_m: 0.0,
        },
        PlaneDocument {
            normal: [0.0, 0.0, -1.0],
            offset_m: -0.6,
        },
    ];
    let config = parse_configuration(
        &include_str!("fixtures/two-blocks/config.toml")
            .replace("schema_version = 2", "schema_version = 1"),
    )
    .unwrap();
    let reference = resolve_scene(&legacy, Some(&config), None).unwrap();
    assert_eq!(scene.particles, reference.particles);
    assert_eq!(scene.planes, reference.planes);
    assert_eq!(scene.comparison_key(), reference.comparison_key());
    let cross_support = |particles: &[maelstrom::scene::Particle]| {
        particles[..64].iter().any(|a| {
            particles[64..].iter().any(|b| {
                let delta = [
                    f64::from(a.position_m.x) - f64::from(b.position_m.x),
                    f64::from(a.position_m.y) - f64::from(b.position_m.y),
                    f64::from(a.position_m.z) - f64::from(b.position_m.z),
                ];
                delta.iter().map(|x| x * x).sum::<f64>()
                    < f64::from(scene.configuration.derived.support_radius_m).powi(2)
            })
        })
    };
    assert!(!cross_support(&scene.particles));
    let mut solver =
        Simulation::from_scene_with_backend(&scene, SearchBackend::UniformGrid).unwrap();
    let mut interacted = false;
    for completed in 1..=480 {
        let outcome = solver.step().unwrap();
        assert_eq!(outcome.completed_steps, completed);
        interacted |= cross_support(solver.snapshot().particles());
    }
    assert!(
        interacted,
        "the example must actually bring particles from both initial blocks into support"
    );
    for particle in solver.snapshot().particles() {
        for plane in &scene.planes {
            assert!(plane.contains(particle.position_m).unwrap());
        }
    }
}

fn lattice(name: &str, origin: &str, counts: &str) -> String {
    format!(
        "\n[[particles.groups]]\nname='{name}'\nconstruction='lattice'\norigin_m={origin}\ncounts={counts}\nendpoint_rule='counted_half_open'\norder='x_then_y_then_z'\ninitial_velocity_m_s=[1.0,0.0,0.0]\n"
    )
}

#[test]
fn groups_parse_and_flatten_in_document_order_with_independent_expected_positions() {
    let source = format!(
        "{}{}{}",
        prefix(),
        lattice("z-first", "[0.0,0.0,0.0]", "[1,1,2]"),
        lattice("a-second", "[0.5,0.0,0.0]", "[2,1,1]")
    );
    let doc = parse_scene(&source).unwrap();
    let scene = resolve_scene(&doc, None, None).unwrap();
    assert_eq!(scene.particles.len(), 4);
    let expected = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.125],
        [0.5, 0.0, 0.0],
        [0.625, 0.0, 0.0],
    ];
    for (i, position) in expected.into_iter().enumerate() {
        assert_eq!(scene.particles[i].identity, i as u64);
        assert_eq!(
            scene.particles[i].position_m,
            Vec3::new(position[0], position[1], position[2])
        );
        assert_eq!(scene.particles[i].velocity_m_s, Vec3::new(1.0, 0.0, 0.0));
    }
    assert_eq!(
        scene,
        resolve_scene(&parse_scene(&source).unwrap(), None, None).unwrap()
    );
    let encoded = toml::to_string_pretty(&doc).unwrap();
    assert_eq!(doc, parse_scene(&encoded).unwrap());
    assert!(matches!(doc.particles, ParticleSource::Groups { .. }));
    assert!(
        parse_scene(&source.replace("name='z-first'", "name='z-first'\nunknown_field=2")).is_err()
    );
}

#[test]
fn translated_decimal_lattice_groups_match_standalone_construction() {
    let source = format!(
        "{}{}",
        prefix().replace("particle_spacing_m=0.125", "particle_spacing_m=0.025"),
        lattice("fluid", "[0.1,0.75,0.1]", "[20,25,20]")
    );
    let grouped = parse_scene(&source).unwrap();
    let mut standalone = grouped.clone();
    let ParticleSource::Groups { groups } = &grouped.particles else {
        panic!()
    };
    standalone.particles = groups[0].source.clone();
    let expected = resolve_scene(&standalone, None, None).unwrap();
    assert_eq!(expected.particles.len(), 10_000);
    assert_eq!(
        resolve_scene(&grouped, None, None).unwrap().particles,
        expected.particles
    );
}

#[test]
fn decimal_lattice_validation_still_rejects_collapsed_axes_and_cross_group_overlap() {
    let source = format!(
        "{}{}{}",
        prefix().replace("particle_spacing_m=0.125", "particle_spacing_m=0.025"),
        lattice("left", "[0.1,0.75,0.1]", "[20,25,20]"),
        lattice("right", "[1.1,0.75,0.1]", "[20,25,20]")
    );
    let mut doc = parse_scene(&source).unwrap();
    assert_eq!(
        resolve_scene(&doc, None, None).unwrap().particles.len(),
        20_000
    );
    let ParticleSource::Groups { groups } = &mut doc.particles else {
        panic!()
    };
    let ParticleSource::Lattice { origin_m, .. } = &mut groups[1].source else {
        panic!()
    };
    *origin_m = [0.11, 0.75, 0.1];
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::OverlappingParticles { .. })
    ));
    let collapsed = source.replace("[1.1,0.75,0.1]", "[10000000.0,0.75,0.1]");
    assert!(resolve_scene(&parse_scene(&collapsed).unwrap(), None, None).is_err());
}

#[test]
fn lattice_range_exemptions_do_not_extend_to_explicit_sources() {
    let source = format!(
        "{}{}",
        prefix(),
        lattice("grid", "[0.0,0.0,0.0]", "[1,1,2]")
    );
    let mut doc = parse_scene(&source).unwrap();
    let ParticleSource::Groups { groups } = &mut doc.particles else {
        panic!()
    };
    groups.push(explicit("empty", &[]));
    groups.push(explicit(
        "listed",
        &[(2, [2.0, 0.0, 0.0]), (3, [2.0625, 0.0, 0.0])],
    ));
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::OverlappingParticles {
            first: 2,
            second: 3
        })
    ));
    let ParticleSource::Groups { groups } = &mut doc.particles else {
        panic!()
    };
    groups.pop();
    groups.push(explicit("listed", &[(2, [0.0625, 0.0, 0.0])]));
    assert!(matches!(
        resolve_scene(&doc, None, None),
        Err(SceneError::OverlappingParticles {
            first: 0,
            second: 2
        })
    ));
}
