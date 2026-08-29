use std::{
    fs,
    path::PathBuf,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use maelstrom::{
    NumericalError,
    config::{ConfigurationOverlay, DerivedAssertions, FieldSource, NumericalOverrides},
    math::Vec3,
    scene::{
        ConfigurationSelection, ConfigurationSource, LatticeEndpointRule, LatticeOrder,
        MAX_EXPLICIT_PARTICLES, MAX_LATTICE_PARTICLES, MAX_PLANES, ParticleDocument,
        ParticleSource, PlaneDocument, RandomnessPolicy, SceneDocument, SceneError,
        load_resolved_scene, parse_scene, resolve_scene,
    },
};

const DEFAULTS_TOML: &str = include_str!("../config/defaults-v1.toml");
const EXTERNAL_SCENE_TOML: &str = include_str!("../scenes/falling-block-06.toml");
static CURRENT_DIRECTORY_LOCK: Mutex<()> = Mutex::new(());

struct CurrentDirectoryGuard {
    original: PathBuf,
    temporary_root: PathBuf,
}

impl Drop for CurrentDirectoryGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
        let _ = fs::remove_dir_all(&self.temporary_root);
    }
}

fn base_document(particles: ParticleSource) -> SceneDocument {
    SceneDocument {
        schema_version: 1,
        name: "scene-contract-test".to_owned(),
        description: "Focused scene contract input".to_owned(),
        randomness: RandomnessPolicy::None,
        allow_coincident_particles: false,
        configuration: ConfigurationSelection {
            source: ConfigurationSource::BuiltInDefaults,
            path: None,
            overrides: None,
        },
        particles,
        planes: Vec::new(),
    }
}

fn explicit_particle(identity: u64, position_m: [f32; 3]) -> ParticleDocument {
    ParticleDocument {
        identity,
        position_m,
        velocity_m_s: [0.0, 0.0, 0.0],
    }
}

fn explicit_document(items: Vec<ParticleDocument>) -> SceneDocument {
    base_document(ParticleSource::Explicit { items })
}

fn lattice_document(origin_m: [f32; 3], counts: [u32; 3]) -> SceneDocument {
    base_document(ParticleSource::Lattice {
        first_identity: 10,
        origin_m,
        counts,
        spacing_m: 0.05,
        endpoint_rule: LatticeEndpointRule::CountedHalfOpen,
        order: LatticeOrder::XThenYThenZ,
        initial_velocity_m_s: [1.0, 2.0, 3.0],
    })
}

fn plane(normal: [f32; 3], offset_m: f32) -> PlaneDocument {
    PlaneDocument { normal, offset_m }
}

#[test]
fn lattice_counts_positions_identities_and_endpoint_rule_are_exact() {
    let document = lattice_document([1.0, 2.0, 3.0], [2, 2, 2]);
    let scene = resolve_scene(&document, None, None).expect("small lattice should resolve");
    let expected = [
        (10, [1.0, 2.0, 3.0]),
        (11, [1.0, 2.0, 3.05]),
        (12, [1.0, 2.05, 3.0]),
        (13, [1.0, 2.05, 3.05]),
        (14, [1.05, 2.0, 3.0]),
        (15, [1.05, 2.0, 3.05]),
        (16, [1.05, 2.05, 3.0]),
        (17, [1.05, 2.05, 3.05]),
    ];

    assert_eq!(scene.particles.len(), expected.len());
    for (particle, (identity, position)) in scene.particles.iter().zip(expected) {
        assert_eq!(particle.identity, identity);
        assert_eq!(
            particle.position_m,
            Vec3::new(position[0], position[1], position[2])
        );
        assert_eq!(particle.velocity_m_s, Vec3::new(1.0, 2.0, 3.0));
        assert_ne!(particle.position_m.x, 1.1);
        assert_ne!(particle.position_m.y, 2.1);
        assert_ne!(particle.position_m.z, 3.1);
    }
}

#[test]
fn explicit_identity_and_plane_document_order_are_preserved() {
    let mut document = explicit_document(vec![
        explicit_particle(4, [0.0, 0.0, 0.0]),
        explicit_particle(9, [0.05, 0.0, 0.0]),
    ]);
    document.planes = vec![
        plane([1.0, 0.0, 0.0], 0.0),
        plane([0.0, 1.0, 0.0], 0.0),
        plane([0.0, 0.0, 1.0], 0.0),
    ];
    let scene = resolve_scene(&document, None, None).expect("ordered scene should resolve");

    assert_eq!(
        scene
            .particles
            .iter()
            .map(|particle| particle.identity)
            .collect::<Vec<_>>(),
        [4, 9]
    );
    assert_eq!(scene.planes[0].normal(), Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(scene.planes[1].normal(), Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(scene.planes[2].normal(), Vec3::new(0.0, 0.0, 1.0));
}

#[test]
fn duplicate_or_descending_identities_are_rejected() {
    let duplicate = parse_scene(include_str!(
        "fixtures/invalid/scene-duplicate-identity.toml"
    ))
    .expect("malformed semantic fixture must still parse");
    assert!(matches!(
        resolve_scene(&duplicate, None, None),
        Err(SceneError::ParticleIdentityOrder {
            previous: 7,
            next: 7
        })
    ));
    let descending = explicit_document(vec![
        explicit_particle(2, [0.0, 0.0, 0.0]),
        explicit_particle(1, [0.05, 0.0, 0.0]),
    ]);
    assert!(matches!(
        resolve_scene(&descending, None, None),
        Err(SceneError::ParticleIdentityOrder {
            previous: 2,
            next: 1
        })
    ));
}

#[test]
fn non_finite_particles_planes_and_offsets_are_rejected() {
    let invalid_position = explicit_document(vec![explicit_particle(0, [f32::NAN, 0.0, 0.0])]);
    assert!(matches!(
        resolve_scene(&invalid_position, None, None),
        Err(SceneError::Numerical(
            NumericalError::NonFiniteVector { .. }
        ))
    ));
    let mut invalid_velocity = explicit_particle(0, [0.0, 0.0, 0.0]);
    invalid_velocity.velocity_m_s = [0.0, f32::INFINITY, 0.0];
    assert!(matches!(
        resolve_scene(&explicit_document(vec![invalid_velocity]), None, None),
        Err(SceneError::Numerical(
            NumericalError::NonFiniteVector { .. }
        ))
    ));
    let mut invalid_normal = explicit_document(Vec::new());
    invalid_normal.planes = vec![plane([0.0, f32::INFINITY, 0.0], 0.0)];
    assert!(matches!(
        resolve_scene(&invalid_normal, None, None),
        Err(SceneError::Numerical(
            NumericalError::NonFiniteVector { .. }
        ))
    ));
    let mut invalid_offset = explicit_document(Vec::new());
    invalid_offset.planes = vec![plane([0.0, 1.0, 0.0], f32::NAN)];
    assert!(matches!(
        resolve_scene(&invalid_offset, None, None),
        Err(SceneError::Numerical(
            NumericalError::NonFiniteScalar { .. }
        ))
    ));
}

#[test]
fn zero_non_unit_and_contradictory_planes_are_rejected() {
    for normal in [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]] {
        let mut document = explicit_document(Vec::new());
        document.planes = vec![plane(normal, 0.0)];
        assert!(resolve_scene(&document, None, None).is_err());
    }
    let mut contradictory = explicit_document(Vec::new());
    contradictory.planes = vec![plane([0.0, 1.0, 0.0], 1.0), plane([0.0, -1.0, 0.0], 0.0)];
    assert!(matches!(
        resolve_scene(&contradictory, None, None),
        Err(SceneError::ContradictoryPlanes {
            first: 0,
            second: 1
        })
    ));

    let mut narrow_feasible_wedge = explicit_document(Vec::new());
    narrow_feasible_wedge.planes = vec![
        plane([1.0, 0.0, 0.0], 1.0),
        plane([-0.999_999_5, 0.001, 0.0], -0.9),
    ];
    resolve_scene(&narrow_feasible_wedge, None, None)
        .expect("near-antiparallel planes still have a feasible intersection");

    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    let mut three_way_contradiction = explicit_document(Vec::new());
    three_way_contradiction.planes = vec![
        plane([1.0, 0.0, 0.0], 1.0),
        plane([0.0, 1.0, 0.0], 1.0),
        plane([-diagonal, -diagonal, 0.0], -diagonal),
    ];
    assert!(matches!(
        resolve_scene(&three_way_contradiction, None, None),
        Err(SceneError::NoFeasiblePlaneIntersection)
    ));
}

#[test]
fn overlapping_particles_require_the_explicit_coincidence_exception() {
    let close = explicit_document(vec![
        explicit_particle(0, [0.0, 0.0, 0.0]),
        explicit_particle(1, [0.025, 0.0, 0.0]),
    ]);
    assert!(matches!(
        resolve_scene(&close, None, None),
        Err(SceneError::OverlappingParticles {
            first: 0,
            second: 1
        })
    ));
    let coincident = explicit_document(vec![
        explicit_particle(0, [0.0, 0.0, 0.0]),
        explicit_particle(1, [0.0, 0.0, 0.0]),
    ]);
    assert!(matches!(
        resolve_scene(&coincident, None, None),
        Err(SceneError::OverlappingParticles { .. })
    ));
    let mut permitted = coincident;
    permitted.allow_coincident_particles = true;
    resolve_scene(&permitted, None, None).expect("explicit exact coincidence is valid");

    let rounded_reference_spacing = explicit_document(vec![
        explicit_particle(0, [0.2, 0.0, 0.0]),
        explicit_particle(1, [0.25, 0.0, 0.0]),
    ]);
    resolve_scene(&rounded_reference_spacing, None, None)
        .expect("binary32 representation of one particle diameter is not overlap");
}

#[test]
fn out_of_bounds_particles_are_rejected_and_exact_contact_is_valid() {
    let out_of_bounds = parse_scene(include_str!("fixtures/invalid/scene-out-of-bounds.toml"))
        .expect("malformed semantic fixture must still parse");
    assert!(matches!(
        resolve_scene(&out_of_bounds, None, None),
        Err(SceneError::ParticleOutsidePlane {
            particle: 0,
            plane: 0
        })
    ));
    let mut contact = explicit_document(vec![explicit_particle(0, [0.0, 0.0, 0.0])]);
    contact.planes = vec![plane([0.0, 1.0, 0.0], 0.0)];
    let contact = resolve_scene(&contact, None, None).expect("exact plane contact is valid");
    assert_eq!(contact.particles[0].position_m, Vec3::ZERO);
}

#[test]
fn invalid_lattice_geometry_and_impractical_counts_are_rejected() {
    for counts in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        assert!(matches!(
            resolve_scene(&lattice_document([0.0, 0.0, 0.0], counts), None, None),
            Err(SceneError::EmptyLatticeAxis { .. })
        ));
    }
    let mut spacing = lattice_document([0.0, 0.0, 0.0], [1, 1, 1]);
    if let ParticleSource::Lattice { spacing_m, .. } = &mut spacing.particles {
        *spacing_m = 0.04;
    }
    assert!(matches!(
        resolve_scene(&spacing, None, None),
        Err(SceneError::ParticleSpacingMismatch { .. })
    ));
    let excessive = lattice_document([0.0, 0.0, 0.0], [1_001, 1_000, 1]);
    assert!(matches!(
        resolve_scene(&excessive, None, None),
        Err(SceneError::ParticleCountTooLarge {
            maximum: MAX_LATTICE_PARTICLES,
            ..
        })
    ));
    let mut identity_overflow = lattice_document([0.0, 0.0, 0.0], [2, 1, 1]);
    if let ParticleSource::Lattice { first_identity, .. } = &mut identity_overflow.particles {
        *first_identity = u64::MAX;
    }
    assert!(matches!(
        resolve_scene(&identity_overflow, None, None),
        Err(SceneError::ParticleIdentityOverflow)
    ));
    let too_many_explicit = explicit_document(
        (0..=MAX_EXPLICIT_PARTICLES)
            .map(|index| explicit_particle(index as u64, [0.0, 0.0, 0.0]))
            .collect(),
    );
    assert!(matches!(
        resolve_scene(&too_many_explicit, None, None),
        Err(SceneError::ParticleCountTooLarge {
            maximum: MAX_EXPLICIT_PARTICLES,
            ..
        })
    ));
    let mut too_many_planes = explicit_document(Vec::new());
    too_many_planes.planes = vec![plane([0.0, 1.0, 0.0], 0.0); MAX_PLANES + 1];
    assert!(matches!(
        resolve_scene(&too_many_planes, None, None),
        Err(SceneError::TooManyPlanes {
            maximum: MAX_PLANES,
            ..
        })
    ));
    let collapsed = lattice_document([f32::MAX, 0.0, 0.0], [2, 1, 1]);
    assert!(matches!(
        resolve_scene(&collapsed, None, None),
        Err(SceneError::LatticeCoordinateCollapse { axis: "x", .. })
    ));
}

#[test]
fn focused_test_inputs_have_distinct_exact_construction() {
    let mut interior = lattice_document([-0.05, -0.05, -0.05], [3, 3, 3]);
    if let ParticleSource::Lattice {
        first_identity,
        initial_velocity_m_s,
        ..
    } = &mut interior.particles
    {
        *first_identity = 0;
        *initial_velocity_m_s = [0.0, 0.0, 0.0];
    }
    let empty = explicit_document(Vec::new());
    let single = explicit_document(vec![explicit_particle(0, [0.0, 0.0, 0.0])]);
    let pair = explicit_document(vec![
        explicit_particle(0, [0.0, 0.0, 0.0]),
        explicit_particle(1, [0.05, 0.0, 0.0]),
    ]);
    let support = explicit_document(vec![
        explicit_particle(0, [0.0, 0.0, 0.0]),
        explicit_particle(1, [0.1, 0.0, 0.0]),
    ]);
    let mut coincident = explicit_document(vec![
        explicit_particle(0, [0.0, 0.0, 0.0]),
        explicit_particle(1, [0.0, 0.0, 0.0]),
    ]);
    coincident.allow_coincident_particles = true;
    let mut contact = single.clone();
    contact.planes = vec![plane([0.0, 1.0, 0.0], 0.0)];
    let mut projection = explicit_document(vec![explicit_particle(0, [0.0, 0.025, 0.0])]);
    projection.planes = vec![plane([0.0, 1.0, 0.0], 0.0)];

    let cases = [
        (interior.clone(), 27),
        (empty, 0),
        (single, 1),
        (pair, 2),
        (support.clone(), 2),
        (coincident, 2),
        (contact, 1),
        (projection.clone(), 1),
    ];
    for (document, expected_count) in cases {
        let first = resolve_scene(&document, None, None).expect("focused input should resolve");
        let second = resolve_scene(&document, None, None).expect("focused input should repeat");
        assert_eq!(first.particles.len(), expected_count);
        assert_eq!(first, second);
    }

    let interior = resolve_scene(&interior, None, None).expect("interior lattice");
    assert_eq!(interior.particles[13].identity, 13);
    assert_eq!(interior.particles[13].position_m, Vec3::ZERO);
    let support = resolve_scene(&support, None, None).expect("support boundary");
    assert_eq!(
        support.particles[1].position_m - support.particles[0].position_m,
        Vec3::new(0.1, 0.0, 0.0)
    );
    let projection = resolve_scene(&projection, None, None).expect("projection input");
    assert_eq!(
        projection.planes[0]
            .project(Vec3::new(0.0, -0.025, 0.0))
            .expect("valid projection"),
        Vec3::ZERO
    );
}

#[test]
fn scene_schema_unknown_fields_missing_values_and_randomness_are_rejected() {
    let unsupported = EXTERNAL_SCENE_TOML.replacen("schema_version = 1", "schema_version = 2", 1);
    assert!(parse_scene(&unsupported).is_err());
    let unknown = EXTERNAL_SCENE_TOML.replacen(
        "randomness = \"none\"",
        "randomness = \"none\"\nunknown_scene_field = true",
        1,
    );
    assert!(parse_scene(&unknown).is_err());
    let missing = EXTERNAL_SCENE_TOML.replacen(
        "description = \"Six-by-six-by-six fluid lattice inside a static six-plane box\"\n",
        "",
        1,
    );
    assert!(parse_scene(&missing).is_err());
    let random = EXTERNAL_SCENE_TOML.replacen("randomness = \"none\"", "randomness = \"pcg32\"", 1);
    assert!(parse_scene(&random).is_err());
    let empty_name =
        EXTERNAL_SCENE_TOML.replacen("name = \"falling-block-06\"", "name = \"  \"", 1);
    assert!(parse_scene(&empty_name).is_err());
}

#[test]
fn scene_overrides_use_neutral_configuration_layers_and_record_provenance() {
    let mut document = explicit_document(Vec::new());
    document.configuration.overrides = Some(ConfigurationOverlay {
        numerical: NumericalOverrides {
            external_acceleration_m_s2: Some([0.0, -1.62, 0.0]),
            ..Default::default()
        },
        derived: DerivedAssertions::default(),
    });
    let scene = resolve_scene(&document, None, None).expect("supported scene override");
    assert_eq!(
        scene
            .configuration
            .sources
            .get("external_acceleration_m_s2"),
        Some(&FieldSource::Scene)
    );
    assert_eq!(
        scene.configuration.numerical.external_acceleration_m_s2.y,
        -1.62
    );
}

#[test]
fn configuration_paths_resolve_relative_to_the_scene_file_not_current_directory() {
    let _lock = CURRENT_DIRECTORY_LOCK.lock().expect("directory lock");
    let original_directory = std::env::current_dir().expect("current directory");
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "maelstrom-scene-path-{}-{unique}",
        std::process::id()
    ));
    let _directory_guard = CurrentDirectoryGuard {
        original: original_directory,
        temporary_root: root.clone(),
    };
    let scenes = root.join("scenes");
    let configuration = root.join("configuration");
    let unrelated = root.join("unrelated");
    fs::create_dir_all(&scenes).expect("scene directory");
    fs::create_dir_all(&configuration).expect("configuration directory");
    fs::create_dir_all(&unrelated).expect("unrelated directory");
    fs::write(configuration.join("defaults-v1.toml"), DEFAULTS_TOML)
        .expect("configuration fixture");
    let scene_input = EXTERNAL_SCENE_TOML.replacen(
        "path = \"../config/defaults-v1.toml\"",
        "path = \"../configuration/defaults-v1.toml\"",
        1,
    );
    let scene_path = scenes.join("scene.toml");
    fs::write(&scene_path, scene_input).expect("scene fixture");

    std::env::set_current_dir(&unrelated).expect("change current directory");
    let loaded = load_resolved_scene(&scene_path, None).expect("relative selection should load");

    assert_eq!(loaded.particles.len(), 216);
}

#[test]
fn absolute_configuration_references_are_rejected() {
    let mut document = explicit_document(Vec::new());
    document.configuration.source = ConfigurationSource::File;
    document.configuration.path = Some(
        PathBuf::from("/machine/specific/config.toml")
            .display()
            .to_string(),
    );
    assert!(matches!(
        resolve_scene(&document, None, None),
        Err(SceneError::InvalidConfigurationSelection { .. })
    ));

    document.configuration.path = None;
    assert!(matches!(
        resolve_scene(&document, None, None),
        Err(SceneError::InvalidConfigurationSelection { .. })
    ));

    document.configuration.path = Some("configuration.toml".to_owned());
    assert!(matches!(
        resolve_scene(&document, None, None),
        Err(SceneError::MissingConfigurationDocument)
    ));

    document.configuration.source = ConfigurationSource::BuiltInDefaults;
    assert!(matches!(
        resolve_scene(&document, None, None),
        Err(SceneError::InvalidConfigurationSelection { .. })
    ));
}

#[test]
fn repeated_construction_and_failure_leave_accepted_inputs_unchanged() {
    let document = lattice_document([0.0, 0.0, 0.0], [2, 2, 2]);
    let accepted = resolve_scene(&document, None, None).expect("valid scene");
    let snapshot = accepted.clone();
    let repeat = resolve_scene(&document, None, None).expect("repeat scene");
    assert_eq!(
        accepted.canonical_inspection(),
        repeat.canonical_inspection()
    );

    let invalid = explicit_document(vec![explicit_particle(0, [f32::NAN, 0.0, 0.0])]);
    assert!(resolve_scene(&invalid, None, None).is_err());
    assert_eq!(accepted, snapshot);
}

#[test]
fn comparison_metadata_uses_resolved_inputs_not_names_or_descriptions() {
    let first_document = explicit_document(vec![explicit_particle(0, [0.0, 0.0, 0.0])]);
    let mut renamed_document = first_document.clone();
    renamed_document.name = "a-different-display-name".to_owned();
    renamed_document.description = "Different explanatory prose".to_owned();
    let second_document = explicit_document(vec![explicit_particle(0, [0.05, 0.0, 0.0])]);
    let first = resolve_scene(&first_document, None, None).expect("first scene");
    let renamed = resolve_scene(&renamed_document, None, None).expect("renamed scene");
    let second = resolve_scene(&second_document, None, None).expect("second scene");
    assert_eq!(first.comparison_key(), renamed.comparison_key());
    assert_ne!(first.canonical_inspection(), renamed.canonical_inspection());
    assert_ne!(first.comparison_key(), second.comparison_key());
}
