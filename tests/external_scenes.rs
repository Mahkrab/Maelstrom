use std::{fs, path::PathBuf};

use maelstrom::{
    config::FieldSource,
    math::Vec3,
    scene::{
        ConfigurationSource, LatticeEndpointRule, LatticeOrder, ParticleSource, RandomnessPolicy,
        load_resolved_scene, parse_scene,
    },
};

#[derive(Clone, Copy)]
struct SceneCase {
    file_name: &'static str,
    axis_count: u32,
    particle_count: usize,
    box_dimensions_m: [f32; 3],
    last_position_m: [f32; 3],
}

const SCENE_CASES: [SceneCase; 7] = [
    SceneCase {
        file_name: "falling-block-04.toml",
        axis_count: 4,
        particle_count: 64,
        box_dimensions_m: [0.6, 0.8, 0.6],
        last_position_m: [0.350_000_02, 0.55, 0.350_000_02],
    },
    SceneCase {
        file_name: "falling-block-06.toml",
        axis_count: 6,
        particle_count: 216,
        box_dimensions_m: [0.700_000_05, 0.900_000_04, 0.700_000_05],
        last_position_m: [0.45, 0.65, 0.45],
    },
    SceneCase {
        file_name: "falling-block-08.toml",
        axis_count: 8,
        particle_count: 512,
        box_dimensions_m: [0.8, 1.0, 0.8],
        last_position_m: [0.55, 0.75, 0.55],
    },
    SceneCase {
        file_name: "falling-block-10.toml",
        axis_count: 10,
        particle_count: 1_000,
        box_dimensions_m: [0.9, 1.1, 0.9],
        last_position_m: [0.650_000_04, 0.85, 0.650_000_04],
    },
    SceneCase {
        file_name: "falling-block-12.toml",
        axis_count: 12,
        particle_count: 1_728,
        box_dimensions_m: [1.0, 1.2, 1.0],
        last_position_m: [0.75, 0.950_000_05, 0.75],
    },
    SceneCase {
        file_name: "falling-block-16.toml",
        axis_count: 16,
        particle_count: 4_096,
        box_dimensions_m: [1.2, 1.400_000_1, 1.2],
        last_position_m: [0.95, 1.15, 0.95],
    },
    SceneCase {
        file_name: "falling-block-20.toml",
        axis_count: 20,
        particle_count: 8_000,
        box_dimensions_m: [1.4, 1.6, 1.4],
        last_position_m: [1.15, 1.35, 1.15],
    },
];

fn scene_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scenes")
        .join(file_name)
}

#[test]
fn every_external_scene_uses_the_ordinary_versioned_loader() {
    let expected_normals = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    let mut expected_configuration_key = None;

    for case in SCENE_CASES {
        let path = scene_path(case.file_name);
        let input = fs::read_to_string(&path).expect("checked-in scene input");
        let document = parse_scene(&input).expect("versioned scene document");

        assert_eq!(document.schema_version, 1);
        assert_eq!(document.randomness, RandomnessPolicy::None);
        assert!(!document.allow_coincident_particles);
        assert_eq!(document.configuration.source, ConfigurationSource::File);
        assert_eq!(
            document.configuration.path.as_deref(),
            Some("../config/defaults-v1.toml")
        );
        assert!(document.configuration.overrides.is_none());
        match document.particles {
            ParticleSource::Lattice {
                first_identity,
                origin_m,
                counts,
                spacing_m,
                endpoint_rule,
                order,
                initial_velocity_m_s,
            } => {
                assert_eq!(first_identity, 0);
                assert_eq!(origin_m, [0.2, 0.4, 0.2]);
                assert_eq!(counts, [case.axis_count; 3]);
                assert_eq!(spacing_m.to_bits(), 0.05_f32.to_bits());
                assert_eq!(endpoint_rule, LatticeEndpointRule::CountedHalfOpen);
                assert_eq!(order, LatticeOrder::XThenYThenZ);
                assert_eq!(initial_velocity_m_s, [0.0, 0.0, 0.0]);
            }
            ParticleSource::Explicit { .. } => {
                panic!("selected scene must remain a counted lattice")
            }
        }
        assert_eq!(document.planes.len(), expected_normals.len());
        for (plane, normal) in document.planes.iter().zip(expected_normals) {
            assert_eq!(
                Vec3::new(plane.normal[0], plane.normal[1], plane.normal[2]),
                normal
            );
        }
        assert_eq!(document.planes[0].offset_m, 0.0);
        assert_eq!(document.planes[1].offset_m, -case.box_dimensions_m[0]);
        assert_eq!(document.planes[2].offset_m, 0.0);
        assert_eq!(document.planes[3].offset_m, -case.box_dimensions_m[1]);
        assert_eq!(document.planes[4].offset_m, 0.0);
        assert_eq!(document.planes[5].offset_m, -case.box_dimensions_m[2]);

        let scene = load_resolved_scene(&path, None).expect("ordinary public scene path");
        assert_eq!(scene.particles.len(), case.particle_count);
        assert_eq!(scene.planes.len(), 6);
        assert_eq!(
            scene.particles.first().expect("first particle").position_m,
            Vec3::new(0.2, 0.4, 0.2)
        );
        assert_eq!(
            scene.particles.last().expect("last particle").position_m,
            Vec3::new(
                case.last_position_m[0],
                case.last_position_m[1],
                case.last_position_m[2]
            )
        );
        assert_eq!(scene.particles[1].position_m, Vec3::new(0.2, 0.4, 0.25));
        assert_eq!(
            scene.particles[case.axis_count as usize].position_m,
            Vec3::new(0.2, 0.450_000_02, 0.2)
        );
        assert_eq!(
            scene.particles[(case.axis_count * case.axis_count) as usize].position_m,
            Vec3::new(0.25, 0.4, 0.2)
        );
        for (index, particle) in scene.particles.iter().enumerate() {
            assert_eq!(particle.identity, index as u64);
            assert_eq!(particle.velocity_m_s, Vec3::ZERO);
        }
        assert!(
            scene
                .configuration
                .sources
                .values()
                .all(|source| *source == FieldSource::ConfigurationFile)
        );
        match &expected_configuration_key {
            Some(expected) => assert_eq!(scene.configuration.comparison_key(), *expected),
            None => expected_configuration_key = Some(scene.configuration.comparison_key()),
        }
    }
}

#[test]
fn repeated_external_scene_resolution_is_structurally_identical() {
    for case in SCENE_CASES {
        let path = scene_path(case.file_name);
        let first = load_resolved_scene(&path, None).expect("first resolution");
        let second = load_resolved_scene(&path, None).expect("second resolution");

        assert_eq!(first, second);
        assert_eq!(first.canonical_inspection(), second.canonical_inspection());
        assert_eq!(first.comparison_key(), second.comparison_key());
    }
}

#[test]
fn differently_sized_external_scenes_have_distinct_comparison_identities() {
    let keys = SCENE_CASES.map(|case| {
        load_resolved_scene(&scene_path(case.file_name), None)
            .expect("selected external scene")
            .comparison_key()
    });

    for first in 0..keys.len() {
        for second in first + 1..keys.len() {
            assert_ne!(keys[first], keys[second]);
        }
    }
}
