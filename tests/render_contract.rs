//! Projection checks use known screen coordinates and exact plane geometry rather
//! than assuming that a plausible picture establishes orientation or world scale.
#![cfg(feature = "render")]
use maelstrom::{
    boundary::Plane,
    math::Vec3,
    render::{BoundaryLines, Camera, Settings},
    scene::Particle,
};

fn camera() -> Camera {
    Camera {
        eye_m: [0.0, 0.0, 2.0],
        target_m: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        vertical_fov_deg: 90.0,
        near_m: 1.0,
        far_m: 3.0,
    }
}
fn particle(position: Vec3) -> Particle {
    Particle {
        identity: 0,
        position_m: position,
        velocity_m_s: Vec3::ZERO,
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}
#[test]
fn camera_has_known_orientation_scale_and_depth() {
    let c = camera();
    let centre = c.project([0.0, 0.0, 0.0], 200, 100).unwrap();
    close(centre[0], 100.0);
    close(centre[1], 50.0);
    close(centre[2], 0.75);
    let right = c.project([1.0, 0.0, 0.0], 200, 100).unwrap();
    close(right[0], 125.0);
    let up = c.project([0.0, 1.0, 0.0], 200, 100).unwrap();
    close(up[1], 25.0);
    close(c.project([0.0, 0.0, 1.0], 200, 100).unwrap()[2], 0.0);
    close(c.project([0.0, 0.0, -1.0], 200, 100).unwrap()[2], 1.0);
}
#[test]
fn invalid_camera_and_strict_render_documents_are_rejected() {
    for source in [
        "width=10",
        "schema_version=3",
        "schema_version=1\nwidth=0",
        "schema_version=1\nheight=8193",
        "schema_version=1\nstep_stride=0",
        "schema_version=1\nformat='jpeg'",
        "schema_version=1\nunknown=true",
        "schema_version=1\n[camera]\nnear_m=nan",
        "schema_version=1\n[camera]\nup=[0.0,0.0,0.0]",
        "schema_version=1\n[camera]\nfar_m=0.001",
        "schema_version=1\n[camera]\nvertical_fov_deg=1.0",
        "schema_version=1\n[camera]\neye_m=[0.3,0.4,0.3]",
        "schema_version=1\n[camera]\nup=[0.9,0.6,0.9]",
    ] {
        assert!(Settings::parse(source).is_err(), "accepted {source}");
    }
    let merged = Settings::parse("schema_version=1\nwidth=800\n[camera]\nfar_m=200.0").unwrap();
    assert_eq!(merged.height, 720);
    assert_eq!(merged.camera.eye_m, Camera::default().eye_m);
    assert_eq!(merged.camera.far_m, 200.0);
}
#[test]
fn fitting_empty_single_small_and_large_geometry_is_finite() {
    for (particles, radius) in [
        (vec![], 0.025),
        (vec![particle(Vec3::ZERO)], 0.025),
        (vec![particle(Vec3::new(1e20, 0.0, 0.0))], 1e18),
        (vec![particle(Vec3::new(1e-5, 0.0, 0.0))], 1e-6),
    ] {
        let mut c = Camera::default();
        c.fit(&particles, radius, 16.0 / 9.0).unwrap();
        c.validate(16.0 / 9.0).unwrap();
        for p in &particles {
            let v = c
                .project(
                    [
                        p.position_m.x.into(),
                        p.position_m.y.into(),
                        p.position_m.z.into(),
                    ],
                    1600,
                    900,
                )
                .unwrap();
            assert!((0.0..1600.0).contains(&v[0]));
            assert!((0.0..900.0).contains(&v[1]));
            assert!((0.0..1.0).contains(&v[2]));
        }
    }
}
#[test]
fn arbitrary_planes_are_clipped_to_frozen_initial_box() {
    let plane = Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 1e-6).unwrap();
    let b = BoundaryLines::new(&[], &[plane], 0.05).unwrap();
    assert_eq!(b.lower_m, [-1.0; 3]);
    assert_eq!(b.upper_m, [1.0; 3]);
    assert_eq!(b.vertices_m.len(), 8);
    for point in &b.vertices_m {
        close(point[0], 0.0);
        assert_eq!(point[1].abs(), 1.0);
        assert_eq!(point[2].abs(), 1.0);
    }
    let n = (0.5f32).sqrt();
    let plane = Plane::new(Vec3::new(n, n, 0.0), 0.0, 1e-6).unwrap();
    let b = BoundaryLines::new(&[], &[plane], 0.05).unwrap();
    assert_eq!(b.vertices_m.len(), 8);
    for p in b.vertices_m {
        close(p[0] + p[1], 0.0);
        assert!(p.iter().all(|x| x.abs() <= 1.0));
    }
    let outside = Plane::new(Vec3::new(0.0, 1.0, 0.0), 10.0, 1e-6).unwrap();
    assert!(
        BoundaryLines::new(&[], &[outside], 0.05)
            .unwrap()
            .vertices_m
            .is_empty()
    );
}

#[test]
fn failed_camera_changes_preserve_the_previous_valid_view() {
    let mut c = camera();
    let before = c.clone();
    assert!(c.orbit(f64::INFINITY, 0.0).is_err());
    assert_eq!(c, before);
    assert!(c.pan(f64::INFINITY, 0.0).is_err());
    assert_eq!(c, before);
    assert!(c.zoom(f64::MAX).is_err());
    assert_eq!(c, before);
    assert!(c.fit(&[], 0.05, f64::NAN).is_err());
    assert_eq!(c, before);
}
