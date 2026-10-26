//! Exact energy cases and independently specified domain corners check visual
//! calculations without drawing or deriving expectations from the production fit.

use maelstrom::{
    boundary::Plane,
    math::Vec3,
    presentation::{Camera, ParticleColours, domain_bounds},
    scene::Particle,
};

fn particle(velocity: Vec3) -> Particle {
    Particle {
        identity: 42,
        position_m: Vec3::ZERO,
        velocity_m_s: velocity,
    }
}
fn plane(n: [f32; 3], offset: f32) -> Plane {
    Plane::new(Vec3::new(n[0], n[1], n[2]), offset, 1e-6).unwrap()
}
fn box_planes() -> Vec<Plane> {
    vec![
        plane([1., 0., 0.], -2.),
        plane([-1., 0., 0.], -6.),
        plane([0., 1., 0.], 0.),
        plane([0., -1., 0.], -8.),
        plane([0., 0., 1.], -4.),
        plane([0., 0., -1.], -2.),
    ]
}

#[test]
fn energy_uses_mass_all_velocity_components_and_fixed_clamping() {
    let colours = ParticleColours::KineticEnergy {
        min_energy_j: 2.,
        max_energy_j: 6.,
    };
    let map = colours.prepare(2.).unwrap();
    assert_eq!(map.endpoints(), ([0., 0., 1.], [1., 0., 0.]));
    for (velocity, expected) in [
        (Vec3::ZERO, 0.),
        (Vec3::new(1., 1., 0.), 0.),
        (Vec3::new(0., 0., 2.), 0.5),
        (Vec3::new(1., 1., 2.), 1.),
        (Vec3::new(4., 0., 0.), 1.),
    ] {
        let p = particle(velocity);
        let before = p;
        assert_eq!(map.fraction(&p).unwrap(), expected);
        assert_eq!(p, before);
    }
    assert_eq!(
        colours
            .prepare(1.)
            .unwrap()
            .fraction(&particle(Vec3::new(0., 0., 2.)))
            .unwrap(),
        0.
    );
    let tiny = ParticleColours::KineticEnergy {
        min_energy_j: 0.,
        max_energy_j: f64::from_bits(1),
    };
    assert_eq!(
        tiny.prepare(f64::from(f32::MAX))
            .unwrap()
            .fraction(&particle(Vec3::new(f32::MAX, f32::MAX, f32::MAX)))
            .unwrap(),
        1.
    );
    for mass in [0., -1., f64::INFINITY, f64::NAN] {
        assert!(colours.prepare(mass).is_err());
    }
    for velocity in [f32::NAN, f32::INFINITY] {
        assert!(
            map.fraction(&particle(Vec3::new(velocity, 0., 0.)))
                .is_err()
        );
    }
}

#[test]
fn fixed_colour_is_linearised_and_default_preserves_legacy_constants() {
    assert_eq!(
        ParticleColours::default().prepare(1.).unwrap().endpoints(),
        (
            [0.0241576, 0.346704, 0.693872],
            [0.0241576, 0.346704, 0.693872]
        )
    );
    let map = ParticleColours::Fixed {
        colour: "#0080ff".into(),
    }
    .prepare(1.)
    .unwrap();
    let (low, high) = map.endpoints();
    assert_eq!(low, high);
    assert_eq!((low[0], low[2]), (0., 1.));
    assert!((low[1] - 0.2158605).abs() < 1e-7);
    assert_eq!(map.fraction(&particle(Vec3::new(99., 0., 0.))).unwrap(), 0.);
    for colour in ["red", "#12345g", "#00000000", "é23456"] {
        assert!(
            ParticleColours::Fixed {
                colour: colour.into()
            }
            .prepare(1.)
            .is_err()
        );
    }
}

#[test]
fn bounded_box_tetrahedron_and_unbounded_intersections_are_distinguished() {
    assert_eq!(
        domain_bounds(&box_planes()),
        Some(([-2., 0., -4.], [6., 8., 2.]))
    );
    // x,y,z >= 0 and x+y+z <= sqrt(3); vertices lie on the axes.
    let n = 1.0f32 / 3.0f32.sqrt();
    let tetra = [
        plane([1., 0., 0.], 0.),
        plane([0., 1., 0.], 0.),
        plane([0., 0., 1.], 0.),
        plane([-n, -n, -n], -1.),
    ];
    let (lo, hi) = domain_bounds(&tetra).unwrap();
    assert_eq!(lo, [0.; 3]);
    for bound in hi {
        assert!((bound - 1.0 / f64::from(n)).abs() < 1e-12);
    }
    // This wedge has a vertex, but extends indefinitely in positive z.
    let wedge = [
        plane([1., 0., 0.], 0.),
        plane([0., 1., 0.], 0.),
        plane([0., 0., 1.], 0.),
        plane([-1., 0., 0.], -1.),
    ];
    assert!(domain_bounds(&wedge).is_none());
    assert!(domain_bounds(&box_planes()[..4]).is_none());
    assert!(domain_bounds(&[]).is_none());
    let mut impossible = box_planes();
    impossible.push(plane([1., 0., 0.], 10.));
    assert!(domain_bounds(&impossible).is_none());
}

#[test]
fn fitting_uses_complete_domain_at_portrait_and_landscape_aspects() {
    for (width, height) in [(200, 800), (1600, 900)] {
        let mut camera = Camera::default();
        camera
            .fit_scene(
                &[particle(Vec3::ZERO)],
                &box_planes(),
                0.025,
                f64::from(width) / f64::from(height),
            )
            .unwrap();
        for (actual, expected) in camera.target_m.into_iter().zip([2., 4., -1.]) {
            // Radius inflation can round the centre by a few f64 ulps in metres.
            assert!((actual - expected).abs() < 1e-12);
        }
        for x in [-2., 6.] {
            for y in [0., 8.] {
                for z in [-4., 2.] {
                    let p = camera.project([x, y, z], width, height).unwrap();
                    assert!((0.0..f64::from(width)).contains(&p[0]));
                    assert!((0.0..f64::from(height)).contains(&p[1]));
                    assert!((0.0..1.0).contains(&p[2]));
                }
            }
        }
        let before = camera.clone();
        assert!(camera.fit_scene(&[], &box_planes(), 0.025, 0.).is_err());
        assert_eq!(camera, before);
        camera.fit_scene(&[], &[], 0.025, 1.).unwrap();
        assert_eq!(camera, Camera::default());
    }
}
