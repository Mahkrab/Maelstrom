//! Independent geometry and adversarial pair comparisons qualify sparse discovery.
//! Explicit rows catch missing cells even if the reference search shared a bug.

use maelstrom::{
    NumericalError,
    math::Vec3,
    neighbours::{brute_force, uniform_grid},
};

fn rows(positions: &[Vec3], radius: f32, expected: &[&[usize]]) {
    let found = uniform_grid(positions, radius).unwrap();
    assert_eq!(found.particle_count(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(found.for_particle(index).unwrap(), *expected, "row {index}");
    }
}

#[test]
fn every_adjacent_face_edge_and_corner_is_searched() {
    // At radius 0.1 the selected cell width is 0.125. Cross each of the 26
    // boundaries separately; each pair is less than sqrt(3)*0.04 apart.
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                if [x, y, z] == [0, 0, 0] {
                    continue;
                }
                let a = [x, y, z].map(|d| match d {
                    -1 => 0.01,
                    1 => 0.115,
                    _ => 0.0625,
                });
                let b =
                    std::array::from_fn::<_, 3, _>(|axis| a[axis] + [x, y, z][axis] as f32 * 0.04);
                rows(
                    &[Vec3::new(a[0], a[1], a[2]), Vec3::new(b[0], b[1], b[2])],
                    0.1,
                    &[&[1], &[0]],
                );
                // Translate across the negative origin as well.
                rows(
                    &[
                        Vec3::new(a[0] - 0.25, a[1] - 0.25, a[2] - 0.25),
                        Vec3::new(b[0] - 0.25, b[1] - 0.25, b[2] - 0.25),
                    ],
                    0.1,
                    &[&[1], &[0]],
                );
            }
        }
    }
}

#[test]
fn strict_support_coincidence_and_identity_order_have_explicit_rows() {
    let inside = f32::from_bits(1.0_f32.to_bits() - 1);
    let outside = f32::from_bits(1.0_f32.to_bits() + 1);
    rows(
        &[
            Vec3::ZERO,
            Vec3::new(-0.0, 0.0, 0.0),
            Vec3::new(inside, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(outside, 0.0, 0.0),
        ],
        1.0,
        &[&[1, 2], &[0, 2], &[0, 1, 3, 4], &[2, 4], &[2, 3]],
    );
    // Identity order deliberately opposes spatial cell order.
    rows(
        &[
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::new(-0.02, 0.0, 0.0),
            Vec3::new(0.0, -0.02, 0.0),
            Vec3::ZERO,
        ],
        0.1,
        &[&[1, 2, 3], &[0, 2, 3], &[0, 1, 3], &[0, 1, 2]],
    );
    rows(&[], 1.0, &[]);
    rows(
        &[Vec3::new(f32::MAX, -f32::MAX, 0.0)],
        f32::from_bits(1),
        &[&[]],
    );
}

#[test]
fn dense_sparse_and_extreme_finite_inputs_keep_all_particles() {
    for points in [
        vec![Vec3::ZERO; 128],
        (0..128)
            .map(|i| Vec3::new(i as f32 * 1000.0, -(i as f32) * 100.0, 0.0))
            .collect(),
        vec![Vec3::new(f32::MAX, 0.0, 0.0); 4],
        vec![Vec3::new(-f32::MAX, f32::MAX, -0.0); 4],
    ] {
        for radius in [f32::from_bits(1), 0.1, 10000.0, f32::MAX] {
            assert_eq!(uniform_grid(&points, radius), brute_force(&points, radius));
        }
    }
    // In real geometry this pair lies far outside the radius. Binary32 squared
    // distance underflows to zero, so the inherited search includes it.
    rows(
        &[Vec3::ZERO, Vec3::new(2.0_f32.powi(-75), 0.0, 0.0)],
        f32::from_bits(1),
        &[&[1], &[0]],
    );
}

#[test]
fn overflow_checks_include_distant_pairs_and_preserve_the_first_error() {
    for points in [
        vec![Vec3::ZERO, Vec3::new(1.0e20, 0.0, 0.0)],
        vec![
            Vec3::new(f32::MAX, 0.0, 0.0),
            Vec3::new(-f32::MAX, 0.0, 0.0),
        ],
        vec![
            Vec3::ZERO,
            Vec3::new(1.0e20, 0.0, 0.0),
            Vec3::new(f32::MAX, 0.0, 0.0),
        ],
    ] {
        assert!(brute_force(&points, 0.1).is_err());
        assert_eq!(uniform_grid(&points, 0.1), brute_force(&points, 0.1));
    }
    // The box diagonal overflows but none of these actual pair distances do.
    let a = 1.2e19;
    let points = [
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, 0.0, a),
    ];
    rows(&points, 0.1, &[&[], &[], &[]]);
    rows(&points, f32::MAX, &[&[1, 2], &[0, 2], &[0, 1]]);
}

#[test]
fn invalid_inputs_fail_before_construction_and_do_not_poison_later_searches() {
    for radius in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            uniform_grid(&[], radius).unwrap_err().to_string(),
            brute_force(&[], radius).unwrap_err().to_string()
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for axis in 0..3 {
            let mut p = [0.0; 3];
            p[axis] = value;
            assert!(matches!(
                uniform_grid(&[Vec3::new(p[0], p[1], p[2])], 0.1),
                Err(NumericalError::NonFiniteVector { .. })
            ));
        }
    }
    for _ in 0..3 {
        rows(
            &[Vec3::new(-0.25, 0.0, 0.0), Vec3::new(-0.24, 0.0, 0.0)],
            0.1,
            &[&[1], &[0]],
        );
        rows(&[Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)], 0.1, &[&[], &[]]);
        rows(&[], 0.1, &[]);
    }
}

fn random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn deterministic_generated_rows_match_the_reference_across_scales_and_translations() {
    let mut state = 0x917a_54e3;
    for scale in [
        2.0_f32.powi(-75),
        2.0_f32.powi(-65),
        0.001,
        0.1,
        1.0,
        1000.0,
        1.0e17,
    ] {
        for translation in [-20.0, 0.0, 15.0] {
            let points: Vec<_> = (0..192)
                .map(|_| {
                    let a = std::array::from_fn::<_, 3, _>(|_| {
                        ((random(&mut state) % 4001) as f32 / 1000.0 - 2.0 + translation) * scale
                    });
                    Vec3::new(a[0], a[1], a[2])
                })
                .collect();
            assert_eq!(uniform_grid(&points, scale), brute_force(&points, scale));
        }
    }
    // Cover the complete finite exponent domain with arbitrary mantissas and
    // signs. Successful and failing searches must agree, not just small boxes.
    for _ in 0..512 {
        let points: Vec<_> = (0..12)
            .map(|_| {
                let a = std::array::from_fn::<_, 3, _>(|_| {
                    let bits = random(&mut state);
                    f32::from_bits(if bits & 0x7f80_0000 == 0x7f80_0000 {
                        bits ^ 0x0080_0000
                    } else {
                        bits
                    })
                });
                Vec3::new(a[0], a[1], a[2])
            })
            .collect();
        let radius = f32::from_bits((random(&mut state) % 0x7f7f_ffff) + 1);
        assert_eq!(uniform_grid(&points, radius), brute_force(&points, radius));
    }
}

#[test]
fn power_of_two_faces_and_support_limits_agree_across_the_finite_exponent_domain() {
    for exponent in -149..=127 {
        let power = f32::from_bits(if exponent < -126 {
            1 << (exponent + 149)
        } else {
            ((exponent + 127) as u32) << 23
        });
        let below = f32::from_bits(power.to_bits() - 1);
        let above = f32::from_bits(power.to_bits() + 1);
        let points = [
            Vec3::ZERO,
            Vec3::new(-0.0, 0.0, 0.0),
            Vec3::new(below, 0.0, 0.0),
            Vec3::new(power, 0.0, 0.0),
            Vec3::new(above, 0.0, 0.0),
            Vec3::new(-power, 0.0, 0.0),
            Vec3::new(-above, -power, 0.0),
            Vec3::new(power, power, power),
        ];
        for radius in [below, power, above].into_iter().filter(|h| *h > 0.0) {
            assert_eq!(
                uniform_grid(&points, radius),
                brute_force(&points, radius),
                "exponent {exponent}; radius {radius}"
            );
        }
    }
}
