//! Independent stage calculations check the SI scaling, while bounded falling
//! scenes guard against the particle collapse caused by weak regularisation.

mod support;

use std::{collections::BTreeSet, path::Path};

use maelstrom::{
    config::{ResolvedConfiguration, parse_configuration_overlay, resolve_effective_configuration},
    fluid::{InteractionSets, compute_jacobi_stage},
    math::Vec3,
    scene::load_resolved_scene,
    simulation::SerialSimulation,
};
use support::{assert_absolute, assert_vec_absolute, poly6_f64};

type Vector = [f64; 3];

fn configuration(overrides: &str) -> ResolvedConfiguration {
    let overlay = parse_configuration_overlay(overrides).unwrap();
    resolve_effective_configuration(None, None, Some(&overlay)).unwrap()
}

fn subtract(a: Vector, b: Vector) -> Vector {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn length(a: Vector) -> f64 {
    a.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn gradient(a: Vector, b: Vector, h: f64) -> Vector {
    let displacement = subtract(a, b);
    let r = length(displacement);
    if r == 0.0 || r >= h {
        [0.0; 3]
    } else {
        let factor = -45.0 * (h - r).powi(2) / (std::f64::consts::PI * h.powi(6) * r);
        displacement.map(|value| value * factor)
    }
}

fn reference_density(spacing: f64, h: f64) -> f64 {
    // Scan beyond support rather than sharing production's lattice loop bounds.
    (-4..=4)
        .flat_map(|x| (-4..=4).flat_map(move |y| (-4..=4).map(move |z| (x, y, z))))
        .map(|(x, y, z)| poly6_f64(spacing * f64::from(x * x + y * y + z * z).sqrt(), h))
        .sum()
}

struct ExpectedStage {
    densities: Vec<f64>,
    constraints: Vec<f64>,
    multipliers: Vec<f64>,
    corrections: Vec<Vector>,
}

fn independent_stage(positions: &[Vector], config: &ResolvedConfiguration) -> ExpectedStage {
    let model = &config.numerical;
    let h = f64::from(model.particle_spacing_m) * f64::from(model.support_radius_ratio);
    let rest = reference_density(f64::from(model.particle_spacing_m), h);
    let epsilon = f64::from(model.relaxation_scale) / h.powi(2);
    let strength = if model.artificial_pressure_enabled {
        f64::from(model.artificial_pressure_strength_scale) * h.powi(2)
    } else {
        0.0
    };
    let reference = poly6_f64(
        f64::from(model.artificial_pressure_reference_separation_ratio) * h,
        h,
    );
    let mut densities = Vec::new();
    let mut constraints = Vec::new();
    let mut multipliers = Vec::new();
    for (i, &p) in positions.iter().enumerate() {
        let density: f64 = positions
            .iter()
            .map(|&q| poly6_f64(length(subtract(p, q)), h))
            .sum();
        let constraint = density / rest - 1.0;
        let mut owner = [0.0; 3];
        let mut denominator = epsilon;
        for (j, &q) in positions.iter().enumerate() {
            if i != j {
                let g = gradient(p, q, h).map(|value| value / rest);
                for axis in 0..3 {
                    owner[axis] += g[axis];
                    denominator += g[axis] * g[axis];
                }
            }
        }
        denominator += owner.iter().map(|value| value * value).sum::<f64>();
        densities.push(density);
        constraints.push(constraint);
        multipliers.push(-constraint / denominator);
    }
    let corrections = positions
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let mut correction = [0.0; 3];
            for (j, &q) in positions.iter().enumerate() {
                if i != j {
                    let r = length(subtract(p, q));
                    let pressure = -strength
                        * (poly6_f64(r, h) / reference)
                            .powf(f64::from(model.artificial_pressure_exponent));
                    let g = gradient(p, q, h);
                    for axis in 0..3 {
                        correction[axis] +=
                            (multipliers[i] + multipliers[j] + pressure) * g[axis] / rest;
                    }
                }
            }
            correction.map(|value| value * f64::from(model.correction_scale))
        })
        .collect();
    ExpectedStage {
        densities,
        constraints,
        multipliers,
        corrections,
    }
}

fn all_pairs(count: usize) -> InteractionSets {
    InteractionSets::new(
        (0..count)
            .map(|i| (0..count).filter(|&j| i != j).collect())
            .collect(),
    )
    .unwrap()
}

#[test]
fn asymmetric_stage_matches_independent_equations_across_resolutions_and_relaxation() {
    for spacing in [0.025, 0.05, 0.1] {
        for relaxation in [0.000001, 1.0, 10.0, 20.0] {
            for pressure in [false, true] {
                let config = configuration(&format!(
                    "[numerical]\nparticle_spacing_m = {spacing}\nrelaxation_scale = {relaxation}\nartificial_pressure_enabled = {pressure}"
                ));
                let positions: Vec<_> = [
                    [0.0, 0.0, 0.0],
                    [0.8, 0.2, -0.1],
                    [1.7, -0.15, 0.3],
                    [0.4, 1.4, -0.3],
                    [3.8, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
                .map(|p| Vec3::new(p[0] * spacing, p[1] * spacing, p[2] * spacing))
                .into();
                let input: Vec<_> = positions
                    .iter()
                    .map(|p| [p.x, p.y, p.z].map(f64::from))
                    .collect();
                let expected = independent_stage(&input, &config);
                let actual = compute_jacobi_stage(
                    &positions,
                    &all_pairs(positions.len()),
                    config.pbf_parameters().unwrap(),
                )
                .unwrap();
                let h = config.derived.support_radius_m;
                let rest = reference_density(f64::from(spacing), f64::from(spacing) * 2.0);
                for i in 0..positions.len() {
                    // Relative density rounding scales as h^-3; multipliers as
                    // h^2 and corrections as h. The allowances stay dimensioned.
                    assert_absolute(
                        actual.normalised_densities[i],
                        expected.densities[i],
                        rest as f32 * 3.0e-6,
                    );
                    assert_absolute(actual.constraints[i], expected.constraints[i], 3.0e-6);
                    // Isolated particles under the legacy epsilon have very
                    // large multipliers despite their exactly zero correction.
                    let multiplier_tolerance =
                        (h * h * 5.0e-5).max(expected.multipliers[i].abs() as f32 * 5.0e-6);
                    assert_absolute(
                        actual.multipliers[i],
                        expected.multipliers[i],
                        multiplier_tolerance,
                    );
                    assert_vec_absolute(actual.corrections[i], expected.corrections[i], h * 5.0e-5);
                }
            }
        }
    }
}

#[test]
fn default_regularisation_limits_sparse_near_support_corrections() {
    for spacing in [0.025, 0.05, 0.1] {
        let config = configuration(&format!("[numerical]\nparticle_spacing_m = {spacing}"));
        let h = config.derived.support_radius_m;
        for ratio in [0.75, 0.9, 0.95, 0.99, 0.999] {
            let positions = [Vec3::ZERO, Vec3::new(ratio * h, 0.0, 0.0)];
            let stage =
                compute_jacobi_stage(&positions, &all_pairs(2), config.pbf_parameters().unwrap())
                    .unwrap();
            let correction = stage.corrections[0].length_squared().sqrt();
            // A quarter of support leaves these initially separated particles
            // ordered, without relying on collision projection to catch them.
            assert!(
                correction < 0.25 * h,
                "spacing {spacing}, ratio {ratio}: correction {correction}, h {h}"
            );
            assert_eq!(stage.corrections[1], -stage.corrections[0]);
        }
    }
}

#[test]
fn legacy_relaxation_reproduces_large_corrections_in_independent_arithmetic() {
    let config = configuration("[numerical]\nrelaxation_scale = 0.000001\ncorrection_scale = 1.0");
    let h = config.derived.support_radius_m;
    let positions = [Vec3::ZERO, Vec3::new(0.9 * h, 0.0, 0.0)];
    let input: Vec<_> = positions
        .iter()
        .map(|p| [p.x, p.y, p.z].map(f64::from))
        .collect();
    let expected = independent_stage(&input, &config);
    let stage =
        compute_jacobi_stage(&positions, &all_pairs(2), config.pbf_parameters().unwrap()).unwrap();
    // Both precisions predict a displacement over forty support radii. This
    // distinguishes weak regularisation from a binary32-only implementation bug.
    assert!(length(expected.corrections[0]) > 40.0 * f64::from(h));
    assert!(stage.corrections[0].length_squared().sqrt() > 40.0 * h);
    assert_vec_absolute(stage.corrections[0], expected.corrections[0], h * 0.001);
}

#[test]
fn regularised_defaults_prevent_catastrophic_collapse_in_small_falling_scenes() {
    for name in ["falling-block-04.toml", "falling-block-06.toml"] {
        let scene = load_resolved_scene(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("scenes")
                .join(name),
            None,
        )
        .unwrap();
        let mut solver = SerialSimulation::from_scene(&scene).unwrap();
        solver.run_steps(120).unwrap();
        let particles = solver.snapshot().particles();
        let distinct: BTreeSet<_> = particles
            .iter()
            .map(|p| [p.position_m.x, p.position_m.y, p.position_m.z].map(f32::to_bits))
            .collect();
        assert_eq!(
            distinct.len(),
            particles.len(),
            "coincident particle collapse in {name}"
        );
        let h = f64::from(scene.configuration.numerical.particle_spacing_m) * 2.0;
        let rest = reference_density(
            f64::from(scene.configuration.numerical.particle_spacing_m),
            h,
        );
        let positions: Vec<_> = particles
            .iter()
            .map(|p| [p.position_m.x, p.position_m.y, p.position_m.z].map(f64::from))
            .collect();
        let mut mean_error = 0.0;
        for &p in &positions {
            let density: f64 = positions
                .iter()
                .map(|&q| poly6_f64(length(subtract(p, q)), h))
                .sum();
            // These are coarse collapse guards, not a fluid-quality certificate.
            // They allow deficient free surfaces but reject doubled density.
            assert!(
                density / rest < 2.0,
                "excess compression in {name}: {}",
                density / rest
            );
            mean_error += (density / rest - 1.0).abs() / positions.len() as f64;
        }
        assert!(
            mean_error < 0.25,
            "large overall density error in {name}: {mean_error}"
        );
    }
}
