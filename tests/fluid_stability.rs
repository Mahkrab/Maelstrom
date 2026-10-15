//! Separation and energy checks complement density error: particles can obtain
//! a plausible density by clumping, and projection can inject motion into a
//! stationary block. Small fixtures keep impact and convergence checks bounded.

#[path = "support/serial_oracle.rs"]
mod oracle;
mod support;

use maelstrom::{
    boundary::Plane,
    config::{ResolvedConfiguration, parse_configuration_overlay, resolve_effective_configuration},
    fluid::compute_jacobi_stage,
    math::Vec3,
    neighbours::brute_force,
    scene::Particle,
    simulation::SerialSimulation,
};
use support::{assert_vec_absolute, poly6_f64};

fn configuration(spacing: f32, relaxation: Option<f32>, gravity: bool) -> ResolvedConfiguration {
    let g = if gravity { -9.81 } else { 0.0 };
    let relaxation =
        relaxation.map_or_else(String::new, |value| format!("relaxation_scale = {value}\n"));
    let overlay = parse_configuration_overlay(&format!(
        "[numerical]\nparticle_spacing_m = {spacing}\n{relaxation}external_acceleration_m_s2 = [0.0, {g}, 0.0]"
    )).unwrap();
    resolve_effective_configuration(None, None, Some(&overlay)).unwrap()
}

fn block(side: usize, spacing: f32, compression: f32) -> Vec<Particle> {
    let mut particles = Vec::new();
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                particles.push(Particle {
                    identity: particles.len() as u64,
                    position_m: Vec3::new(
                        4.0 * spacing + x as f32 * spacing * compression,
                        8.0 * spacing + y as f32 * spacing * compression,
                        4.0 * spacing + z as f32 * spacing * compression,
                    ),
                    velocity_m_s: Vec3::ZERO,
                });
            }
        }
    }
    particles
}

fn walls(side: usize, spacing: f32) -> Vec<Plane> {
    [
        (Vec3::new(1.0, 0.0, 0.0), 0.0),
        (Vec3::new(-1.0, 0.0, 0.0), -(side as f32 + 8.0) * spacing),
        (Vec3::new(0.0, 1.0, 0.0), 0.0),
        (Vec3::new(0.0, -1.0, 0.0), -(side as f32 + 12.0) * spacing),
        (Vec3::new(0.0, 0.0, 1.0), 0.0),
        (Vec3::new(0.0, 0.0, -1.0), -(side as f32 + 8.0) * spacing),
    ]
    .map(|(normal, offset)| Plane::new(normal, offset, 1.0e-6).unwrap())
    .into()
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    [
        f64::from(a.x) - f64::from(b.x),
        f64::from(a.y) - f64::from(b.y),
        f64::from(a.z) - f64::from(b.z),
    ]
    .iter()
    .map(|value| value * value)
    .sum::<f64>()
    .sqrt()
}

#[derive(Debug)]
struct Health {
    minimum_separation: f64,
    maximum_speed: f64,
    mean_kinetic: f64,
    mean_energy: f64,
    mean_compression: f64,
    maximum_compression: f64,
    mean_squared_constraint: f64,
}

fn health(particles: &[Particle], config: &ResolvedConfiguration) -> Health {
    let spacing = f64::from(config.numerical.particle_spacing_m);
    let h = spacing * f64::from(config.numerical.support_radius_ratio);
    // A larger cube independently derives the reference, with strict support
    // enforced by the scalar kernel rather than production's lattice bounds.
    let rest: f64 = (-4..=4)
        .flat_map(|x| (-4..=4).flat_map(move |y| (-4..=4).map(move |z| (x, y, z))))
        .map(|(x, y, z)| poly6_f64(spacing * f64::from(x * x + y * y + z * z).sqrt(), h))
        .sum();
    let mut result = Health {
        minimum_separation: f64::INFINITY,
        maximum_speed: 0.0,
        mean_kinetic: 0.0,
        mean_energy: 0.0,
        mean_compression: 0.0,
        maximum_compression: 0.0,
        mean_squared_constraint: 0.0,
    };
    for (i, p) in particles.iter().enumerate() {
        let squared_speed = [p.velocity_m_s.x, p.velocity_m_s.y, p.velocity_m_s.z]
            .map(f64::from)
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        result.maximum_speed = result.maximum_speed.max(squared_speed.sqrt());
        result.mean_kinetic += squared_speed / (2.0 * particles.len() as f64);
        result.mean_energy += (squared_speed / 2.0
            - f64::from(config.numerical.external_acceleration_m_s2.y) * f64::from(p.position_m.y))
            / particles.len() as f64;
        let density: f64 = particles
            .iter()
            .map(|q| poly6_f64(distance(p.position_m, q.position_m), h))
            .sum();
        let constraint = density / rest - 1.0;
        let compression = constraint.max(0.0);
        result.mean_squared_constraint += constraint * constraint / particles.len() as f64;
        result.mean_compression += compression / particles.len() as f64;
        result.maximum_compression = result.maximum_compression.max(compression);
        for q in &particles[i + 1..] {
            result.minimum_separation = result
                .minimum_separation
                .min(distance(p.position_m, q.position_m));
        }
    }
    result
}

fn assert_separation(values: &Health, spacing: f32) {
    // Half the initial spacing is an anti-clumping guard, not a particle radius.
    assert!(
        values.minimum_separation >= 0.5 * f64::from(spacing),
        "lost separation: {values:?}"
    );
}

#[test]
fn sparse_pairs_remain_ordered_during_each_correction() {
    for spacing in [0.025, 0.05, 0.1] {
        let config = configuration(spacing, None, false);
        let h = config.derived.support_radius_m;
        for ratio in [0.5, 0.75, 0.9, 0.99] {
            let mut positions = [Vec3::ZERO, Vec3::new(ratio * h, 0.0, 0.0)];
            let fixed = brute_force(&positions, h).unwrap();
            for _ in 0..4 {
                let stage =
                    compute_jacobi_stage(&positions, &fixed, config.pbf_parameters().unwrap())
                        .unwrap();
                for (position, correction) in positions.iter_mut().zip(stage.corrections) {
                    *position += correction;
                }
                assert!(
                    positions[0].x < positions[1].x,
                    "pair crossed at spacing {spacing}, initial ratio {ratio}: {positions:?}"
                );
            }
        }
    }
}

#[test]
fn intermediate_regularisation_still_inverts_pairs_and_injects_motion() {
    let mut config = configuration(0.05, Some(1.0), false);
    config.numerical.solver_iterations = 4;
    config.numerical.correction_scale = 1.0;
    let initial = vec![
        Particle {
            identity: 0,
            position_m: Vec3::ZERO,
            velocity_m_s: Vec3::ZERO,
        },
        Particle {
            identity: 1,
            position_m: Vec3::new(0.05, 0.0, 0.0),
            velocity_m_s: Vec3::ZERO,
        },
    ];
    let expected = oracle::step(&initial, &config, &[], 1, oracle::Variant::default());
    assert!(expected.positions[0][0] > expected.positions[1][0]);
    let positions: Vec<_> = initial.iter().map(|p| p.position_m).collect();
    let fixed = brute_force(&positions, config.derived.support_radius_m).unwrap();
    let stage = compute_jacobi_stage(&positions, &fixed, config.pbf_parameters().unwrap()).unwrap();
    for ((&position, correction), &reference) in positions
        .iter()
        .zip(stage.corrections)
        .zip(&expected.positions)
    {
        assert_vec_absolute(position + correction, reference, 5.0e-6);
    }
    let mut solver = SerialSimulation::new(&config, block(4, 0.05, 1.0), vec![]).unwrap();
    let _ = solver.step().unwrap();
    let values = health(solver.snapshot().particles(), &config);
    let budget = 0.5 * (0.2 * 0.05 / f64::from(config.numerical.fixed_timestep_s)).powi(2);
    assert!(values.mean_kinetic > budget);
}

#[test]
fn stationary_blocks_limit_correction_induced_motion_and_settle() {
    for spacing in [0.025, 0.05, 0.1] {
        let config = configuration(spacing, None, false);
        let mut solver = SerialSimulation::new(&config, block(4, spacing, 1.0), vec![]).unwrap();
        let dt = f64::from(config.numerical.fixed_timestep_s);
        // The RMS displacement budget is one fifth of spacing per step. Convert
        // it to specific kinetic energy so the guard scales with resolution.
        let energy_budget = 0.5 * (0.2 * f64::from(spacing) / dt).powi(2);
        let mut peak_kinetic = 0.0_f64;
        for _ in 0..120 {
            let _ = solver.step().unwrap();
            let values = health(solver.snapshot().particles(), &config);
            assert_separation(&values, spacing);
            assert!(
                values.maximum_speed * dt <= 0.5 * f64::from(spacing),
                "large correction displacement: {values:?}"
            );
            assert!(
                values.mean_kinetic <= energy_budget,
                "large correction energy: {values:?}"
            );
            peak_kinetic = peak_kinetic.max(values.mean_kinetic);
        }
        let final_values = health(solver.snapshot().particles(), &config);
        assert!(
            final_values.mean_kinetic < 0.01 * peak_kinetic,
            "stationary block did not settle: {final_values:?}"
        );
    }
}

fn assert_impact(side: usize, spacing: f32, steps: u64, asymmetric: bool) {
    let config = configuration(spacing, None, true);
    let mut initial = block(side, spacing, 1.0);
    if asymmetric {
        initial[5].position_m.x += 0.1 * spacing;
        initial[side * side * side - 1].velocity_m_s =
            Vec3::new(3.0 * spacing, 0.0, -2.0 * spacing);
    }
    verify_impact(&config, initial, walls(side, spacing), steps);
}

fn verify_impact(
    config: &ResolvedConfiguration,
    initial: Vec<Particle>,
    boundaries: Vec<Plane>,
    steps: u64,
) {
    let spacing = config.numerical.particle_spacing_m;
    let initial_energy = health(&initial, config).mean_energy;
    let dt = f64::from(config.numerical.fixed_timestep_s);
    let correction_budget = 0.5 * (0.2 * f64::from(spacing) / dt).powi(2);
    let mut solver = SerialSimulation::new(config, initial, boundaries.clone()).unwrap();
    let mut reached_floor = false;
    for step in 1..=steps {
        let outcome = solver.step().unwrap();
        assert_eq!(outcome.completed_steps, step);
        let particles = solver.snapshot().particles();
        let values = health(particles, config);
        assert_separation(&values, spacing);
        // Monitor excess density separately: a deficient free surface must not
        // hide compression or be treated as a demand to attract isolated fluid.
        assert!(
            values.maximum_compression < 0.35,
            "excess compression: {values:?}"
        );
        assert!(
            values.mean_energy <= initial_energy + correction_budget,
            "impact gained excessive energy: {values:?}"
        );
        for (identity, particle) in particles.iter().enumerate() {
            assert_eq!(particle.identity, identity as u64);
            assert!(particle.position_m.is_finite() && particle.velocity_m_s.is_finite());
            for boundary in &boundaries {
                assert!(boundary.contains(particle.position_m).unwrap());
            }
            reached_floor |= particle.position_m.y == 0.0;
        }
    }
    assert!(reached_floor, "fixture did not exercise floor impact");
    let final_values = health(solver.snapshot().particles(), config);
    assert!(
        final_values.mean_kinetic < 0.01 * initial_energy,
        "impact did not settle: {final_values:?}"
    );
}

#[test]
fn lateral_and_spreading_blocks_survive_wall_and_floor_contact() {
    let config = configuration(0.05, None, true);
    let mut lateral = block(4, 0.05, 1.0);
    for particle in &mut lateral {
        particle.velocity_m_s = Vec3::new(1.0, 0.0, 0.35);
    }
    verify_impact(&config, lateral, walls(4, 0.05), 360);
    let mut spreading = block(4, 0.05, 1.0);
    for particle in &mut spreading {
        particle.velocity_m_s = Vec3::new(
            (particle.position_m.x - 0.275) * 3.0,
            0.0,
            (particle.position_m.z - 0.275) * 3.0,
        );
    }
    let mut boundaries = walls(4, 0.05);
    boundaries[1] = Plane::new(Vec3::new(-1.0, 0.0, 0.0), -2.0, 1.0e-6).unwrap();
    boundaries[5] = Plane::new(Vec3::new(0.0, 0.0, -1.0), -2.0, 1.0e-6).unwrap();
    verify_impact(&config, spreading, boundaries, 360);
}

#[test]
fn matched_time_refinement_converges_in_settled_shape_without_growing_energy_budget() {
    let mut shapes = Vec::new();
    for factor in [1, 2, 4] {
        let overlay =
            parse_configuration_overlay(&format!("[numerical]\nrefine_timestep={factor}")).unwrap();
        let config = resolve_effective_configuration(None, None, Some(&overlay)).unwrap();
        let initial = block(4, 0.05, 1.0);
        let initial_health = health(&initial, &config);
        let initial_energy = initial_health.mean_energy;
        // Hold this physical budget fixed while dt changes. Recomputing it from
        // refined dt would permit quadratically increasing correction energy.
        let h = f64::from(config.numerical.particle_spacing_m) * 2.0;
        let epsilon = f64::from(config.numerical.relaxation_scale) / (h * h);
        let dt = f64::from(config.numerical.fixed_timestep_s);
        // Include a scalar soft-constraint energy allowance for the initially
        // deficient surface. scale/dt² is invariant under matched refinement.
        // This is a screening allowance, not an exact pressure energy model.
        let constraint_allowance = initial_health.mean_squared_constraint
            * f64::from(config.numerical.correction_scale)
            / (2.0 * epsilon * dt * dt);
        let budget = 0.5 * (0.2 * 0.05 / (1.0 / 120.0_f64)).powi(2) + constraint_allowance;
        let mut solver = SerialSimulation::new(&config, initial, walls(4, 0.05)).unwrap();
        for _ in 0..360 * factor {
            let _ = solver.step().unwrap();
            let particles = solver.snapshot().particles();
            let energy: f64 = particles
                .iter()
                .map(|p| {
                    let v = [p.velocity_m_s.x, p.velocity_m_s.y, p.velocity_m_s.z].map(f64::from);
                    0.5 * v.iter().map(|value| value * value).sum::<f64>()
                        + 9.81 * f64::from(p.position_m.y)
                })
                .sum::<f64>()
                / particles.len() as f64;
            assert!(
                energy <= initial_energy + budget,
                "factor {factor}: excess energy {energy}"
            );
        }
        let particles = solver.snapshot().particles();
        let values = health(particles, &config);
        assert_separation(&values, 0.05);
        assert!(values.maximum_compression < 0.35);
        let centre = [0, 1, 2].map(|axis| {
            particles
                .iter()
                .map(|p| [p.position_m.x, p.position_m.y, p.position_m.z][axis] as f64)
                .sum::<f64>()
                / particles.len() as f64
        });
        let radius = (particles
            .iter()
            .map(|p| {
                (f64::from(p.position_m.x) - centre[0]).powi(2)
                    + (f64::from(p.position_m.z) - centre[2]).powi(2)
            })
            .sum::<f64>()
            / particles.len() as f64)
            .sqrt();
        shapes.push([centre[1], radius]);
    }
    let change = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    assert!(change(shapes[0], shapes[1]) < 0.1 * 0.05);
    assert!(change(shapes[1], shapes[2]) < change(shapes[0], shapes[1]));
}

#[test]
fn changing_only_timestep_reproduces_the_correction_energy_defect() {
    let raw = parse_configuration_overlay(
        "[numerical]\nfixed_timestep_s=0.0020833335\nexternal_acceleration_m_s2=[0.0,0.0,0.0]",
    )
    .unwrap();
    let config = resolve_effective_configuration(None, None, Some(&raw)).unwrap();
    let mut solver = SerialSimulation::new(&config, block(4, 0.05, 1.0), vec![]).unwrap();
    let _ = solver.step().unwrap();
    let budget = 0.5 * (0.2 * 0.05 / (1.0 / 120.0_f64)).powi(2);
    assert!(health(solver.snapshot().particles(), &config).mean_kinetic > budget);
    let reference = configuration(0.05, None, false);
    let initial = health(&block(4, 0.05, 1.0), &reference);
    let epsilon = f64::from(reference.numerical.relaxation_scale) / (0.1_f64 * 0.1);
    let allowance = initial.mean_squared_constraint / (2.0 * epsilon * (1.0 / 120.0_f64).powi(2));
    assert!(health(solver.snapshot().particles(), &config).mean_kinetic > budget + allowance);
}

#[test]
fn refined_timestep_matches_independent_position_and_velocity_equations() {
    let overlay = parse_configuration_overlay(
        "[numerical]\nrefine_timestep=4\nsolver_iterations=4\nexternal_acceleration_m_s2=[0.0,0.0,0.0]",
    )
    .unwrap();
    let config = resolve_effective_configuration(None, None, Some(&overlay)).unwrap();
    let initial = vec![
        Particle {
            identity: 3,
            position_m: Vec3::ZERO,
            velocity_m_s: Vec3::ZERO,
        },
        Particle {
            identity: 19,
            position_m: Vec3::new(0.04, 0.015, 0.0),
            velocity_m_s: Vec3::ZERO,
        },
        Particle {
            identity: 800,
            position_m: Vec3::new(0.07, -0.01, 0.02),
            velocity_m_s: Vec3::ZERO,
        },
    ];
    let expected = oracle::step(&initial, &config, &[], 4, oracle::Variant::default());
    let mut solver = SerialSimulation::new(&config, initial, vec![]).unwrap();
    let _ = solver.step().unwrap();
    for ((actual, &position), &velocity) in solver
        .snapshot()
        .particles()
        .iter()
        .zip(&expected.positions)
        .zip(&expected.velocities)
    {
        assert_vec_absolute(actual.position_m, position, 5.0e-6);
        assert_vec_absolute(
            actual.velocity_m_s,
            velocity,
            5.0e-6 / config.numerical.fixed_timestep_s,
        );
    }
}

#[test]
fn falling_blocks_keep_separation_and_bound_energy_across_resolutions() {
    for spacing in [0.025, 0.05, 0.1] {
        assert_impact(4, spacing, 120, false);
    }
}

#[test]
fn perturbed_block_keeps_separation_during_asymmetric_impact() {
    assert_impact(4, 0.05, 120, true);
}

#[test]
fn small_blocks_remain_settled_over_three_seconds() {
    for side in [4, 6] {
        assert_impact(side, 0.05, 360, false);
    }
}

#[test]
fn compressed_blocks_reduce_excess_density_without_iteration_oscillation() {
    for (iterations, scale, relaxation) in [(4, 1.0, Some(10.0)), (8, 1.0, None)] {
        for spacing in [0.025, 0.05, 0.1] {
            let mut config = configuration(spacing, relaxation, false);
            config.numerical.solver_iterations = iterations as u32;
            config.numerical.correction_scale = scale;
            let initial = block(4, spacing, 0.8);
            let mut particles = initial.clone();
            let initial_values = health(&particles, &config);
            assert!(initial_values.mean_compression > 0.25);
            let expected = oracle::step(
                &initial,
                &config,
                &[],
                iterations,
                oracle::Variant::default(),
            );
            let in_place = oracle::step(
                &initial,
                &config,
                &[],
                iterations,
                oracle::Variant {
                    in_place: true,
                    ..oracle::Variant::default()
                },
            );
            assert!(
                oracle::maximum_difference(&expected.positions, &in_place.positions)
                    > f64::from(config.derived.support_radius_m) * 1.0e-4
            );
            let mut positions: Vec<_> = particles.iter().map(|p| p.position_m).collect();
            let fixed = brute_force(&positions, config.derived.support_radius_m).unwrap();
            for iteration in 1..=4 * iterations {
                let stage =
                    compute_jacobi_stage(&positions, &fixed, config.pbf_parameters().unwrap())
                        .unwrap();
                for (p, c) in positions.iter_mut().zip(stage.corrections) {
                    *p += c;
                }
                for (p, &position) in particles.iter_mut().zip(&positions) {
                    p.position_m = position;
                }
                if [iterations, 2 * iterations, 4 * iterations].contains(&iteration) {
                    let values = health(&particles, &config);
                    assert_separation(&values, spacing);
                    assert!(
                        values.mean_compression < 0.1 * initial_values.mean_compression,
                        "mean compression did not converge at iteration {iteration}: {values:?}"
                    );
                    assert!(
                        values.maximum_compression < 0.25 * initial_values.maximum_compression,
                        "maximum compression did not converge at iteration {iteration}: {values:?}"
                    );
                }
                if iteration == iterations {
                    // Compare the configured passes with independently derived
                    // binary64 positions; more passes above are an untimed study.
                    for (actual, &reference) in positions.iter().zip(&expected.stages[iterations]) {
                        assert_vec_absolute(
                            *actual,
                            reference,
                            config.derived.support_radius_m * 5.0e-5,
                        );
                    }
                }
            }
            let mut solver = SerialSimulation::new(&config, initial, vec![]).unwrap();
            let _ = solver.step().unwrap();
            for ((actual, &position), &velocity) in solver
                .snapshot()
                .particles()
                .iter()
                .zip(&expected.positions)
                .zip(&expected.velocities)
            {
                let position_tolerance = config.derived.support_radius_m * 5.0e-5;
                assert_vec_absolute(actual.position_m, position, position_tolerance);
                assert_vec_absolute(
                    actual.velocity_m_s,
                    velocity,
                    position_tolerance / config.numerical.fixed_timestep_s,
                );
            }
        }
    }
}

#[test]
fn interior_equilibrium_has_negligible_correction() {
    for spacing in [0.025, 0.05, 0.1] {
        let config = configuration(spacing, None, false);
        let positions: Vec<_> = (-3..=3)
            .flat_map(|x| {
                (-3..=3).flat_map(move |y| {
                    (-3..=3).map(move |z| {
                        Vec3::new(x as f32 * spacing, y as f32 * spacing, z as f32 * spacing)
                    })
                })
            })
            .collect();
        let fixed = brute_force(&positions, config.derived.support_radius_m).unwrap();
        let stage =
            compute_jacobi_stage(&positions, &fixed, config.pbf_parameters().unwrap()).unwrap();
        // The centre and its immediate neighbours have filled supports. It is
        // therefore protected from the finite block's free-surface deficiency.
        let centre = 3 * 49 + 3 * 7 + 3;
        assert!(stage.constraints[centre].abs() < 3.0e-6);
        assert!(stage.corrections[centre].length_squared().sqrt() < spacing * 1.0e-5);
    }
}
