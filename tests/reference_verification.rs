//! Frozen external inputs and complete accepted outputs guard the serial oracle.
//! Independent binary64 equations and health checks qualify the golden data;
//! replay agreement alone would only show that an implementation repeats itself.

#[path = "support/serial_oracle.rs"]
mod oracle;
mod support;

use std::{fs, path::PathBuf};

use maelstrom::{
    config::parse_configuration_overlay,
    math::Vec3,
    scene::{Particle, ResolvedScene, load_resolved_scene},
    simulation::SerialSimulation,
    workflow::{
        CompletionStatus, RecordedCflLimit, RunRequest, RunResult, SnapshotRecord, replay_inputs,
        run,
    },
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use support::{assert_absolute, assert_vec_absolute, poly6_f64};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    family: String,
    configuration: String,
    configuration_sha256: String,
    implementation_revision: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    scene: String,
    sha256: String,
    particles: usize,
    refine_timestep: u32,
    steps: u64,
    checkpoints: Vec<u64>,
    golden: Option<String>,
    golden_sha256: Option<String>,
}

fn path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn manifest() -> Manifest {
    serde_json::from_str(include_str!("../reference/manifest.json")).unwrap()
}

fn scene(case: &Case) -> ResolvedScene {
    let overlay = parse_configuration_overlay(&format!(
        "[numerical]\nrefine_timestep = {}",
        case.refine_timestep
    ))
    .unwrap();
    load_resolved_scene(&path(&case.scene), Some(&overlay)).unwrap()
}

fn request(case: &Case) -> RunRequest {
    RunRequest {
        steps: case.steps,
        checkpoints: case.checkpoints.clone(),
    }
}

fn document(result: &RunResult) -> Value {
    serde_json::from_str(&result.to_json().unwrap()).unwrap()
}

fn assert_hash(relative: &str, expected: &str) {
    let actual: String = Sha256::digest(fs::read(path(relative)).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(actual, expected, "frozen input changed: {relative}");
}

// JSON numbers retain binary32 round trips, but ordinary JSON equality treats
// signed zero as equal. Walk numbers explicitly so serial changes cannot hide it.
fn assert_exact(actual: &Value, expected: &Value, location: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => assert_eq!(
            (a.as_f64().unwrap() as f32).to_bits(),
            (b.as_f64().unwrap() as f32).to_bits(),
            "binary32 mismatch at {location}: {actual} versus {expected}"
        ),
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "array length at {location}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert_exact(a, b, &format!("{location}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, a) in a {
                assert_exact(a, &b[key], &format!("{location}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "mismatch at {location}"),
    }
}

fn numerical_document(mut value: Value) -> Value {
    // Build context is recorded, but does not change the physical experiment.
    for key in [
        "build_revision",
        "target_os",
        "target_arch",
        "debug_assertions",
    ] {
        value["metadata"].as_object_mut().unwrap().remove(key);
    }
    value
}

#[test]
fn frozen_family_has_exact_inputs_and_one_declared_scalability_sequence() {
    let manifest = manifest();
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.family, "falling-block-refined-v1");
    assert_eq!(
        manifest.implementation_revision,
        "54c5176fe0475b39e53759f11aab355c4c815583"
    );
    assert_hash(&manifest.configuration, &manifest.configuration_sha256);
    assert_eq!(manifest.cases.len(), 8);
    let counts = [27, 64, 216, 512, 1_000, 1_728, 4_096, 8_000];
    let mut keys = Vec::new();
    for (index, case) in manifest.cases.iter().enumerate() {
        assert_hash(&case.scene, &case.sha256);
        let input = scene(case);
        assert_eq!(input.particles.len(), counts[index]);
        assert_eq!(case.particles, counts[index]);
        let numerical = &input.configuration.numerical;
        assert_eq!(numerical.solver_iterations, 8);
        assert_eq!(numerical.relaxation_scale, 20.0);
        assert_eq!(numerical.particle_spacing_m, 0.05);
        assert_eq!(numerical.support_radius_ratio, 2.0);
        assert!(numerical.artificial_pressure_enabled);
        assert_eq!(case.golden.is_some(), index < 2);
        if let Some(golden) = &case.golden {
            assert_hash(golden, case.golden_sha256.as_ref().unwrap());
        } else {
            assert!(case.golden_sha256.is_none());
        }
        if index == 0 {
            assert_eq!(case.steps, 12);
            assert_eq!(case.checkpoints, [0, 1, 4, 12]);
            assert_eq!(case.refine_timestep, 1);
            assert_eq!(numerical.external_acceleration_m_s2, Vec3::ZERO);
        } else {
            assert_eq!(case.steps, 480);
            assert_eq!(case.checkpoints, [0, 1, 120, 240, 480]);
            assert_eq!(case.refine_timestep, 4);
            assert_eq!(
                numerical.fixed_timestep_s.to_bits(),
                (1.0_f32 / 480.0).to_bits()
            );
            assert_eq!(numerical.correction_scale, 0.0625);
            assert_eq!(input.planes.len(), 6);
        }
        keys.push(input.comparison_key());
    }
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), manifest.cases.len());
}

#[test]
fn golden_runs_replay_and_repeat_every_accepted_value_exactly() {
    let manifest = manifest();
    for case in manifest.cases.iter().filter(|case| case.golden.is_some()) {
        let json = fs::read_to_string(path(case.golden.as_ref().unwrap())).unwrap();
        let golden: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            golden["metadata"]["build_revision"],
            manifest.implementation_revision
        );
        assert_eq!(golden["metadata"]["debug_assertions"], false);
        let (recorded, recorded_request) = replay_inputs(&json).unwrap();
        let input = scene(case);
        assert_eq!(recorded.comparison_key(), input.comparison_key());
        assert_eq!(recorded_request, request(case));
        let before = input.clone();
        for result in [
            run(&input, request(case)),
            run(&input, request(case)),
            run(&recorded, recorded_request),
        ] {
            assert_eq!(result.status(), CompletionStatus::Completed);
            assert_eq!(result.completed_steps(), case.steps);
            assert_eq!(result.warning_count(), 0);
            assert_exact(
                &numerical_document(document(&result)),
                &numerical_document(golden.clone()),
                &case.scene,
            );
        }
        assert_eq!(input, before, "comparison inputs must remain immutable");
    }
}

#[test]
fn golden_results_replay_through_the_public_cli_in_both_formats() {
    for case in manifest().cases.iter().filter(|case| case.golden.is_some()) {
        let golden_path = path(case.golden.as_ref().unwrap());
        let expected: Value =
            serde_json::from_str(&fs::read_to_string(&golden_path).unwrap()).unwrap();
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_maelstrom"))
            .arg("replay")
            .arg(&golden_path)
            .args(["--format", "json", "--quiet"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_exact(
            &numerical_document(actual),
            &numerical_document(expected),
            "CLI replay",
        );
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_maelstrom"))
            .arg("replay")
            .arg(golden_path)
            .args(["--format", "text", "--quiet"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let summary = String::from_utf8(output.stdout).unwrap();
        let result = run(&scene(case), request(case));
        assert_eq!(summary, result.summary());
    }
}

#[test]
fn frozen_reference_cancellation_preserves_state_at_every_stage_and_retries_exactly() {
    let manifest = manifest();
    let input = scene(&manifest.cases[1]);
    let mut setup = SerialSimulation::from_scene(&input).unwrap();
    for _ in 0..120 {
        let _ = setup.step().unwrap();
    }
    // Start from the moving pre-impact checkpoint, retaining the refined
    // timestep and all eight iterations rather than a historical work count.
    let accepted = setup.snapshot().particles().to_vec();
    let mut clean =
        SerialSimulation::new(&input.configuration, accepted.clone(), input.planes.clone())
            .unwrap();
    let mut polls = 0;
    let expected = clean
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    for cancel_at in 1..=polls {
        let mut solver =
            SerialSimulation::new(&input.configuration, accepted.clone(), input.planes.clone())
                .unwrap();
        let before = SnapshotRecord::capture(solver.snapshot())
            .to_json()
            .unwrap();
        let mut seen = 0;
        assert!(matches!(
            solver.step_with_cancellation(|| {
                seen += 1;
                seen == cancel_at
            }),
            Err(maelstrom::simulation::StepError::Cancelled)
        ));
        assert_eq!(
            SnapshotRecord::capture(solver.snapshot())
                .to_json()
                .unwrap(),
            before
        );
        assert_eq!(solver.step().unwrap(), expected);
        assert_eq!(
            SnapshotRecord::capture(solver.snapshot())
                .to_json()
                .unwrap(),
            SnapshotRecord::capture(clean.snapshot()).to_json().unwrap()
        );
    }
}

#[test]
fn small_lattice_every_transition_matches_independent_equations() {
    let manifest = manifest();
    let case = &manifest.cases[0];
    let input = scene(case);
    let mut solver = SerialSimulation::from_scene(&input).unwrap();
    for step in 1..=case.steps {
        let expected = oracle::step(
            solver.snapshot().particles(),
            &input.configuration,
            &[],
            8,
            oracle::Variant::default(),
        );
        assert_eq!(expected.stages.len(), 9);
        if step == 1 {
            let in_place = oracle::step(
                solver.snapshot().particles(),
                &input.configuration,
                &[],
                8,
                oracle::Variant {
                    in_place: true,
                    ..Default::default()
                },
            );
            assert!(oracle::maximum_difference(&expected.positions, &in_place.positions) > 1.0e-6);
        }
        let _ = solver.step().unwrap();
        for (index, actual) in solver.snapshot().particles().iter().enumerate() {
            // Compare a complete transition from the same accepted input. An
            // independently evolved binary64 trajectory amplifies rounding
            // differences and is a different precision experiment. Full serial
            // trajectories are checked bit for bit against the frozen outputs.
            assert_vec_absolute(actual.position_m, expected.positions[index], 3.0e-6);
            assert_vec_absolute(actual.velocity_m_s, expected.velocities[index], 4.0e-4);
        }
        assert_eq!(solver.snapshot().completed_steps(), step);
    }
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    let delta = [
        f64::from(a.x) - f64::from(b.x),
        f64::from(a.y) - f64::from(b.y),
        f64::from(a.z) - f64::from(b.z),
    ];
    delta.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn independent_errors(particles: &[Particle], input: &ResolvedScene) -> Vec<f64> {
    let spacing = f64::from(input.configuration.numerical.particle_spacing_m);
    let h = spacing * 2.0;
    let rest: f64 = (-3..=3)
        .flat_map(|x| (-3..=3).flat_map(move |y| (-3..=3).map(move |z| (x, y, z))))
        .map(|(x, y, z)| poly6_f64(spacing * f64::from(x * x + y * y + z * z).sqrt(), h))
        .sum();
    particles
        .iter()
        .map(|p| {
            particles
                .iter()
                .map(|q| poly6_f64(distance(p.position_m, q.position_m), h))
                .sum::<f64>()
                / rest
                - 1.0
        })
        .collect()
}

fn energy(particles: &[Particle]) -> f64 {
    particles
        .iter()
        .map(|p| {
            let v = [p.velocity_m_s.x, p.velocity_m_s.y, p.velocity_m_s.z].map(f64::from);
            0.5 * v.iter().map(|x| x * x).sum::<f64>() + 9.81 * f64::from(p.position_m.y)
        })
        .sum::<f64>()
        / particles.len() as f64
}

#[test]
fn golden_falling_trajectory_has_independent_health_audits_and_cfl_evidence() {
    let manifest = manifest();
    let case = &manifest.cases[1];
    let input = scene(case);
    let result = run(&input, request(case));
    let mut solver = SerialSimulation::from_scene(&input).unwrap();
    let initial_errors = independent_errors(&input.particles, &input);
    let model = &input.configuration.numerical;
    let dt = f64::from(model.fixed_timestep_s);
    let h = f64::from(model.particle_spacing_m) * 2.0;
    let epsilon = f64::from(model.relaxation_scale) / h.powi(2);
    let soft_energy = initial_errors.iter().map(|e| e * e).sum::<f64>()
        / initial_errors.len() as f64
        * f64::from(model.correction_scale)
        / (2.0 * epsilon * dt * dt);
    // The displacement allowance stays at the unrefined value, rather than
    // growing as dt shrinks. This is the existing qualification energy screen.
    let allowance = 0.5 * (0.2 * 0.05 / (1.0 / 120.0_f64)).powi(2) + soft_energy;
    let initial_energy = energy(&input.particles);
    let mut touched_floor = false;
    for diagnostic in result.steps() {
        let speed = solver
            .snapshot()
            .particles()
            .iter()
            .map(|p| {
                let v = [
                    f64::from(p.velocity_m_s.x),
                    f64::from(p.velocity_m_s.y) - 9.81 * dt,
                    f64::from(p.velocity_m_s.z),
                ];
                v.iter().map(|x| x * x).sum::<f64>().sqrt()
            })
            .fold(0.0, f64::max);
        assert_absolute(diagnostic.maximum_speed_m_s, speed, 2.0e-6);
        let limit = 0.4 * f64::from(model.particle_spacing_m) / speed;
        match diagnostic.cfl_limit {
            RecordedCflLimit::Finite(actual) => assert_absolute(actual, limit, 1.0e-7),
            RecordedCflLimit::Unbounded => panic!("gravity supplies a non-zero prediction speed"),
        }
        assert_eq!(diagnostic.warning, dt > limit);
        assert!(!diagnostic.warning);
        let outcome = solver.step().unwrap();
        assert_eq!(
            outcome.cfl.maximum_speed.to_bits(),
            diagnostic.maximum_speed_m_s.to_bits()
        );
        let particles = solver.snapshot().particles();
        for (i, p) in particles.iter().enumerate() {
            assert!(p.position_m.is_finite() && p.velocity_m_s.is_finite());
            touched_floor |= p.position_m.y == 0.0;
            for q in &particles[i + 1..] {
                assert!(distance(p.position_m, q.position_m) >= 0.025);
            }
        }
        assert!(energy(particles) <= initial_energy + allowance);
        let errors = independent_errors(particles, &input);
        assert!(errors.iter().all(|&error| error < 0.35));
        if let Some(checkpoint) = result
            .checkpoints()
            .iter()
            .find(|c| c.snapshot().completed_steps() == solver.snapshot().completed_steps())
        {
            assert_eq!(
                checkpoint.snapshot(),
                &SnapshotRecord::capture(solver.snapshot())
            );
        }
    }
    assert!(
        touched_floor,
        "the golden must exercise contact, not only free fall"
    );
    for checkpoint in result.checkpoints() {
        let particles: Vec<_> = checkpoint
            .snapshot()
            .particles()
            .iter()
            .map(|p| p.particle())
            .collect();
        let errors = independent_errors(&particles, &input);
        let audit = checkpoint.density().unwrap();
        assert_eq!(audit.sample_count, case.particles);
        // About 2 ppm allows binary32 reductions while independently checking
        // the full accepted support, including the deficient free surface.
        assert_absolute(
            audit.mean_error.unwrap(),
            errors.iter().map(|x| x.abs()).sum::<f64>() / errors.len() as f64,
            2.0e-6,
        );
        assert_absolute(
            audit.maximum_error.unwrap(),
            errors.iter().map(|x| x.abs()).fold(0.0, f64::max),
            2.0e-6,
        );
    }
    let quiet = run(
        &input,
        RunRequest {
            steps: case.steps,
            checkpoints: vec![],
        },
    );
    assert_eq!(quiet.final_snapshot(), result.final_snapshot());
    assert_eq!(quiet.steps(), result.steps());
}

#[test]
fn exact_comparison_detects_signed_zero_and_non_final_checkpoint_changes() {
    assert!(
        std::panic::catch_unwind(|| assert_exact(&Value::from(-0.0), &Value::from(0.0), "zero"))
            .is_err()
    );
    let expected = serde_json::json!({"checkpoints": [{"x": 1.0}], "final": 2.0});
    let mut changed = expected.clone();
    changed["checkpoints"][0]["x"] = Value::from(1.0001);
    assert!(std::panic::catch_unwind(|| assert_exact(&changed, &expected, "run")).is_err());
}

#[test]
fn measured_reference_runs_preserve_every_numerical_bit() {
    use maelstrom::workflow::{MeasurementMode, run_measured_with_cancellation};

    for case in manifest().cases.iter().filter(|case| case.golden.is_some()) {
        let input = scene(case);
        let golden: Value =
            serde_json::from_str(&fs::read_to_string(path(case.golden.as_ref().unwrap())).unwrap())
                .unwrap();
        for mode in [MeasurementMode::Total, MeasurementMode::Stages] {
            let measured = run_measured_with_cancellation(&input, request(case), mode, || false);
            let timings = measured.measurements().unwrap();
            assert_eq!(timings.steps.len(), case.steps as usize);
            assert!(timings.failed_attempt_s.is_none());
            for step in &timings.steps {
                assert!(step.total_s.is_finite() && step.total_s >= 0.0);
                assert_eq!(step.stages.is_some(), mode == MeasurementMode::Stages);
                if let Some(stages) = step.stages {
                    let values = [
                        stages.validation_and_acceleration_s,
                        stages.prediction_s,
                        stages.neighbour_search_s,
                        stages.jacobi_s,
                        stages.projection_s,
                        stages.acceptance_s,
                    ];
                    assert!(values.iter().all(|s| s.is_finite() && *s >= 0.0));
                    assert!(values.iter().sum::<f64>() <= step.total_s + 1e-12);
                }
            }
            let mut actual = document(&measured);
            actual.as_object_mut().unwrap().remove("measurements");
            assert_exact(
                &numerical_document(actual),
                &numerical_document(golden.clone()),
                "measured golden",
            );
        }
    }
}

#[test]
fn measured_cancellation_keeps_poll_order_and_successful_work_separate() {
    use maelstrom::workflow::{
        MeasurementMode, run_measured_with_cancellation, run_with_cancellation,
    };
    let input = scene(&manifest().cases[1]);
    let request = RunRequest {
        steps: 2,
        checkpoints: vec![0, 1, 2],
    };
    let mut polls = 0;
    let mut one_step = SerialSimulation::from_scene(&input).unwrap();
    let _ = one_step
        .step_with_cancellation(|| {
            polls += 1;
            false
        })
        .unwrap();
    for cancel_at in [1, 3, polls, polls + 1, polls + 5, polls * 2] {
        let mut count = 0;
        let expected = run_with_cancellation(&input, request.clone(), || {
            count += 1;
            count == cancel_at
        });
        for mode in [MeasurementMode::Total, MeasurementMode::Stages] {
            let mut actual_count = 0;
            let actual = run_measured_with_cancellation(&input, request.clone(), mode, || {
                actual_count += 1;
                actual_count == cancel_at
            });
            assert_eq!(actual_count, count);
            assert_eq!(actual.status(), CompletionStatus::Cancelled);
            let timings = actual.measurements().unwrap();
            assert_eq!(timings.steps.len(), actual.completed_steps() as usize);
            assert!(timings.failed_attempt_s.unwrap() >= 0.0);
            let mut value = document(&actual);
            value.as_object_mut().unwrap().remove("measurements");
            assert_exact(&value, &document(&expected), "cancelled measured run");
        }
    }
    for mode in [MeasurementMode::Total, MeasurementMode::Stages] {
        let zero = run_measured_with_cancellation(
            &input,
            RunRequest {
                steps: 0,
                checkpoints: vec![0],
            },
            mode,
            || panic!("zero steps must not poll"),
        );
        assert!(zero.measurements().unwrap().steps.is_empty());
        assert_eq!(zero.completed_steps(), 0);
        assert!(zero.measurements().unwrap().failed_attempt_s.is_none());
    }
}

#[test]
fn measured_empty_invalid_and_numerically_failed_runs_are_explicit() {
    use maelstrom::workflow::{MeasurementMode, run_measured_with_cancellation};
    for mode in [MeasurementMode::Total, MeasurementMode::Stages] {
        let mut input = scene(&manifest().cases[1]);
        input.particles.clear();
        let empty = run_measured_with_cancellation(
            &input,
            RunRequest {
                steps: 2,
                checkpoints: vec![0, 2],
            },
            mode,
            || false,
        );
        assert_eq!(empty.status(), CompletionStatus::Completed);
        assert_eq!(empty.measurements().unwrap().steps.len(), 2);
        assert_eq!(empty.checkpoints()[1].density().unwrap().sample_count, 0);

        let mut input = scene(&manifest().cases[1]);
        input.particles[0].velocity_m_s.x = f32::MAX;
        input.configuration.numerical.external_acceleration_m_s2.x = f32::MAX;
        let failed = run_measured_with_cancellation(
            &input,
            RunRequest {
                steps: 2,
                checkpoints: vec![],
            },
            mode,
            || false,
        );
        assert_eq!(failed.status(), CompletionStatus::SimulationFailure);
        assert_eq!(failed.completed_steps(), 0);
        assert!(failed.measurements().unwrap().steps.is_empty());
        assert!(failed.measurements().unwrap().failed_attempt_s.unwrap() >= 0.0);
        assert_eq!(
            failed.final_snapshot().unwrap().particles()[0].particle(),
            input.particles[0]
        );

        input.configuration.numerical.fixed_timestep_s = 0.0;
        let invalid = run_measured_with_cancellation(
            &input,
            RunRequest {
                steps: 1,
                checkpoints: vec![],
            },
            mode,
            || false,
        );
        assert_eq!(invalid.status(), CompletionStatus::InputValidationFailure);
        assert!(invalid.measurements().unwrap().steps.is_empty());
        assert!(invalid.measurements().unwrap().failed_attempt_s.is_none());
    }
}
