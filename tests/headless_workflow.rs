//! Independent audit equations and exact recorded-state comparisons exercise the
//! boundary between accepted simulation work, observation and publication.

mod support;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use maelstrom::{
    boundary::Plane,
    config::{FieldSource, parse_configuration_overlay, resolve_effective_configuration},
    diagnostics::audit_density,
    math::Vec3,
    scene::{Particle, RandomnessPolicy, ResolvedScene, load_resolved_scene},
    simulation::SerialSimulation,
    workflow::{
        CompletionStatus, OutputFormat, RunRequest, RunResult, SnapshotRecord, replay_inputs, run,
        run_with_cancellation,
    },
};
use support::{assert_absolute, poly6_f64};

fn scene(overlay: &str, particles: Vec<Particle>, planes: Vec<Plane>) -> ResolvedScene {
    let overlay = parse_configuration_overlay(overlay).unwrap();
    ResolvedScene {
        schema_version: 1,
        name: "small input".into(),
        description: "independent fixture".into(),
        randomness: RandomnessPolicy::None,
        configuration: resolve_effective_configuration(None, None, Some(&overlay)).unwrap(),
        particles,
        planes,
    }
}

fn particle(identity: u64, x: f32) -> Particle {
    Particle {
        identity,
        position_m: Vec3::new(x, 0.0, 0.0),
        velocity_m_s: Vec3::ZERO,
    }
}

fn request(steps: u64, checkpoints: &[u64]) -> RunRequest {
    RunRequest {
        steps,
        checkpoints: checkpoints.to_vec(),
    }
}

fn particle_bits(particles: impl IntoIterator<Item = Particle>) -> Vec<(u64, [u32; 6])> {
    particles
        .into_iter()
        .map(|p| {
            (
                p.identity,
                [
                    p.position_m.x,
                    p.position_m.y,
                    p.position_m.z,
                    p.velocity_m_s.x,
                    p.velocity_m_s.y,
                    p.velocity_m_s.z,
                ]
                .map(f32::to_bits),
            )
        })
        .collect()
}

fn snapshot_bits(snapshot: &SnapshotRecord) -> Vec<(u64, [u32; 6])> {
    particle_bits(snapshot.particles().iter().map(|p| p.particle()))
}

fn document(result: &RunResult) -> serde_json::Value {
    serde_json::from_str(&result.to_json().unwrap()).unwrap()
}

#[test]
fn complete_set_audit_includes_self_and_uses_mean_of_absolute_errors() {
    let input = scene(
        "",
        vec![
            particle(2, 0.0),
            particle(8, 0.5),
            particle(31, 1.0),
            particle(97, 3.0),
        ],
        vec![],
    );
    let solver = SerialSimulation::from_scene(&input).unwrap();
    let before = particle_bits(solver.snapshot().particles().iter().copied());
    let actual = audit_density(solver.snapshot(), 1.0, 2.0).unwrap();
    let self_weight = poly6_f64(0.0, 1.0);
    let pair_weight = poly6_f64(0.5, 1.0);
    let densities = [
        self_weight + pair_weight,
        self_weight + 2.0 * pair_weight,
        self_weight + pair_weight,
        self_weight,
    ];
    let errors = densities.map(|density| (density / 2.0 - 1.0).abs());
    assert_eq!(actual.sample_count, 4);
    // Four binary32 kernel evaluations and an ordered sum need less than 1e-6
    // absolute error here; the allowance is far below the fixture's differences.
    assert_absolute(
        actual.mean_error.unwrap(),
        errors.iter().sum::<f64>() / 4.0,
        1.0e-6,
    );
    assert_absolute(
        actual.maximum_error.unwrap(),
        errors.into_iter().fold(0.0, f64::max),
        1.0e-6,
    );
    let wrong_signed_mean = (densities.iter().sum::<f64>() / 8.0 - 1.0).abs();
    assert!((f64::from(actual.mean_error.unwrap()) - wrong_signed_mean).abs() > 0.01);
    assert_eq!(
        particle_bits(solver.snapshot().particles().iter().copied()),
        before
    );
}

#[test]
fn audit_strict_support_and_coincident_self_contribution_match_independent_values() {
    let input = scene(
        "",
        vec![particle(0, 0.0), particle(1, 0.0), particle(2, 1.0)],
        vec![],
    );
    let solver = SerialSimulation::from_scene(&input).unwrap();
    let result = audit_density(solver.snapshot(), 1.0, 1.0).unwrap();
    let w = poly6_f64(0.0, 1.0);
    assert_absolute(
        result.mean_error.unwrap(),
        (2.0 * (2.0 * w - 1.0) + w - 1.0) / 3.0,
        2.0e-6,
    );
    assert_absolute(result.maximum_error.unwrap(), 2.0 * w - 1.0, 2.0e-6);

    let single = SerialSimulation::from_scene(&scene("", vec![particle(0, 0.0)], vec![])).unwrap();
    let result = audit_density(single.snapshot(), 1.0, 1.0).unwrap();
    assert_eq!(result.sample_count, 1);
    assert_absolute(result.mean_error.unwrap(), w - 1.0, 1.0e-6);
    assert_eq!(result.mean_error, result.maximum_error);
}

fn independent_rest_density(h: f64, spacing: f64) -> f64 {
    let mut total = 0.0;
    for x in -3..=3 {
        for y in -3..=3 {
            for z in -3..=3 {
                let distance = f64::from(x * x + y * y + z * z).sqrt() * spacing;
                total += poly6_f64(distance, h);
            }
        }
    }
    total
}

#[test]
fn checkpoint_audit_discovers_neighbours_absent_from_solver_membership() {
    let input = scene(
        "[numerical]\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]",
        vec![particle(0, -0.2), particle(1, 0.04)],
        vec![Plane::new(Vec3::new(1.0, 0.0, 0.0), 0.0, 0.0).unwrap()],
    );
    let result = run(&input, request(1, &[1]));
    assert!(result.status().is_complete());
    let positions = result.final_snapshot().unwrap();
    assert_eq!(positions.particles()[0].particle().position_m.x, 0.0);
    assert_eq!(positions.particles()[1].particle().position_m.x, 0.04);
    let h = f64::from(input.configuration.derived.support_radius_m);
    let rest = independent_rest_density(
        h,
        f64::from(input.configuration.numerical.particle_spacing_m),
    );
    let expected = ((poly6_f64(0.0, h) + poly6_f64(f64::from(0.04_f32), h)) / rest - 1.0).abs();
    let retained_membership_error = (poly6_f64(0.0, h) / rest - 1.0).abs();
    let audit = result.checkpoints()[0].density().unwrap();
    assert_eq!(audit.sample_count, 2);
    assert_absolute(audit.mean_error.unwrap(), expected, 1.0e-6);
    assert_absolute(audit.maximum_error.unwrap(), expected, 1.0e-6);
    assert!((expected - retained_membership_error).abs() > 0.1);
}

#[test]
fn empty_audits_and_unbounded_cfl_are_explicit_in_both_outputs() {
    let input = scene("", vec![], vec![]);
    let result = run(&input, request(2, &[0, 2]));
    assert_eq!(result.status(), CompletionStatus::Completed);
    assert_eq!(result.completed_steps(), 2);
    for checkpoint in result.checkpoints() {
        let density = checkpoint.density().unwrap();
        assert_eq!(density.sample_count, 0);
        assert_eq!(density.mean_error, None);
        assert_eq!(density.maximum_error, None);
    }
    let json = document(&result);
    assert_eq!(json["steps"][0]["cfl_limit"]["kind"], "unbounded");
    assert!(json["checkpoints"][0]["density"]["mean_error"].is_null());
    assert!(json["checkpoints"][0]["density"]["maximum_error"].is_null());
    assert!(
        result
            .summary()
            .contains("0 samples; mean relative density error unavailable; maximum unavailable")
    );
    assert!(result.summary().contains("tightest CFL limit: unbounded"));
    let zero = run(&input, request(0, &[0]));
    assert_eq!(zero.status(), CompletionStatus::Completed);
    assert_eq!(zero.checkpoints().len(), 1);
    assert!(zero.summary().contains("no accepted steps"));
}

#[test]
fn large_finite_density_error_is_only_diagnostic() {
    let input = scene("", (0..100).map(|id| particle(id, 0.0)).collect(), vec![]);
    let result = run(&input, request(1, &[1]));
    assert_eq!(result.status(), CompletionStatus::Completed);
    let density = result.checkpoints()[0].density().unwrap();
    assert_eq!(density.sample_count, 100);
    assert!(density.mean_error.unwrap() > 10.0);
    assert!(result.failures().is_empty());
}

#[test]
fn earlier_cfl_warning_survives_later_unbounded_steps_and_summary() {
    let mut moving = particle(0, 0.0);
    moving.velocity_m_s.y = -10.0;
    let input = scene(
        "[numerical]\nfixed_timestep_s = 0.125\nexternal_acceleration_m_s2 = [0.0, 0.0, 0.0]",
        vec![moving],
        vec![Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0).unwrap()],
    );
    let result = run(&input, request(3, &[3]));
    assert_eq!(result.status(), CompletionStatus::CompletedWithWarnings);
    assert_eq!(result.warning_count(), 1);
    assert!(result.steps()[0].warning);
    assert!(!result.steps()[1].warning);
    assert!(!result.steps()[2].warning);
    assert_eq!(result.steps()[0].maximum_speed_m_s, 10.0);
    let json = document(&result);
    assert_eq!(json["status"], "completed_with_warnings");
    assert_eq!(json["steps"][0]["cfl_limit"]["kind"], "finite");
    assert_eq!(json["steps"][1]["cfl_limit"]["kind"], "unbounded");
    assert_eq!(
        json["input"]["configuration"]["numerical"]["fixed_timestep_s"],
        0.125
    );
    assert!(result.summary().contains("CFL warning steps: 1"));
}

#[test]
fn observation_and_output_selection_leave_every_accepted_checkpoint_bitwise_unchanged() {
    let input = load_resolved_scene(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes/falling-block-04.toml"),
        None,
    )
    .unwrap();
    let observed = run(&input, request(12, &[0, 1, 4, 12]));
    let bare = run(&input, request(12, &[]));
    assert!(observed.status().is_complete());
    assert_eq!(observed.status(), bare.status());
    assert_eq!(observed.steps(), bare.steps());
    assert_eq!(
        observed.metadata().unwrap().run_id,
        bare.metadata().unwrap().run_id
    );
    assert_eq!(
        snapshot_bits(observed.final_snapshot().unwrap()),
        snapshot_bits(bare.final_snapshot().unwrap())
    );
    let mut solver = SerialSimulation::from_scene(&input).unwrap();
    for checkpoint in observed.checkpoints() {
        while solver.snapshot().completed_steps() < checkpoint.snapshot().completed_steps() {
            let outcome = solver.step().unwrap();
            assert_eq!(
                outcome.cfl.maximum_speed.to_bits(),
                observed.steps()[outcome.completed_steps as usize - 1]
                    .maximum_speed_m_s
                    .to_bits()
            );
        }
        assert_eq!(
            snapshot_bits(checkpoint.snapshot()),
            particle_bits(solver.snapshot().particles().iter().copied())
        );
    }
    let before = snapshot_bits(observed.final_snapshot().unwrap());
    assert!(!observed.to_json().unwrap().is_empty());
    assert!(!observed.summary().is_empty());
    assert_eq!(snapshot_bits(observed.final_snapshot().unwrap()), before);
}

#[test]
fn numerical_failure_preserves_prior_steps_warnings_and_last_accepted_snapshot() {
    let input = scene(
        "[numerical]\nfixed_timestep_s = 1.0\nexternal_acceleration_m_s2 = [1.0e19, 0.0, 0.0]",
        vec![particle(0, 0.0)],
        vec![],
    );
    let result = run(&input, request(3, &[1, 2, 3]));
    let accepted = run(&input, request(1, &[]));
    assert_eq!(result.status(), CompletionStatus::SimulationFailure);
    assert_eq!(result.completed_steps(), 1);
    assert_eq!(result.steps().len(), 1);
    assert_eq!(result.warning_count(), 1);
    assert_eq!(result.checkpoints().len(), 1);
    assert_eq!(
        snapshot_bits(result.final_snapshot().unwrap()),
        snapshot_bits(accepted.final_snapshot().unwrap())
    );
    let json = document(&result);
    assert_eq!(json["status"], "simulation_failure");
    assert_eq!(json["completed_steps"], "1");
    assert_eq!(json["request"]["steps"], "3");
    assert!(!result.status().is_complete());
}

#[test]
fn cancellation_retains_prior_work_and_poll_order_is_independent_of_observation() {
    let input = scene("", vec![particle(0, 0.0)], vec![]);
    let mut solver = SerialSimulation::from_scene(&input).unwrap();
    let mut polls_per_step = 0;
    assert_eq!(
        solver
            .step_with_cancellation(|| {
                polls_per_step += 1;
                false
            })
            .unwrap()
            .completed_steps,
        1
    );
    let mut polls = 0;
    let observed = run_with_cancellation(&input, request(3, &[0, 1, 3]), || {
        polls += 1;
        polls > polls_per_step + 5
    });
    let mut bare_polls = 0;
    let bare = run_with_cancellation(&input, request(3, &[]), || {
        bare_polls += 1;
        bare_polls > polls_per_step + 5
    });
    assert_eq!(polls, bare_polls);
    assert_eq!(observed.status(), CompletionStatus::Cancelled);
    assert_eq!(observed.completed_steps(), 1);
    assert_eq!(observed.checkpoints().len(), 2);
    assert_eq!(
        snapshot_bits(observed.final_snapshot().unwrap()),
        snapshot_bits(bare.final_snapshot().unwrap())
    );
    assert_eq!(observed.steps(), bare.steps());
    let immediate = run_with_cancellation(&input, request(2, &[]), || true);
    assert_eq!(immediate.status(), CompletionStatus::Cancelled);
    assert_eq!(immediate.completed_steps(), 0);
    assert_eq!(
        snapshot_bits(immediate.final_snapshot().unwrap()),
        particle_bits(input.particles)
    );
}

#[test]
fn input_and_checkpoint_validation_failures_are_distinct_from_execution_failure() {
    let valid = scene("", vec![particle(0, 0.0)], vec![]);
    for checkpoints in [vec![2, 1], vec![1, 1], vec![4]] {
        let result = run(
            &valid,
            RunRequest {
                steps: 3,
                checkpoints,
            },
        );
        assert_eq!(result.status(), CompletionStatus::InputValidationFailure);
        assert_eq!(result.completed_steps(), 0);
        assert!(result.final_snapshot().is_none());
        assert!(result.input().is_none());
    }
    let mut invalid = valid.clone();
    invalid.particles[0].position_m.x = f32::NAN;
    let result = run(&invalid, request(1, &[]));
    assert_eq!(result.status(), CompletionStatus::InputValidationFailure);
    let json = document(&result);
    assert_eq!(json["status"], "input_validation_failure");
    assert_eq!(json["application_version"], env!("CARGO_PKG_VERSION"));
    assert!(json["input"].is_null());
    assert!(replay_inputs(&result.to_json().unwrap()).is_err());
    invalid = valid;
    invalid.schema_version = 999;
    assert_eq!(
        run(&invalid, request(0, &[])).status(),
        CompletionStatus::InputValidationFailure
    );
}

#[test]
fn audit_failure_does_not_reject_or_corrupt_the_last_accepted_state() {
    let input = scene("", vec![particle(0, 0.0), particle(1, 1.0e20)], vec![]);
    let without = run(&input, request(0, &[]));
    assert_eq!(without.status(), CompletionStatus::Completed);
    let audited = run(&input, request(0, &[0]));
    assert_eq!(audited.status(), CompletionStatus::AuditFailure);
    assert_eq!(audited.completed_steps(), 0);
    assert_eq!(audited.checkpoints().len(), 1);
    assert!(audited.checkpoints()[0].density().is_none());
    assert_eq!(
        snapshot_bits(audited.final_snapshot().unwrap()),
        snapshot_bits(without.final_snapshot().unwrap())
    );
    assert!(!audited.status().is_complete());
    assert_eq!(document(&audited)["status"], "audit_failure");
    assert!(audited.summary().contains("density audit failed"));

    let solver = SerialSimulation::from_scene(&scene("", vec![], vec![])).unwrap();
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(audit_density(solver.snapshot(), value, 1.0).is_err());
        assert!(audit_density(solver.snapshot(), 1.0, value).is_err());
    }
}

#[test]
fn full_record_replays_identical_bits_with_names_and_provenance_preserved() {
    let input = load_resolved_scene(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes/falling-block-04.toml"),
        None,
    )
    .unwrap();
    let original = run(&input, request(6, &[0, 3, 6]));
    let json = original.to_json().unwrap();
    let (restored, request) = replay_inputs(&json).unwrap();
    assert_eq!(restored, input);
    assert_eq!(
        restored.configuration.comparison_key(),
        input.configuration.comparison_key()
    );
    let repeated = run(&restored, request);
    assert_eq!(repeated.to_json().unwrap(), json);
    assert_eq!(
        snapshot_bits(original.final_snapshot().unwrap()),
        snapshot_bits(repeated.final_snapshot().unwrap())
    );
    for checkpoint in original.checkpoints() {
        let decoded = SnapshotRecord::from_json(&checkpoint.snapshot().to_json().unwrap()).unwrap();
        assert_eq!(
            decoded.completed_steps(),
            checkpoint.snapshot().completed_steps()
        );
        assert_eq!(
            snapshot_bits(&decoded),
            snapshot_bits(checkpoint.snapshot())
        );
    }
}

#[test]
fn replay_uses_all_recorded_numerical_values_and_canonical_identity_order() {
    let input = scene(
        "[numerical]\nfixed_timestep_s = 0.02\nexternal_acceleration_m_s2 = [0.25, -1.0, 0.0]\nparticle_spacing_m = 0.1\nsupport_radius_ratio = 1.5\nphysical_rest_density_kg_m3 = 800.0\nrelaxation_scale = 10.0\nartificial_pressure_enabled = false\nartificial_pressure_strength_scale = 0.02\nartificial_pressure_reference_separation_ratio = 0.3\nartificial_pressure_exponent = 3.0",
        vec![particle(2, 1.0), particle(u64::MAX, 2.0)],
        vec![Plane::new(Vec3::new(0.0, 1.0, 0.0), -0.2, 0.0).unwrap()],
    );
    let original = run(&input, request(3, &[0, 1, 3]));
    assert!(original.status().is_complete());
    let (restored, request) = replay_inputs(&original.to_json().unwrap()).unwrap();
    assert_eq!(restored, input);
    let repeated = run(&restored, request.clone());
    assert_eq!(original.to_json().unwrap(), repeated.to_json().unwrap());
    let mut reversed = restored;
    reversed.particles.reverse();
    let reordered = run(&reversed, request);
    assert_eq!(
        original.metadata().unwrap().input_id,
        reordered.metadata().unwrap().input_id
    );
    assert_eq!(original.to_json().unwrap(), reordered.to_json().unwrap());
}

#[test]
fn physical_identity_excludes_labels_provenance_and_observation_but_tracks_inputs() {
    let input = scene("", vec![particle(42, 0.0)], vec![]);
    let original = run(&input, request(0, &[]));
    let mut renamed = input.clone();
    renamed.name = "different display name\nwith controls".into();
    renamed.description = "different display context".into();
    renamed.configuration.name = "new configuration name".into();
    renamed
        .configuration
        .sources
        .insert("fixed_timestep_s", FieldSource::Scene);
    let observed = run(&renamed, request(0, &[0]));
    assert_eq!(
        original.metadata().unwrap().run_id,
        observed.metadata().unwrap().run_id
    );
    assert_eq!(
        original.metadata().unwrap().input_id,
        observed.metadata().unwrap().input_id
    );
    assert_eq!(observed.input().unwrap().resolve().unwrap(), renamed);
    assert_ne!(original.to_json().unwrap(), observed.to_json().unwrap());
    let longer = run(&input, request(1, &[]));
    assert_ne!(
        original.metadata().unwrap().run_id,
        longer.metadata().unwrap().run_id
    );
    assert_eq!(
        original.metadata().unwrap().input_id,
        longer.metadata().unwrap().input_id
    );
    renamed.particles[0].position_m.x = 0.01;
    let different = run(&renamed, request(0, &[]));
    assert_ne!(
        original.metadata().unwrap().input_id,
        different.metadata().unwrap().input_id
    );
    assert_eq!(
        original.metadata().unwrap().configuration_id,
        different.metadata().unwrap().configuration_id
    );
}

#[test]
fn machine_records_preserve_signed_zero_subnormals_large_values_and_u64_identities() {
    let mut p = particle(u64::MAX, -0.0);
    p.position_m.y = f32::from_bits(1);
    p.position_m.z = f32::MAX;
    p.velocity_m_s = Vec3::new(
        f32::MIN_POSITIVE,
        -f32::from_bits(1),
        f32::from_bits(0x3f80_0001),
    );
    let input = scene("", vec![p], vec![]);
    let result = run(&input, request(0, &[]));
    assert_eq!(result.status(), CompletionStatus::Completed);
    let json = result.to_json().unwrap();
    let value = document(&result);
    assert_eq!(
        value["input"]["particles"][0]["identity"],
        u64::MAX.to_string()
    );
    let (replayed, _) = replay_inputs(&json).unwrap();
    assert_eq!(particle_bits(replayed.particles), particle_bits([p]));
    let snapshot = result.final_snapshot().unwrap();
    let restored = SnapshotRecord::from_json(&snapshot.to_json().unwrap()).unwrap();
    assert_eq!(snapshot_bits(&restored), particle_bits([p]));
    // Parsing JSON through a binary64 intermediate is a common consumer path.
    let nested = serde_json::to_string(&value["final_snapshot"]).unwrap();
    assert_eq!(
        snapshot_bits(&SnapshotRecord::from_json(&nested).unwrap()),
        particle_bits([p])
    );
}

#[test]
fn malformed_or_inconsistent_recorded_inputs_are_rejected_before_replay() {
    let result = run(&scene("", vec![particle(1, 0.0)], vec![]), request(0, &[]));
    let valid = document(&result);
    let mutations: &[fn(&mut serde_json::Value)] = &[
        |v| v["schema_version"] = 9.into(),
        |v| v["backend"] = "unsupported-backend".into(),
        |v| v["input"]["schema_version"] = 9.into(),
        |v| v["input"]["configuration"]["derived"]["particle_mass_kg"] = 1.0.into(),
        |v| v["input"]["particles"][0]["identity"] = "01".into(),
        |v| v["input"]["particles"][0]["position_m"]["x"] = 0.5.into(),
        |v| v["input"]["configuration"]["numerical"]["extra"] = true.into(),
        |v| v["request"]["checkpoints"] = serde_json::json!(["1"]),
        |v| v["input"]["particles"][0]["position_m"]["x"] = serde_json::Value::Null,
        |v| v["input"]["comparison_key"] = "fabricated".into(),
    ];
    for mutate in mutations {
        let mut invalid = valid.clone();
        mutate(&mut invalid);
        assert!(replay_inputs(&invalid.to_string()).is_err());
    }
    let mut snapshot = valid["final_snapshot"].clone();
    snapshot["schema_version"] = 9.into();
    assert!(SnapshotRecord::from_json(&snapshot.to_string()).is_err());
}

static TEMP_COUNT: AtomicU64 = AtomicU64::new(0);
struct TempDirectory(PathBuf);
impl TempDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "maelstrom-workflow-test-{}-{}",
            std::process::id(),
            TEMP_COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn publication_is_complete_no_clobber_and_failure_preserves_state_and_history() {
    let directory = TempDirectory::new();
    let input = scene("", vec![particle(0, 0.0)], vec![]);
    let mut result = run(&input, request(1, &[1]));
    let accepted = snapshot_bits(result.final_snapshot().unwrap());
    let output = directory.0.join("result.json");
    let before = result.to_json().unwrap();
    result.publish(&output, OutputFormat::Json).unwrap();
    assert_eq!(fs::read_to_string(&output).unwrap(), before);
    assert!(result.publish(&output, OutputFormat::Text).is_err());
    assert_eq!(fs::read_to_string(&output).unwrap(), before);
    assert_eq!(result.status(), CompletionStatus::OutputFailure);
    assert_eq!(snapshot_bits(result.final_snapshot().unwrap()), accepted);
    assert_eq!(result.completed_steps(), 1);
    let failure_record = directory.0.join("failure.json");
    result.publish(&failure_record, OutputFormat::Json).unwrap();
    let published: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&failure_record).unwrap()).unwrap();
    assert_eq!(published["status"], "output_failure");
    assert_eq!(published["failures"][0]["status"], "output_failure");
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 2);

    let mut cancelled = run_with_cancellation(&input, request(3, &[]), || true);
    let missing = directory.0.join("missing/result.json");
    assert!(cancelled.publish(&missing, OutputFormat::Json).is_err());
    assert!(!missing.exists());
    assert_eq!(cancelled.failures().len(), 2);
    assert_eq!(cancelled.failures()[0].status, CompletionStatus::Cancelled);
    assert_eq!(
        cancelled.failures()[1].status,
        CompletionStatus::OutputFailure
    );
    assert_eq!(cancelled.completed_steps(), 0);
}
