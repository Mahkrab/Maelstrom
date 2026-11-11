//! Independent wire fixtures and fault cases protect the observation boundary.
//! Tests own their directories; no hardware or existing user cache is required.
use super::*;
use crate::{
    math::Vec3,
    scene::{Particle, load_resolved_scene},
    workflow::SnapshotRecord,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "maelstrom-cache-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn scene() -> crate::scene::ResolvedScene {
    load_resolved_scene(Path::new("reference/small-lattice.toml"), None).unwrap()
}
fn bake_at(path: &Path, steps: u64, stride: u64) {
    let scene = scene();
    bake(
        BakeRequest {
            scene: &scene,
            steps,
            stride,
            options: Default::default(),
            destination: path,
        },
        || false,
        |_, _| {},
    )
    .unwrap();
}

#[test]
fn wire_fixture_preserves_offsets_signed_zero_and_maximum_identity() {
    let particles = vec![
        Particle {
            identity: 1,
            position_m: Vec3::new(1.0, -0.0, 3.0),
            velocity_m_s: Vec3::new(7.0, 8.0, 9.0),
        },
        Particle {
            identity: u64::MAX,
            position_m: Vec3::new(4.0, 5.0, 6.0),
            velocity_m_s: Vec3::new(-0.0, 11.0, 12.0),
        },
    ];
    let snapshot = SnapshotRecord::from_particles(particles.clone(), 23).unwrap();
    // Derived generic JSON serialisation is independent of the streaming writer.
    let canonical: [u8; 32] = Sha256::digest(snapshot.to_json().unwrap().as_bytes()).into();
    let request = hex(&[9; 32]);
    let mut expected = Vec::new();
    expected.extend(b"MSPCACHE");
    expected.extend([1, 0, 0, 0]);
    expected.extend(160u32.to_le_bytes());
    for v in [7u64, 23, 2, 48, 48, 160, 184] {
        expected.extend(v.to_le_bytes());
    }
    expected.extend([0; 8]);
    expected.extend([9; 32]);
    expected.extend(canonical);
    expected.extend([0; 16]);
    for bits in [
        0x3f800000u32,
        0x80000000,
        0x40400000,
        0x40800000,
        0x40a00000,
        0x40c00000,
        0x40e00000,
        0x41000000,
        0x41100000,
        0x80000000,
        0x41300000,
        0x41400000,
    ] {
        expected.extend(bits.to_le_bytes());
    }
    assert_eq!(snapshot.canonical_sha256().unwrap(), canonical);
    assert_eq!(
        format::encode(&snapshot, 7, &request, canonical).unwrap(),
        expected
    );
    let receipt = Receipt {
        schema_version: 1,
        index: 7,
        step: "23".into(),
        bytes: 208,
        file_sha256: digest(&expected),
        snapshot_sha256: hex(&canonical),
    };
    let decoded = format::decode(&expected, &receipt, &request, &particles)
        .unwrap()
        .0;
    assert_eq!(decoded.to_json().unwrap(), snapshot.to_json().unwrap());
    assert_eq!(
        decoded.particles()[0].particle().position_m.y.to_bits(),
        0x80000000
    );
    assert_eq!(decoded.particles()[1].particle().identity, u64::MAX);
    for offset in [0, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 112, 144] {
        let mut changed = expected.clone();
        changed[offset] ^= 1;
        let r = Receipt {
            file_sha256: digest(&changed),
            ..Receipt {
                schema_version: 1,
                index: 7,
                step: "23".into(),
                bytes: 208,
                file_sha256: String::new(),
                snapshot_sha256: hex(&canonical),
            }
        };
        assert!(
            format::decode(&changed, &r, &request, &particles).is_err(),
            "header offset {offset}"
        );
    }
    let mut nonfinite = expected.clone();
    nonfinite[160..164].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    let r = Receipt {
        file_sha256: digest(&nonfinite),
        ..receipt
    };
    assert!(format::decode(&nonfinite, &r, &request, &particles).is_err());
}

#[test]
fn schedule_has_exact_endpoints_and_rejects_unbounded_frame_counts() {
    let s = Schedule::new(5, 2).unwrap();
    assert_eq!(
        (0..s.len()).map(|i| s.step(i).unwrap()).collect::<Vec<_>>(),
        [0, 2, 4, 5]
    );
    assert_eq!(Schedule::new(0, 7).unwrap().len(), 1);
    assert!(s.step(4).is_none());
    assert!(Schedule::new(1, 0).is_err());
    assert!(Schedule::new(u64::MAX, 1).is_err());
    assert_eq!(Schedule::new(1_000_000, 1).unwrap().len(), 1_000_001);
}

#[test]
fn bake_round_trips_every_selected_accepted_state_and_diagnostic() {
    let t = Temp::new();
    let path = t.0.join("cache");
    bake_at(&path, 5, 2);
    let cache = Cache::open(&path, false).unwrap();
    assert!(cache.is_complete());
    assert_eq!(cache.frame_count(), 4);
    cache.validate_full(|| false).unwrap();
    let mut solver = crate::simulation::Simulation::from_scene(&scene()).unwrap();
    for step in 0..=5 {
        if step > 0 {
            let _ = solver.step().unwrap();
        }
        if let Some(i) = (0..cache.frame_count()).find(|&i| cache.step(i) == Some(step)) {
            let expected = SnapshotRecord::capture(solver.snapshot());
            let frame = cache.frame(i).unwrap();
            assert_eq!(
                frame.snapshot().to_json().unwrap(),
                expected.to_json().unwrap()
            );
            assert_eq!(
                frame.canonical_sha256(),
                digest(expected.to_json().unwrap().as_bytes())
            );
        }
    }
    let before = fs::read(path.join("cache.json")).unwrap();
    assert!(
        bake(
            BakeRequest {
                scene: &scene(),
                steps: 5,
                stride: 2,
                options: Default::default(),
                destination: &path
            },
            || false,
            |_, _| {}
        )
        .is_err()
    );
    assert_eq!(fs::read(path.join("cache.json")).unwrap(), before);
}

#[test]
fn cancellation_keeps_only_receipt_backed_prefix_and_never_completes() {
    let t = Temp::new();
    let path = t.0.join("cancel");
    let flag = Arc::new(AtomicBool::new(false));
    let signal = flag.clone();
    let result = bake(
        BakeRequest {
            scene: &scene(),
            steps: 10,
            stride: 2,
            options: Default::default(),
            destination: &path,
        },
        || flag.load(Ordering::Relaxed),
        |n, _| {
            if n == 2 {
                signal.store(true, Ordering::Relaxed);
            }
        },
    );
    assert!(matches!(result, Err(CacheError::Cancelled)));
    assert!(!path.join("cache.json").exists());
    assert!(path.join("failure.json").is_file());
    assert!(Cache::open(&path, false).is_err());
    let partial = Cache::open(&path, true).unwrap();
    assert!(!partial.is_complete());
    assert_eq!(partial.frame_count(), 2);
    partial.validate_full(|| false).unwrap();
    assert!(partial.frame(2).is_err());
}

#[test]
fn asynchronous_write_failure_retains_previous_receipt_without_deadlock() {
    let t = Temp::new();
    let path = t.0.join("failure");
    let mut blocked = false;
    let error = bake(
        BakeRequest {
            scene: &scene(),
            steps: 6,
            stride: 2,
            options: Default::default(),
            destination: &path,
        },
        || false,
        |n, _| {
            if n == 1 && !blocked {
                fs::create_dir(path.join("frames").join(frame_name(1))).unwrap();
                blocked = true;
            }
        },
    )
    .unwrap_err();
    assert!(matches!(error, CacheError::Io(_)));
    assert!(!path.join("cache.json").exists());
    let partial = Cache::open(&path, true).unwrap();
    assert_eq!(partial.frame_count(), 1);
    partial.validate_full(|| false).unwrap();
}

#[test]
fn changed_receipts_payloads_and_truncation_cannot_reuse_a_digest() {
    let t = Temp::new();
    let path = t.0.join("cache");
    bake_at(&path, 2, 1);
    let cache = Cache::open(&path, false).unwrap();
    let frame_path = path.join("frames").join(frame_name(1));
    let bytes = fs::read(&frame_path).unwrap();
    let mut changed = bytes.clone();
    changed[160] ^= 1;
    fs::write(&frame_path, changed).unwrap();
    assert!(cache.frame(1).is_err());
    fs::write(&frame_path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(cache.frame(1).is_err());
    fs::write(&frame_path, &bytes).unwrap();
    let receipt_path = path.join("receipts").join(receipt_name(1));
    let mut receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
    receipt["snapshot_sha256"] = serde_json::json!(hex(&[0; 32]));
    fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    assert!(cache.frame(1).is_err());
    assert!(Cache::open(&path, false).is_err());
}

#[cfg(unix)]
#[test]
fn symlink_frames_are_rejected_even_when_the_target_has_valid_bytes() {
    let t = Temp::new();
    let path = t.0.join("cache");
    bake_at(&path, 0, 1);
    let cache = Cache::open(&path, false).unwrap();
    let f = path.join("frames").join(frame_name(0));
    let other = t.0.join("other");
    fs::rename(&f, &other).unwrap();
    std::os::unix::fs::symlink(other, &f).unwrap();
    assert!(cache.frame(0).is_err());
}

#[test]
fn missing_completion_is_explicit_even_if_all_frames_were_written() {
    let t = Temp::new();
    let path = t.0.join("cache");
    bake_at(&path, 0, 1);
    fs::remove_file(path.join("cache.json")).unwrap();
    assert!(Cache::open(&path, false).is_err());
    let partial = Cache::open(&path, true).unwrap();
    assert_eq!(partial.frame_count(), 1);
    assert!(!partial.is_complete());
    fs::remove_file(path.join("receipts").join(receipt_name(0))).unwrap();
    assert_eq!(Cache::open(&path, true).unwrap().frame_count(), 0);
}

#[test]
fn empty_initial_observation_has_a_valid_zero_payload_and_no_diagnostics() {
    let t = Temp::new();
    let path = t.0.join("empty");
    let mut scene = scene();
    scene.particles.clear();
    bake(
        BakeRequest {
            scene: &scene,
            steps: 0,
            stride: 4,
            options: Default::default(),
            destination: &path,
        },
        || false,
        |_, _| {},
    )
    .unwrap();
    let cache = Cache::open(&path, false).unwrap();
    assert_eq!(cache.frame_count(), 1);
    assert_eq!(
        fs::metadata(path.join("frames").join(frame_name(0)))
            .unwrap()
            .len(),
        160
    );
    assert_eq!(
        fs::metadata(path.join("diagnostics.jsonl")).unwrap().len(),
        0
    );
    assert!(cache.frame(0).unwrap().snapshot().particles().is_empty());
    cache.validate_full(|| false).unwrap();
}

#[test]
fn full_validation_detects_a_self_consistent_false_canonical_claim() {
    let t = Temp::new();
    let path = t.0.join("external");
    bake_at(&path, 0, 1);
    let frame_path = path.join("frames").join(frame_name(0));
    let mut bytes = fs::read(&frame_path).unwrap();
    bytes[112..144].fill(0);
    fs::write(&frame_path, &bytes).unwrap();
    let receipt_path = path.join("receipts").join(receipt_name(0));
    let mut receipt: Receipt = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
    receipt.file_sha256 = digest(&bytes);
    receipt.snapshot_sha256 = hex(&[0; 32]);
    let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
    fs::write(&receipt_path, &receipt_bytes).unwrap();
    let completion_path = path.join("cache.json");
    let mut completion: Completion =
        serde_json::from_slice(&fs::read(&completion_path).unwrap()).unwrap();
    completion.receipts_sha256 = digest(&receipt_bytes);
    completion.cache_id = result_id(
        &completion.request_sha256,
        &completion.receipts_sha256,
        &completion.diagnostics_sha256,
    );
    fs::write(&completion_path, serde_json::to_vec(&completion).unwrap()).unwrap();
    // A checksum is not authentication. Full validation checks the semantic
    // canonical claim even when an external producer rewrites all integrity metadata.
    let cache = Cache::open(&path, false).unwrap();
    assert!(cache.validate_full(|| false).is_err());
}

#[test]
fn huge_partial_schedule_indexes_only_the_committed_prefix_and_can_cancel_open() {
    let t = Temp::new();
    let path = t.0.join("sparse");
    bake_at(&path, 0, 1);
    fs::remove_file(path.join("cache.json")).unwrap();
    fs::remove_file(path.join("receipts/frame-00000000.json")).unwrap();
    let mut request: Request =
        serde_json::from_slice(&fs::read(path.join("request.json")).unwrap()).unwrap();
    request.steps = (u64::MAX - 1).to_string();
    fs::write(
        path.join("request.json"),
        serde_json::to_vec_pretty(&request).unwrap(),
    )
    .unwrap();
    let cache = Cache::open(&path, true).unwrap();
    assert_eq!(cache.frame_count(), 0);
    assert_eq!(cache.info()["expected_frames"], u64::MAX);
    assert!(matches!(
        Cache::open_with_cancellation(&path, true, || true),
        Err(CacheError::Cancelled)
    ));
    assert!(!path.join("cache.json").exists());
}

#[test]
fn frame_size_is_checked_without_allocating_and_schedules_cross_old_boundaries() {
    assert!(format::size(usize::MAX).is_err());
    assert!(format::size(isize::MAX as usize / 24).is_err());
    for count in [999_999, 1_000_000, 1_000_001] {
        let schedule = Schedule::new(count - 1, 1).unwrap();
        assert_eq!(schedule.len(), count);
        assert_eq!(schedule.step(count - 1), Some(count - 1));
        assert_eq!(
            format::size(count as usize).unwrap(),
            160 + 24 * count as usize
        );
    }
    let schedule = Schedule::new(u64::MAX, 2).unwrap();
    assert_eq!(schedule.step(schedule.len() - 1), Some(u64::MAX));
    assert_eq!(schedule.step(schedule.len() - 2), Some(u64::MAX - 1));
}

#[test]
#[ignore = "resource qualification: writes and fully validates a million-particle cache"]
fn million_particle_cache_streams_large_input_and_full_validation() {
    let t = Temp::new();
    let path = t.0.join("large");
    let mut scene = scene();
    scene.planes.clear();
    scene.particles = (0..1_000_001u64)
        .map(|identity| Particle {
            identity,
            position_m: Vec3::new(
                (identity % 101) as f32 * 0.125,
                (identity / 101 % 100) as f32 * 0.125,
                (identity / 10100) as f32 * 0.125,
            ),
            velocity_m_s: Vec3::new(-0.0, 0.0, 0.0),
        })
        .collect();
    bake(
        BakeRequest {
            scene: &scene,
            steps: 0,
            stride: 1,
            options: Default::default(),
            destination: &path,
        },
        || false,
        |_, _| {},
    )
    .unwrap();
    assert!(fs::metadata(path.join("input.json")).unwrap().len() > 256 * 1024 * 1024);
    let expected = SnapshotRecord::capture(
        crate::simulation::Simulation::from_scene(&scene)
            .unwrap()
            .snapshot(),
    )
    .canonical_sha256()
    .unwrap();
    drop(scene);
    let cache = Cache::open(&path, false).unwrap();
    assert_eq!(cache.scene().particles.len(), 1_000_001);
    assert_eq!(cache.frame(0).unwrap().canonical_sha256(), hex(&expected));
    cache.validate_full(|| false).unwrap();
    assert!(matches!(
        cache.validate_full(|| true),
        Err(CacheError::Cancelled)
    ));
    println!(
        "input_bytes={} frame_bytes={}",
        fs::metadata(path.join("input.json")).unwrap().len(),
        fs::metadata(path.join("frames/frame-00000000.mpc"))
            .unwrap()
            .len()
    );
}

#[test]
fn diagnostic_file_bound_matches_the_checked_per_record_bound() {
    let t = Temp::new();
    let path = t.0.join("diagnostics");
    bake_at(&path, 1, 1);
    let original = fs::read(path.join("diagnostics.jsonl")).unwrap();
    let mut completion: Completion =
        serde_json::from_slice(&fs::read(path.join("cache.json")).unwrap()).unwrap();
    for padding in [1024, 2048, 4096] {
        let mut line = vec![b' '; padding];
        line.extend_from_slice(&original);
        completion.diagnostics_sha256 = digest(&line);
        completion.cache_id = result_id(
            &completion.request_sha256,
            &completion.receipts_sha256,
            &completion.diagnostics_sha256,
        );
        fs::write(path.join("diagnostics.jsonl"), &line).unwrap();
        fs::write(
            path.join("cache.json"),
            serde_json::to_vec_pretty(&completion).unwrap(),
        )
        .unwrap();
        assert_eq!(Cache::open(&path, false).is_ok(), line.len() <= 4096);
    }
}

#[test]
fn opening_cancels_during_a_streamed_input_document() {
    use std::io::Write;
    let t = Temp::new();
    let path = t.0.join("cancel-read");
    bake_at(&path, 0, 1);
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(path.join("input.json"))
        .unwrap();
    file.write_all(&[b' '; 128 * 1024]).unwrap();
    drop(file);
    let mut polls = 0;
    let result = Cache::open_with_cancellation(&path, false, || {
        polls += 1;
        polls > 5
    });
    assert!(matches!(result, Err(CacheError::Cancelled)));
    assert!(polls > 5);
}

#[test]
#[ignore = "resource qualification: streams diagnostics for over a million accepted steps"]
fn long_empty_bake_has_bounded_diagnostics_and_exact_terminal_step() {
    let t = Temp::new();
    let path = t.0.join("long");
    let mut scene = scene();
    scene.particles.clear();
    bake(
        BakeRequest {
            scene: &scene,
            steps: 1_000_001,
            stride: 500_000,
            options: Default::default(),
            destination: &path,
        },
        || false,
        |_, _| {},
    )
    .unwrap();
    let cache = Cache::open(&path, false).unwrap();
    assert_eq!(cache.frame_count(), 4);
    assert_eq!(
        (0..4).map(|i| cache.step(i).unwrap()).collect::<Vec<_>>(),
        [0, 500_000, 1_000_000, 1_000_001]
    );
    assert_eq!(
        cache.frame(3).unwrap().snapshot().completed_steps(),
        1_000_001
    );
    cache.validate_full(|| false).unwrap();
    println!(
        "diagnostics_bytes={}",
        fs::metadata(path.join("diagnostics.jsonl")).unwrap().len()
    );
}
