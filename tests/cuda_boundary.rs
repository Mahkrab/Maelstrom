//! The device suite is opt-in and fails if the requested hardware is absent.
//! Default CPU checks exercise explicit rejection without any toolkit or driver.

use maelstrom::cuda::{CudaContext, CudaError};
use std::process::Command;

#[test]
fn invalid_probe_requests_are_rejected_by_the_application() {
    for args in [
        vec!["cuda-check", "--repetitions", "0"],
        vec!["cuda-check", "--device", "-1"],
        vec![
            "run",
            "falling-block-04",
            "--steps",
            "1",
            "--execution",
            "cuda",
        ],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_maelstrom"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
    assert!(matches!(
        CudaContext::new(u32::MAX),
        Err(CudaError::InvalidDevice { .. })
    ));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_builds_report_disabled_cuda_without_falling_back() {
    assert!(matches!(
        CudaContext::new(0),
        Err(CudaError::FeatureDisabled)
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_maelstrom"))
        .args(["cuda-check", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(3));
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--features cuda"));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device; run explicitly on the selected hardware"]
fn hardware_layout_lifetimes_and_public_application() {
    use maelstrom::cuda::Probe;
    for _ in 0..8 {
        let mut first = CudaContext::new(0).unwrap();
        let mut second = CudaContext::new(0).unwrap();
        assert_eq!(first.device().ordinal, 0);
        assert!(first.device().memory_bytes > 0);
        for length in [0, 1, 2, 127, 128, 129, 257, 1025] {
            let input: Vec<_> = (0..length)
                .map(|index| Probe {
                    identity: u64::MAX - index as u64,
                    position: [index as f32, -0.0, -2.5],
                    marker: index as u32,
                })
                .collect();
            let original = input.clone();
            let result = first.roundtrip(&input).unwrap();
            assert_eq!(result.len(), length);
            for (index, value) in result.iter().enumerate() {
                // Integer addition gives an independent exact oracle for this
                // bounded range; y checks signed zero and z a fixed exact value.
                assert_eq!(value.identity, u64::MAX - index as u64);
                assert_eq!(
                    value.position.map(f32::to_bits),
                    [((index + 1) as f32).to_bits(), 0, (-3.0_f32).to_bits()]
                );
                assert_eq!(value.marker, index as u32 ^ 0xa5a5_5a5a);
            }
            assert_eq!(input, original);
            for (actual, original) in input.iter().zip(&original) {
                assert_eq!(
                    actual.position.map(f32::to_bits),
                    original.position.map(f32::to_bits)
                );
            }
        }
        first.close().unwrap();
        // Closing one isolated context must not invalidate its surviving peer.
        assert_eq!(
            second
                .roundtrip(&[Probe {
                    identity: 7,
                    position: [-2.0, 2.5, 1.0],
                    marker: 0
                }])
                .unwrap(),
            vec![Probe {
                identity: 7,
                position: [-1.0, -5.0, 0.5],
                marker: 0xa5a5_5a5a
            }]
        );
        // Exercise implicit Drop as well as explicit teardown.
        drop(second);
    }
    assert!(matches!(
        CudaContext::new(i32::MAX as u32),
        Err(CudaError::InvalidDevice { .. })
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_maelstrom"))
        .args(["cuda-check", "--repetitions", "5", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["report"]["verified"], true);
    assert_eq!(report["report"]["repetitions"], 5);
    assert_eq!(report["report"]["records_per_repetition"], 257);
    let unavailable = Command::new(env!("CARGO_BIN_EXE_maelstrom"))
        .env("CUDA_VISIBLE_DEVICES", "")
        .args(["cuda-check", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(unavailable.status.code(), Some(3));
    assert!(unavailable.stdout.is_empty());
    let message = String::from_utf8_lossy(&unavailable.stderr);
    assert!(
        message.contains("Initialise") || message.contains("unavailable"),
        "{message}"
    );
}
