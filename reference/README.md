# Numerical comparison fixtures

This directory is verification material, not the user's scene or compiled-run library.
`manifest.json` identifies the exact input bytes, numerical requests and size sequence.
Its two golden JSON results contain complete resolved inputs; they do not contain
performance measurements.

## Run the checks

From the Maelstrom source checkout:

```sh
cargo test --locked --offline --test reference_verification
cargo test --release --locked --offline
```

The reference target checks fixture hashes, signed-zero-aware exact repeatability,
public numerical reruns, independent equations, density/CFL evidence and numerical
health. Other targets cover membership, Jacobi staging, planes and transactional
failure/cancellation. Build context is not a numerical value.

To run the main request explicitly through an installed executable:

```sh
maelstrom run ./scenes/falling-block-04.toml --backend brute-force --unpaced \
  --refine-timestep 4 --steps 480 --checkpoints 0,1,120,240,480 \
  --format json --output candidate.json
maelstrom rerun ./reference/falling-block-v1.json --format text
```

Use a new output path. The benchmark family uses 64, 216, 512, 1,000, 1,728, 4,096 and
8,000 particles; preserve its exact refinement, work count and checkpoints. The small
unforced lattice uses `small-lattice.toml`, refinement 1, twelve steps and checkpoints
0/1/4/12. Larger family requests are explicit correctness workloads, not all ordinary
test cases or demonstrated maximum capacity.

## Provenance and interpretation

The goldens were generated at `54c5176fe0475b39e53759f11aab355c4c815583` using Rust/Cargo
1.98.1, an optimised x86_64 WSL2 Ubuntu 24.04.5 build. Their hashes are in the manifest.
`family-review-v1.json` records one complete CLI request per family size, its input
identity and result hash. All seven completed 480 steps without warnings/failures.
It is not a golden trajectory or an every-step health study for every size.

To reproduce historical bytes, use the recorded implementation in a separately
authorised checkout and its recorded build context. Whole-file hashes include revision,
target and profile; current-build results should be compared numerically, not assumed
to have the same file hash. Tests never regenerate or silently replace goldens.

The fixtures qualify the recorded numerical method, not engineering fluid accuracy or
bitwise agreement on every compiler/device. Complete binary64 trajectories can diverge
through accumulated rounding; independent equation checks start each transition from
the same accepted input. Parallel implementations retain contract-owned quantity budgets
and exact structural/status requirements, not arbitrary blanket tolerances.

The owning research policy is in the parent EPQ repository at
`docs/project/development/milestones/00/reference-verification.md`. It defines comparison
and disagreement reporting. Revisions need an explicit decision, independent checks
and new provenance; old results remain identifiable. Ordinary user scenes and execution
preferences do not silently redefine this experiment.
