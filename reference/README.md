# Serial comparison fixtures

`manifest.json` freezes the source bytes, run requests and scalability order. The two golden JSON
files are ordinary CLI results with self-contained resolved inputs. They were generated from the
unchanged numerical implementation at `54c5176fe0475b39e53759f11aab355c4c815583`, using Rust/Cargo
1.98.1, an optimised build on x86_64 WSL2 Ubuntu 24.04.5. Their SHA-256 hashes are in the manifest.
No performance measurements are encoded in these fixtures.

`family-review-v1.json` retains the local completion and checkpoint diagnostics for one public CLI
execution of every manifest case, alongside input identities and complete-result hashes. All seven
falling blocks completed 480 steps without warnings or failures. This compact report is not a golden
trajectory for every size and does not extend the every-step health checks beyond the primary fixture.
Its optional recorded build revisions are preserved as observed; absent values remain explicit.

Run the complete verification suite from the repository root:

```sh
cargo test --locked --offline
cargo test --locked --offline --release
```

The focused `reference_verification` integration target checks fixture hashes, exact repeatability,
public CLI replay, independent equations, density audits, CFL reporting and numerical health.
It compares every accepted output value in both golden results, allowing only build-context fields
to differ. Golden comparisons include signed-zero bits and intermediate checkpoints. Other integration
targets verify kernels, input validation, neighbour membership, Jacobi staging, projection, cancellation
and failure atomicity.

To exercise the main comparison request through the public CLI:

```sh
cargo run --locked --offline --release -- run scenes/falling-block-04.toml \
  --refine-timestep 4 --steps 480 --checkpoints 0,1,120,240,480 \
  --format json --output /tmp/falling-block-candidate.json
cargo run --locked --offline --release -- replay reference/falling-block-v1.json --format text
```

Output paths must be new files. To exercise the full size sequence, substitute the external scene
paths listed in the manifest, retaining its exact refinement, steps and checkpoint list. These larger
requests are explicit correctness workloads; the ordinary suite runs the two small golden fixtures.

To reproduce a golden capture, check out its recorded implementation in a separate authorised
checkout, build with `MAELSTROM_BUILD_REVISION` set to that full commit, and run the manifest's request
with `--format json --quiet`. The unforced lattice uses `reference/small-lattice.toml`, refinement 1,
12 steps and checkpoints `0,1,4,12`. Compare numerical results across build contexts; a whole-file
hash also includes the recorded revision, target and build profile. Verification never regenerates
golden files. Replacing one requires an explicit change to the frozen experiment, review against the
independent checks and new provenance and hashes.

Future backends reuse the complete recorded inputs and checkpoint outputs. The owning EPQ record,
`docs/project/development/milestones/00/reference-verification.md`, defines exact structural matches,
per-quantity parallel tolerances and the required disagreement report. The serial golden is an oracle
for the recorded numerical method, not a guarantee of engineering fluid accuracy or cross-platform
bitwise agreement. Independent binary64 trajectories can amplify rounding differences; the lattice
equation checks therefore compare each complete transition from the same accepted input, while
golden comparisons verify the complete binary32 trajectory exactly.
