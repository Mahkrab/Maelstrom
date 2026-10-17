# Running and reproducing a headless simulation

Maelstrom runs a serial binary32 Position-Based Fluids solver without a display. Configuration and
scene resolution precede stepping; observations and output read accepted state. No graphical,
threaded or CUDA backend is selected by these commands.

## Build and discovery

From the Maelstrom checkout, build with stable Rust and Cargo. The retained reference was verified
with Rust/Cargo 1.98.1 on x86_64 WSL2 Ubuntu 24.04.5. Offline commands require the locked dependencies
to be cached; run `cargo fetch --locked` first if necessary.

```sh
cargo build --release --locked --offline
./target/release/maelstrom --help
./target/release/maelstrom --version
./target/release/maelstrom scenes
./target/release/maelstrom run --help
```

Examples below use `maelstrom` on PATH. Either substitute `./target/release/maelstrom` or install
with `cargo install --path . --locked --offline`. Bundled names are `falling-block-04`, `-06`, `-08`,
`-10`, `-12`, `-16` and `-20`, containing 64 through 8,000 particles. They select current unrefined
defaults; the frozen comparison request adds explicit refinement.

## Validate and inspect

```sh
maelstrom validate
maelstrom validate falling-block-04
maelstrom validate ./scenes/falling-block-04.toml --config ./config/defaults-v4.toml
maelstrom inspect falling-block-04 --format json
maelstrom inspect --set 'external_acceleration_m_s2=[0.0, 0.0, 0.0]'
```

`validate` and `inspect` accept configuration alone and perform no timesteps. The
[input specification](inputs.md) describes schema versions, units, every field, validation and
precedence. Use `inspect` to check effective settings before attributing a result to a configuration.
`--set` accepts repeatable primary-field TOML literals; derived assertions belong in documents.

## Execute and retain results

```sh
maelstrom run falling-block-04 --steps 12 --checkpoints 0,6,12
maelstrom run ./scenes/falling-block-04.toml --config ./config/defaults-v4.toml \
  --duration 0.1 --format json --output result.json --snapshot final.json
maelstrom replay result.json --format json --output repeated.json
```

Run requires a named or external scene and exactly one limit: `--steps` is an unsigned 64-bit count;
`--duration` is a finite non-negative number of simulated seconds, rounded up to whole effective
timesteps. It is not a process timeout. Zero steps observes initial state. Checkpoints must be
unique, ascending and within the resolved step count; none are selected by default. Checkpoint zero
observes the initial state. Other checkpoints copy accepted state and rebuild a fresh density audit
over every particle; they do not reuse solver neighbour lists.

Output paths must name new files in existing directories. Existing files and symlinks are refused,
including destinations that appear after preflight. Publication writes and syncs a temporary file
beside the destination, then uses a non-replacing hard link for atomic visibility. The filesystem
must support hard links. This does not guarantee directory-entry durability after a crash. Result
and snapshot files are published separately; a failure can leave one complete file already present.
Choose new destinations when retrying and inspect retained failure history.

Ctrl-C requests cooperative cancellation. The solver checks between stages, keeps the last complete
accepted step and discards partial work. A long discovery or Jacobi stage can delay cancellation.
The CLI attempts to publish a cancelled result and final accepted snapshot to the requested new paths.

`replay` validates the recorded schema, inputs, comparison identities and backend, then executes the
original request from its initial state. It requires neither original TOML files nor their paths.
It reruns the requested step limit even if the original was interrupted; it does not resume a final
snapshot or certify the saved outcome. Replay is unmeasured. Compare outputs using the declared
numerical policy rather than elapsed time or variable build metadata.

## Output contract

Text is the default presentation. `--format json` selects versioned machine-readable documents;
`--output FILE` directs run/replay results to a file instead of stdout. `--snapshot FILE` separately
publishes the final accepted particle state as JSON. Requested data goes to stdout; diagnostics,
warnings and errors go to stderr. `--quiet` suppresses optional diagnostics, retaining data and
errors; `--verbose` adds input/execution context. The two flags conflict. Neither changes numerical
inputs, accepted state or recorded warnings. There are no prompts or terminal-dependent progress bars.
Use versioned JSON for automation; human-readable summary wording is not a byte-level compatibility
guarantee. The implemented schema readers and backend checks determine replay compatibility.

Result schema 1 contains these responsibilities:

| Field | Meaning |
| --- | --- |
| `schema_version`, `application_version`, `backend` | Document, package and numerical implementation identity |
| `metadata` | Run/input/configuration hashes and build context, or null when input validation failed |
| `input` | Complete resolved configuration, provenance, initial particles and ordered planes, or null before admission |
| `request` | Requested steps and checkpoints |
| `completed_steps` | Number of fully accepted timesteps, even when the whole request failed |
| `status` | Whole-request disposition, including observations and attempted publication |
| `steps` | One post-acceleration maximum speed, CFL limit and warning flag per accepted step |
| `checkpoints` | Accepted snapshots with independently rebuilt mean/maximum absolute relative density errors |
| `final_snapshot` | Last accepted state, including initial state when no step was accepted; unavailable for invalid input |
| `failures` | Ordered failure classifications, accepted-step counts and context |
| `measurements` | Optional separate versioned timing object; absent on ordinary runs |

Particle identities, requested/accepted step counts and checkpoint step numbers use canonical decimal **strings** in run
and snapshot records to preserve all `u64` values. Finite physical values are JSON numbers that
round-trip binary32. Vectors in JSON use `{ "x": ..., "y": ..., "z": ... }`, whereas TOML uses
three-element arrays. A zero-speed CFL limit is `{ "kind": "unbounded" }`; a finite limit also has
`"seconds"`. Empty successful density audits have zero samples and null metrics. An unavailable or
failed audit is distinct from a zero density error. A standalone snapshot has schema version,
completed steps and ordered particles; it is an observation, not a replayable full input document.

The run hash identifies the backend, application, physical input and requested step count. It is
repeatable rather than invocation-unique. Checkpoint selection and presentation do not change it;
always retain the full request alongside the hash. Build revision is null unless supplied through
`MAELSTROM_BUILD_REVISION` when compiling. A recorded revision is provenance, not proof of a clean
tree or a successful verification run.

## Completion and failure

| Exit | Meaning |
| --- | --- |
| `0` | Discovery, inspection, validation or completed simulation; completion may retain CFL warnings |
| `1` | Process setup failure, such as inability to install the Ctrl-C handler |
| `2` | Invalid command usage, limit or checkpoint request |
| `3` | Invalid input, configuration, scene or replay record |
| `4` | Simulation or correctness-audit failure |
| `5` | Encoding/publication or output-channel failure |
| `130` | Cooperative cancellation |

Run statuses are `completed`, `completed_with_warnings`, `input_validation_failure`,
`simulation_failure`, `audit_failure`, `output_failure` and `cancelled`. Publication failure can
supersede an earlier status while retaining earlier failures. Errors before execution may produce
only a stderr diagnostic. Check process exit status and the complete record; accepted-step count
alone does not prove success. CFL warnings preserve the fixed timestep and successful acceptance,
but disqualify a warning-free baseline sample. Non-finite numerical work fails atomically.

## Frozen comparison and measurements

From the checkout, the canonical correctness request is:

```sh
maelstrom run ./scenes/falling-block-04.toml --refine-timestep 4 \
  --steps 480 --checkpoints 0,1,120,240,480 --format json --output reference-run.json
```

`reference/manifest.json` fixes input and golden hashes for this family and an additional 27-particle
unforced lattice. The comparison family uses eight iterations, timestep `0.0020833334 s` and complete
correction scale `0.0625`. The unrefined defaults are a different request and can report CFL warnings.
The reference tests compare all numerical bits, including signed zero and intermediate checkpoints,
while excluding variable build context. They also perform independent equation and health checks.

`--measure` adds monotonic elapsed seconds around each complete transactional solver step, retaining
accepted `total_s` samples and a separate `failed_attempt_s`. `--measure --stage-profile` additionally
records validation/acceleration, prediction, neighbour search, Jacobi, projection and acceptance
intervals. Profiling adds clocks and must be reported separately from primary total-only timings.
Iteration intervals accumulate across the configured work. Zero-step runs have no timing samples.

Total timing includes mandatory validation, CFL diagnostics, allocation and cancellation polls.
Loading, solver construction, fresh audits, snapshots, workflow bookkeeping, formatting and output
remain outside it. A single invocation is a raw sample. The EPQ `serial-baseline-v1` protocol owns
controlled warm-up, three samples, a separate profile, timeouts and aggregation. Important comparisons
run on the primary PC using the same frozen inputs and release settings; local test timings are
not a replacement baseline. No process-memory claim follows from these measurements.

## Verify the reference

From this checkout, use `cargo test --all-targets --all-features --locked --offline` and repeat with
`--release`; run doctests separately with `cargo test --doc --all-features --locked --offline` in
both profiles. `cargo test --test reference_verification --locked --offline` focuses on hashes,
goldens, independent equations, cancellation and measurement invariance. Keep numerical assertions
outside timing comparisons. The EPQ milestone record retains the baseline, numerical limitations
and the exact obligations inherited by replacement neighbour search.
