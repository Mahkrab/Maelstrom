# Maelstrom

A deterministic, headless Position-Based Fluids simulator. Run `cargo run --release -- --help`
for scene discovery, validation, inspection, execution and replay.

Neighbour discovery is selectable: `run --backend brute-force` is the default reference;
`run --backend uniform-grid` uses sparse occupied cells with the same serial numerical work
and particle reduction order. Saved results identify the selection, and `replay` restores it.
Both remain available for comparison; dense support can still require quadratic work and storage.

The [usage guide](docs/usage.md) covers building, CLI journeys, replay, JSON records, output
publication and failure codes. The [input specification](docs/inputs.md) documents every supported
configuration field, scene construction, validation and precedence rule. Frozen correctness inputs
and golden results are indexed by [the reference manifest](reference/manifest.json).

Elapsed solver measurements are opt-in:

```sh
cargo run --release --locked --offline -- run falling-block-04 --refine-timestep 4 \
  --steps 480 --checkpoints 0,1,120,240,480 --measure --format json
```

`--measure` adds a versioned `measurements` object to the ordinary replayable result. It retains one
`total_s` value per accepted step and a separate `failed_attempt_s` for an unsuccessful attempt.
An unmeasured result has no additional fields. Replay uses the numerical inputs and request and
runs without measurements. Zero-step runs produce an empty timing list.

`--measure --stage-profile` additionally records validation/acceleration, prediction, neighbour search,
Jacobi, projection and acceptance intervals in seconds. Iteration intervals accumulate across the
configured solver work. Profiling adds internal clock overhead and should be reported separately
from primary total-only samples. Totals include the complete transactional timestep and mandatory
CFL diagnostic; loading, solver construction, snapshots, fresh density audits, formatting and output
are outside them. Measurements never change the timestep or numerical acceptance rules.

Use optimised builds and record inputs, compiler, hardware, warm-up, repetition and variability when
reporting performance. A single measured invocation is a raw sample, not a controlled comparison.
