# Maelstrom

A deterministic, headless Position-Based Fluids simulator. Run `cargo run --release -- --help`
for scene discovery, validation, inspection, execution and replay.

Neighbour discovery is selectable: `run --backend brute-force` is the default reference;
`run --backend uniform-grid` uses sparse occupied cells with the same serial numerical work
and particle reduction order. Add `--execution threaded --threads N` to distribute Jacobi
particle work across a private reusable pool. Threaded execution requires the grid; serial remains
the default. Saved results identify discovery and worker settings, and `replay` restores them.
Both remain available for comparison; dense support can still require quadratic work and storage.

The [usage guide](docs/usage.md) covers building, CLI journeys, replay, JSON records, output
publication and failure codes. The [input specification](docs/inputs.md) documents every supported
configuration field, scene construction, validation and precedence rule. Frozen correctness inputs
and golden results are indexed by [the reference manifest](reference/manifest.json).

The optional [CUDA solver](docs/cuda.md) builds with `--features cuda`. Select it with
`run --backend uniform-grid --execution cuda --device 0 --block-size 128`.
It executes discovery and numerical stages on a private device stream, retaining atomic accepted
snapshots and public replay. `cuda-check` remains an independent resource probe. Ordinary CPU
builds need no CUDA toolkit.

The [application controller](docs/application.md) lets a non-visual caller pause,
single-step, restart and cancel a worker-owned solver through bounded commands
and owned accepted observations. It is available in ordinary CPU-only builds.

The optional [interactive viewer](docs/viewer.md) builds with `--features viewer`.
Launch `view falling-block-04 --steps 480 --software` for an explicitly selected
software Vulkan renderer, or omit `--software` to require hardware graphics.
Its camera, controls and diagnostics consume accepted observations independently
of numerical execution. The default build remains headless.

The optional [offline exporter](docs/export.md) builds with `--features render`.
Use `export saved` to draw stored observations, `export rerun` to rerun recorded
inputs, or `export direct` to start a scene. Each sequence has explicit accepted
step sampling, safe publication and a final provenance manifest. No window is required.
With `--features viewer`, use `maelstrom play <export-directory>` to watch those
saved images without rerunning physics or drawing the particles again. Playback
supports pause, frame seeking and speed changes; its camera is the exported one.

Elapsed solver measurements are opt-in:

```sh
cargo run --release --locked --offline -- run falling-block-04 --refine-timestep 4 \
  --steps 480 --checkpoints 0,1,120,240,480 --measure --format json
```

`--measure` adds a versioned `measurements` object to the ordinary replayable result. It retains one
`total_s` value per accepted step and a separate `failed_attempt_s` for an unsuccessful attempt.
An unmeasured result omits the timing object. Replay uses the numerical inputs and request and
runs without measurements. Zero-step runs produce an empty timing list.

`--measure --stage-profile` additionally records validation/acceleration, prediction, neighbour search,
Jacobi, projection and acceptance intervals in seconds. Iteration intervals accumulate across the
configured solver work. Profiling adds internal clock overhead and should be reported separately
from primary total-only samples. Totals include the complete transactional timestep and mandatory
CFL diagnostic; loading, solver construction, snapshots, fresh density audits, formatting and output
are outside them. Measurements never change the timestep or numerical acceptance rules.

Use optimised builds and record inputs, compiler, hardware, warm-up, repetition and variability when
reporting performance. A single measured invocation is a raw sample, not a controlled comparison.
