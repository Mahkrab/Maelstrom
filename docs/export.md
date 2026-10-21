# Exporting accepted observations

Build with `--features render` for PNG sequences without a window or display
server. Vulkan, its loader and a compatible graphics driver are still required.
Default builds retain the headless solver and have no graphics dependencies.
`viewer` includes `render`; `cuda` remains an independent simulation feature.
Run the commands below from the Maelstrom directory. Each output directory must
be new, with an existing parent.

The examples use the installed `maelstrom` command. To install export and playback
from this checkout, run `cargo install --locked --path . --features viewer`.
Add `cuda` to the feature list only if new CUDA simulation is needed. A headless
installation does not expose the optional export/play commands.

## Watching a saved sequence

This command **loads already rendered images**. It does not load a numerical scene,
run a solver, initialise CUDA or render particles again:

```sh
maelstrom play frames-direct
```

The argument is a completed export directory or its `manifest.json`, not a raw
simulation result. The `viewer` feature and a window-capable Vulkan device are
required. Hardware graphics is the default; select `--software` explicitly for
software Vulkan. Copy the whole directory to another machine to watch it there;
the original simulation backend and CUDA device are not needed.

Playback follows the recorded simulated timestamps at 1x, independently of how
long the original computation took. `--speed 0.25` plays at quarter speed;
`--paused` opens on the initial image, and `--repeat` loops. Space pauses/resumes,
Left/Right seek one recorded frame and pause, Home/End select the endpoints, R
restarts, +/- change speed, and Escape closes. Speed must be finite and within
0.01–100. The title shows the selected frame, accepted step, simulated time and
speed. Click the window before using keyboard controls.

The final image stays visible. A loop holds it for the preceding recorded interval
before returning to the start; a one-frame sequence stays static. Resizing preserves
aspect ratio with black letterboxing. The camera is baked into these images, so
it cannot be orbited during playback. Changing it requires a new image export;
available saved observations can be rendered again without a numerical rerun.

Only completed schema-1 manifests are accepted. The player checks order, schedule,
timestamps and safe frame names, then verifies each image's recorded PNG and pixel
hashes when loading it. Missing, changed or malformed images stop playback; nothing
is regenerated or overwritten. The manifest is limited to 64 MiB, encoded frames to
300 MiB and decoded dimensions to 8192 per side. The exporter's one-million-frame
ceiling does not guarantee that a manifest fits the player's separate size limit.
Images are decoded on demand and one texture is reused, rather than loading the
whole history. Slow decoding or
presentation may skip displayed images to catch up; seeking still accesses every
recorded image. There is no interpolation or fixed-FPS conversion.

A raw result containing only sparse checkpoints is not a full animation. Export
those stored states explicitly, or choose a one-time numerical rerun to capture
the missing states. After that export, `play` never needs the computation again.

## Creating a sequence once

Choose the numerical source explicitly. This command **loads saved observations**
and does no new simulation work:

```sh
maelstrom export saved reference/small-lattice-v1.json \
  --step-stride 12 --output frames-saved --software --fit-camera
```

This command **starts a fresh simulation from the recorded initial inputs**,
retaining the full recorded request and execution resources:

```sh
maelstrom export rerun reference/falling-block-v1.json \
  --step-stride 120 --output frames-rerun --software --fit-camera
```

Every available source checkpoint and final snapshot is compared with the newly
accepted state. Serial comparisons preserve every binary32 bit, including signed
zero. Threaded/CUDA comparisons use the recorded comparison policy's absolute
component budgets: position `1e-5 m`, velocity `0.005 m/s`, with exact identities
and step counts. A mismatch fails the export and retains expected/actual snapshots
in `comparison-failure.json`. Rerunning does not resume or certify the original
outcome; original failures and disposition remain in provenance.

This command **starts a fresh simulation from a scene**:

```sh
maelstrom export direct falling-block-04 \
  --steps 5 --mode threaded-grid --workers 2 --step-stride 2 \
  --output frames-direct --software --fit-camera
```

Direct execution accepts the ordinary `--config`, repeatable `--set` and
`--refine-timestep` input selections. Modes are `brute-force`, `serial-grid`
(default), `threaded-grid` and `cuda-grid`. CPU workers require threaded grid;
`--device` and `--block-size` require CUDA grid. CUDA requires `--features
cuda,render`, a toolkit when building and a supported runtime device. There is
no automatic CPU fallback. Saved CUDA observations can be rendered with a CPU-only
build because their original simulation device is never constructed.

Graphics defaults to hardware Vulkan adapter zero. `--software` explicitly
requires the software class; `--graphics-adapter N` selects within that class.
The manifest identifies the actual adapter, driver and API separately from
simulation execution. Local qualification uses llvmpipe software Vulkan on
x86_64 WSL2 Linux. Hardware Vulkan export, native Windows, macOS, browsers and
other graphics APIs have not been qualified.

## Schedule and presentation

For `S` accepted steps and positive stride `K`, frames sample `0,K,2K,... <= S`,
then `S` if it was absent. `S=5,K=2` produces steps `[0,2,4,5]`; zero steps
produces one initial frame. Frame indices are contiguous from zero and names
are `frame-00000000.png`, `frame-00000001.png`, and so on. The checked limit is
one million frames. Simulated timestamps are `step * fixed_timestep_s`; a
sequence has no implicit playback FPS; `play` uses these timestamps. There is no interpolation, wall-clock
sampling or duplication of observations to fill missing steps.

Saved mode requires every scheduled state to exist as an explicitly stored
checkpoint or the final snapshot. Initial inputs alone cannot supply frame zero.
Missing indices fail before output creation. Incomplete source results require
`--last-step S` explicitly; that prefix must have all scheduled observations.
The manifest retains the original request, completed count, status and failures.
A completed image sequence of that prefix does not claim original run completion.

Select strict TOML render schema 1 with `--render-config FILE`. Built-in defaults
are overridden by that file, then explicit `--width`, `--height` and
`--step-stride` options. Camera fields merge individually with defaults. There
is no implicit configuration lookup. For example:

```toml
schema_version = 1
width = 640
height = 360
step_stride = 120
[camera]
eye_m = [1.2, 1.0, 1.2]
target_m = [0.3, 0.4, 0.3]
vertical_fov_deg = 45.0
```

Dimensions must be 1–8192 and fit actual graphics limits. Only opaque RGBA8 sRGB
PNG is supported. Camera, sphere radius, lighting and clipped boundary depiction
are shared with the [viewer](viewer.md); exported images have no diagnostic
overlay. `--fit-camera` resolves the initial geometry once and records the result.
No interactive camera history is consulted. Presentation settings do not change
physical input identity.

## Publication, reconstruction and failures

The destination is created exclusively. Existing files, directories and symlinks
are rejected. Each image is encoded completely, synced, then published by a
hard link without replacement. The filesystem must support hard links. At most
one frame is being drawn/read back/encoded at a time, and stepping waits until
publication finishes. Slow output cannot discard a scheduled state or alter the
solver's timestep. Per-frame records, accepted diagnostics and selected input
records remain in memory; particle history is not accumulated internally.

`progress.json` always says `complete=false`. It records published frames and
any caught failure. Cancellation (Ctrl+C), solver, graphics, encoding and disk
errors leave those files in place. An additional failure to record progress is
reported alongside the first error. Process termination may leave stale progress
or temporary files. These files never prove completion. Choose a new directory
to retry; automatic resume and cleanup are not provided.

`manifest.json` is published last without replacement, only after every expected
frame is closed and hashed. It alone establishes sequence completion. It includes
full resolved physical inputs and identities, effective execution and source
outcomes, frame/step/time mappings, PNG and decoded-pixel hashes, observation
hashes, camera/display box, renderer/shader identity, compiled source/lock hashes,
commit/dirty state at build time, compiler/target/features and graphics/encoder
context. Missing build information is explicit. Destination labels contain only
the final directory name. Source content hashes describe the compiled source;
Git metadata is supplementary and may be unavailable or stale between builds.

The manifest's input/backend/execution/request fields are accepted by the ordinary
replay input loader. To **start a new numerical rerun** that reconstructs its
sampled observations, use a build supporting its recorded execution mode:

```sh
maelstrom replay frames-direct/manifest.json --format json \
  --output reconstructed-result.json
```

Compare each reconstructed checkpoint's portable snapshot hash with the frame's
`snapshot_sha256`. Saved image rendering still uses result records rather than
image manifests, so this new result can subsequently be passed to `export saved`.
Original source outcomes remain separate from the manifest's rendered-prefix
request. Solver step, graphics/readback and encoding durations have distinct
fields; they are local diagnostics, not a benchmark or an interactive FPS claim.

Repeated exports with the same binary, observations, resolved settings and
qualified adapter/driver produce identical decoded RGBA pixels. PNG byte hashes
are retained too; encoding bytes may differ across encoder versions. No
cross-device or cross-driver pixel equality is promised. A useful picture does
not establish fluid-model accuracy.
