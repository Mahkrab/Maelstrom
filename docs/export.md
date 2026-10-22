# Compile once and play later

A compiled run is a directory containing rendered PNG frames and a versioned manifest.
Its images are ready to watch; playback does not reconstruct particles, run a solver or
initialise CUDA. Install `--features viewer` for both compilation and playback, or
`--features render` for windowless compilation alone.

## Create a compiled run

```sh
maelstrom compile builtin:falling-block-04 --backend serial-grid --refine-timestep 4 \
  --steps 480 --frame-stride 4 --name water
maelstrom runs list
maelstrom play <run-id>
```

Compilation runs a fresh simulation, samples accepted steps and renders them without a
window. Vulkan and a compatible graphics driver are still required. `--software`
intentionally selects software Vulkan; it is not a solver backend.

Without `--output`, a new ID is created in the run library printed by `maelstrom paths`.
Set `MAELSTROM_RUNS_DIR` or `--runs-dir` to relocate that library. An explicit
`--output ./frames` uses that new directory instead; its parent must exist.
`--name` is a display label, not a replacement policy or an ambiguous lookup key.
Commands select exact IDs or explicit paths. Existing output is never overwritten.

## Watch or transfer it

```sh
maelstrom play <run-id> --paused
maelstrom play ./copied-run --speed 0.25 --repeat
maelstrom runs show <run-id>
maelstrom runs validate <run-id>
```

Copy the **whole directory**, not just its manifest. Input values and frame references
are portable; the original scene library and original simulation device are not needed.
`runs validate` checks the frame schedule and decodes/hashes every PNG without a window.
Playback validates each frame as it loads it and never writes to the bundle.

Playback uses recorded simulated timestamps. Space pauses/resumes, Left/Right seek and
pause, Home/End select endpoints, R restarts, +/- change speed and Escape closes.
Click the window before using keys. Speed is finite, 0.01–100. A loop holds the last
image for the previous interval before restarting. Slow presentation can skip displayed
images to catch up; every stored frame remains seekable. There is no interpolation.

The camera is baked into the images. Resizing preserves aspect ratio rather than
redrawing particles. Choose a different camera by creating another image sequence.
Missing or changed images are errors, never a reason to recompute the simulation.

## Render an existing numerical result

```sh
maelstrom export saved ./result.json --frame-stride 240 --output ./saved-frames
```

This renders only stored observations. The stride must select available checkpoints;
the usual 0/1/120/240/480 history supports stride 240, not stride 120, because 360 is
absent. Missing observations fail before publication. Partial results require an
explicit `--last-step` within the accepted prefix.

`rerun <result-or-run>` deliberately recomputes recorded input from zero. It is not
resumption, rendering saved observations or playing images. The old `export direct`
and `export rerun` spellings remain hidden compatibility routes; prefer `compile`
for fresh scenes and `export saved` when the necessary observations already exist.

## Sampling and image settings

Use `--frame-stride N` to select accepted step indices, always including initial and
final state. It must be positive; schedules are bounded to one million frames.
`--width`, `--height` and `--render-config` control presentation, not physics.
The [viewer guide](viewer.md) describes camera files and graphics selection.

Images are opaque RGBA8 PNGs, with stable ordering and names such as
`frame-00000000.png`. The manifest records resolved inputs, request and effective
execution, accepted diagnostics, simulated times, frame/observation hashes, camera,
build and graphics/encoder context. `run.json` adds the optional compilation label.
The manifest alone is sufficient for playback and numerical reruns.

Reproducibility means identical decoded pixels for the qualified binary, adapter/driver,
settings and accepted observations. Different drivers/devices need not produce identical
pixels. Encoded PNG bytes and decoded pixels are hashed separately. The player bounds
manifests to 64 MiB, PNGs to 300 MiB and dimensions to 8192 per side; the export frame-count
ceiling is not a guarantee that every possible manifest fits the player.

## Failures and costs

Frames publish only after they are closed and hashed. A completed `manifest.json`
appears only after all frames succeed. A cancelled or failed compilation retains
`progress.json` and any published prefix, explicitly incomplete; it is not normal
playback input. Never infer completion from a directory or a few images alone.

Compilation uses a lossless synchronous frame boundary, unlike the live viewer's
coalesced observations. Numerical outcome and image publication are distinct.
Solver work, graphics/readback and PNG encoding have separate manifest timings;
total wall time includes additional work and is not an interchangeable solver benchmark.
