# Commands and execution

Use an installed `maelstrom` executable for ordinary work. `maelstrom --help` lists
commands; `maelstrom <command> --help` gives arguments, defaults and options without
examples. `maelstrom doctor` reports installed features without starting a simulation.

## Installation and capabilities

From the Maelstrom checkout:

```sh
cargo install --path . --locked --features viewer
maelstrom doctor
maelstrom doctor graphics
```

This installs an optimised executable with live viewing, compilation and saved playback.
Use `--features viewer,cuda` to include NVIDIA simulation. For a headless CPU executable,
omit `--features`; `--features render` adds windowless compilation without a window library.
Offline installation additionally requires cached dependencies.

Viewing/compilation use Vulkan or OpenGL ES. WSLg or an X11/Wayland display is required for windows,
not for windowless compilation. Use `doctor graphics --software` and the corresponding
`--software` launch option when intentionally selecting software Vulkan. CUDA simulation
and graphics adapters are independent. Hardware Vulkan is preferred; accelerated OpenGL
is also supported, including WSLg's D3D12 driver. See [CUDA setup](cuda.md) and
[graphics troubleshooting](viewer.md#graphics-on-wslg).

If commands are missing from an older installation, reinstall this checkout. A Cargo
development build and the installed command are different executables; changing build
features can also replace the executable under `target/release`. Development-only
commands are `cargo build --release --locked` and `cargo test --locked`.

## Pick the right operation

| Operation | Command | Computes physics? |
| --- | --- | --- |
| Execute headlessly | `run <scene>` | Yes, fresh input |
| Execute with a live window | `view <scene>` | Yes, fresh input |
| Produce saved playback output | `compile <scene>` | Yes, plus image rendering |
| Watch compiled output | `play <run-id-or-path>` | No |
| Recompute recorded input | `rerun <result-or-run>` | Yes, from step zero |
| Render stored numerical observations | `export saved <result.json>` | No |

`validate` checks inputs without stepping; `inspect` shows effective numerical, run and presentation
inputs, provenance and selected execution. Schema-2 scenes can save a `[run]` limit, allowing
`run water`, `view water` and `compile water` without repeating duration or step flags. Explicit
limits replace the saved selection. A numerical JSON result is not a playable image sequence.

## Execution selection

All fresh execution commands accept the same `--backend` and resource options:

| Backend | Resource policy |
| --- | --- |
| `auto` (default) | Honour explicit CPU/CUDA resources; otherwise prefer an available CUDA device, then serial grid |
| `brute-force` | Serial all-pairs discovery, retained for comparison/demonstration |
| `serial-grid` | Grid discovery and calling-thread numerical work |
| `threaded-grid` | Grid with a private reusable pool; `--threads N` selects its exact size |
| `cuda` | Explicit CUDA device; `--device N` and `--block-size N` select resources |

`auto --threads N` selects threaded grid; `auto --device N` or `--block-size N`
requires CUDA. These resource families cannot be combined. Without an explicit thread
count, threaded execution uses available process parallelism, capped at 256. CUDA blocks
support 32/64/128/256 lanes, default 128. The default device ordinal is zero.

Auto is a resource preference, not a universal fastest-backend guarantee. Small scenes can
run faster on serial grid. Once CUDA is selected, runtime failure never triggers a CPU retry.
Results record concrete execution settings; `rerun` restores them instead of selecting again.
For comparisons, always name the backend and resources explicitly.

```sh
maelstrom run builtin:falling-block-04 --backend serial-grid --steps 12 --unpaced
maelstrom run builtin:falling-block-04 --backend threaded-grid --threads 3 --steps 12 --unpaced
maelstrom inspect builtin:falling-block-04 --backend cuda --format json
```

## Numerical modes

`run`, `view`, `compile`, `inspect` and `validate` accept
`--numerical-mode reference`, `fast`, `experimental`, `paper-baseline` or `scene`. **Local scenes use their saved numerical settings by default; built-in scenes default to reference.**
Opt into fast with `--numerical-mode fast`, or save
`numerical_mode = "fast"` in the application preferences at
`$XDG_CONFIG_HOME/maelstrom/config.toml` (normally `~/.config/maelstrom/config.toml`):

```toml
schema_version = 1
numerical_mode = "fast"
```

The command-line mode overrides that preference. Explicit `--set` values override
the preset. Use `--numerical-mode scene` (or `numerical_mode = "scene"` in preferences)
to preserve a custom scene/configuration schedule without a preset. Reruns retain their recorded numerical values. Reference, fast and experimental change only timestep,
Jacobi count, correction scale and relaxation scale; geometry, mass, gravity,
kernel support and velocity-effect settings retain their existing ownership.
Enabled velocity effects keep their coefficients; changing the step rate also
changes how often those effects are applied per simulated second.

| Mode | Steps / simulated second | Jacobi iterations / step | Correction scale | Relaxation scale |
|---|---:|---:|---:|---:|
| `reference` | 480 | 8 | 0.0625 | 20 |
| `fast` | 240 | 8 | 0.5 | 2 |
| `experimental` | 240 | 8 | 0.25 | 20 |
| `paper-baseline` | 240 | 3 | 1.0 | 20 |

Fast mode reduces numerical work and changes the physical trajectory. It is a
throughput/quality trade-off, not a promise of the same fluid response. Both modes
retain the existing PBF physical limitations. Fast mode can improve compression
and separation while adding projection energy or changing shear response.

Experimental mode retains reference regularisation and Jacobi count while halving the step rate. It matches the approximate response `iterations × correction_scale / dt²`
of reference, using four times the correction per pass. This is an empirical visual
calibration, not a convergence law: impact detail and trajectories differ, and
legacy refinement still scales timestep and correction together. Its qualification
uses disabled velocity effects; enabled effects need separate calibration.

`--numerical-mode paper-baseline` starts a separate published-equation PBF policy:
240 Hz, three Jacobi passes, full position corrections and nominal equal particle
mass `rho0 * spacing^3`. XSPH is enabled at rest-volume coefficient 0.01; vorticity
confinement is optional and disabled initially. The selected regularisation is
`20/h^2` and artificial pressure is `0.1*h^2`, with reference separation `0.2*h`
and exponent four. These dimensional choices are Maelstrom calibration, not
numerical constants fully specified by the paper. Reference remains the default.

For this formulation `--refine-timestep N` changes only timestep; it never
attenuates position corrections. `--set correction_scale` must remain 1.0.
Resolved records retain `pbf_formulation = "paper2013"`, so reruns use their
recorded equations and constants. Existing reference, fast and experimental
inputs retain their historical mass and refinement policy.

The assessed three-pass baseline remains a comparison option: large impact leaves
more persistent overlaps and fast outliers. Experimental remains the practical
candidate for the tested mixture of scenes.

These commands start new simulations for five simulated seconds. The first uses
reference mode unless an application preference selects another mode; the second
explicitly selects fast:

```sh
maelstrom run builtin:falling-block-20 --duration 5 --unpaced
maelstrom run builtin:falling-block-20 --numerical-mode fast --duration 5 --unpaced
```

Built-in scenes also save the reference schedule in their configuration files.
Their output stride is eight, giving 60 Hz at 480 steps per simulated second.
For 60 Hz output with fast or experimental mode, add `--frame-stride 4` to `compile`.

Explicit `--set` values take precedence over a preset. Presets reset inherited
timestep refinement to one; an explicit `--refine-timestep N` still divides the
selected timestep by N and, for empirical formulations, its correction scale by N². For example, fast mode
with refinement two runs at 480 Hz with correction 0.125 and relaxation 2. Inspect
the resolved values before comparing a modified preset. CFL remains a warning;
it neither inserts timesteps nor switches modes. Numerical errors and cancellation
retain the last accepted state through the existing transaction boundary.

The viewer's **Numerics** draft field accepts `fast`, `reference`, `experimental`,
`paper-baseline`, or blank for
the application default/preference. `scene` explicitly preserves the selected
scene/configuration schedule. Apply the draft to start a new generation. Editing the
field does not modify an active run. The entered step count stays fixed: five
simulated seconds require 1,200 fast steps or 2,400 reference steps. The selected
generation displays its effective rate, iterations, correction and relaxation.

Recorded runs store resolved numerical values and selection provenance. `rerun`
recomputes those recorded values from zero, even if current mode defaults change;
it does not turn a fast recording into a reference run. Playback uses saved images.

## Limits, pacing and results

Fresh execution uses the saved scene limit unless `--steps N` or `--duration SECONDS` replaces it.
Without a saved limit, supply one of those flags. Duration
means simulated time, rounded up to whole fixed timesteps, not an execution timeout.
Zero steps is valid. Headless `run` targets 1x simulation time unless `--unpaced` is
selected. It never changes timestep size, iterations or accepted state to catch up.
Compilation is unpaced; live viewing has independent redraw cadence.

```sh
maelstrom run builtin:falling-block-04 --backend serial-grid --steps 12 --unpaced \
  --checkpoints 0,6,12 --format json --output result.json --snapshot final.json
maelstrom rerun ./result.json --format json --output repeated.json
```

Checkpoints must be unique, ascending and within the request. They copy accepted state
and independently audit density; no checkpoints are selected by default. Audits can be
expensive and are outside solver timing. The final accepted snapshot is retained even
when no checkpoints are requested. Ctrl-C cooperatively retains accepted state and exits
130; forcibly killing a process cannot guarantee output.

Output files must be new and their parent directories must exist. Existing files and
symlinks are never replaced. Without `--output`, the requested text/JSON result goes to
stdout. Progress, selected resources, warnings and failures go to stderr.
`--quiet` suppresses optional diagnostics, not errors or requested data; `--verbose`
adds input context. `--color auto|always|never` controls diagnostic colour only.

Numerical results use schema 1: resolved input, request, execution identity, metadata,
accepted diagnostics, requested checkpoints, final snapshot, failures and status.
Large step counts/identities use decimal strings. Supported historical records retain
their numerical values; unsupported arithmetic identities are rejected, not reinterpreted.

| Exit | Meaning |
| --- | --- |
| 0 | Completed, possibly with explicitly recorded CFL warnings |
| 2 | Invalid command/argument combination |
| 3 | Input validation or unavailable required capability |
| 4 | Simulation or audit failure, with accepted state retained |
| 5 | Output/publication failure |
| 130 | Cooperative cancellation |

## Solver measurements

```sh
maelstrom run builtin:falling-block-04 --backend serial-grid --refine-timestep 4 \
  --steps 480 --checkpoints 0,1,120,240,480 --unpaced --measure --format json
```

`--measure` records complete accepted-step elapsed seconds and any failed-attempt time;
add `--stage-profile` for instrumented stage intervals. Use `--unpaced`; pacing waits,
loading, solver construction, checkpoints, density audits, rendering and publication
are not solver-step time. Construction/teardown are separate fields. Zero-step requests
have an empty step-timing list. Reruns are unmeasured unless a new experiment explicitly
requests measurement through fresh execution.

A single invocation is a sample, not a controlled comparison. Record optimised build,
source/binary, hardware, numerical input, warm-up, repetition, variability and timer.
Do not interpret steps per second as rendered FPS or memory-context data as peak memory.

## Compatibility

`replay` remains a hidden alias for numerical `rerun`; bare `scenes` still lists scenes,
and `cuda-check` remains a hidden compatibility command for `doctor cuda`. Old
`uniform-grid --execution serial|threaded|cuda`, `--mode`, `--workers` and `cuda-grid`
spellings remain accepted. Legacy `--execution` requests retain unpaced execution.
New examples use only the shared backend vocabulary. Historical saved-result and
image-manifest formats are unchanged.


Local scene packages can be launched directly:

```sh
maelstrom view user:water
maelstrom run user:water
maelstrom compile user:water
maelstrom play --last
```

The first three commands start fresh simulations from
`~/.local/share/maelstrom/scenes/water/scene.toml` (or the selected scene library).
An adjacent `config.toml` supplies the numerical configuration without a reference
inside the scene, and `render.toml` supplies presentation settings automatically.
An explicit `--config` takes priority over the adjacent file; without either,
the scene's configuration reference or defaults apply. Scene overlays and explicit
CLI overrides retain their precedence. An explicitly saved application numerical
mode also applies; use `--numerical-mode scene` to select saved scene numerics.

`play --last` plays saved images from the most recently completed compile,
including a compile with an explicit output directory. Failed or cancelled compiles
do not replace it. Playback does not run the simulation again.
Compile reports the scheduled frame count and nominal render Hz, including initial
and final states. A final short interval can differ from the nominal cadence.
Interactive terminals show one updating progress bar on stderr; redirected output
contains the diagnostics and final result without terminal control sequences.
