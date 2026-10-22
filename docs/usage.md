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

Viewing/compilation use Vulkan. WSLg or an X11/Wayland display is required for windows,
not for windowless compilation. Use `doctor graphics --software` and the corresponding
`--software` launch option when intentionally selecting software Vulkan. CUDA simulation
and graphics adapters are independent. See [CUDA setup](cuda.md).

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

`validate` checks inputs without stepping; `inspect` shows effective numerical inputs,
provenance and selected execution. A numerical JSON result is not a playable image sequence.

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

## Limits, pacing and results

Fresh execution requires exactly one of `--steps N` or `--duration SECONDS`. Duration
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
