# Maelstrom

A 3D particle-fluid simulator with CPU and optional CUDA execution, live viewing,
and compiled image sequences you can play without computing the simulation again.

## Install

From this source checkout, install an optimised executable with viewing and compilation:

```sh
cargo install --path . --locked --features viewer
maelstrom doctor
```

For NVIDIA simulation, install with `--features viewer,cuda` instead. CUDA needs a
compatible toolkit at build time and driver/device at runtime. A CPU-only installation
uses no CUDA toolkit. See [installation and troubleshooting](docs/usage.md).

## Start here

```sh
maelstrom scenes list
maelstrom scenes create water --from builtin:falling-block-04
maelstrom view water --refine-timestep 4 --steps 480 --paused
```

Edit the new scene's `scene.toml`, `config.toml` and `render.toml` under the location printed by
`maelstrom paths`. Installed examples remain untouched. [Scenes and inputs](docs/inputs.md)
explains the geometry and numerical settings.

For two interacting blocks inside a box, create a copy with
`maelstrom scenes create collision --from builtin:two-blocks`. Its named initial groups share
one fluid model; the scene file supplies box bounds instead of hand-written planes.

## Compute once, watch later

```sh
maelstrom compile water --refine-timestep 4 --steps 480 --frame-stride 4
maelstrom runs list
maelstrom play <run-id>
```

`compile` runs the solver and renders images. `play` only reads those images: no
simulation, particle rendering or CUDA is needed. Copy the complete run directory
to another machine to watch it there. See [compiled runs and playback](docs/export.md).

For headless execution, use `maelstrom run water --steps 120`. It targets 1x simulated
time; `--unpaced` removes waiting. A slow solver reports lag rather than skipping work.
`view` starts a fresh live simulation; `rerun` recomputes recorded inputs from zero.
Neither is saved playback.

## Choose the next guide

- [Commands, execution and measurement](docs/usage.md)
- [Scene library, geometry and configuration](docs/inputs.md)
- [Live viewer and controls](docs/viewer.md)
- [Compiled output, playback and saved-observation export](docs/export.md)
- [CUDA installation and numerical limits](docs/cuda.md)
- [Developer application API](docs/application.md)
- [Comparison fixtures and numerical verification](reference/README.md)

The executable's help lists commands and options without examples. Examples live in
these guides. Backend improvements do not establish physical accuracy or a particular
real-time particle capacity; measurements need a declared workload and environment.
