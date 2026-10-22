# View a live simulation

`view` starts a **fresh simulation**. To watch something already compiled, use
[playback](export.md) instead. Install with `--features viewer`; a Vulkan driver,
display and supported native keyboard libraries are required.

```sh
maelstrom view builtin:falling-block-04 --backend serial-grid --steps 120 --paused
maelstrom view water --backend threaded-grid --threads 4 --steps 120
maelstrom view water --backend cuda --device 0 --steps 120
```

Use `--paused` to inspect step zero before starting. Without it, the new run starts
advancing immediately. The simulation's fixed timestep is independent of redraw rate.
A renderer can skip intermediate observations but cannot skip solver work or change state.

## Scene and camera selection

Names resolve through the same [scene library](inputs.md) as `run` and `compile`.
Execution flags and numerical override precedence are also shared. `--graphics-adapter N`
selects a hardware Vulkan adapter; `--software` selects the software class instead.
These indices do not select the CUDA simulation device. The reported adapter identity
is authoritative because enumeration can differ between machines.

Use `--render-config ./camera.toml` for presentation settings. This minimal file
selects a fixed eye and target; omitted settings retain presentation defaults:

```toml
schema_version = 1
width = 1280
height = 720
step_stride = 1

[camera]
eye_m = [1.5, 1.2, 1.5]
target_m = [0.3, 0.4, 0.3]
up = [0.0, 1.0, 0.0]
vertical_fov_deg = 45.0
near_m = 0.001
far_m = 100.0
```

Eye, target and clipping distances are in metres; the field of view is in degrees.
The up direction must not be parallel to the view. Unknown fields and invalid/non-finite
cameras fail without changing accepted simulation state.
`--fit-camera` fits the initial particles once; later movement can leave that view,
so choose a wider fixed camera to cover a complete trajectory. Empty scenes use a
one-metre fallback box. Camera calculations use f64 before view-relative graphics f32;
extreme ranges can still lose precision or clip.

## Controls

Click the window to give it keyboard focus. The window does not take existing focus
away from another application.

| Control | Action |
| --- | --- |
| Space / N | Pause or resume / attempt one timestep while paused |
| R / C | Reset original inputs / cancel and retain accepted state |
| O | Inspect the labelled prior outcome; controls still address the current run |
| D | Request an independent density audit of the displayed accepted snapshot |
| Tab, Enter, Delete, Escape | Choose, edit, clear or abandon a draft field |
| A | Validate/apply the complete draft as a new run |
| 1–7 / B | Choose a bundled size / cycle explicit simulation backends in the draft |
| Left drag or arrows | Orbit |
| Right drag | Pan |
| Wheel or +/- | Zoom |
| F | Fit initial particles explicitly |
| H / G | Toggle diagnostics / recreate the surface on the same graphics device |
| Escape outside editing, or close | Request cooperative shutdown |

The draft and selected request are separate. Invalid draft validation preserves the
selected run. Construction failure creates a failed generation without borrowing old
particles; O can inspect retained prior state. Relative edited paths resolve through
the invocation/scene path rules. Launch numerical overlays remain active during edits.

The panel reports status, effective resources, accepted/requested steps, simulated
time, particle count, CFL warnings and failures. Density auditing uses a separate owned
snapshot; it is neither solver timing nor a redraw side effect. Display FPS is only
presentation cadence, not solver throughput.

## Limits and recovery

Particles are opaque spheres, with static boundaries shown as lines; there is no
photorealistic surface reconstruction or interpolation. Graphics consumes accepted
state only. A display/surface failure must not become a numerical acceptance decision.
Shutdown can wait for synchronous numerical/driver work; no universal deadline is promised.

The supported verification environment is x86_64 Linux/WSL2, WSLg and software Vulkan.
Hardware Vulkan and other platforms require their own qualification. Linux commonly
needs `libxkbcommon0`; X11 additionally needs `libxkbcommon-x11-0` and `libxcb-xkb1`.
Run `doctor graphics` for an adapter check; that alone does not test native window events.
