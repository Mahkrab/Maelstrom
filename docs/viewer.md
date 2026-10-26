# View a live simulation

`view` starts a **fresh simulation**. To watch something already compiled, use
[playback](export.md) instead. Install with `--features viewer`; a Vulkan or OpenGL ES driver,
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
selects a hardware graphics adapter; `--software` selects the software class instead.
Hardware Vulkan is preferred, followed by hardware OpenGL ES; software is never
selected implicitly. Windowed selection includes surface compatibility.
These indices do not select the CUDA simulation device. The reported adapter identity
is authoritative because enumeration can differ between machines.

The viewer automatically loads `render.toml` beside the selected scene. Use
`--render-config ./camera.toml` to replace that file for one launch. This minimal legacy file
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
Without a selected fixed camera, the viewer fits once to the finite plane-bounded domain,
or to the initial particles if the domain is unbounded. An empty unbounded scene uses the
built-in fixed camera. `--fit-camera` overrides a saved fixed camera; `--no-fit-camera` uses
that camera or the built-in fallback. The flags conflict. F explicitly refits the current
accepted particles or the finite domain using the window's current aspect ratio. Resizing
changes the projection, without moving your camera. Camera calculations use f64 before
view-relative graphics f32; extreme ranges can still lose precision or clip.

New scene packages use render schema 2. `camera_mode = "fit"` is its default; `"fixed"` requires
a `[camera]` table, and supplying that table without a mode also selects fixed behaviour. Schema 1
continues to use its fixed-camera defaults.

Press **S** outside field editing to save the current camera to the displayed editable scene's
`render.toml`. This writes fixed mode and all camera fields while retaining unrelated settings
and comments. It upgrades a legacy render document to schema 2. Publication is atomic; invalid,
read-only or symbolic-link targets and detected concurrent edits are rejected. The panel reports
success or failure. The viewer and solver continue while a separate worker saves the camera.

`--camera-output ./chosen-render.toml` chooses an explicit save destination; its parent must exist.
`--render-config` only selects input and does not redirect saving. Bundled scenes require an editable
copy through `scenes create` before saving. Launching, orbiting, fitting, resizing and temporary CLI
overrides never write files. Home restores the camera loaded or fitted for the displayed generation,
or the last camera successfully saved for it. R resets the simulation while retaining the current view.

## Particle colours

The same `render.toml` controls live viewing and compilation. For a fixed sRGB colour:

```toml
schema_version = 2
[particles]
colour_mode = "fixed"
colour = "#00aaff"
```

To colour accepted motion from blue to red:

```toml
schema_version = 2
[particles]
colour_mode = "kinetic_energy"
min_energy_j = 0.0
max_energy_j = 0.001
```

Energy is per-particle kinetic energy, `½m|v|²`, in joules, using the resolved physical particle
mass and accepted velocity. The saved interval is fixed across frames: values at or below its
minimum are blue, values at or above its maximum red. Intermediate values interpolate in linear
RGB before sphere lighting and sRGB encoding. Halfway is equal linear red and blue, rather than
a temperature reading. Choose bounds for your scene's mass and speeds; the interval is not
rescaled automatically. It requires finite `0 <= min_energy_j < max_energy_j`.

The default remains fixed `#2b9fd9`. These settings never change accepted state. Headless `run`
does not read them. Colours and camera are baked into compiled images; `play` cannot change them
without new rendering from particle observations. Reopen a scene, or apply its draft again, to read
external edits to its presentation file.

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
| F / Home | Refit domain or current particles / restore loaded or last saved camera |
| S | Save the current camera for the displayed editable scene |
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

The checked graphics paths are software Vulkan and WSLg OpenGL/D3D12 on the RTX 4070
and Intel integrated graphics. The PC's live-view and saved-playback windows have also
been checked through X11, including rendered pixels and clean window closure.
Hardware Vulkan and other platforms require their own qualification. Linux commonly
needs `libxkbcommon0`; X11 additionally needs `libxkbcommon-x11-0` and `libxcb-xkb1`.
Run `doctor graphics` for an adapter check; that alone does not test native window events.

## Graphics on WSLg

CUDA availability does not establish graphics availability. WSLg can expose accelerated
OpenGL through Mesa's D3D12 driver while Vulkan exposes only CPU rendering. Maelstrom
supports that OpenGL path; `doctor graphics --format json` reports the actual API,
adapter and device type. `--software` deliberately requests CPU graphics.

In a WSLg session where the default Mesa selection is software, check the installed
D3D12 driver explicitly:

```sh
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA maelstrom doctor graphics
```

Use an adapter-name substring matching your Windows GPU; `NVIDIA` is not appropriate
for every machine. For SSH into a running WSLg desktop, windows also need its display
connection. When the corresponding sockets exist, `DISPLAY=:0` and
`WAYLAND_DISPLAY=/mnt/wslg/runtime-dir/wayland-0` select that desktop. The window opens
on the remote PC's Windows desktop, not on the SSH client's display. Preserve an
existing working display selection rather than overriding it globally.

`MAELSTROM_WINDOW_BACKEND=x11` or `wayland` explicitly selects the window transport
for Maelstrom alone. The default, `auto`, follows the available display environment.
This is separate from OpenGL/Vulkan and CPU/CUDA selection. On the primary PC,
X11 is configured because native Wayland tests encountered a WSLg compositor restart;
live viewing and playback passed through X11 with hardware OpenGL.
