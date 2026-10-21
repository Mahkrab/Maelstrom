# Viewing accepted particles

Build the optional viewer on x86_64 Linux/WSL2. It uses winit for the window and
Vulkan through wgpu for drawing; simulation execution remains independent.
The default Cargo build has no graphics dependencies or display requirement.
`view` **starts a new simulation**; it is not historical playback. To watch a
completed PNG export without simulation or particle rendering, use
`maelstrom play <export-directory>` instead; see [saved playback](export.md#watching-a-saved-sequence).

```sh
cargo run --locked --features viewer -- view falling-block-04 --steps 480
cargo run --locked --features viewer -- view ./scene.toml --config ./config.toml --steps 120 --mode threaded-grid --workers 4
cargo run --locked --features cuda,viewer -- view falling-block-04 --steps 480 --mode cuda-grid --device 0 --block-size 128
```

A launch starts paused at accepted step zero. The window leaves existing keyboard
focus alone; click it before using keyboard controls. A zero-step request completes at
zero. Select `--mode brute-force`, `serial-grid`, `threaded-grid` or `cuda-grid`;
CUDA requires its independent feature, toolkit at build time and runtime device.
Explicit CUDA requests fail visibly when unavailable. There is no CPU fallback.

Graphics defaults to hardware Vulkan adapter zero. `--graphics-adapter N` selects
an index within that device class. `--software` explicitly selects the software
class instead, for example Mesa llvmpipe. The selected name, driver and surface
format are reported to stderr. Adapter enumeration can differ between machines;
the printed identity describes the device actually used. CUDA ordinals do not
select a graphics device. Native Windows, macOS, browsers and non-Vulkan backends
have not been qualified.

WSLg supplies the Wayland display used for local interaction checks. A Vulkan
loader and compatible driver are required. Linux keyboard support requires
`libxkbcommon0`; X11 additionally needs `libxkbcommon-x11-0` and `libxcb-xkb1`.
The native keyboard libraries are checked before window creation. Missing
displays, native libraries or selected adapters produce launch errors. The
`render` feature alone supports shared drawing without pulling in winit; the
[offline exporter](export.md) uses that feature for windowless PNG sequences.

## Choosing and controlling a run

The window panel shows the draft and the selected run separately. Use Tab to
choose a draft field, Enter to edit it, Delete to clear it, then type and press
Enter to save. Escape abandons an edit. Scene accepts a bundled name or an
external TOML path; Config accepts an explicit base file, or blank to keep the
scene's selection. Relative launch and edited paths resolve against the invocation
directory. The existing defaults/file/scene/`--set` precedence and scene-relative
configuration references apply. Launch-time `--set` and refinement overlays
remain in effect when editing the draft.

| Control | Result |
| --- | --- |
| A | Validate and apply the whole draft as a new paused run |
| 1–7 | Select a bundled falling-block size in the draft |
| B | Cycle the draft through the four simulation modes |
| Space | Pause or resume the selected active run |
| N | Attempt exactly one timestep while paused |
| R | Reset the selected original inputs, ignoring unapplied draft edits |
| C | Cancel, retaining the last accepted state |
| O | Toggle inspection of the labelled prior accepted outcome |
| D | Request a density audit of the displayed accepted snapshot |
| Left drag or arrow keys | Orbit around the camera target |
| Right drag | Pan eye and target together |
| Wheel or +/− | Zoom without changing world coordinates |
| F | Explicitly fit the displayed run's initial particles |
| H | Hide/show the diagnostic panel |
| G | Recreate the presentation surface using the same graphics device |
| Escape outside editing, or window close | Request cooperative worker teardown |

Failed draft validation keeps the selected run. A worker construction failure
shows a new failed generation without borrowing particles from the old one. O
can inspect the retained prior accepted outcome, explicitly labelled PRIOR;
controls still target the selected run. Input fields and pending resource settings
do not describe effective resources until construction succeeds.

The panel reports status, accepted/requested steps, simulated seconds, particle
count, effective backend/resources, cumulative CFL warnings, latest CFL and any
terminal failure. Density auditing runs on one separate thread against an owned
snapshot; its result identifies the generation and step even after reset. It is
outside solver work and is never triggered by redraws. Redraw FPS describes
presentation cadence only. Solver timing and simulation throughput are not
measured by this viewer.

## Camera and failure behaviour

Particles are opaque blue spheres, with radius half the configured spacing,
in the original metres and positive-Y-up coordinates. Colours and lighting do
not encode pressure or density. Static constraints appear only as plane outlines
clipped to a frozen display box derived from initial particle bounds. A plane
outside that box has no visible outline; the depiction adds no physical walls.
The fixed sphere mesh has 16 longitude and 12 latitude divisions, with depth
testing, one sample per pixel and no blending or temporal interpolation.

The configured camera remains fixed until changed explicitly. Use F or launch
with `--fit-camera` to frame the initial particles. Empty scenes use a one-metre
fallback box. Camera calculations use f64 metres before converting view-relative
coordinates to graphics f32; accepted particle storage stays untouched. Views
that cannot be represented finitely are rejected and preserve the previous
camera. Very extreme dynamic ranges can still lose display precision or clip;
that does not alter the simulation.

Select `--render-config render.toml` for strict render schema 1. Built-in defaults
are overridden by that file, then explicit `--width`/`--height` arguments. There
is no implicit home or working-directory configuration. Camera fields merge
individually with defaults. For example:

```toml
schema_version = 1
width = 1280
height = 720
[camera]
eye_m = [1.2, 1.0, 1.2]
target_m = [0.3, 0.4, 0.3]
vertical_fov_deg = 45.0
```

Dimensions must be 1–8192 and within the actual device limit. Unknown fields,
versions, invalid vectors or clipping distances fail validation. `format="png"`
and positive `step_stride` are shared with the offline exporter; this viewer
uses drawable dimensions and camera settings.

Resize recreates the depth target; zero drawable extent and occlusion suspend
drawing. Surface loss recreates the surface, outdated/suboptimal surfaces are
reconfigured, and acquisition timeout skips presentation. Device/validation/resource
failures request cancellation and continue pumping events while teardown finishes.
The worker owns all solver and CUDA resources until destruction. Close also waits
for a requested density audit to finish. Synchronous graphics calls, solver stages
and audits have no promised shutdown deadline. A hung driver or forced process
termination cannot guarantee a retained outcome.

[Application control](application.md) explains coalescing observations and command
acknowledgements. Display may skip any intermediate snapshots; numerical work
always executes the complete finite request with the unchanged fixed timestep.
`THIRD-PARTY-NOTICES.txt` retains upstream notices for the locked Linux dependency
graph, including build dependencies. Preserve these notices when distributing
a viewer binary; other target graphs need their own review.
