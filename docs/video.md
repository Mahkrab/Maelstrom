# Explore saved motion and export video

Build with `cargo build --release --features viewer` for all commands. `--features render`
supports PNG and MP4 export without a window. FFmpeg is an external executable, not a Rust
build dependency. Install a build with the `libx264` encoder on PATH, or use `--ffmpeg PATH`.
`ffmpeg -encoders` should list `libx264`; a missing executable or encoder produces an actionable
error before frame delivery. The commands use argument-based subprocess execution.

The following bake starts a **new simulation**. Every later command loads saved observations
or images and executes no fluid timesteps; none resumes a solver.

```sh
maelstrom cache bake builtin:large-impact --duration 5 --backend cuda --frame-stride 4 -o impact-cache
maelstrom view --cache impact-cache --paused --camera-output chosen-view.toml
# Press S in the window to publish chosen-view.toml.
maelstrom view --cache impact-cache --render-config chosen-view.toml --paused
maelstrom view --cache impact-cache --cache-playback frame-accurate
maelstrom render impact-cache --render-config chosen-view.toml --frame 150 -o preview
maelstrom video cache impact-cache --render-config chosen-view.toml -o impact.mp4 --fps 60 --crf 18

# The lossless image path remains supported, and images need no source cache afterwards.
maelstrom render impact-cache --render-config chosen-view.toml -o impact-images
maelstrom video images impact-images -o from-images.mp4 --fps 60 --crf 18
```

Every output path must be new, including the video provenance sidecar (`impact.mp4.json`).
Camera saves also create new files: choose a new `--camera-output` for another viewpoint.
Load the saved file with `--render-config` in the viewer, PNG renderer or video exporter.
No original scene file is needed. Cache view rejects numerical/backend/step options.

## Window controls and timing

| Control | Action |
| --- | --- |
| Space | Play/pause |
| N / B, or period / comma | Next/previous exact cached frame, paused |
| J, index, Enter | Jump to any zero-based cache-frame index, paused |
| Drag bottom seek strip | Scrub cached frames, paused |
| R / Home | Restart at frame zero |
| End | Last frame, paused |
| L | Toggle loop |
| `[` / `]` | Halve/double playback speed, 1/16× to 16× |
| T | Switch real-time/frame-accurate timing, anchored at the displayed frame |
| Left drag / arrows | Orbit |
| Right drag | Pan |
| Wheel / plus / minus | Zoom |
| F | Fit frozen initial geometry |
| P | Switch indexed/analytic impostor preview |
| S | Save full current presentation to `--camera-output` |
| Escape | Close (or cancel index entry) |

The window title and panel report displayed frame, simulation timestamp, play/pause, loading,
speed, loop and geometry choice. Playback uses the **recorded simulation timestamps**, including
an irregular terminal interval. It never interpolates or changes particles. The solver frequency
(e.g. 240 Hz), cache sampling frequency (normally 60 Hz at stride 4) and display refresh are
different clocks. Client submissions follow the surface/driver; they do not prove physical display FPS.

**Real-time playback is the default** (`--cache-playback real-time`). Monotonic elapsed time and
the speed multiplier choose the most recent recorded observation at or before the target time.
Intermediate presentations may be skipped. During a disk stall, time continues while the viewer
holds the latest available state and accepts camera input. A completed older read can still update
the display; subsequent reads catch up with the current target. Slow storage can therefore leave
the displayed state behind the requested timeline. No missing state is invented or interpolated.

**Frame-accurate playback** (`--cache-playback frame-accurate`, or T) presents every sequential
observation, advancing at most once per redraw. It can run slower than requested. Disk stalls hold
both the displayed state and playback progression; camera input continues. Single-frame stepping,
reverse stepping, arbitrary seeking and paused inspection always request the exact selected state
in either policy. A seek supersedes read-ahead; one in-flight read may finish before its target loads.

Initial loading and explicit seeks anchor the clock when their exact observation arrives. Pausing
preserves fractional playback time; resuming adds no paused wall time, and speed changes settle
elapsed time at the old multiplier first. Looping holds the terminal observation for the final
recorded gap, then wraps to zero; a one-frame cache uses 1/60 s. Without looping the final observation
is loaded and playback pauses. In real-time mode a long stall may skip whole loop presentations.
These controls never change cache bytes or execute fluid physics.

Incomplete caches require `view --cache DIR --allow-partial`. This opens only the committed prefix;
the cache remains incomplete. Missing or corrupted observations are errors rather than skipped data.

`--cache-memory-mb 128` is the default **particle read-ahead/workspace budget**, not a total-process
RSS limit. A conservative 256 bytes per particle per slot covers binary decode and conversion;
two workspace/display slots are reserved, with two to eight read-ahead frames. One latest completed
read is retained to permit progress when its target was superseded. Budgets too small
for the minimum are rejected; there is no whole-animation load. Frozen input/identity metadata,
receipt index, graphics resources and window resources have separate existing bounds. Only one
background reader performs disk work. Shutdown joins its current read; a stalled filesystem can
delay shutdown but does not block camera input during normal viewing. Cache metadata is validated
at opening and frame bytes are validated before use.

## Reproducing the camera

The save includes explicit pixel dimensions, perspective eye/target/up, vertical FOV, near/far
clipping distances, colour mode and colour/energy range. It writes a validated schema-2 fixed
presentation using the existing TOML schema, so it does not depend on scene preferences or
transient camera defaults. Dimensions reflect the drawable area at save time. Retain the same
aspect ratio to preserve framing; changing only the output aspect changes horizontal framing.
Only perspective projection is supported by the existing renderer.

The cached viewer defaults to **indexed** geometry, the same geometry as offline PNG and cached
video export. Both modes share `Graphics::draw`, camera basis, relative-coordinate conversion,
vertical-FOV projection, aspect and clipping maths. Camera geometry therefore reproduces exactly
at matching dimensions. Indexed preview is the appropriate composition check. P or
`--preview-renderer impostor` selects the live viewer's analytic sphere representation for smoother
spheres. Analytic surfaces and triangle meshes differ in silhouette, depth and normals, so their
pixels are not expected to match. Surface format/driver/presentation and the viewer's diagnostic
overlay can also differ from an offscreen PNG. Renderer selection is a preview choice; the
portable configuration describes the supported indexed output. Use `render --frame INDEX` for
an overlay-free lossless output preview. H.264 also introduces lossy colour/subsampling changes.

## Video policy and reliability

`video cache` requires an explicit **fixed** `--render-config`. It streams indexed RGBA readback
to FFmpeg, without PNG files. `video images` consumes a validated completed Maelstrom PNG bundle
(directory or `manifest.json`, schema 1 or 2); it validates ordered PNG and decoded-pixel hashes
and never requires the cache. This command does not accept arbitrary unmanifested numbered PNGs.

Both workflows accept `--width`, `--height`, `--fps`, `--crf`, `--first-frame`, `--last-frame`,
`--every` and `--omit-terminal`. Frame bounds are inclusive, zero-based **source-frame indices**;
`--every` never appends an unselected last frame. Invalid/out-of-range/empty selections fail.
For `video images`, indices address the image bundle's order. A sparse bundle retains its
original cache indices/timestamps in its source manifest; they are not new image ordinals.
Images are scaled by FFmpeg when output dimensions differ; cache frames are rendered directly at
the selected dimensions. H.264 output uses libx264, MP4, yuv420p and fast-start metadata. Output
dimensions must be even, between 2 and 8192. FPS must be finite in (0, 1000]. CRF ranges from 0 to
51, with 18 the default (lower values increase quality/size). Yuv420p output is not a replacement
for lossless RGBA PNGs. See [FFmpeg's format documentation](https://ffmpeg.org/ffmpeg-formats.html)
for rawvideo input and MP4 muxing.

**Default: encode every selected observation once for one video-frame period.** A 240 Hz,
five-second bake with stride 4 retains 301 observations (steps 0 through 1200). At 60 FPS this
is 301/60 = 5.016666… seconds; the final observation remains visible for a full frame period.
Use `--omit-terminal` explicitly for 300/60 = 5 seconds: it omits the final *selected* observation
from this video only. The cache, image schedule and all identities remain intact. The provenance
records the omission policy. No duration option silently resamples or drops endpoints. For a
sparse or irregular selection, CFR video time is selected ordinal/FPS, while original simulation
timestamps remain separate in provenance. It is intentionally a constant-frame-rate slideshow of
selected observations, not an interpolation or a claim that sparse frames retain physical pacing.

Frame delivery retains one producer frame, one queued frame and one writer frame, plus graphics
readback/decode and FFmpeg's own encoder buffers. FFmpeg stderr is drained with a 64 KiB tail.
Cancellation is checked during queue backpressure and encoder shutdown, kills/reaps FFmpeg,
joins pipe threads and removes staging files. Graphics readback and a single disk decode finish
before the next cancellation check. Encoder nonzero exit or a frame validation failure prevents
publication. A private sibling staging directory holds the temporary MP4; successful completion
syncs it and publishes through exclusive hard links. Existing destinations are never overwritten.
Caught failures/cancellation remove incomplete outputs. Abrupt process/OS death can leave a hidden
`.maelstrom-video-*` staging directory; automatic recovery and power-loss atomic publication of
the MP4/JSON pair are not claimed. Publication requires a filesystem supporting hard links, as
does the existing output publisher.

The sidecar records FFmpeg version/path, codec, CRF/pixel format, dimensions, frame selection,
original simulation timestamps, video timestamps/duration, timing policy and elapsed encoding
wall time. Cached video records cache identity, timestep bits and resolved presentation. PNG
conversion records the source manifest SHA-256 and its resolved presentation; the manifest binds
original image and snapshot identities. No video throughput or primary-PC speedup claim is made.

## Verification boundary

Local WSL verification on 10 November 2026 passed 67 library tests, 56 CLI tests across
`cli_usability`, `cli_workflow` and `saved_settings_cli`, all three opt-in cache graphics/video
workflow tests, and the real-FFmpeg encoder tests. FFprobe checked codec, dimensions and frame
count. Tests cover explicit endpoint omission, invalid ranges/FPS/dimensions, missing executable
or libx264, encoder failure, cancellation, corrupt input and output preservation/cleanup.
Headless and render-only builds were checked separately; the viewer build was exercised in a window.

A scripted X11 journey on llvmpipe Vulkan exercised stepping, reverse stepping, arbitrary seeking,
play/pause, speed, camera orbit/zoom, preview switching, save and reload. The saved camera was
accepted by independent PNG and MP4 export. At matching frame 17 and 640×480 dimensions,
140,160 pixels in the scene region outside the viewer controls matched the indexed PNG exactly.
This is a same-device indexed regression observation, not a general cross-driver pixel guarantee
or a claim of equality with analytic impostors. X11 testing supplied Ubuntu's documented
`libxkbcommon-x11-0` and `libxcb-xkb1` libraries from a temporary extracted package directory;
the system installation was not changed.

Primary-PC qualification on 10 November 2026 used release builds on the Ryzen 9 7900X / RTX 4070,
WSL2 and WSLg's hardware Mesa D3D12 OpenGL path. Real-time playback at 8,000, 32,000 and 110,592
particles followed recorded time at both 1280×720 and 1920×1080. Requested 4× playback advanced
at approximately 3.98–3.99×, including the terminal hold. Frame-accurate mode retained sequential
states and accepted slower playback. Client submissions are not verified physical display FPS.

The 110,592-particle, 1920×1080, 120-second looping soak held sampled process RSS at 228.20 MiB.
Camera controls, rapid seeks, resize, saved-camera reload, missing/corrupt frames, explicit partial
viewing, repeated open/close and delayed reads were exercised. A saved-camera indexed capture and
offline PNG agreed at every one of 160,000 compared pixels outside the controls. Native X11/XTest
routing was exercised; human input feel and Windows desktop focus/scan-out remain unverified.

Both video paths passed independent H.264/yuv420p decoding, duration, endpoint and sparse-selection
checks at 640×480, 1280×720 and 1920×1080. At the large count, a 301-frame direct export took
3.97 / 4.67 / 5.83 seconds at those resolutions; PNG conversion took 0.97 / 1.27 / 1.97 seconds,
excluding PNG production. These are individual complete-process observations, not a general speedup
or hardware-encoding claim. FFmpeg used CPU libx264. Its default encoder buffers reached about
1.2 GiB at 1080p, separate from the application's roughly 206 MiB sampled peak during direct export
and from the cache read-ahead budget. Cancellation during real encoding reaped the child and
removed incomplete output. Missing executables/codecs and encoder failure were rejected cleanly.

Match video selection/FPS to the cache's sampling cadence when recorded-time pacing is wanted.
For example, a five-second 120 Hz cache has 601 observations: default 60 FPS produces 10.016667
seconds. `--every 2 --fps 60` selects 301 existing observations for 5.016667 seconds; `--fps 120`
keeps all 601 for 5.008333 seconds. Both endpoint policies remain explicit.

The parent EPQ project retains the hardware protocol, individual traces, source/binary identities,
GPU timestamp diagnostics and limitations in `performance-cached-playback.md`. Cold storage,
permanent filesystem/device stalls, physical display delivery, other operating systems and power-loss
durability remain unqualified. Shutdown still joins a current read. Existing cache identities,
solver/CUDA sources, PNG schedules and bundle compatibility are unchanged.
