# Reusable particle caches

`cache bake` starts a new simulation and retains selected accepted particle states. `render`
consumes those states without constructing a simulation or CUDA context. A cache is an observation
archive for visualisation, **not a solver restart checkpoint**. No command continues a solver from
one of its frames.

```sh
# New simulation, no graphics or PNG work; both endpoints are retained.
maelstrom cache bake builtin:large-impact --duration 5 --backend cuda --frame-stride 4 -o impact-cache
maelstrom cache inspect impact-cache
maelstrom cache validate impact-cache --full

# Existing observations, no fluid steps. Each output directory must be new.
maelstrom render impact-cache --render-config front.toml -o impact-front
maelstrom render impact-cache --render-config side.toml --width 1920 --height 1080 -o impact-side
maelstrom render impact-cache --frame 150 --render-config front.toml -o impact-still
maelstrom render impact-cache --first-frame 30 --last-frame 150 --every 2 -o impact-excerpt
maelstrom play impact-front

# New simulation followed by rendering, retaining a reusable cache.
maelstrom compile builtin:large-impact --duration 5 --frame-stride 4 --backend cuda \
  --cache-output another-cache -o another-animation
```

Bare `compile` retains its existing direct pipeline and schema-1 PNG bundles. `--cache-output`
selects staged orchestration explicitly. The image output and cache must be different new
directories. A successful cache survives later graphics or image-output failure.

Baking works in headless builds, including `--features cuda`; rendering requires `render` or
`viewer`. Backend preferences, experimental numerical defaults and explicit overrides work as
they do for `run` and `compile`. Render commands accept no simulation backend or numerical override.
Their default presentation is the built-in view; use `--render-config` to choose the scene's saved
camera/colours. Standalone caches contain physical geometry and scaling, not camera preferences.
The staged `compile` convenience preserves its ordinary scene-package presentation resolution.

Frame numbers passed to `render` are **cache-frame indices**, starting at zero. `--every` selects
an increment in those indices; the inclusive last index is not appended when the increment does
not reach it. `--frame-stride` belongs to simulation sampling, not cache rendering. A 240 Hz solver
with stride 4 produces 301 observations for 1,200 steps: initial state, every fourth accepted state
and final state. It does not retain all 1,201 boundaries or generate new intermediate states later.

## Format and identity

The version-1 directory contains:

- `request.json`: immutable schema/kind, canonical decimal step/stride strings, particle count,
  exact timestep bits, execution/backend/build provenance and input/identity file digests.
- `input.json`: existing portable `RecordedInput`, with fully resolved numerics, initial particles,
  stable identities and static planes. Reading never consults original scene paths or presets.
- `identities.bin`: ascending, unique little-endian u64 identities, one per row, stored once.
- `frames/frame-00000000.mpc`: independently addressable raw binary frames.
- `receipts/frame-00000000.json`: committed frame index/step/length, binary SHA-256 and the existing
  canonical snapshot SHA-256. Filenames are derived from integer indices, never supplied by receipts.
- `diagnostics.jsonl`: one existing accepted-step diagnostic per line, streamed in accepted order.
- `cache.json`: the final completion marker, request/receipt-chain/diagnostic hashes, frame/step
  counts, warnings, content identity and overlapping stage measurements.
- `failure.json`, when possible after an error: explicitly incomplete outcome with separate accepted
  step and committed frame counts. It is never a completion marker.

Each frame begins with this **160-byte little-endian header**:

| Offset | Bytes | Meaning |
| ---: | ---: | --- |
| 0 | 8 | ASCII `MSPCACHE` |
| 8 | 2 + 2 | Major 1, minor 0 |
| 12 | 4 | Header size 160 |
| 16 | 8 | Cache-frame index |
| 24 | 8 | Accepted timestep index |
| 32 | 8 | Particle count N |
| 40 | 8 | Uncompressed payload length, 24N |
| 48 | 8 | Stored payload length, 24N |
| 56 | 8 | Position array file offset, 160 |
| 64 | 8 | Velocity array file offset, 160 + 12N |
| 72 | 4 + 4 | Codec 0 (raw), flags 0 |
| 80 | 32 | SHA-256 of exact `request.json` bytes |
| 112 | 32 | Canonical snapshot SHA-256 |
| 144 | 16 | Reserved, all zero |

The payload has N XYZ f32 positions followed by N XYZ f32 velocities. Values retain the original
bits, including signed zero; no quantisation, spatial sorting or coordinate conversion occurs.
Positions are metres, velocities metres per second, right-handed with positive Y upward. Row i
uses identity i from the shared table. Birth/death, compression, other codecs and nonzero flags
are unsupported and rejected. A frame has exactly 160 + 24N bytes; trailing bytes are invalid.

The 160-byte header deliberately extends the review's suggested 128-byte header: the canonical
digest is inside the binary checksum, binding the digest and particle bytes together. Canonical
provenance still hashes the **unchanged standalone pretty snapshot JSON**, including velocities,
using the retained streaming writer. Binary checksums never substitute for it.

The reader checks input/identity digests, exact bit timestep, compact schedule, receipt chain and
completed diagnostic stream. It pins the checked receipt digests in memory, then checks the selected
frame's exact length, header, request binding, binary checksum and finite ordered particles before
constructing an immutable `ValidatedFrame`. That sealed value supplies its bound canonical digest
to the existing encoder. Changing a receipt after opening the cache cannot change that proof.
`cache validate --full` additionally reconstructs and independently recomputes every canonical digest.
Ordinary rendering avoids that repeated JSON formatting. Checksums establish integrity relative to
the cache metadata; they are not signatures. Use full validation for externally produced caches.

Request identity includes exact inputs, build/execution provenance, timestep, duration and sampling.
Result identity additionally includes ordered receipts and accepted diagnostics. Labels and camera
changes cannot make one physical cache supply states that were never recorded. File versions and
required features are strict; future format conversion must be explicit.

The particle payload for 301 frames is 57,792,000 bytes at 8,000 particles and 798,916,608 bytes at
110,592 particles, plus 160 bytes/frame, static identities, resolved inputs and small receipts.
Producer memory holds at most one pending particle frame, a packing buffer, fixed diagnostic/hash
buffers and the solver/input state. It does not retain particle history or per-step diagnostic arrays.
Readers retain typed inputs and pin receipt digests in a private temporary index (32 bytes per
committed frame), without allocating for the uncommitted schedule;
particle arrays are loaded on demand. CLI image export retains bounded pixel/encoder work and
streams its frame index and manifest through temporary storage.

## Completion, cancellation and recovery

The cache writer owns one asynchronous CPU job. Canonical hashing, packing, binary hashing and
synced frame preparation overlap subsequent solver steps. The caller applies bounded backpressure
and publishes receipts in order. The initial receipt closes before stepping; a simulation never
waits for a PNG. All accepted numerical diagnostics and rollback/cancellation points are preserved.

Files are exclusively staged, synced and linked into their final names. A receipt is published
only after its complete frame, and `cache.json` only after the whole schedule and diagnostic stream
close. Unix cache directories are synced at publication boundaries. Other platforms retain process-
failure semantics without claiming directory durability. No power-loss qualification is implied.
The writer is closed and joined on every return; cancellation discards any uncommitted CPU result.
A prepared frame without its receipt is an orphan and is ignored during prefix inspection.

```sh
# Existing, incomplete observations; explicitly choose the available range.
maelstrom cache inspect interrupted-cache --allow-partial
maelstrom cache validate interrupted-cache --allow-partial --full
maelstrom render interrupted-cache --allow-partial --last-frame 42 -o partial-preview
```

A completed prefix render means that the selected **image request** completed; its source manifest
still records `source_cache.complete = false`. An incomplete cache never becomes complete by having
all numbered files present. Missing/corrupt receipts or frames fail validation, existing destinations
are never overwritten, and a failed image sequence retains `progress.json` and its published prefix.
Retry rendering into a new output directory, optionally selecting only the missing range. Automatic
in-place image repair and solver continuation are not supported.

Cache-derived image sequences use **schema 2**, kind `maelstrom-cache-render`, with explicit cache
selection, timestep bits and source identity. Their PNG bytes, RGBA and snapshot hash meanings,
indexed renderer, shader and encoder policies are unchanged. `play` and `runs validate` accept both
schemas. Images remain playable and independently verifiable after the source cache is moved or
deleted. Existing `export saved`, `export rerun`, result JSON and schema-1 bundles retain their meaning.
Recorded-input reruns start from zero; they do not load a cache as solver state.

## Costs and limits

The first staged export adds particle writes/reads and separate setup. A cache is valuable for
repeated presentation, but is not a promise of a faster first export. Baked output records solver,
capture, diagnostics, canonical hashing, packing, binary hashing, file/receipt publication and writer
wait intervals. Image output records cache reads/validation, graphics/readback, encoding, hashes and
publication. Worker intervals overlap caller intervals and must not be added as disjoint wall time.
Complete process measurements include startup, completion publication and teardown.

One GPU owns simulation during baking and graphics during rendering. This path does not introduce
concurrent CUDA/graphics workers, compression or surface reconstruction. See
[interactive playback and MP4 export](video.md) for cache viewing,
portable cameras and bounded direct video encoding. The live solver viewer retains its existing
analytic impostor behaviour.


Input JSON and identity tables stream through their original SHA-256 checks; input documents have
no fixed file-size cutoff. The typed initial state and its legacy comparison key still require
memory proportional to particle count. Receipt indexes use temporary disk space, so opening a
cache can fail clearly if that storage is unavailable. No source file or format identity changes.
`Cache::open_with_cancellation` supports cancellation while reading input, receipts and diagnostics;
`Cache::open` remains the convenience API. Full validation processes one observation at a time,
including an independent canonical snapshot rehash. See [resource limits](scalability.md).
