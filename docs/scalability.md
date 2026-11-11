# Scalability and resource limits

Particle count, accepted timesteps, saved observations, file bytes and memory are separate
quantities. A million timesteps need not produce a million frames: scheduling always includes the
initial and terminal observations, with the selected stride between them. There is no application
one-million-particle or one-million-frame ceiling.

## Limits and their owners

| Owner | Previous restriction | Assessment and current behaviour |
| --- | --- | --- |
| `scene/resolved.rs`, `particles.rs`, `groups.rs` | 1,000,000 total/lattice particles | Arbitrary. Replaced by checked count products, `usize` conversion, Rust's contiguous allocation range and fallible reservation. The exported `MAX_*_PARTICLES` constants now describe `isize::MAX / size_of::<Particle>()`, not usable physical memory. |
| `scene/particles.rs` | 4,096 standalone explicit particles | Protective against quadratic validation, but unnecessarily restrictive. Larger inputs reuse spatial separation checks, including coincidence and finite-distance checks. Existing small inputs retain their diagnostic order. Extremely remote standalone coordinates use the original pairwise fallback. |
| `scene/groups.rs` | 256 named sources | Arbitrary. Removed; uniqueness, ascending identities, checked totals and allocation failures remain. `MAX_PARTICLE_GROUPS` describes addressable document storage. |
| `cache/format.rs`, `export/mod.rs`, `playback/sequence.rs` | 1,000,000 scheduled/indexed frames | Arbitrary. Removed. Both schedules are compact. Cache counts use `u64`; image indexes additionally require `usize`. Zero stride and count overflow still fail. |
| `cli/inputs.rs` | 64 MiB scene TOML | Arbitrary file-size cutoff. Removed for regular scene files, with fallible document-buffer growth bounded by the opened file extent. TOML parsing still materialises its document. |
| `cache/reader.rs` | 256 MiB input JSON | Arbitrary and inconsistent with large cache producers. Removed through streaming typed parsing and hashing. The reader checks the observed opened-file extent rather than following indefinitely growing input. |
| `cache/reader.rs` | Receipt hashes reserved for the whole expected schedule | Unnecessary memory growth, especially for partial caches. Only committed receipts enter a private disk index, at 32 bytes per frame. Every later read still checks the pinned receipt hash. |
| `cache/reader.rs` | Whole identity table buffered | Unnecessary. Checked identities and their original digest stream in row order. |
| `cache/reader.rs` | Diagnostic file ceiling of 1,024 bytes per timestep, with a separate 4,096-byte row check | Inconsistent protective bounds. The file bound now derives from the retained 4,096-byte row budget, using saturating arithmetic; every record and trailing byte is still checked. |
| `export/`, `playback/`, `cli/video.rs` | Frame/diagnostic/provenance arrays; repeated full progress rewrites | Replaced on CLI paths by temporary indexes, streamed manifests/sidecars and geometric progress snapshots. PNG playback's clock reads its disk index rather than retaining a timestamp array. |
| `playback/sequence.rs`, `export/cached.rs` | Absolute timestamp ceiling of 10^12 seconds | Arbitrary. Removed. Timestamps must remain finite; image/video mappings must remain non-negative and strictly increasing at their recorded precision. |
| `render/gpu.rs` | Requested default 256 MiB graphics-buffer limit | Artificial relative to adapters supporting larger buffers. The graphics device now requests the adapter's supported buffer limit. Host size/count and buffer checks run before centre-vector allocation. |
| `scene/geometry.rs`, `container.rs` | 32 planes | Retained protective authoring budget: feasibility validation enumerates plane triples and tests candidates against all planes, with worst-case fourth-order work. It is not a CUDA plane-array limit. Changing this geometry algorithm is outside this revision. |
| `config/validation.rs` | Support ratio at most 8; 1–64 solver iterations | Retained numerical-policy and validation-work budgets. These govern the supported model, not particle count or run duration; this revision does not change numerical algorithms. |
| `execution.rs` | At most 256 CPU workers | Retained process thread budget; independent of workload size. |
| `presentation/`, `playback/`, `export/video.rs` | 8192 pixels per image side; 300 MiB PNG file; 272 MiB decoder workspace | Retained explicit image/decoder resource budgets. Graphics texture limits can be lower. H.264 yuv420p still requires even output dimensions. |
| Cache request/completion and receipt parsing; CLI settings readers | 1 MiB small metadata/settings documents; 2,048-byte cache receipts; JSON recursion protection | Retained protective input budgets for bounded schemas, not scene/frame histories. Unknown fields, unsafe paths and unsupported schemas remain rejected where required. |
| `viewer/cached.rs` | Default 128 MiB read-ahead budget; 2–8 queued frames and decode/display allowance | Retained explicit memory policy. Increase the existing `--cache-memory-mb` for larger particle frames. The estimate is 256 bytes per particle per slot; it is not total process RSS. |
| `application/worker.rs`, `viewer/profile.rs` | 1,000,000 opt-in trace rows | Retained profiling memory budget, now with an explicit incomplete-trace diagnostic. Neither profile stops simulation steps or presentations. |

## Format and hardware constraints

Existing scene schemas 1/2, cache schema 1, binary frame version 1, image manifests 1/2 and video
sidecars 1 retain their meanings. No incompatible format revision is introduced. Binary frame size
is exactly `160 + 24 * particle_count` bytes; identities use another `8 * particle_count` bytes in
the initial identity table. Layout arithmetic, stable `u64` identities, finite binary32 values,
checksums, canonical digests and exact accepted-step mappings remain mandatory. Filenames use a
minimum eight-digit index, allowing longer canonical decimal names as counts increase.

Timesteps, identities and cache frame counts must fit `u64`; an endpoint-inclusive schedule can
overflow its frame count even when its final timestep fits. Lattice axis counts remain `u32`, and
binary32 axis positions must remain strictly increasing after the existing position arithmetic.
Grouped spatial cells must fit signed 64-bit coordinates. Duration selection rounds up with the
effective timestep and rejects an unrepresentable `u64` result. Very long image/video sequences
can exhaust distinguishable `f64` timestamps before they exhaust exact step indices; no precision
is reduced to work around this. Image metadata indexing rejects collisions before completion
publication, rather than producing a completed sequence that playback would reject.

CUDA retains `u32` particle indexes, `u64` neighbour offsets and checked byte sizes. Its current
sort padding must fit a `u32` power of two, giving a representational count bound of `2^31` rather
than a promise that such a workload fits a device. Cell-sort scratch uses `64 * padded_count`
bytes alone. Neighbour storage contains every actual entry, including checked tiled-row padding;
no neighbour cap is introduced. CUDA allocations, launch dimensions, available VRAM and device
errors can impose earlier limits. Allocation failures retain the existing retry/rollback boundary.
Native kernels, numerical modes and retained CUDA optimisations are unchanged.

Graphics draws require a `u32` instance count and a centre buffer of 16 bytes per particle within
the selected device's supported buffer size. Textures, readback and host upload buffers require
additional storage. The adapter's advertised limit does not guarantee that a particular allocation
will succeed. Simulation and graphics are independently constrained.

## Memory, publication and failure behaviour

Cache/frame validation keeps one decoded observation and bounded I/O workspace at a time. It does
not retain all particle frames. Metadata indexes move history-length storage to the temporary
filesystem: 32 bytes per cache receipt and 120 bytes per PNG frame, plus streamed diagnostic and
video provenance logs. Set the operating system's existing temporary-directory location when that
storage needs a different disk. No persistent index or new application configuration is required.
Unix scratch files are exclusively created with private permissions and immediately unlinked;
other platforms remove them when closed. A missing/full temporary filesystem is a normal reported
failure. Scene and manifest reads are bounded by their opened regular-file extent so concurrent
growth cannot create an indefinite input stream. Opening caches and sequences requires writable temporary storage even when the source is
read-only.

Large particle vectors, CPU working arrays, record decoding, comparison-key construction during
baking/replay, geometry uploads, decoded pixels and PNG output use checked reservations on the
changed paths. This does not guarantee survival of an
operating-system OOM kill, standard allocation in legacy clone/inspection paths, or every allocation
inside third-party parsers, graphics and encoder libraries. The initial state and legacy comparison-key string still grow with particle count.

CLI manifests and video sidecars stream to exclusively published files. Existing library APIs
returning complete JSON values deliberately materialise those return values; use the documented
summary APIs for bounded history storage. Explicit inspection, legacy result JSON, `runs show`,
saved-result rendering and recorded-input reruns still materialise their selected numerical record.
They have no new size cutoff, and are not the recommended route for long baked histories.

Progress snapshots may lag a running PNG export; handled failure/cancellation flushes the complete
published prefix where storage remains writable. No completion marker is published after a failed
frame, receipt, checksum, encoder or staging operation. Existing partial-cache and exclusive-output
rules remain. Abrupt process death and storage failure can prevent the final progress update.

## Remaining bottlenecks

Simulation state and the legacy input/comparison key require memory proportional to particle count;
dense neighbour sets can require quadratic storage/work. Plane feasibility, extreme-coordinate
explicit fallback and the unchanged legacy in-memory result APIs remain expensive. Temporary
indexes trade RAM for disk space and I/O (in addition to source-file and inode capacity), and cache opening still verifies every committed receipt
and completed diagnostic. Rendering and encoding retain per-frame GPU/readback/pixel costs.
No result here establishes a universal feasible maximum or production speedup for large scenes.

The parent project's scalability assessment records the exact validation environments, synthetic
fixtures, matched measurements and their limitations.

## Rust API notes

The CLI and persisted schemas remain compatible. The compact schedule changes the Rust
`FrameSchedule.steps` field from a `Vec<u64>` to `StepIndices`: use `.len()`, `.get(index)` and
`.iter()`; iteration now yields step values. `FrameSchedule` still serialises the same `steps`
array, but no longer derives arbitrary-array deserialisation/equality. `Sequence::timestamps()`
now returns a `Result<Vec<f64>, RenderError>` so index I/O and allocation failures are explicit;
`.timestamp(index)` avoids that array entirely. Full-document export APIs retain their return shape,
while the new summary APIs avoid retaining histories. MP4 `--format json` streams the complete
published sidecar, preserving the existing response and provenance fields.
