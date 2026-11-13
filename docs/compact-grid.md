# CUDA compact uniform grid

Normal neighbour discovery builds a compact uniform grid on the resource's existing
CUDA context and stream. It replaces operation 13 and the cell-key lower-bound
searches. This checkout's previous operation 13 was already a tiled merge sort;
the original individual bitonic stages remain available to existing regression tests.
There is no Runtime API dependency or new full-particle host observation.

The PTX-only build compiles `native/kernels.cu`, then the C++ bridge loads its named
entry points through the Driver API. A host CUB device-scan dispatch would not fit
that compilation boundary without changing the host build/dependency model. The
implementation instead follows the existing integer prefix approach: 256-thread
block scans, a block scanning contiguous chunks of block totals, and carry addition.
The largest admitted grid has 65,537 block totals, at most 257 per lane in the
upper scan. Shared-memory size is fixed, all lanes participate in barriers, and
all sums are exact u32 integers bounded by particle count (currently at most 2^31).

`SolverSettings` extends the private by-value Rust/C++ ABI with signed grid minima,
u32 dimensions and cell count, and a u64 tile-offset base. Its checked size is 152
bytes. `SolverBuffers`, the FFI entry points in `native.rs`/`probe.h`, and the fourteen
persistent storage slots are unchanged. The compact entry point takes the same
three kernel arguments as the other solver entry points.

The cell buffer, slot 10, contains these u32 arrays in order, where M is admitted
cell count and N is particle count:

| Array | Length | Purpose |
| --- | ---: | --- |
| counts | M | histogram, cleared each build |
| offsets | M + 1 | exclusive range boundaries, final value N |
| cursors | M | mutable scatter cursors, initialised from offsets |
| particle cell IDs | N | avoid repeating coordinate conversion during scatter |
| grouped particle indices | N | permutation of 0..N, grouped by cell |
| block totals | ceil((M + 1) / 256) | reusable integer scan scratch |

After rounding up to eight-byte alignment, the allocation contains u64 neighbour
row tile offsets of length `ceil(N / 32) + 1`. CSR builds reserve the same small
trailer for a simple shared layout. No cell record or padding sentinel is generated
on this path. The first attempt defers slot 10 allocation until prediction bounds
select its representation. Index capacity grows geometrically when bounds exceed its reservation, rather
than reallocating for every small cell-count change. Growth is capped at the
larger of the maximum admitted grid layout and the sorted layout for that particle
count, using checked byte arithmetic. Switching to a smaller domain retains the
previous high-water capacity.

Operation 26 submits operations 27–32: clear counts, assign/histogram, exclusive
block scans, exclusive block-total scan, carry addition plus cursor initialisation,
and atomic scatter. Neighbour count and fill read `offsets[cell]..offsets[cell+1]`
directly. Both are scheduled in grouped-cell order. The width remains the existing
power of two greater than or equal to support radius and the large-coordinate
safety bound, which proves the adjacent-cell search radius is one. Negative cell
coordinates use floor in double precision, matching the existing implementation;
clipping precedes unsigned flattening. Malformed coordinates or bounds in the
native interface record the existing neighbour-storage diagnostic before count
can access grouped indices. Build errors own diagnostic slot two, separate from
count (zero) and neighbour-prefix (one). Discovery gates only on the completed
build slot; every count/fill row still participates in its own minimum-error
reduction, preserving lowest-identity selection across blocks.

Atomic scatter does not specify order within a cell. The existing complete
per-particle neighbour-row sort is mandatory. Subsequent PBF passes consume the
same ascending identity rows, preserving floating-point reduction order. Self
handling, strict rounded distance predicate, unsafe-diagonal pair validation,
`Checked`, numerical formulas, cancellation boundaries, accepted-state promotion,
velocity effects and the existing GPU neighbour-list prefix are unchanged.

Dense-grid admission requires `M <= 16,777,216` and `M <= max(4096, 8*N)`. Signed
coordinate conversion is checked before casts, dimension products use checked
u64 arithmetic, and allocation sizes use checked u128/usize arithmetic. The
native bridge independently validates dimensions, product, width and layout.
These limits bound the grid by both volume and particle population; they are a
memory policy, not physical acceptance criteria. Domains outside them use the
previous sorted GPU implementation, with no CPU execution fallback. Empty input
sets the single row offset to zero without launching a grid. Degenerate bounds
produce a one-cell grid. Allocation exhaustion retains existing retry/error
semantics and never accepts partial state. Allocation failure does not trigger a
second, potentially larger sorted allocation.

Explicit neighbour lists can still require quadratic storage when particles are
coincident or densely packed. No neighbour cap, truncation or precision reduction
is introduced. The compact grid bounds spatial-index storage, not total simulation
storage. Geometric neighbour-buffer growth and temporary old/new allocations during
growth can exceed steady-state reservations.

Set `MAELSTROM_CUDA_SPATIAL_INDEX=sort` before starting a process to force the old
sorted path. An unset variable, `grid`, or other values select automatic grid
admission with sparse fallback. This diagnostic policy is read once when each
CUDA solver is created; it does not change physical configuration or recorded
inputs. Tests use per-solver overrides to compare both paths without mutating the
process environment.

## Reproduction

Run from the Maelstrom repository on the selected CUDA measurement machine:

```sh
CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --lib cuda:: -- --include-ignored
MAELSTROM_CUDA_SPATIAL_INDEX=sort CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --lib cuda:: -- --include-ignored
CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --test cuda_execution --test cuda_boundary -- --include-ignored
MAELSTROM_CUDA_SPATIAL_INDEX=sort CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --test cuda_execution --test cuda_boundary -- --include-ignored
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I/usr/local/cuda/include tests/native/tests.cpp -ldl -o /tmp/maelstrom-native-tests
/tmp/maelstrom-native-tests
```

Build the attribution harness and the complete Simulation example. `PTX` must be
the `roundtrip.ptx` from this exact build; use a dedicated `CARGO_TARGET_DIR` when
multiple source exports share a machine. These commands start new simulations;
they do not load or continue saved observations.

```sh
CUDA_HOME=/usr/local/cuda cargo build --locked --offline --release --features cuda --example compact_grid_step_bench
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I/usr/local/cuda/include tests/native/compact_grid_bench.cpp -ldl -o /tmp/compact-grid-bench
# Set this to the matching target/release/build/maelstrom-*/out/roundtrip.ptx.
PTX=/absolute/path/to/roundtrip.ptx
python3 tests/native/compact_grid_bench.py /tmp/compact-grid-bench "$PTX" /tmp/grid-attribution
python3 tests/native/compact_grid_bench.py target/release/examples/compact_grid_step_bench "$PTX" /tmp/grid-steps --steps 12 --warmup 5
MAELSTROM_GRID_BENCH_PROFILE=1 target/release/examples/compact_grid_step_bench 1048576 normal 3
```

The collector defaults to five alternating pairs for each of 65,536, 262,144,
1,048,576, 2,097,152 and 4,194,304 particles and three distributions. It preserves
raw stdout/stderr and JSON samples with executable/PTX hashes and GPU information.
The native harness warms a complete discovery before each sample. CUDA events
measure index construction, count, fill and row sort separately. Wall time covers
complete discovery including the existing neighbour prefix, total readback,
allocation and checked completion. Stage-event samples are separate from ordinary
wall samples; do not sum them into a claimed production timestep.

The native distributions use unit radius/width and lattice spacings 0.6, 0.4 and
4.0. They isolate indexing with fixed geometry. Complete Simulation measurements
use the actual resolved physical configuration, refinement four, eight iterations,
spacings 0.05, 0.035 and 0.4 metres, default effects, and no planes. The default
example warms one accepted step. The delivered qualification uses
five warm-up steps and twelve measured accepted steps, without stage profiling.
`--sort-executable /path/to/control` compares the candidate with an independently
built original source control instead of forcing sort in the candidate binary.
`--warmup 5 --steps 12` selects the delivered qualification protocol. Keep other
protocols separate rather than pooling their samples. Host allocation warm-up can
outlast the first device warm-up step.

Every final position/velocity and identity is hashed outside timing; the collector
requires exact final-state equality across all matched policies and repetitions.
These short lattice runs do not measure long-duration impact or rendering.

`reserved_device_bytes` is the high-water explicit allocation sum, including the
device diagnostic allocation. `observed_device_growth_bytes` uses Driver API free
memory before allocations and after warm-up, including allocator/JIT costs and
potential unrelated device activity; it is not a sampled whole-device peak.
Production Simulation reservations can differ because neighbour storage grows
geometrically across changing timesteps. Use separate memory tracing when peak
transient usage is the question.
