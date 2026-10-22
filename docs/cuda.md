# CUDA setup and execution

CUDA is an optional NVIDIA simulation backend, independent of the graphics device used
for viewing or image compilation. A CPU build needs neither a CUDA toolkit nor a device.
Saved-image playback never requires CUDA, even when CUDA produced the simulation.

## Install and check

From this checkout on native x86_64 Linux/WSL2:

```sh
CUDA_HOME=/usr/local/cuda MAELSTROM_CUDA_ARCH=89 \
  cargo install --path . --locked --features viewer,cuda
maelstrom doctor cuda --device 0 --repetitions 3
```

Use CUDA 13 headers and nvcc, a toolkit-supported host compiler, C++17/libstdc++ and ar.
`CUDA_HOME` is an absolute toolkit path, defaulting to `/usr/local/cuda`, independent
of PATH. `MAELSTROM_CUDA_ARCH` selects the virtual capability, default 89. The runtime
device must support that target and the driver must accept the emitted PTX.
`CXX` and `AR` select executables, not commands containing additional arguments.
No build step installs a toolkit or changes drivers.

The retained hardware qualification used RTX 4070/WSL2, CUDA 13.3.73, GCC/G++ 13.3.0,
Rust/Cargo 1.96.0 and Windows driver 610.88. These are observed versions, not a promise
that every compiler/driver combination produces identical results. Cross-compilation
and other platforms are currently rejected by the native build.

Under WSL, use the Windows NVIDIA driver exposed to Linux. Do not put toolkit stub
libraries on the runtime library path. The bridge loads `libcuda.so.1` dynamically;
missing symbols, devices, incompatible targets and module loading failures are explicit.
Driver JIT belongs to context setup, not solver-step time.

## Select CUDA deliberately

```sh
maelstrom run builtin:falling-block-04 --backend cuda --device 0 --block-size 128 \
  --refine-timestep 4 --steps 480 --unpaced --format json --output cuda-run.json
maelstrom rerun ./cuda-run.json --format json
```

Supported blocks are 32, 64, 128 and 256 lanes. CUDA uses grid discovery and rejects
CPU thread counts. Explicit CUDA cannot fall back to CPU. Auto may probe device zero
before choosing a backend; after selection, a driver failure still fails the run.
Records retain the ordinal, launch size, device properties and arithmetic identity.
An ordinal on another machine need not identify the same GPU.

`doctor cuda` is an independent resource/layout check, not a fluid benchmark. It
verifies 257 records against known values across independent context lifetimes.
Its transform is `(x+1, y*-2, z-0.5)`, marker XOR `0xa5a55a5a`, with identity unchanged.
Passing the probe does not qualify a fluid workload or physical accuracy.

## Implementation and numerical limits

Each solver owns a context, module, private stream and persistent scratch.
It constructs, uses and destroys them on one thread; the resource is neither Send nor
Sync. Native calls restore the previous context and complete borrowed host transfers
before returning. The private stream participates in legacy-stream ordering so
pageable uploads finish before dependent kernels read them. No shared primary-context
reset or process-wide stream is used.

Device data uses separate packed binary32 vectors (12 bytes), planes (16 bytes) and
cell entries (32 bytes); probe records use a checked 24-byte layout. u64 identities
remain in stable host order. The GPU sorts occupied cells and searches 27 neighbouring
cells using conservative power-of-two widths and strict binary32 distance membership.
An unsafe bounding diagonal retains distant-pair overflow checks rather than silently
discarding them.

Fixed prediction membership uses contiguous u32 neighbour indices and `N + 1` u64
row offsets. A counting pass determines exact row lengths; checked host prefixes
size reusable storage before a second pass fills and sorts each row by identity.
Traversal visits those indices, inserting self once at its ordered position where
required. Storage follows actual membership rather than an all-pairs matrix; dense
input can still require quadratic space and is never silently capped or truncated.

Each attempt uploads accepted state, returns velocities for the host CFL diagnostic
and predictions for bounds checks, then retains iterative working arrays on device.
Failing lanes select the first particle's checked quantity with an integer atomic
minimum instead of returning the complete error array. Each Jacobi calculation and
candidate/projection pair is submitted as a stream-ordered group, with one result
slot per pass. A kernel skips work if an earlier pass failed; later slots cannot
replace the earlier error. One readback returns the group's results in stage order.
Successful passes leave their sentinels untouched; the four-slot record is reset for the first
attempt and after a numerical failure. The following synchronous readback needs no
additional wait after kernel completion. Numerical checks and their stage,
particle and expression ordering remain unchanged. Final state still returns for inspection.
The integer cell-sort passes are submitted in order to the private stream and completed
as one batch. Their kernels and dependencies are unchanged; submission failure stops
the batch, drains outstanding work and invalidates the context before returning.
Whole-stage Jacobi dependencies and ordered plane projection remain intact. The host
validates and atomically accepts a complete step; scratch never becomes accepted state
merely because a kernel finished.

Ordered binary32 sums, denormals, round-to-nearest division/square root and disabled
contraction preserve the supported arithmetic. Native power alternatives diverged on
the retained trajectories. The narrow positive-base/exponent implementation in
`native/power.cuh` follows the tested host approximation using binary64 intermediates;
its Arm-derived code retains the MIT licence in `native/power-LICENSE`.
This does not promise agreement with every libm or permit blanket fast-math.

Cancellation is cooperative between completed stages. Numerical failure or cancellation
retains old accepted state; retry reloads it and rebuilds membership. Input rejection
and device-allocation exhaustion may be retried. Other driver faults poison the
resource, including for later empty requests; construct a new solver rather than
resetting the process or silently substituting CPU. Actual device-loss recovery and
worst-case cancellation latency remain unqualified. Consuming close reports teardown
errors; Drop attempts cleanup but cannot report them.

## Developer verification

```sh
CUDA_HOME=/usr/local/cuda cargo test --locked --features cuda
CUDA_HOME=/usr/local/cuda cargo test --locked --features cuda --test cuda_boundary -- --include-ignored
```

Ordinary feature-enabled tests do not imply that opt-in hardware checks ran. Native
injected-driver tests cover ownership, failure precedence, draining, retry, cleanup and
context restoration; they are not real device-fault recovery tests. Host ASan/UBSan
and device memory diagnostics establish different things. The retained PC Compute
Sanitizer attempt could not initialise its WDDM debugger interface, so no passing
device-memory diagnostic follows.

Measured step totals include required transfers, launches, synchronisation and atomic
acceptance. Context/module setup and teardown are separate; audits, output, graphics
and pacing are outside solver timing. Coarse host stage intervals do not isolate
kernel time or prove which overhead dominates. See [measurement use](usage.md#solver-measurements).
