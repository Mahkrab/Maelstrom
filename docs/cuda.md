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
device must support that target, mapped page-locked host memory and the emitted PTX.
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
row offsets. Count and fill assign adjacent lanes to particles in sorted cell order, improving
locality without reordering particle state. Each lane still writes its owner's identity-indexed
row and error record; candidate visitation and final ascending row order are unchanged.
A counting pass determines exact row lengths, followed on the same stream by
checked integer prefix kernels. Block scans, a scan of block totals and carry addition produce
the exact exclusive row offsets. Counts are checked against the row bound before use; only the
final u64 total returns for checked host allocation of reusable storage before filling. Offsets
stay on device. These kernels reuse correction scratch after the bounds summary is consumed. A stream-ordered integer sorting kernel
then orders each row by identity. Rows of up to 256 entries use a complete warp;
larger rows retain heapsort. Fill and sorting share one completion and diagnostic readback.
Traversal visits those indices, inserting self once at its ordered position where
required.
Density and owner-gradient accumulation share a traversal and checked geometry. Each
accumulator keeps its identity-ordered additions. Owner and prefix errors are deferred
until the complete density/constraint and reciprocal checks have finished, preserving
original diagnostic priority. The higher-identity squared-gradient suffix remains a
separate traversal; no geometry survives a change of position snapshot. Storage follows actual membership rather than an all-pairs matrix; dense
input can still require quadratic space and is never silently capped or truncated.

In the grouped Jacobi path, that traversal also performs active-distance validation.
Its first geometry error is tracked separately from density and multiplier errors,
even when an earlier numerical expression has already failed. Three numerical values
and their deferred error word are staged in candidate scratch. A following kernel
publishes them only after the entire validation pass has succeeded. Consequently a
distance error at a later particle still precedes any numerical error at an earlier
particle, and validation failure leaves density, constraint, multiplier and correction
outputs untouched. The separate validation and numerical calls retain their original
behaviour. Candidate scratch grows from 12 to 16 bytes per particle and is reused for
position and velocity candidates afterwards. This removes one complete neighbour
traversal without a per-neighbour cache, additional launch or host boundary.

The first attempt uploads accepted state and static planes and reserves fixed storage.
Later attempts reuse accepted device vectors. Acceleration and prediction emit compact block
summaries into finished correction scratch: checked speed maxima for CFL and coordinate extrema
for spatial bounds. Only these summaries return for host decisions; particle vectors remain
on device until final snapshot download. CFL lengths use the existing ordered binary32 dot
product and round-to-nearest square root. The host retains the shared limit/warning arithmetic.
Acceleration and CFL errors occupy separate diagnostic slots so stage priority is unchanged.
Failing lanes select the first particle's checked quantity with an integer atomic
minimum instead of returning the complete error array. Each Jacobi calculation and
candidate/projection pair is submitted as a stream-ordered group, with one result
slot per pass. A kernel skips work if an earlier pass failed; later slots cannot
replace the earlier error. Device records remain the ordering authority. Winning
error minima also update a private 32-byte mapped host packet; successful lanes
never access host memory. The CPU reads that packet only after the existing
stream completion, retaining stage order without a separate CUDA readback.
CPU resets occur after previous checked calls completed. GPU-only atomics select
the minima; the host never polls or competes with those updates. The device
allocation gains an eight-byte mapped address, uploaded during context setup.
The context owns both allocations and releases them at destruction, including
when failed completion leaves access uncertain. No borrowed host pointer is
retained by submitted work.
Successful passes leave their sentinels untouched; the four-slot record is reset for the first
attempt and after a numerical failure. Resets and unchecked kernel calls submit work without
waiting. The bridge tracks pending work on the private stream: diagnostic observations wait
after submission, while host transfers and growing reservations wait before accessing storage.
Reservations within capacity and empty transfers require no driver call. Synchronous host
copies finish their borrows before returning and add no empty stream wait. Numerical checks and their stage,
particle and expression ordering remain unchanged. Final state still returns for inspection.
Cell generation, spatial sorting and neighbour counting run in order on the private stream,
with one completion at the checked count. Spatial sorting first orders block-sized tiles in
shared memory, then merges adjacent runs through two equally sized cell buffers. The final
merge always lands at the original cell-array address, without a copy-back pass. Ordering
uses all three signed 64-bit coordinates and the particle-index tie-breaker; padding sentinels
are retained. A separate sorting entry point keeps its shared-memory allocation out of the
numerical kernels. At block size 128 this takes seven sorting launches for 8,000 particles
and eleven for 110,592, replacing 91 and 153 global bitonic passes. The second buffer costs
32 bytes per padded entry (256 KiB and 4 MiB respectively), retained across timesteps. Their kernels and dependencies are
unchanged. Immediate submission failure stops the batch and drains outstanding work;
completion faults take precedence and invalidate the context before returning. An unchecked
submission can report success before a device fault is observed; the next diagnostic or
host access reports that completion failure. Failed completion prevents copies, allocation
replacement and acceptance, retaining uncertain storage until context destruction.
Whole-stage Jacobi dependencies and ordered plane projection remain intact. The host
validates a complete step and passes its final cancellation poll before exchanging accepted
and working device vectors and installing the host snapshot. Promotion makes no driver calls.
Scratch never becomes accepted state merely because a kernel finished. Host snapshots remain
current after every accepted step, with the same identity, finiteness and plane checks before
acceptance. Final position/velocity downloads remain necessary for the borrowed, infallible
snapshot API; this change does not introduce lazy snapshots or thread-shared device storage.

Ordered binary32 sums, denormals, round-to-nearest division/square root and disabled
contraction preserve the supported arithmetic. Native power alternatives diverged on
the retained trajectories. The narrow positive-base/exponent implementation in
`native/power.cuh` follows the tested host approximation using binary64 intermediates;
its Arm-derived code retains the MIT licence in `native/power-LICENSE`.
For exponent four, the fast path reconstructs the retained binary64 quartic and its
binary32 rounding with integer products. A binary32 input has a 24-bit significand:
its square fits 48 bits and its fourth power fits 96. The high and low halves of
that integer product reproduce the retained round-to-nearest 53-bit significand,
then the same 24-bit result and rounding-boundary guard. The fast path's acceptance
set and accepted output bits are unchanged; the remaining inputs still use the
retained approximation. No position-correction accumulation or error check changes.
Inputs near a rounding midpoint, subnormal or extreme outputs, and other exponents
use the retained approximation. This preserves its bits rather than substituting a
different `powf` result. `native/power_exhaustive.cu` compares every non-negative
binary32 word with the reference and demonstrates why an unguarded quartic differs.
It also checks every guard decision and accepted value against the retained
binary64 implementation.
This does not promise agreement with every libm or permit blanket fast-math.

Enabled XSPH uses a separate complete velocity pass and a subsequent replacement pass. Rows read
final projected positions and the immutable reconstructed velocity snapshot; no thread reads another
thread's partially smoothed output. The candidate buffer is reused after projection. Disabled or
zero strength launches neither pass. The [input guide](inputs.md#fluid-motion-controls) describes
coefficient units and the effect of timestep refinement.

Cancellation is cooperative between completed stages. Numerical failure or cancellation
retains old accepted host/device state; retry reuses those device inputs and rebuilds membership. Input rejection
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
and device memory diagnostics establish different things. Compute
Sanitizer results must identify the actual tool and checks run; unavailable
diagnostics and passing host tests do not establish device-memory coverage.

Measured step totals include required transfers, launches, synchronisation and atomic
acceptance. Context/module setup and teardown are separate; audits, output, graphics
and pacing are outside solver timing. Coarse host stage intervals do not isolate
kernel time or prove which overhead dominates. See [measurement use](usage.md#solver-measurements).

Optional vorticity confinement runs after velocity reconstruction and before XSPH. A complete curl
pass reuses finished Jacobi scratch; a second pass computes confinement into candidate velocities,
then a separate pass replaces the velocity snapshot. XSPH also uses separate candidate and replacement
passes. Disabled or zero-strength effects launch none of their passes. Both effects retain ordered
fixed neighbour rows and strict final-distance support; no extra discovery or device allocation is
introduced. Strengths and numerical revisions are part of recorded physical inputs.

The exhaustive power check is a standalone CUDA test, separate from the Driver API
solver and its production dependencies. It uses the toolkit's Runtime API only for
test setup, launches and counters. Build it with the solver's precise flags; its
optional argument selects 32, 64, 128 or 256 threads:

```sh
/usr/local/cuda/bin/nvcc --std=c++17 --fmad=false --ftz=false --prec-div=true \
  --prec-sqrt=true -O3 --gpu-architecture=compute_89 native/power_exhaustive.cu \
  -o /tmp/maelstrom-power-exhaustive
/tmp/maelstrom-power-exhaustive 128
```

Final plane containment is checked alongside velocity reconstruction, using the same
binary32 dot-product and subtraction order as the host predicate. A private 16-byte
trailer in each velocity allocation contains an aligned earliest particle/plane
failure record. Projection resets it before reconstruction; velocity effects touch
only the numerical vector prefix. The existing final velocity download returns
the trailer too, adding 16 bytes without another launch, transfer or stream wait.
Both allocations retain their trailer when accepted and working buffers swap.

The host still validates particle identities and finite state before acceptance.
A failed containment certificate runs the original host particle/plane loop at its
original boundary, preserving numerical errors, first-failure ordering and the
priority of earlier velocity-effect failures. Successful certificates avoid that
second traversal. Cancellation polling and accepted-state promotion are unchanged.
The empty-particle case allocates and reads no trailer.
