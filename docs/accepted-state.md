# CUDA host acceptance storage

The CUDA backend retains a complete synchronous host snapshot. `Simulation::snapshot`
and `Snapshot::particles` remain infallible borrows; cloning still creates an independent
checkpoint. Worker observations, headless clients, export and the viewer consume those
host particles. Rendering converts positions to camera-relative centres and reads
velocities for energy colours, then uploads its own wgpu buffer; worker observation
clones and render packing stay outside solver step clocks. No CUDA operation is deferred into a snapshot query.

`CudaSolver` owns reusable position (`[f32; 3]`) and velocity (`u32`) staging vectors.
The velocity allocation includes the existing four-word containment trailer. Driver-only
`cuMemcpyDtoH` completes before either vector is read or borrowed storage can move.
The existing HostAccess boundary drains pending stream work before storage access.
Checked velocity operations normally already completed it. No stream/event/ABI/kernel
changes, extra observations, pinned allocations or new synchronization are introduced.

A private `host_particles` vector holds the spare snapshot allocation. Packing reads
staging by reference, copies the existing identities and preserves float bits. After
packing, ownership moves into `CudaStepOutput`; Simulation runs the same ordered host
validation, containment fallback and last cancellation poll. Device promotion remains
before the infallible host replacement. Only then is the previous snapshot's vector
returned to the solver. The public snapshot is never used as writable scratch.

There are two particle allocations after successful warm-up, alternating accepted/spare,
and two raw staging allocations. At fixed particle count, successful steady-state steps
neither allocate nor free these four vectors. CPU backend storage is unchanged. The
particle count is fixed by Simulation's public interface, so exact reservations are
bounded by its count; geometric growth is unnecessary here. Reusable download helpers
also handle growing/shrinking requests: reserve uses checked Vec arithmetic before
changing length, and capacity is only changed by successful allocation. Existing elements
are not initialised again at unchanged length. All types are plain Copy transfer types.

Allocation failure returns the existing HostAllocation error before commit. A transfer
failure may modify staging but cannot modify the current snapshot. Native uncertain-
completion poisoning, completion-error priority and resource lifetime are unchanged.
If validation/cancellation rejects a successfully materialised output, that private output
can be dropped; retry may allocate a new spare. Retention guarantees steady-state reuse
for successful steps, not zero allocation after arbitrary failures. No fallible operation
is added after device promotion. No background destruction/threading is used.

Build and test from this repository on a CUDA machine:

```sh
CUDA_HOME=/usr/local/cuda cargo clippy --locked --offline --features cuda --all-targets -- -D warnings
CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --lib cuda:: -- --include-ignored --test-threads=1
CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --test cuda_boundary --test cuda_execution -- --include-ignored
MAELSTROM_CUDA_SPATIAL_INDEX=sort CUDA_HOME=/usr/local/cuda cargo test --locked --offline --features cuda --lib cuda:: -- --include-ignored --test-threads=1
```

The focused host-storage checks are in native.rs and residency_tests.rs. The test-only
thread-local allocation hook fails a selected growing reservation without changing the
production allocator. Native Driver-injection tests continue to own transfer/completion,
poisoning, allocation recovery and failure-priority checks.

The EPQ `accepted-state-v1` evidence package owns measured attribution, independent
control identities, benchmark results, rejected alternatives and limitations. Diagnostic
instrumentation is applied only to isolated source exports; production has no timers,
allocator tracing or environment switches for this change. The complete-step benchmark
example hashes final state and diagnostic outcomes after timing. Its commands start
new simulations, using the same short evolving lattices as compact-grid qualification.
