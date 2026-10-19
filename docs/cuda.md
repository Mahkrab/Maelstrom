# Building and running the CUDA solver

CUDA is an optional execution resource. `run --backend uniform-grid --execution cuda` executes
discovery, acceleration, prediction, every Jacobi pass, projection and velocity reconstruction on
the selected GPU. The host coordinates stages, checks diagnostics and publishes accepted state.
`cuda-check` remains a separate layout probe. CPU selection and historical replay retain their meaning.

## Prerequisites and build

CPU builds need stable Rust/Cargo and the locked dependencies. They do not inspect CUDA settings,
invoke a native compiler, link GPU libraries or require a device:

```sh
cargo build --release --locked --offline --no-default-features
cargo test --locked --offline --no-default-features
```

For CUDA, use native x86_64 Linux (including WSL2), CUDA 13 headers and `nvcc`, a toolkit-supported
host compiler, a C++17 compiler with libstdc++, and `ar`. Cross-compilation and other platforms are
rejected. The qualified combination is Ubuntu 24.04.5/WSL2, Rust/Cargo 1.96.0, CUDA 13.3.73 and
GCC/G++ 13.3.0. The primary PC has an RTX 4070, compute capability 8.9, with Windows driver 610.88
exposing driver API version 13030. These are tested versions, not auto-installed tools. Offline
builds require a populated Cargo cache; `cargo fetch --locked` supplies missing Rust packages.

```sh
CUDA_HOME=/usr/local/cuda MAELSTROM_CUDA_ARCH=89 cargo build --release --locked --offline --features cuda
./target/release/maelstrom cuda-check --device 0 --repetitions 8 --format json
```

`CUDA_HOME` defaults to `/usr/local/cuda` and must be an absolute toolkit directory containing
`bin/nvcc` and `include/cuda.h`. It is independent of the shell PATH. `MAELSTROM_CUDA_ARCH` is the
numeric virtual compute capability, default `89`; nvcc rejects unsupported targets. `CXX` and `AR`
select compiler/archive executables, without embedded arguments. The nvcc host compiler uses the
toolkit's ordinary discovery. No build step downloads native tools or changes the driver.

With the feature enabled, Cargo's build script compiles `native/kernels.cu` to PTX using
`--ptx --std=c++17 --gpu-architecture=compute_89 -O3 --fmad=false --ftz=false --prec-div=true
--prec-sqrt=true` (substitute the selected architecture). The bridge uses
`-std=c++17 -O2 -fPIC -Wall -Wextra -Werror`. Cargo embeds the NUL-terminated PTX and links the local
bridge archive, libstdc++ and libdl. There are no new Cargo dependencies or default features.

The bridge loads the system `libcuda.so.1` at runtime, resolves required Driver API symbols, and
loads embedded PTX into an explicitly owned context. It uses `cuCtxCreate_v4` and requires a
CUDA 13-compatible driver that understands the toolkit's emitted PTX. The device must meet the
selected virtual capability. Missing libraries/symbols, device visibility, incompatible capability
and PTX loading failures are explicit errors; driver JIT compilation belongs to context setup.
Do not put toolkit stub libraries on the runtime library path. Under WSL, use the Windows driver
exposed to Linux; installing or changing drivers is outside the build process.

## Solver storage, arithmetic and completion

```sh
./target/release/maelstrom run falling-block-04 --backend uniform-grid --execution cuda \
  --device 0 --block-size 128 --refine-timestep 4 --steps 480 --checkpoints 0,1,120,240,480 \
  --format json --output cuda-run.json
./target/release/maelstrom replay cuda-run.json --format json --output cuda-replay.json
```

Device zero and 128 lanes per block are defaults. Blocks of 32, 64, 128 and 256 lanes are supported;
CUDA rejects CPU worker settings and brute-force discovery. Each solver owns a context and persistent
scratch buffers. Identity stays in the host's sorted u64 particle records; device indices refer to
that immutable order. Separate packed binary32 position/velocity arrays use 12 bytes per vector,
planes 16 bytes and cell entries 32 bytes, with layout assertions on both sides of the interface.

Each attempt uploads the last accepted positions, velocities and ordered planes. Acceleration
velocities return for the shared host CFL check; predictions return for the conservative bounding
diagonal and cell-width calculation. The GPU constructs and bitonically sorts occupied-cell entries,
then searches all 27 adjacent cells. Power-of-two widths are at least the radius and `2^-72`, and
bound indices by `2^50`, covering binary32 distance underflow and negative boundaries. An unsafe
bounding diagonal requests reference-ordered distant-pair validation on the device. Strict binary32
distance alone admits membership. Cells and indices determine sort order independently of scheduling.

Fixed prediction membership uses `N * ceil(N/32)` u32 words. There is no occupancy or neighbour-row
cap. Each lane exclusively owns one row and reads set bits in ascending identity order; self is
excluded from membership and inserted once at its ordered position for density and squared-direction
sums. Every Jacobi pass filters current distances against the same fixed rows. Dense input retains
all neighbours; the quadratic membership storage remains a deliberate scalability limitation
(8,000 particles use 8,000,000 bytes for these words). Counts, padding, multiplication, slot capacity
and allocation failures are checked before access. Failed allocation cannot publish a truncated step.

Kernel launches on one stream separate active-distance checks, complete multipliers, corrections,
physical-density checks, candidate formation and projection. Each launch is synchronised before
its errors are read. Per-particle error slots are inspected in serial stage/particle order;
threads do not race to publish an error. Ordered plane projection retains exact axis contact and
velocity reconstruction includes the complete corrected displacement.

Binary32 operations retain ordered sums, round-to-nearest division/square root, denormals and
disabled contraction. No fast-math is enabled. The artificial-pressure power uses binary64
intermediates and one binary32 return, as the host library does internally. Native CUDA `powf`
and a binary64 library-power alternative both caused trajectory disagreement, so `native/power.cuh`
adapts the positive-base, positive-finite-exponent path of
[Arm optimized-routines at f67e7daa5457646fad9f75580263dba654c988aa](https://github.com/ARM-software/optimized-routines/tree/f67e7daa5457646fad9f75580263dba654c988aa/math).
Only its power approximation and tables are retained, with the MIT licence in `native/power-LICENSE`;
no runtime library or package was added. Zero/subnormal bases and overflow/underflow remain explicit.
This choice reproduces the tested host approximation, including rounding cases where a more accurate
power would differ. It does not promise bitwise agreement with every host libm or device.

Working positions and multipliers stay resident through the iterative solver. Error slots return
after checked stages; final positions and velocities return before host state/plane validation and
the last cancellation poll. The host accepted snapshot is the sole commit authority. Cancellation
polls match CPU stage boundaries, after pending work finishes; a panic or error leaves the old state
available. Retry uploads that state and rebuilds membership, discarding stale scratch. Driver errors
follow the invalidation policy below; poisoned resources require a new solver. Recovery from actual
device loss is unqualified, with no reset or automatic retry.

CUDA records use `cuda-uniform-grid-pbf-binary32-<iterations>-jacobi-v1`, with `four` for four passes.
Execution settings contain device ordinal and lanes; metadata adds GPU name, memory, capability,
driver API and embedded target. Run identity includes ordinal, lanes and target; physical input identity
does not. `replay_options` and the public CLI restore CUDA settings, while older replay helpers reject
execution settings they cannot retain. Replay does not certify a record or promise the same hardware
at an ordinal on another machine.

Ordinary, total-only and stage-profile paths share numerical work. Timers include each completed
operation, scratch allocation/growth, transfers, synchronisation, validation and host acceptance.
Stage values are coordinated host elapsed intervals, never summed worker clocks or isolated device
kernel times. Context/module setup and context destruction are outside step totals; fresh brute-force
density audits, snapshot recording and publication remain outside. These boundaries describe
instrumentation and do not establish a performance comparison. Measured records also expose
`solver_construction_s` and `solver_teardown_s` outside the step totals. Construction includes
validated host state, context/module loading (and any driver JIT) and stream setup as one combined
cost. Persistent device buffers are first reserved inside the first step. Teardown consumes the
solver and releases its context and buffers; ordinary unmeasured records retain their existing shape.

## Ownership and failure contract

`cuda::CudaContext::new(ordinal)` selects a process-visible device ordinal. It owns a separate
driver context, one module and a non-blocking stream. It is neither `Send` nor `Sync`; operations
push its context and restore the caller's previous context. Interleaved resources remain independent.
There is no shared primary-context reset, global stream or automatic device/CPU selection.

`roundtrip(&[Probe])` accepts immutable records and returns independent output only after allocation,
synchronous upload, kernel submission, stream completion, synchronous download and buffer release
succeed. Host/device storage uses a checked 24-byte, eight-byte-aligned C layout: u64 identity at
offset 0, three binary32 coordinates at offset 8, u32 marker at offset 20. Rust and CUDA/C++ assert
these properties. No simulator struct or Rust collection crosses the ABI. The count fits u32, byte
arithmetic fits usize/isize, and the kernel checks its widened lane index against the count. Each
block has 128 lanes; empty input performs no allocation, transfer or launch.

The probe maps coordinates to `(x + 1, y * -2, z - 0.5)`, XORs the marker with `0xa5a55a5a` and leaves
identity unchanged. This is diagnostic arithmetic, with no fluid numerical acceptance policy.
`cuda-check` verifies 257 records against fixed independent values over independent context lifetimes.
Its JSON contains selected device, memory, driver API version, target and verification count.

`CudaError` distinguishes unavailable features/devices, capacity, host allocation, verification,
invalidated resources and driver failures by `CudaStage` plus native code. Immediate launch and
completion failures have separate stages. A submitted stream is drained even after an immediate
launch error; a completion fault takes precedence if draining also fails. Input/host-allocation
rejection and device-allocation exhaustion allow retry; all other driver failures invalidate the
resource. Subsequent requests return `Poisoned`, including empty requests. Partial output is discarded.
Storage retained after uncertain completion stays owned until context destruction; no asynchronous
operation borrows host memory on return.

`close(self)` consumes the resource and reports context-destruction errors. `Drop` attempts cleanup
but cannot report errors. The context releases its module, stream and outstanding allocations.
Teardown failure consumes the handle; recovery from real device loss is unqualified. No process-wide
reset or unsafe retry is attempted. `cuda-check` on a CPU-only build exits 3 with rebuild instructions.
Invalid arguments exit 2; device/driver/probe failures exit 3 with stderr and no success document.
No CPU result is substituted.

## Verification

CPU-only tests exercise disabled-feature and invalid-request behaviour without a toolkit or device.
Feature-enabled host tests need no GPU. The hardware test must be explicitly selected and fails
when the requested device is absent:

```sh
CUDA_HOME=/usr/local/cuda cargo test --features cuda --locked --offline
CUDA_HOME=/usr/local/cuda cargo test --features cuda --locked --offline --test cuda_boundary -- --include-ignored
CUDA_HOME=/usr/local/cuda cargo test --release --features cuda --locked --offline --test cuda_boundary -- --include-ignored
CUDA_HOME=/usr/local/cuda cargo test --release --features cuda --locked --offline \
  --lib --test cuda_execution -- --include-ignored --test-threads 1
c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined -g \
  -I /usr/local/cuda/include native/tests.cpp -ldl -o /tmp/maelstrom-native-tests
/tmp/maelstrom-native-tests
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode 90 \
  ./target/release/maelstrom cuda-check --repetitions 8
```

The native executable injects failures through private driver function pointers, with no production
fault switches. It checks partial construction, context restoration, allocation retry, transfer,
submission/completion separation, invalidation, cleanup and consuming teardown errors. ASan/UBSan
check these host ownership paths; they do not establish driver-fault recovery. Hardware tests check
exact identities/markers, signed zero, empty/single and uneven launches, repeated calls and simultaneous
resource lifetimes. The PC's Compute Sanitizer currently fails to initialise its WDDM debugger
interface; its attempted run is not a passing device memory diagnostic.

Ownership uses NVIDIA's [context API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/group__CUDA__CTX.html),
[memory API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/group__CUDA__MEM.html)
and [stream API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/group__CUDA__STREAM.html).
Flags follow the [nvcc reference](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-compiler-driver-nvcc/index.html).
