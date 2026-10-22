// Each handle owns an isolated driver context. Calls push/pop that context so
// independent handles never reset a shared primary context or steal its stream.
#include "probe.h"
#include "solver.h"
#include <cuda.h>
#include <dlfcn.h>
#include <limits>
#include <new>
#include <cmath>

#if CUDA_VERSION < 13000
#error "The CUDA bridge requires CUDA 13 headers; see docs/cuda.md"
#endif

// Resolve the system driver at runtime: CUDA-enabled binaries can still report
// an actionable error on machines without libcuda, rather than failing to load.
#define DRIVER_FUNCTIONS(X) \
    X(cuInit, "cuInit") X(cuDeviceGetCount, "cuDeviceGetCount") \
    X(cuDeviceGet, "cuDeviceGet") X(cuDeviceGetName, "cuDeviceGetName") \
    X(cuDeviceGetAttribute, "cuDeviceGetAttribute") \
    X(cuDeviceTotalMem, "cuDeviceTotalMem_v2") X(cuDriverGetVersion, "cuDriverGetVersion") \
    X(cuCtxCreate, "cuCtxCreate_v4") X(cuCtxDestroy, "cuCtxDestroy_v2") \
    X(cuCtxPushCurrent, "cuCtxPushCurrent_v2") X(cuCtxPopCurrent, "cuCtxPopCurrent_v2") \
    X(cuModuleLoadData, "cuModuleLoadData") X(cuModuleGetFunction, "cuModuleGetFunction") \
    X(cuStreamCreate, "cuStreamCreate") X(cuStreamSynchronize, "cuStreamSynchronize") \
    X(cuMemAlloc, "cuMemAlloc_v2") X(cuMemFree, "cuMemFree_v2") \
    X(cuMemcpyHtoD, "cuMemcpyHtoD_v2") X(cuMemcpyDtoH, "cuMemcpyDtoH_v2") \
    X(cuLaunchKernel, "cuLaunchKernel")

struct Driver {
    void *library = nullptr;
#define DECLARE(name, symbol) decltype(&name) name##_fn = nullptr;
    DRIVER_FUNCTIONS(DECLARE)
#undef DECLARE
    Driver() = default;
    Driver(const Driver &) = delete;
    Driver &operator=(const Driver &) = delete;
    ~Driver() { if (library) dlclose(library); }
    Status load() {
        library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library) return {-1, 1};
#define RESOLVE(name, symbol) \
        name##_fn = reinterpret_cast<decltype(name##_fn)>(dlsym(library, symbol)); \
        if (!name##_fn) return {-2, 1};
        DRIVER_FUNCTIONS(RESOLVE)
#undef RESOLVE
        return {0, 0};
    }
};

struct Resource {
    Driver driver;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUfunction solver = nullptr;
    CUdeviceptr slots[14]{};
    std::size_t capacities[14]{};
    CUdeviceptr buffer = 0;
    bool poisoned = false;
    ~Resource() {
        // Context destruction releases its module, stream and device storage,
        // including allocations retained after an uncertain completion failure.
        if (context) driver.cuCtxDestroy_fn(context);
    }
};

// Stage codes are the C ABI shared with src/cuda/native.rs. A failure before
// context creation publishes no handle. After creation, destruction owns cleanup.
static Status initialise(Resource &r, int ordinal, const char *ptx,
                         int target, DeviceInfo *info) {
    auto &d = r.driver;
    CUresult code;
#define CHECK(call, stage) if ((code = (call)) != CUDA_SUCCESS) return {code, stage}
    CHECK(d.cuInit_fn(0), 2);
    int count = 0;
    CHECK(d.cuDeviceGetCount_fn(&count), 3);
    if (ordinal < 0 || ordinal >= count) return {CUDA_ERROR_INVALID_DEVICE, 3};
    CUdevice device;
    CHECK(d.cuDeviceGet_fn(&device, ordinal), 3);
    CHECK(d.cuDeviceGetName_fn(info->name, sizeof(info->name), device), 3);
    CHECK(d.cuDeviceGetAttribute_fn(&info->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), 3);
    CHECK(d.cuDeviceGetAttribute_fn(&info->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), 3);
    std::size_t memory = 0;
    CHECK(d.cuDeviceTotalMem_fn(&memory, device), 3);
    info->memory_bytes = memory;
    CHECK(d.cuDriverGetVersion_fn(&info->driver_version), 3);
    if (info->major * 10 + info->minor < target) return {CUDA_ERROR_NO_BINARY_FOR_GPU, 3};
    CHECK(d.cuCtxCreate_fn(&r.context, nullptr, 0, device), 4);
    // Creation pushes the new context; always restore the previous one, even
    // when module loading or stream creation fails.
    Status status{0, 0};
    if ((code = d.cuModuleLoadData_fn(&r.module, ptx)) != CUDA_SUCCESS) status = {code, 5};
    if (!status.code && (code = d.cuModuleGetFunction_fn(&r.kernel, r.module, "roundtrip")) != CUDA_SUCCESS) status = {code, 5};
    if (!status.code && (code = d.cuModuleGetFunction_fn(&r.solver, r.module, "solver")) != CUDA_SUCCESS) status = {code, 5};
    // Pageable uploads may return before DMA finishes on the legacy stream.
    // This private stream must inherit that dependency before reading the data.
    if (!status.code && (code = d.cuStreamCreate_fn(&r.stream, CU_STREAM_DEFAULT)) != CUDA_SUCCESS) status = {code, 6};
    CUcontext previous;
    code = d.cuCtxPopCurrent_fn(&previous);
    if (!status.code && code != CUDA_SUCCESS) status = {code, 13};
    return status;
}

extern "C" Status maelstrom_cuda_open(int ordinal, const char *ptx,
                                      int target, DeviceInfo *info, void **out) {
    *out = nullptr;
    auto *r = new (std::nothrow) Resource;
    if (!r) return {CUDA_ERROR_OUT_OF_MEMORY, 4};
    Status status = r->driver.load();
    if (!status.code) status = initialise(*r, ordinal, ptx, target, info);
    if (status.code) { delete r; return status; }
    *out = r;
    return {0, 0};
}

static Status roundtrip(Resource &r, const Probe *input, Probe *output, std::size_t count) {
    if (r.poisoned) return {-3, 14};
    // Defend the native trust boundary too; the Rust caller checks independently.
    if (count > std::numeric_limits<unsigned int>::max() ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(Probe)) return {-4, 7};
    if (!count) return {0, 0};
    if (!input || !output) return {CUDA_ERROR_INVALID_VALUE, 7};
    auto &d = r.driver;
    auto code = d.cuCtxPushCurrent_fn(r.context);
    if (code != CUDA_SUCCESS) { r.poisoned = true; return {code, 13}; }
    const auto bytes = count * sizeof(Probe);
    Status status{0, 0};
    CUdeviceptr allocation = 0;
    code = d.cuMemAlloc_fn(&allocation, bytes);
    if (code != CUDA_SUCCESS) status = {code, 7};
    else r.buffer = allocation;
    if (!status.code && (code = d.cuMemcpyHtoD_fn(r.buffer, input, bytes)) != CUDA_SUCCESS) status = {code, 8};
    if (!status.code) {
        auto length = static_cast<unsigned int>(count);
        void *args[] = {&r.buffer, &length};
        // 128 lanes are supported by the declared target; the quotient is
        // bounded by u32 input count and fits the device grid-x limit.
        code = d.cuLaunchKernel_fn(r.kernel, static_cast<unsigned int>((count + 127) / 128),
                                   1, 1, 128, 1, 1, 0, r.stream, args, nullptr);
        if (code != CUDA_SUCCESS) status = {code, 9};
    }
    // Submission and completion are distinct errors. Always drain a submitted
    // stream before touching storage, including after an immediate launch error.
    if (!status.code || status.stage == 9) {
        code = d.cuStreamSynchronize_fn(r.stream);
        if (code != CUDA_SUCCESS) { status = {code, 10}; r.poisoned = true; }
    }
    if (!status.code && (code = d.cuMemcpyDtoH_fn(output, r.buffer, bytes)) != CUDA_SUCCESS) status = {code, 11};
    // Only allocation exhaustion is recoverable. Other driver errors may
    // report older asynchronous faults; never reuse uncertain device state.
    if (status.code && !(status.stage == 7 && status.code == CUDA_ERROR_OUT_OF_MEMORY)) r.poisoned = true;
    if (r.buffer && !r.poisoned) {
        code = d.cuMemFree_fn(r.buffer);
        if (code != CUDA_SUCCESS) { status = {code, 12}; r.poisoned = true; }
        else r.buffer = 0;
    }
    CUcontext previous;
    code = d.cuCtxPopCurrent_fn(&previous);
    if (code != CUDA_SUCCESS) { status = {code, 13}; r.poisoned = true; }
    return status;
}

extern "C" Status maelstrom_cuda_roundtrip(void *handle, const Probe *input,
                                           Probe *output, std::size_t count) {
    return roundtrip(*static_cast<Resource *>(handle), input, output, count);
}

extern "C" Status maelstrom_cuda_close(void *handle) {
    auto *r = static_cast<Resource *>(handle);
    // Destroying an isolated context is the final abandonment boundary. No
    // borrowed host storage is outstanding: transfers here are synchronous.
    const auto code = r->driver.cuCtxDestroy_fn(r->context);
    r->context = nullptr;
    delete r;
    return {code, code == CUDA_SUCCESS ? 0 : 15};
}

// Operations finish their stream before returning. Only allocation exhaustion
// permits driver-resource retry; partial solver output remains private scratch.
template<class Operation> static Status solver_call(Resource &r, Operation operation) {
    if (r.poisoned) return {-3,14};
    auto &d=r.driver;
    auto code=d.cuCtxPushCurrent_fn(r.context);
    if (code!=CUDA_SUCCESS) { r.poisoned=true; return {code,13}; }
    Status status=operation();
    code=d.cuStreamSynchronize_fn(r.stream);
    if (code!=CUDA_SUCCESS) status={code,10};
    if (status.code && !(status.stage==7 && status.code==CUDA_ERROR_OUT_OF_MEMORY)) r.poisoned=true;
    CUcontext previous;
    code=d.cuCtxPopCurrent_fn(&previous);
    if (code!=CUDA_SUCCESS) { status={code,13}; r.poisoned=true; }
    return status;
}

extern "C" Status maelstrom_cuda_storage(void *handle, unsigned int slot,
                                          void *host, std::size_t bytes, unsigned int action) {
    auto &r=*static_cast<Resource *>(handle);
    if (r.poisoned) return {-3,14};
    if (slot>=14 || action>2 || bytes>static_cast<std::size_t>(PTRDIFF_MAX) ||
        (action && bytes && !host)) return {CUDA_ERROR_INVALID_VALUE,7};
    if (action && bytes>r.capacities[slot]) return {CUDA_ERROR_INVALID_VALUE,7};
    return solver_call(r,[&]() -> Status {
        auto &d=r.driver;
        CUresult code=CUDA_SUCCESS;
        if (!action && bytes>r.capacities[slot]) {
            CUdeviceptr replacement=0;
            code=d.cuMemAlloc_fn(&replacement,bytes);
            if (code!=CUDA_SUCCESS) return {code,7};
            if (r.slots[slot]) {
                code=d.cuMemFree_fn(r.slots[slot]);
                if (code!=CUDA_SUCCESS) return {code,12};
            }
            r.slots[slot]=replacement;
            r.capacities[slot]=bytes;
        } else if (action==1 && bytes) code=d.cuMemcpyHtoD_fn(r.slots[slot],host,bytes);
        else if (action==2 && bytes) code=d.cuMemcpyDtoH_fn(host,r.slots[slot],bytes);
        return {code,code==CUDA_SUCCESS ? 0 : (action==1 ? 8 : 11)};
    });
}

extern "C" Status maelstrom_cuda_solver(void *handle, SolverSettings settings,
                                         unsigned int operation, unsigned int threads) {
    auto &r=*static_cast<Resource *>(handle);
    if (r.poisoned) return {-3,14};
    if (operation>13 || ((operation==2 || operation==4 || operation==12) && (!std::isfinite(settings.width) || settings.width<=0)) ||
        (threads!=32 && threads!=64 && threads!=128 && threads!=256) ||
        settings.padded==0 || (settings.padded&(settings.padded-1)) ||
        settings.count>settings.padded || settings.entries>static_cast<std::uint64_t>(PTRDIFF_MAX)/4 ||
        (operation==3 && (!settings.stride || settings.stride>=settings.span ||
        (settings.stride&(settings.stride-1)) || (settings.span&(settings.span-1)) || settings.span>settings.padded)))
        return {CUDA_ERROR_INVALID_VALUE,7};
    std::uint64_t n=settings.count;
    std::uint64_t required[14]={n*12,n*12,n*12,n*12,n*4,n*4,n*4,n*12,n*12,
        static_cast<std::uint64_t>(settings.planes)*16,static_cast<std::uint64_t>(settings.padded)*32,
        settings.entries*4,n*4,(n+1)*8};
    for (unsigned int slot=0;slot<14;++slot) if (required[slot]>r.capacities[slot]) return {CUDA_ERROR_INVALID_VALUE,7};
    if (!settings.count) return {0,0};
    return solver_call(r,[&]() -> Status {
        SolverBuffers buffers;
        static_assert(sizeof(buffers)==sizeof(r.slots));
        // CUdeviceptr holds the 64-bit address representation used by embedded PTX.
        __builtin_memcpy(&buffers,r.slots,sizeof(buffers));
        const bool sort=operation==13;
        unsigned int kernel_operation=sort ? 3 : operation;
        void *args[]={&buffers,&settings,&kernel_operation};
        std::uint64_t count=(operation==2 || kernel_operation==3) ? settings.padded : settings.count;
        auto launch=[&]() -> Status {
            auto code=r.driver.cuLaunchKernel_fn(r.solver,static_cast<unsigned int>((count+threads-1)/threads),
                1,1,threads,1,1,0,r.stream,args,nullptr);
            return {code,code==CUDA_SUCCESS ? 0 : 9};
        };
        if (!sort) return launch();
        // These integer-only passes have no host observations or cancellation
        // polls between them. Stream order supplies their global dependency;
        // solver_call drains the batch even after an immediate launch failure.
        for (std::uint64_t span=2;span<=settings.padded;span*=2) {
            settings.span=static_cast<unsigned int>(span);
            for (settings.stride=settings.span/2;settings.stride;settings.stride/=2) {
                auto status=launch();
                if (status.code) return status;
            }
        }
        return {0,0};
    });
}
