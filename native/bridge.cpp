// Each handle owns an isolated driver context. Calls push/pop that context so
// independent handles never reset a shared primary context or steal its stream.
#include "probe.h"
#include <cuda.h>
#include <dlfcn.h>
#include <limits>
#include <new>

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
    if (!status.code && (code = d.cuStreamCreate_fn(&r.stream, CU_STREAM_NON_BLOCKING)) != CUDA_SUCCESS) status = {code, 6};
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
