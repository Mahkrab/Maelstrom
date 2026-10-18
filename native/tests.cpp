// Faults replace driver functions inside this test executable only. These
// assertions prove ownership transitions, not recovery from actual device loss.
#include "bridge.cpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

struct Observed {
    int fail = 0;
    int pushes = 0;
    int pops = 0;
    int destroys = 0;
    int frees = 0;
    int completions = 0;
    int downloads = 0;
    std::vector<Probe> memory;
} observed;

static CUresult fault(int stage) {
    return observed.fail == stage ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}

static void fake_driver(Driver &d) {
    d.cuInit_fn = [](unsigned int) { return fault(2); };
    d.cuDeviceGetCount_fn = [](int *count) { *count = 1; return CUDA_SUCCESS; };
    d.cuDeviceGet_fn = [](CUdevice *device, int) { *device = 0; return fault(3); };
    d.cuDeviceGetName_fn = [](char *name, int, CUdevice) { std::strcpy(name, "injected"); return CUDA_SUCCESS; };
    d.cuDeviceGetAttribute_fn = [](int *value, CUdevice_attribute attribute, CUdevice) {
        *value = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 8 : 9;
        return CUDA_SUCCESS;
    };
    d.cuDeviceTotalMem_fn = [](std::size_t *bytes, CUdevice) { *bytes = 1 << 20; return CUDA_SUCCESS; };
    d.cuDriverGetVersion_fn = [](int *version) { *version = 13030; return CUDA_SUCCESS; };
    d.cuCtxCreate_fn = [](CUcontext *context, CUctxCreateParams *, unsigned int, CUdevice) {
        if (observed.fail == 4) return CUDA_ERROR_OUT_OF_MEMORY;
        *context = reinterpret_cast<CUcontext>(1); return CUDA_SUCCESS;
    };
    d.cuCtxDestroy_fn = [](CUcontext) { ++observed.destroys; observed.memory.clear(); return fault(15); };
    d.cuCtxPushCurrent_fn = [](CUcontext) { ++observed.pushes; return fault(13); };
    d.cuCtxPopCurrent_fn = [](CUcontext *context) { ++observed.pops; *context = nullptr; return fault(16); };
    d.cuModuleLoadData_fn = [](CUmodule *module, const void *) { *module = reinterpret_cast<CUmodule>(1); return fault(5); };
    d.cuModuleGetFunction_fn = [](CUfunction *function, CUmodule, const char *) { *function = reinterpret_cast<CUfunction>(1); return CUDA_SUCCESS; };
    d.cuStreamCreate_fn = [](CUstream *stream, unsigned int) { *stream = reinterpret_cast<CUstream>(1); return fault(6); };
    d.cuMemAlloc_fn = [](CUdeviceptr *buffer, std::size_t bytes) {
        if (observed.fail == 7) return CUDA_ERROR_OUT_OF_MEMORY;
        observed.memory.resize(bytes / sizeof(Probe)); *buffer = 1; return CUDA_SUCCESS;
    };
    d.cuMemFree_fn = [](CUdeviceptr) { ++observed.frees; if (!observed.fail) observed.memory.clear(); return fault(12); };
    d.cuMemcpyHtoD_fn = [](CUdeviceptr, const void *data, std::size_t bytes) { std::memcpy(observed.memory.data(), data, bytes); return fault(8); };
    d.cuLaunchKernel_fn = [](CUfunction, unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int, unsigned int,
                            CUstream, void **, void **) { return fault(9); };
    d.cuStreamSynchronize_fn = [](CUstream) { ++observed.completions; return fault(10); };
    d.cuMemcpyDtoH_fn = [](void *data, CUdeviceptr, std::size_t bytes) { ++observed.downloads; std::memcpy(data, observed.memory.data(), bytes); return fault(11); };
}

static void ready(Resource &r) {
    fake_driver(r.driver);
    DeviceInfo info{};
    assert(initialise(r, 0, "unused", 89, &info).code == 0);
    assert(observed.pops == 1);
}

int main() {
    for (int stage : {2, 3, 4, 5, 6, 16}) {
        observed = {};
        {
            Resource r;
            fake_driver(r.driver);
            observed.fail = stage;
            DeviceInfo info{};
            const auto status = initialise(r, 0, "unused", 89, &info);
            assert(status.code != 0 && status.stage == (stage == 16 ? 13 : stage));
            assert(observed.pops == (stage >= 5 ? 1 : 0));
        }
        assert(observed.destroys == (stage >= 5 ? 1 : 0));
    }
    for (int stage : {8, 9, 10, 11, 12, 13, 16}) {
        observed = {};
        {
            Resource r;
            ready(r);
            observed.fail = stage;
            Probe input{7, {1, 2, 3}, 4}, output{};
            auto status = roundtrip(r, &input, &output, 1);
            assert(status.code != 0 && status.stage == (stage == 16 ? 13 : stage));
            assert(r.poisoned);
            assert(roundtrip(r, &input, &output, 1).stage == 14);
            assert(observed.downloads == (stage == 11 || stage == 12 || stage == 16 ? 1 : 0));
            assert(observed.completions == (stage == 8 || stage == 13 ? 0 : 1));
            assert(observed.pops == (stage == 13 ? 1 : 2));
        }
        assert(observed.destroys == 1);
    }
    observed = {};
    {
        Resource r;
        ready(r);
        Probe input{}, output{};
        assert(roundtrip(r, nullptr, nullptr, 0).code == 0);
        assert(roundtrip(r, nullptr, &output, 1).stage == 7 && !r.poisoned);
        assert(roundtrip(r, &input, &output, std::numeric_limits<std::size_t>::max()).code == -4);
        observed.fail = 7;
        assert(roundtrip(r, &input, &output, 1).stage == 7 && !r.poisoned);
        assert(observed.frees == 0 && observed.completions == 0);
        observed.fail = 0;
        assert(roundtrip(r, &input, &output, 1).code == 0);
        assert(observed.frees == 1 && observed.completions == 1);
    }
    assert(observed.destroys == 1);
    observed = {};
    {
        Resource r;
        fake_driver(r.driver);
        DeviceInfo info{};
        assert(initialise(r, -1, "unused", 89, &info).code == CUDA_ERROR_INVALID_DEVICE);
        assert(initialise(r, 1, "unused", 89, &info).code == CUDA_ERROR_INVALID_DEVICE);
        assert(initialise(r, 0, "unused", 90, &info).code == CUDA_ERROR_NO_BINARY_FOR_GPU);
        assert(observed.destroys == 0);
    }
    for (int stage : {0, 15}) {
        observed = {};
        auto *r = new Resource;
        ready(*r);
        observed.fail = stage;
        assert(maelstrom_cuda_close(r).stage == stage);
        assert(observed.destroys == 1);
    }
    std::puts("native lifecycle and injected failure checks passed");
}
