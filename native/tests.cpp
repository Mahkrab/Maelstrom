// Faults replace driver functions inside this test executable only. These
// assertions prove ownership transitions, not recovery from actual device loss.
#include "bridge.cpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>

struct Observed {
    int fail = 0;
    int pushes = 0;
    int pops = 0;
    int destroys = 0;
    int frees = 0;
    int completions = 0;
    int downloads = 0;
    bool record_solver = false;
    int fail_launch = 0;
    int resets = 0;
    bool fail_reset = false;
    struct Launch { unsigned int operation, stride, span, grid, block; };
    std::vector<Launch> launches;
    std::map<unsigned int,std::pair<unsigned int,std::uint32_t>> numerical_failures;
    CUdeviceptr next=1;
    std::map<CUdeviceptr,std::vector<unsigned char>> memory;
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
    d.cuModuleGetFunction_fn = [](CUfunction *function, CUmodule, const char *name) {
        *function = reinterpret_cast<CUfunction>(std::strcmp(name,"solver")==0 ? 2 : 1);
        return CUDA_SUCCESS;
    };
    d.cuStreamCreate_fn = [](CUstream *stream, unsigned int flags) {
        assert(flags==CU_STREAM_DEFAULT);
        *stream = reinterpret_cast<CUstream>(1); return fault(6);
    };
    d.cuMemAlloc_fn = [](CUdeviceptr *buffer, std::size_t bytes) {
        if (observed.fail == 7) return CUDA_ERROR_OUT_OF_MEMORY;
        *buffer=observed.next++; observed.memory[*buffer].resize(bytes); return CUDA_SUCCESS;
    };
    d.cuMemFree_fn = [](CUdeviceptr pointer) { ++observed.frees; if (!observed.fail) observed.memory.erase(pointer); return fault(12); };
    d.cuMemcpyHtoD_fn = [](CUdeviceptr pointer, const void *data, std::size_t bytes) { assert(bytes<=observed.memory.at(pointer).size());std::memcpy(observed.memory.at(pointer).data(), data, bytes); return fault(8); };
    d.cuMemsetD8Async_fn = [](CUdeviceptr pointer, unsigned char value, std::size_t bytes, CUstream) {
        ++observed.resets;
        if (observed.fail_reset) return CUDA_ERROR_UNKNOWN;
        assert(bytes==32 && value==0xff && observed.memory.at(pointer).size()==bytes);
        std::memset(observed.memory.at(pointer).data(),value,bytes);
        return CUDA_SUCCESS;
    };
    d.cuLaunchKernel_fn = [](CUfunction function, unsigned int grid, unsigned int, unsigned int,
                            unsigned int block, unsigned int, unsigned int, unsigned int,
                            CUstream, void **args, void **) {
        if (function==reinterpret_cast<CUfunction>(2)) {
            // This host stand-in supplies an independent serial selection oracle;
            // hardware tests exercise the actual competing atomic updates.
            auto buffers=*static_cast<SolverBuffers *>(args[0]);
            auto settings=*static_cast<SolverSettings *>(args[1]);
            auto operation=*static_cast<unsigned int *>(args[2]);
            if (buffers.diagnostic && observed.fail!=9) {
                auto pointer=reinterpret_cast<CUdeviceptr>(buffers.errors);
                auto output=reinterpret_cast<CUdeviceptr>(buffers.diagnostic);
                const auto pass=buffers.first_operation ? operation-buffers.first_operation : 0;
                auto &records=observed.memory.at(output);
                bool skipped=false;
                for (unsigned int earlier=0;earlier<pass;++earlier) {
                    std::uint64_t record;
                    std::memcpy(&record,records.data()+earlier*sizeof(record),sizeof(record));
                    skipped|=record!=UINT64_MAX;
                }
                std::uint64_t result;
                std::memcpy(&result,records.data()+pass*sizeof(result),sizeof(result));
                for (unsigned int i=0;!skipped && i<settings.count;++i) {
                    std::uint32_t code;
                    std::memcpy(&code,observed.memory.at(pointer).data()+i*sizeof(code),sizeof(code));
                    if (buffers.first_operation) {
                        const auto failure=observed.numerical_failures.find(operation);
                        code=failure!=observed.numerical_failures.end() && failure->second.first==i ? failure->second.second : 0;
                    }
                    if (code) { result=std::min(result,(static_cast<std::uint64_t>(i)<<32)|code);break; }
                }
                std::memcpy(records.data()+pass*sizeof(result),&result,sizeof(result));
            }
        }
        if (observed.record_solver) {
            auto settings=*static_cast<SolverSettings *>(args[1]);
            observed.launches.push_back({*static_cast<unsigned int *>(args[2]),
                settings.stride,settings.span,grid,block});
            if (static_cast<int>(observed.launches.size())==observed.fail_launch) return CUDA_ERROR_UNKNOWN;
        }
        return fault(9);
    };
    d.cuStreamSynchronize_fn = [](CUstream) { ++observed.completions; return fault(10); };
    d.cuMemcpyDtoH_fn = [](void *data, CUdeviceptr pointer, std::size_t bytes) { ++observed.downloads;assert(bytes<=observed.memory.at(pointer).size()); std::memcpy(data, observed.memory.at(pointer).data(), bytes); return fault(11); };
}

static void ready(Resource &r) {
    fake_driver(r.driver);
    DeviceInfo info{};
    assert(initialise(r, 0, "unused", 89, &info).code == 0);
    assert(observed.pops == 1);
}

static void sort_batches() {
    for (auto threads : {32U,64U,128U,256U}) {
        observed={};
        Resource r;ready(r);
        SolverSettings settings{};settings.count=5;settings.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,256,0).code==0);
        const auto before=observed.completions;
        observed.record_solver=true;
        // Deliberately invalid incoming pass fields must be replaced by the
        // complete network, rather than inherited from the caller.
        settings.stride=17;settings.span=19;
        assert(maelstrom_cuda_solver(&r,settings,13,threads).code==0);
        const unsigned int expected[][2]={{1,2},{2,4},{1,4},{4,8},{2,8},{1,8}};
        assert(observed.launches.size()==6 && observed.completions==before+1);
        for (unsigned int i=0;i<6;++i) {
            const auto &pass=observed.launches[i];
            assert(pass.operation==3 && pass.stride==expected[i][0] && pass.span==expected[i][1]);
            assert(pass.grid==1 && pass.block==threads);
        }
        observed.launches.clear();settings.count=1;settings.padded=1;
        assert(maelstrom_cuda_solver(&r,settings,13,threads).code==0);
        assert(observed.launches.empty());
        settings.count=0;
        const auto completions=observed.completions;
        assert(maelstrom_cuda_solver(&r,settings,13,threads).code==0);
        assert(observed.launches.empty() && observed.completions==completions);
        settings.padded=3;
        assert(maelstrom_cuda_solver(&r,settings,13,threads).stage==7 && !r.poisoned);
        settings.padded=1;
        assert(maelstrom_cuda_solver(&r,settings,14,threads).stage==7 && !r.poisoned);
    }
    for (auto failure : {9,10,13,16}) {
        observed={};
        Resource r;ready(r);
        SolverSettings settings{};settings.count=5;settings.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,256,0).code==0);
        observed.record_solver=true;
        observed.fail_launch=3;
        observed.fail=failure==9 ? 0 : failure;
        const auto completions=observed.completions,pops=observed.pops;
        auto status=maelstrom_cuda_solver(&r,settings,13,128);
        assert(status.stage==(failure==16 ? 13 : failure) && r.poisoned);
        assert(observed.launches.size()==(failure==13 ? 0 : 3));
        assert(observed.completions==completions+(failure==13 ? 0 : 1));
        assert(observed.pops==pops+(failure==13 ? 0 : 1));
        observed.fail=0;
        assert(maelstrom_cuda_solver(&r,settings,13,128).stage==14);
        assert(observed.launches.size()==(failure==13 ? 0 : 3));
    }
}

static void checked_groups() {
    for (auto first : {5U,9U}) for (auto threads : {32U,64U,128U,256U}) {
        const auto passes=first==5 ? 4U : 2U;
        for (unsigned int fail_at=0;fail_at<=passes;++fail_at) {
            observed={};Resource r;ready(r);
            SolverSettings s{};s.count=5;s.padded=8;
            for (unsigned int slot=0;slot<14;++slot)
                assert(maelstrom_cuda_storage(&r,slot,nullptr,256,0).code==0);
            assert(maelstrom_cuda_begin_attempt(&r).code==0);
            observed.record_solver=true;
            if (fail_at<passes) {
                observed.numerical_failures[first+fail_at]={4,22};
                if (fail_at+1<passes) observed.numerical_failures[first+fail_at+1]={0,1};
            }
            auto waits=observed.completions,copies=observed.downloads;
            std::uint64_t result=0;
            assert(maelstrom_cuda_checked_group(&r,s,first,threads,&result).code==0);
            assert(result==(fail_at==passes ? UINT64_MAX : (std::uint64_t{4}<<32)|22));
            assert(observed.completions==waits+1 && observed.downloads==copies+1);
            assert(observed.launches.size()==passes);
            for (unsigned int pass=0;pass<passes;++pass) {
                assert(observed.launches[pass].operation==first+pass && observed.launches[pass].block==threads);
                std::uint64_t record;
                std::memcpy(&record,observed.memory.at(r.diagnostic).data()+pass*sizeof(record),sizeof(record));
                assert(record==(pass==fail_at ? (std::uint64_t{4}<<32)|22 : UINT64_MAX));
            }
            // A failed group cannot be reused without resetting every slot.
            if (fail_at<passes) assert(maelstrom_cuda_checked_group(&r,s,first,threads,&result).stage==7);
            assert(maelstrom_cuda_begin_attempt(&r).code==0);
            observed.numerical_failures.clear();
            assert(maelstrom_cuda_checked_group(&r,s,first,threads,&result).code==0 && result==UINT64_MAX);
            for (auto invalid : {0U,6U,10U,14U})
                assert(maelstrom_cuda_checked_group(&r,s,invalid,threads,&result).stage==7);
            assert(maelstrom_cuda_checked_group(&r,s,first,threads,nullptr).stage==7);
            s.count=0;waits=observed.completions;copies=observed.downloads;
            assert(maelstrom_cuda_checked_group(&r,s,first,threads,&result).code==0 && result==UINT64_MAX);
            assert(observed.completions==waits && observed.downloads==copies);
        }
    }
    for (auto failure : {9,10,11,13,16}) for (int launch=1;launch<=4;++launch) {
        observed={};Resource r;ready(r);
        SolverSettings s{};s.count=5;s.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,256,0).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        observed.record_solver=true;
        observed.fail=failure==9 ? 0 : failure;
        observed.fail_launch=launch;
        observed.numerical_failures[5]={4,22};
        auto waits=observed.completions;
        std::uint64_t result=17;
        auto status=maelstrom_cuda_checked_group(&r,s,5,128,&result);
        // Readback is never attempted after submission failure. A completion
        // fault supersedes that failure; context-pop failure supersedes both.
        auto expected=failure==11 ? 9 : failure==16 ? 13 : failure;
        assert(status.stage==expected && r.poisoned && result==17);
        assert(observed.completions==waits+(failure==13 ? 0 : 1));
        assert(observed.launches.size()==static_cast<unsigned int>(failure==13 ? 0 : launch));
        assert(maelstrom_cuda_checked_group(&r,s,5,128,&result).stage==14);
        assert(maelstrom_cuda_begin_attempt(&r).stage==14);
    }
    // A readback fault after an otherwise completed group is also fatal.
    observed={};Resource r;ready(r);
    SolverSettings s{};s.count=1;s.padded=1;
    for (unsigned int slot=0;slot<14;++slot)
        assert(maelstrom_cuda_storage(&r,slot,nullptr,256,0).code==0);
    assert(maelstrom_cuda_begin_attempt(&r).code==0);
    observed.fail=11;
    std::uint64_t result=17;
    assert(maelstrom_cuda_checked_group(&r,s,9,128,&result).stage==11 && r.poisoned && result==17);
}

static void checked_solver() {
    for (int failure : {0,9,10,11,13,16}) {
        observed={};
        Resource r;ready(r);
        SolverSettings settings{};settings.count=5;settings.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,256,0).code==0);
        std::uint32_t errors[]={0,0,52,1,0};
        assert(maelstrom_cuda_storage(&r,12,errors,sizeof(errors),1).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        assert(observed.resets==1);
        std::uint64_t result=17;
        for (auto operation : {2U,3U,13U,14U})
            assert(maelstrom_cuda_checked_solver(&r,settings,operation,128,&result).stage==7);
        assert(maelstrom_cuda_checked_solver(&r,settings,0,128,nullptr).stage==7);
        assert(maelstrom_cuda_checked_solver(&r,settings,0,33,&result).stage==7);
        observed.fail=failure;
        const auto completions=observed.completions,downloads=observed.downloads,pops=observed.pops;
        const auto status=maelstrom_cuda_checked_solver(&r,settings,0,128,&result);
        const auto expected=failure==16 ? 13 : failure;
        assert(status.stage==expected && r.poisoned==(failure!=0));
        assert(observed.completions==completions+(failure==13 ? 0 : 1));
        assert(observed.downloads==downloads+((failure==0 || failure==11 || failure==16) ? 1 : 0));
        assert(observed.pops==pops+(failure==13 ? 0 : 1));
        if (!failure) {
            assert(result==((std::uint64_t{2}<<32)|52));
            assert(maelstrom_cuda_checked_solver(&r,settings,0,128,&result).stage==7 && !r.poisoned);
            assert(maelstrom_cuda_begin_attempt(&r).code==0 && observed.resets==2);
            std::memset(errors,0,sizeof(errors));
            assert(maelstrom_cuda_storage(&r,12,errors,sizeof(errors),1).code==0);
            assert(maelstrom_cuda_checked_solver(&r,settings,0,128,&result).code==0 && result==UINT64_MAX);
            auto resets=observed.resets,waits=observed.completions;
            assert(maelstrom_cuda_begin_attempt(&r).code==0);
            assert(observed.resets==resets && observed.completions==waits);
            settings.count=0;
            auto before=observed.completions;
            assert(maelstrom_cuda_checked_solver(&r,settings,0,128,&result).code==0 && result==UINT64_MAX);
            assert(observed.completions==before);
        } else {
            observed.fail=0;
            assert(maelstrom_cuda_checked_solver(&r,settings,0,128,&result).stage==14);
            assert(maelstrom_cuda_begin_attempt(&r).stage==14);
        }
    }
    for (auto failure : {9,10,13,16}) {
        observed={};
        Resource r;ready(r);
        observed.fail=failure==9 ? 0 : failure;
        observed.fail_reset=failure==9;
        const auto completions=observed.completions,pops=observed.pops;
        auto status=maelstrom_cuda_begin_attempt(&r);
        assert(status.stage==(failure==16 ? 13 : failure) && r.poisoned);
        assert(observed.completions==completions+(failure==13 ? 0 : 1));
        assert(observed.pops==pops+(failure==13 ? 0 : 1));
        assert(observed.downloads==0);
        assert(maelstrom_cuda_begin_attempt(&r).stage==14);
    }
    // A completion fault takes precedence even when reset submission fails.
    observed={};
    Resource r;ready(r);
    observed.fail_reset=true;observed.fail=10;
    assert(maelstrom_cuda_begin_attempt(&r).stage==10 && r.poisoned);
}

int main() {
    sort_batches();
    checked_solver();
    checked_groups();
    for (int stage : {2, 3, 4, 5, 6, 7, 16}) {
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
    for (int stage : {7,8,9,10,11,12,13,16}) {
        observed={};
        {
            Resource r;ready(r);
            SolverSettings settings{};settings.count=1;settings.padded=1;
            std::size_t sizes[14]={12,12,12,12,4,4,4,12,12,0,32,0,4,16};
            for (unsigned int slot=0;slot<14;++slot) assert(maelstrom_cuda_storage(&r,slot,nullptr,sizes[slot],0).code==0);
            const auto allocations=observed.memory.size();
            assert(maelstrom_cuda_storage(&r,0,nullptr,12,0).code==0);
            assert(observed.memory.size()==allocations);
            assert(maelstrom_cuda_storage(&r,14,nullptr,0,0).code!=0);
            assert(maelstrom_cuda_solver(&r,settings,0,33).code!=0);
            assert(maelstrom_cuda_solver(&r,settings,4,128).code!=0); // missing cell width
            auto oversized=settings;oversized.entries=std::numeric_limits<std::uint64_t>::max();
            assert(maelstrom_cuda_solver(&r,oversized,0,128).code!=0);
            auto unallocated=settings;unallocated.entries=1;
            assert(maelstrom_cuda_solver(&r,unallocated,0,128).code!=0);
            Vector input{1,2,3},output{};
            observed.fail=stage;
            Status status;
            if (stage==7 || stage==12) status=maelstrom_cuda_storage(&r,0,nullptr,24,0);
            else if (stage==8) status=maelstrom_cuda_storage(&r,0,&input,12,1);
            else if (stage==11) status=maelstrom_cuda_storage(&r,0,&output,12,2);
            else status=maelstrom_cuda_solver(&r,settings,0,128);
            assert(status.code!=0 && status.stage==(stage==16 ? 13 : stage));
            assert(r.poisoned==(stage!=7));
            observed.fail=0;
            if (stage==7) {
                assert(maelstrom_cuda_storage(&r,0,&input,12,1).code==0);
                assert(maelstrom_cuda_storage(&r,0,&output,12,2).code==0);
                assert(output.x==1 && output.y==2 && output.z==3);
                assert(maelstrom_cuda_solver(&r,settings,0,128).code==0);
            } else {
                assert(maelstrom_cuda_storage(&r,0,nullptr,0,0).stage==14);
                assert(maelstrom_cuda_solver(&r,settings,0,128).stage==14);
            }
        }
        assert(observed.destroys==1 && observed.memory.empty());
    }
    std::puts("native lifecycle and injected failure checks passed");
}
