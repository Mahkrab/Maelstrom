// Faults replace driver functions inside this test executable only. These
// assertions prove ownership transitions, not recovery from actual device loss.
#include "bridge.cpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>

struct Observed {
    int fail = 0;
    int pushes = 0;
    int pops = 0;
    int destroys = 0;
    int frees = 0;
    int completions = 0;
    int downloads = 0;
    int uploads = 0;
    bool work_pending = false;
    bool record_solver = false;
    int fail_launch = 0;
    int resets = 0;
    bool fail_reset = false;
    bool fail_host_allocation = false;
    bool fail_host_mapping = false;
    int pinned_allocations = 0;
    int pinned_releases = 0;
    std::map<void *,std::unique_ptr<std::uint64_t[]>> host_memory;
    struct Launch { unsigned int operation, stride, span, grid, block; };
    std::vector<Launch> launches;
    std::map<unsigned int,std::pair<unsigned int,std::uint32_t>> numerical_failures;
    CUdeviceptr next=4096;
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
    d.cuCtxDestroy_fn = [](CUcontext) { ++observed.destroys; observed.work_pending=false; observed.memory.clear();
        observed.pinned_releases+=static_cast<int>(observed.host_memory.size());
        observed.host_memory.clear();return fault(15); };
    d.cuCtxPushCurrent_fn = [](CUcontext) { ++observed.pushes; return fault(13); };
    d.cuCtxPopCurrent_fn = [](CUcontext *context) { ++observed.pops; *context = nullptr; return fault(16); };
    d.cuModuleLoadData_fn = [](CUmodule *module, const void *) { *module = reinterpret_cast<CUmodule>(1); return fault(5); };
    d.cuModuleGetFunction_fn = [](CUfunction *function, CUmodule, const char *name) {
        *function = reinterpret_cast<CUfunction>(std::strcmp(name,"solver")==0 ? 2 : std::strcmp(name,"spatial_sort")==0 ? 3 : std::strcmp(name,"observe_state")==0 ? 4 : std::strcmp(name,"prefix_counts")==0 ? 5 : 1);
        return CUDA_SUCCESS;
    };
    d.cuStreamCreate_fn = [](CUstream *stream, unsigned int flags) {
        assert(flags==CU_STREAM_DEFAULT);
        *stream = reinterpret_cast<CUstream>(1); return fault(6);
    };
    d.cuMemAlloc_fn = [](CUdeviceptr *buffer, std::size_t bytes) {
        if (observed.fail == 7) return CUDA_ERROR_OUT_OF_MEMORY;
        *buffer=observed.next;observed.next+=bytes+4096; observed.memory[*buffer].resize(bytes); return CUDA_SUCCESS;
    };
    d.cuMemAllocHost_fn = [](void **pointer,std::size_t bytes) {
        assert(bytes==diagnostic_records_bytes);
        if (observed.fail_host_allocation) return CUDA_ERROR_OUT_OF_MEMORY;
        auto memory=std::make_unique<std::uint64_t[]>(diagnostic_record_count);
        *pointer=memory.get();observed.host_memory.emplace(*pointer,std::move(memory));
        ++observed.pinned_allocations;return CUDA_SUCCESS;
    };
    d.cuMemHostGetDevicePointer_fn = [](CUdeviceptr *device,void *host,unsigned int flags) {
        assert(!flags && observed.host_memory.count(host)==1);
        if (observed.fail_host_mapping) return CUDA_ERROR_UNKNOWN;
        *device=reinterpret_cast<CUdeviceptr>(host);return CUDA_SUCCESS;
    };
    d.cuMemFree_fn = [](CUdeviceptr pointer) { assert(!observed.work_pending); ++observed.frees; if (!observed.fail) observed.memory.erase(pointer); return fault(12); };
    d.cuMemcpyHtoD_fn = [](CUdeviceptr pointer, const void *data, std::size_t bytes) {
        assert(!observed.work_pending);++observed.uploads;
        auto allocation=observed.memory.upper_bound(pointer);assert(allocation!=observed.memory.begin());--allocation;
        const auto offset=pointer-allocation->first;
        assert(offset<=allocation->second.size() && bytes<=allocation->second.size()-offset);
        std::memcpy(allocation->second.data()+offset,data,bytes);return fault(8);
    };
    d.cuMemsetD8Async_fn = [](CUdeviceptr pointer, unsigned char value, std::size_t bytes, CUstream) {
        observed.work_pending=true;
        ++observed.resets;
        if (observed.fail_reset) return CUDA_ERROR_UNKNOWN;
        assert(bytes==32 && value==0xff && observed.memory.at(pointer).size()==diagnostic_allocation_bytes);
        std::memset(observed.memory.at(pointer).data(),value,bytes);
        return CUDA_SUCCESS;
    };
    d.cuLaunchKernel_fn = [](CUfunction function, unsigned int grid, unsigned int, unsigned int,
                            unsigned int block, unsigned int, unsigned int, unsigned int,
                            CUstream, void **args, void **) {
        observed.work_pending=true;
        if (function==reinterpret_cast<CUfunction>(2)) {
            // This host stand-in supplies an independent serial selection oracle;
            // hardware tests exercise the actual competing atomic updates.
            auto buffers=*static_cast<SolverBuffers *>(args[0]);
            auto settings=*static_cast<SolverSettings *>(args[1]);
            auto operation=*static_cast<unsigned int *>(args[2]);
            if (buffers.diagnostic && observed.fail!=9 && operation!=18) {
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
        if (function==reinterpret_cast<CUfunction>(4) || function==reinterpret_cast<CUfunction>(5)) {
            const auto buffers=*static_cast<SolverBuffers *>(args[0]);
            const auto operation=*static_cast<unsigned int *>(args[2]);
            const auto output=reinterpret_cast<CUdeviceptr>(buffers.diagnostic);
            const auto failure=observed.numerical_failures.find(operation);
            if (failure!=observed.numerical_failures.end()) {
                const auto slot=function==reinterpret_cast<CUfunction>(5) ? 1 : 0;
                const auto record=(static_cast<std::uint64_t>(failure->second.first)<<32)|failure->second.second;
                std::memcpy(observed.memory.at(output).data()+slot*sizeof(record),&record,sizeof(record));
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
    d.cuStreamSynchronize_fn = [](CUstream) {
        ++observed.completions;
        // Publish the serial diagnostic oracle only at completion. A failing
        // completion may still have changed private mapped storage; callers
        // must not expose it until every completion/binding check succeeds.
        for (auto &[pointer,memory]:observed.memory) {
            (void)pointer;
            if (memory.size()!=diagnostic_allocation_bytes) continue;
            CUdeviceptr address=0;
            std::memcpy(&address,memory.data()+diagnostic_records_bytes,sizeof(address));
            const auto host=reinterpret_cast<void *>(address);
            if (observed.host_memory.count(host))
                std::memcpy(host,memory.data(),diagnostic_records_bytes);
        }
        if (observed.fail!=10) observed.work_pending=false;
        return fault(10);
    };
    d.cuMemcpyDtoH_fn = [](void *data, CUdeviceptr pointer, std::size_t bytes) {
        assert(!observed.work_pending);++observed.downloads;
        auto allocation=observed.memory.upper_bound(pointer);assert(allocation!=observed.memory.begin());--allocation;
        const auto offset=pointer-allocation->first;
        assert(offset<=allocation->second.size() && bytes<=allocation->second.size()-offset);
        std::memcpy(data,allocation->second.data()+offset,bytes);return fault(11);
    };
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
            assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
        const auto before=observed.completions;
        observed.record_solver=true;
        // Deliberately invalid incoming pass fields must be replaced by the
        // complete network, rather than inherited from the caller.
        settings.stride=17;settings.span=19;
        assert(maelstrom_cuda_solver(&r,settings,13,threads).code==0);
        assert(observed.launches.size()==1 && observed.completions==before && r.pending);
        const auto &pass=observed.launches[0];
        assert(pass.operation==19 && pass.stride==0 && pass.grid==1 && pass.block==threads);
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
        assert(maelstrom_cuda_solver(&r,settings,18,threads).stage==7 && !r.poisoned);
    }
    for (auto failure : {9,10,13,16}) {
        observed={};
        Resource r;ready(r);
        SolverSettings settings{};settings.count=1025;settings.padded=2048;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,2048*64,0).code==0);
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

static void merge_sort_submissions() {
    for (auto threads : {32U,64U,128U,256U}) {
        unsigned int merges=0;
        for (unsigned int width=threads;width<512;width*=2) ++merges;
        for (unsigned int failure=0;failure<=merges+1;++failure) {
            observed={};Resource r;ready(r);
            SolverSettings s{};s.count=257;s.padded=512;
            for (unsigned int slot=0;slot<14;++slot)
                assert(maelstrom_cuda_storage(&r,slot,nullptr,512*64,0).code==0);
            observed.record_solver=true;observed.fail_launch=static_cast<int>(failure);
            const auto status=maelstrom_cuda_solver(&r,s,13,threads);
            assert(status.stage==(failure ? 9 : 0) && r.poisoned==(failure!=0));
            const auto submitted=failure ? failure : merges+1;
            assert(observed.launches.size()==submitted);
            assert(observed.completions==(failure ? 1 : 0));
            for (unsigned int pass=0;pass<submitted;++pass) {
                const auto &launch=observed.launches[pass];
                assert(launch.operation==(pass ? 20U : 19U));
                const auto selector=pass ? ((merges&1U)^((pass-1)&1U)) : (merges&1U);
                assert(launch.stride==selector && launch.block==threads && launch.grid==512/threads);
                if (pass) assert(launch.span==(threads<<(pass-1)));
            }
            if (failure) assert(maelstrom_cuda_solver(&r,s,13,threads).stage==14);
        }
    }
}

static void containment_storage() {
    // The aligned certificate must fit in both velocity allocations before any
    // submission, including after accepted/working buffers exchange ownership.
    for (auto slot : {1U,3U}) for (auto count : {1U,2U,3U}) {
        observed={};Resource r;ready(r);
        SolverSettings s{};s.count=count;s.padded=4;
        for (unsigned int index=0;index<14;++index)
            assert(maelstrom_cuda_storage(&r,index,nullptr,index==slot ? count*12 : 512,0).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        observed.record_solver=true;
        std::uint64_t result=17;
        assert(maelstrom_cuda_checked_solver(&r,s,24,128,&result).stage==7);
        assert(result==17 && observed.launches.empty() && !r.poisoned);
        assert(maelstrom_cuda_storage(&r,slot,nullptr,count*12+velocity_observation_bytes,0).code==0);
        assert(maelstrom_cuda_checked_solver(&r,s,24,128,&result).code==0);
        assert(result==UINT64_MAX && observed.launches.size()==1 && observed.launches[0].operation==24);
    }
}

static void checked_groups() {
    // The grouped validation pass stages four words per particle. Reject an
    // old vector-sized candidate allocation before submitting any kernel.
    {
        observed={};Resource r;ready(r);
        SolverSettings s{};s.count=5;s.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,slot==8 ? 60 : 512,0).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        observed.record_solver=true;
        std::uint64_t result=17;
        assert(maelstrom_cuda_checked_group(&r,s,5,128,&result).stage==7);
        assert(result==17 && observed.launches.empty() && !r.poisoned);
        assert(maelstrom_cuda_storage(&r,8,nullptr,80,0).code==0);
        assert(maelstrom_cuda_checked_group(&r,s,5,128,&result).code==0 && result==UINT64_MAX);
    }
    for (auto first : {5U,9U}) for (auto threads : {32U,64U,128U,256U}) {
        const auto passes=first==5 ? 4U : 2U;
        for (unsigned int fail_at=0;fail_at<=passes;++fail_at) {
            observed={};Resource r;ready(r);
            SolverSettings s{};s.count=5;s.padded=8;
            for (unsigned int slot=0;slot<14;++slot)
                assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
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
            assert(observed.completions==waits+1 && observed.downloads==copies);
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
            assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        observed.record_solver=true;
        observed.fail=failure==9 ? 0 : failure;
        observed.fail_launch=launch;
        observed.numerical_failures[5]={4,22};
        auto waits=observed.completions;
        std::uint64_t result=17;
        auto status=maelstrom_cuda_checked_group(&r,s,5,128,&result);
        // Results are never published after submission failure. A completion
        // fault supersedes that failure; context-pop failure supersedes both.
        auto expected=failure==11 ? 9 : failure==16 ? 13 : failure;
        assert(status.stage==expected && r.poisoned && result==17);
        assert(observed.completions==waits+(failure==13 ? 0 : 1));
        assert(observed.launches.size()==static_cast<unsigned int>(failure==13 ? 0 : launch));
        assert(maelstrom_cuda_checked_group(&r,s,5,128,&result).stage==14);
        assert(maelstrom_cuda_begin_attempt(&r).stage==14);
    }
    // Diagnostic observation submits no transfer. A subsequent data-copy fault
    // still poisons the resource and leaves the caller output untouched.
    observed={};Resource r;ready(r);
    SolverSettings s{};s.count=1;s.padded=1;
    for (unsigned int slot=0;slot<14;++slot)
        assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
    assert(maelstrom_cuda_begin_attempt(&r).code==0);
    observed.fail=11;
    std::uint64_t result=17;
    assert(maelstrom_cuda_checked_group(&r,s,9,128,&result).code==0 && result==UINT64_MAX);
    assert(observed.downloads==0 && !r.poisoned);
    std::uint64_t output=17;
    assert(maelstrom_cuda_storage(&r,0,&output,sizeof(output),2).stage==11 && r.poisoned);
}

static void row_sort_batches() {
    for (auto threads : {32U,64U,128U,256U}) for (auto count : {1U,5U,17U}) {
        for (int fail_at : {0,1,2}) {
            observed={};
            Resource r;ready(r);
            SolverSettings s{};s.count=count;s.padded=32;s.width=1;
            for (unsigned int slot=0;slot<14;++slot)
                assert(maelstrom_cuda_storage(&r,slot,nullptr,4096,0).code==0);
            std::vector<std::uint32_t> errors(count,0);
            assert(maelstrom_cuda_storage(&r,12,errors.data(),errors.size()*4,1).code==0);
            assert(maelstrom_cuda_begin_attempt(&r).code==0);
            observed.record_solver=true;observed.fail_launch=fail_at;
            const auto waits=observed.completions, downloads=observed.downloads;
            std::uint64_t result=17;
            const auto status=maelstrom_cuda_checked_solver(&r,s,12,threads,&result);
            assert(status.stage==(fail_at ? 9 : 0) && r.poisoned==(fail_at!=0));
            assert(observed.completions==waits+1);
            assert(observed.downloads==downloads);
            assert(observed.launches.size()==static_cast<unsigned int>(fail_at ? fail_at : 2));
            assert(observed.launches[0].operation==12 && observed.launches[0].grid==(count+threads-1)/threads);
            if (observed.launches.size()==2)
                assert(observed.launches[1].operation==18 && observed.launches[1].grid==(32*count+threads-1)/threads);
            if (fail_at) {
                assert(result==17);
                assert(maelstrom_cuda_checked_solver(&r,s,12,threads,&result).stage==14);
            } else {
                assert(result==UINT64_MAX);
                errors[count-1]=52;
                assert(maelstrom_cuda_storage(&r,12,errors.data(),errors.size()*4,1).code==0);
                assert(maelstrom_cuda_checked_solver(&r,s,12,threads,&result).code==0);
                assert(result==((static_cast<std::uint64_t>(count-1)<<32)|52));
            }
        }
    }
}

static void checked_solver() {
    for (int failure : {0,9,10,13,16}) {
        observed={};
        Resource r;ready(r);
        SolverSettings settings{};settings.count=5;settings.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
        std::uint32_t errors[]={0,0,52,1,0};
        assert(maelstrom_cuda_storage(&r,12,errors,sizeof(errors),1).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        assert(observed.resets==1);
        std::uint64_t result=17;
        for (auto operation : {2U,3U,13U,18U})
            assert(maelstrom_cuda_checked_solver(&r,settings,operation,128,&result).stage==7);
        assert(maelstrom_cuda_checked_solver(&r,settings,0,128,nullptr).stage==7);
        assert(maelstrom_cuda_checked_solver(&r,settings,0,33,&result).stage==7);
        observed.fail=failure;
        const auto completions=observed.completions,downloads=observed.downloads,pops=observed.pops;
        const auto status=maelstrom_cuda_checked_solver(&r,settings,0,128,&result);
        const auto expected=failure==16 ? 13 : failure;
        assert(status.stage==expected && r.poisoned==(failure!=0));
        assert(observed.completions==completions+(failure==13 ? 0 : 1));
        assert(observed.downloads==downloads);
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
        if (failure==10) {
            assert(status.code==0 && r.pending && !r.poisoned);
            status=solver_call(r,Boundary::Observe,[]() -> Status { return {0,0}; });
        }
        assert(status.stage==(failure==16 ? 13 : failure) && r.poisoned);
        assert(observed.completions==completions+((failure==13 || failure==16) ? 0 : 1));
        assert(observed.pops==pops+(failure==13 ? 0 : failure==10 ? 2 : 1));
        assert(observed.downloads==0);
        assert(maelstrom_cuda_begin_attempt(&r).stage==14);
    }
    // A completion fault takes precedence even when reset submission fails.
    observed={};
    Resource r;ready(r);
    observed.fail_reset=true;observed.fail=10;
    assert(maelstrom_cuda_begin_attempt(&r).stage==10 && r.poisoned);
}

static void accepted_vectors() {
    observed={};
    Resource r;ready(r);
    for (unsigned int slot=0;slot<4;++slot) {
        assert(maelstrom_cuda_storage(&r,slot,nullptr,12*(slot+1),0).code==0);
        std::fill(observed.memory.at(r.slots[slot]).begin(),
                  observed.memory.at(r.slots[slot]).end(),static_cast<unsigned char>(slot+1));
    }
    const std::vector<CUdeviceptr> pointers(r.slots,r.slots+14);
    const std::vector<std::size_t> capacities(r.capacities,r.capacities+14);
    assert(maelstrom_cuda_accept(&r).stage==7);
    assert(maelstrom_cuda_begin_attempt(&r).code==0);
    r.diagnostic_clear=false;
    assert(maelstrom_cuda_accept(&r).stage==7);
    r.poisoned=true;
    assert(maelstrom_cuda_accept(&r).stage==14);
    r.poisoned=false;r.diagnostic_clear=true;
    assert(maelstrom_cuda_accept(&r).stage==7); // Reset is still pending.
    Vector output{};
    assert(maelstrom_cuda_storage(&r,0,&output,sizeof(output),2).code==0);
    assert(std::equal(pointers.begin(),pointers.end(),r.slots));
    assert(std::equal(capacities.begin(),capacities.end(),r.capacities));
    const auto pushes=observed.pushes,pops=observed.pops,waits=observed.completions;
    const auto uploads=observed.uploads,downloads=observed.downloads,resets=observed.resets;
    assert(maelstrom_cuda_accept(&r).code==0);
    for (unsigned int slot=0;slot<4;++slot) {
        const auto source=(slot+2)%4;
        assert(r.slots[slot]==pointers[source] && r.capacities[slot]==capacities[source]);
        for (auto byte:observed.memory.at(r.slots[slot])) assert(byte==source+1);
    }
    assert(std::equal(pointers.begin()+4,pointers.end(),r.slots+4));
    assert(observed.pushes==pushes && observed.pops==pops && observed.completions==waits);
    assert(observed.uploads==uploads && observed.downloads==downloads && observed.resets==resets);
    const auto accepted=r.slots[0];
    assert(maelstrom_cuda_accept(&r).stage==7 && r.slots[0]==accepted);
    assert(maelstrom_cuda_begin_attempt(&r).code==0);
    assert(maelstrom_cuda_accept(&r).code==0);
    assert(std::equal(pointers.begin(),pointers.end(),r.slots));
    assert(std::equal(capacities.begin(),capacities.end(),r.capacities));
}

// Exercise the cross-call dependencies, including failure at a later host
// observation. The fake driver rejects host copies/frees while work is pending.
static void submission_boundaries() {
    for (unsigned int action : {0U,1U,2U,3U}) {
        observed={};Resource r;ready(r);
        SolverSettings s{};s.count=5;s.padded=8;s.width=1;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
        assert(observed.completions==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0 && r.pending);
        assert(maelstrom_cuda_solver(&r,s,2,128).code==0);
        assert(maelstrom_cuda_solver(&r,s,13,128).code==0);
        assert(observed.completions==0);
        // No-op reservation/empty transfer must preserve the pending batch.
        assert(maelstrom_cuda_storage(&r,0,nullptr,512,0).code==0);
        assert(maelstrom_cuda_storage(&r,0,nullptr,0,2).code==0);
        assert(observed.completions==0 && r.pending);
        Vector value{};
        std::uint64_t diagnostic=0;
        if (action==3)
            assert(maelstrom_cuda_checked_solver(&r,s,4,128,&diagnostic).code==0 && diagnostic==UINT64_MAX);
        else
            assert(maelstrom_cuda_storage(&r,0,action ? &value : nullptr,action ? 12 : 1024,action).code==0);
        assert(observed.completions==1 && !r.pending);
        assert(maelstrom_cuda_storage(&r,0,&value,12,2).code==0);
        assert(observed.completions==1); // Synchronous copy adds no empty wait.
    }
    for (unsigned int action : {0U,1U,2U}) {
        observed={};Resource r;ready(r);
        assert(maelstrom_cuda_storage(&r,0,nullptr,12,0).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        auto copies=observed.uploads+observed.downloads, frees=observed.frees;
        observed.fail=10;
        Vector value{};
        assert(maelstrom_cuda_storage(&r,0,action ? &value : nullptr,action ? 12 : 24,action).stage==10);
        assert(r.poisoned && r.pending && observed.completions==1);
        assert(observed.uploads+observed.downloads==copies && observed.frees==frees);
        assert(maelstrom_cuda_storage(&r,0,nullptr,0,0).stage==14);
    }
    // Even an empty checked observation must complete the queued reset before
    // acceptance; a failed completion cannot publish the sentinel result.
    for (auto failure : {0,10}) {
        observed={};Resource r;ready(r);
        SolverSettings s{};s.padded=1;
        assert(maelstrom_cuda_storage(&r,10,nullptr,64,0).code==0);
        assert(maelstrom_cuda_storage(&r,13,nullptr,8,0).code==0);
        assert(maelstrom_cuda_begin_attempt(&r).code==0);
        assert(maelstrom_cuda_accept(&r).stage==7);
        observed.fail=failure;
        std::uint64_t result=17;
        const auto status=maelstrom_cuda_checked_solver(&r,s,0,128,&result);
        assert(status.stage==failure && observed.completions==1);
        assert(result==(failure ? 17 : UINT64_MAX));
        assert(maelstrom_cuda_accept(&r).stage==(failure ? 14 : 0));
    }
    // Allocation exhaustion after a successful drain remains retryable.
    observed={};Resource r;ready(r);
    assert(maelstrom_cuda_storage(&r,0,nullptr,12,0).code==0);
    assert(maelstrom_cuda_begin_attempt(&r).code==0);
    observed.fail=7;
    const auto original=r.slots[0];
    assert(maelstrom_cuda_storage(&r,0,nullptr,24,0).stage==7);
    assert(!r.poisoned && !r.pending && r.slots[0]==original && r.capacities[0]==12);
    observed.fail=0;
    assert(maelstrom_cuda_storage(&r,0,nullptr,24,0).code==0 && r.capacities[0]==24);
}

static void compact_observation_failures() {
    for (auto prefix : {false,true}) for (auto failure : {0,9,10,13,16}) {
        const auto passes=prefix ? 4 : 1;
        for (int launch=1;launch<=passes;++launch) {
            observed={};Resource r;ready(r);
            SolverSettings s{};s.count=5;s.padded=8;s.width=1;
            for (unsigned int slot=0;slot<14;++slot)
                assert(maelstrom_cuda_storage(&r,slot,nullptr,512,0).code==0);
            assert(maelstrom_cuda_begin_attempt(&r).code==0);
            observed.record_solver=true;
            observed.fail=failure==9 ? 0 : failure;
            observed.fail_launch=failure==9 ? launch : 0;
            std::uint64_t results[]={17,19};
            const auto status=prefix ? maelstrom_cuda_prefix(&r,s,128,1,results) :
                                      maelstrom_cuda_observe(&r,s,0,128,results);
            const auto expected=failure==16 ? 13 : failure;
            assert(status.stage==expected && r.poisoned==(failure!=0));
            assert(observed.completions==(failure==13 ? 0 : 1));
            assert(observed.downloads==0);
            if (failure) assert(results[0]==17 && results[1]==19);
            else {
                assert(results[0]==UINT64_MAX);
                if (!prefix) assert(results[1]==UINT64_MAX);
                assert(observed.launches.size()==static_cast<unsigned int>(passes));
                if (prefix) {
                    assert(observed.launches[0].operation==4);
                    for (int pass=1;pass<4;++pass)
                        assert(observed.launches[pass].operation==20U+pass);
                }
            }
            if (failure) assert(maelstrom_cuda_prefix(&r,s,128,1,results).stage==14);
        }
    }
    // Offset reads must be bounded before forming a pointer or touching host data.
    observed={};Resource r;ready(r);
    std::uint64_t values[]={11,13},out=0;
    assert(maelstrom_cuda_storage(&r,13,nullptr,sizeof(values),0).code==0);
    assert(maelstrom_cuda_storage(&r,13,values,sizeof(values),1).code==0);
    assert(maelstrom_cuda_storage(&r,13,&out,8,2,8).code==0 && out==13);
    const auto reads=observed.downloads;
    assert(maelstrom_cuda_storage(&r,13,&out,8,2,9).stage==7);
    assert(maelstrom_cuda_storage(&r,13,nullptr,0,2,17).stage==7);
    assert(maelstrom_cuda_storage(&r,13,nullptr,0,0,1).stage==7);
    assert(observed.downloads==reads && !r.poisoned);
}

static void mapped_observation_lifetimes() {
    for (unsigned int failure=0;failure<3;++failure) {
        observed={};
        {
            Resource r;fake_driver(r.driver);
            observed.fail_host_allocation=failure==0;
            observed.fail_host_mapping=failure==1;
            observed.fail=failure==2 ? 8 : 0;
            DeviceInfo info{};
            assert(initialise(r,0,"unused",89,&info).stage==(failure==2 ? 8 : 7));
        }
        assert(observed.destroys==1 && observed.memory.empty() && observed.host_memory.empty());
        assert(observed.pinned_allocations==(failure==0 ? 0 : 1));
        assert(observed.pinned_releases==observed.pinned_allocations);
    }
    for (auto failure : {0,9,10,13,16}) {
        observed={};
        auto *r=new Resource;ready(*r);
        SolverSettings s{};s.count=5;s.padded=8;
        for (unsigned int slot=0;slot<14;++slot)
            assert(maelstrom_cuda_storage(r,slot,nullptr,512,0).code==0);
        assert(maelstrom_cuda_begin_attempt(r).code==0);
        std::uint32_t errors[]={0,0,52,1,0};
        assert(maelstrom_cuda_storage(r,12,errors,sizeof(errors),1).code==0);
        const auto host=r->host_diagnostic;
        observed.fail=failure;
        std::uint64_t result=17;
        const auto status=maelstrom_cuda_checked_solver(r,s,0,128,&result);
        assert(status.stage==(failure==16 ? 13 : failure));
        assert(result==(failure ? 17 : (std::uint64_t{2}<<32)|52));
        assert(observed.downloads==0 && observed.host_memory.count(host)==1);
        // A bad completion can already have changed mapped storage. Its
        // lifetime extends past this return until context abandonment.
        if (failure==10) assert(static_cast<std::uint64_t *>(host)[0]==((std::uint64_t{2}<<32)|52));
        if (!failure) {
            CUdeviceptr mapped=0;
            std::memcpy(&mapped,observed.memory.at(r->diagnostic).data()+diagnostic_records_bytes,sizeof(mapped));
            assert(mapped==reinterpret_cast<CUdeviceptr>(host));
            assert(maelstrom_cuda_begin_attempt(r).code==0);
            for (unsigned int k=0;k<diagnostic_record_count;++k)
                assert(static_cast<std::uint64_t *>(host)[k]==UINT64_MAX);
            CUdeviceptr after=0;
            std::memcpy(&after,observed.memory.at(r->diagnostic).data()+diagnostic_records_bytes,sizeof(after));
            assert(after==mapped); // Reset cannot overwrite the device address.
        }
        observed.fail=0;
        assert(maelstrom_cuda_close(r).code==0);
        assert(observed.pinned_releases==1 && observed.host_memory.empty() && observed.memory.empty());
    }
}

int main() {
    mapped_observation_lifetimes();
    compact_observation_failures();
    submission_boundaries();
    accepted_vectors();
    sort_batches();
    merge_sort_submissions();
    row_sort_batches();
    checked_solver();
    containment_storage();
    checked_groups();
    for (int stage : {2, 3, 4, 5, 6, 7, 8, 16}) {
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
            std::size_t sizes[14]={12,12+velocity_observation_bytes,12,12+velocity_observation_bytes,4,4,4,32,12,0,64,0,4,16};
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
            else {
                status=maelstrom_cuda_solver(&r,settings,0,128);
                if (stage==10) {
                    assert(status.code==0 && r.pending);
                    status=maelstrom_cuda_storage(&r,0,&output,12,2);
                }
            }
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
