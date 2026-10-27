// Qualify the guarded exponent-four path against every non-negative binary32
// word. Bounded batches avoid a single long-running watchdog-sensitive kernel.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include "power.cuh"

__global__ void compare_words(std::uint64_t begin, std::uint64_t count,
                              unsigned long long *counters) {
    const auto lane=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=static_cast<std::uint64_t>(gridDim.x)*blockDim.x;
    unsigned long long mismatches=0, first=~0ULL, guarded=0, raw_mismatches=0, raw_first=~0ULL;
    for (auto index=lane;index<count;index+=stride) {
        auto bits=static_cast<unsigned int>(begin+index);
        float x=__uint_as_float(bits), candidate;
        guarded+=try_power_four(x,candidate);
        const auto expected=__float_as_uint(ordered_power_reference(x,4.0f));
        const auto actual=__float_as_uint(ordered_power(x,4.0f));
        if (actual!=expected) { ++mismatches; if (bits<first) first=bits; }
        if (bits<0x7f800000U) {
            double square=static_cast<double>(x)*static_cast<double>(x);
            const auto raw=__float_as_uint(static_cast<float>(square*square));
            if (raw>=0x00800000U && raw<0x7f800000U && raw!=expected) {
                ++raw_mismatches;
                if (bits<raw_first) raw_first=bits;
            }
        }
    }
    if (mismatches) { atomicAdd(counters,mismatches);atomicMin(counters+1,first); }
    if (guarded) atomicAdd(counters+2,guarded);
    if (raw_mismatches) { atomicAdd(counters+3,raw_mismatches);atomicMin(counters+4,raw_first); }
}

static bool checked(cudaError_t result, const char *operation) {
    if (result==cudaSuccess) return true;
    std::fprintf(stderr,"%s: %s\n",operation,cudaGetErrorString(result));
    return false;
}
int main(int argc, char **argv) {
    unsigned int threads=128;
    if (argc==2) {
        char *end=nullptr;
        const auto parsed=std::strtoul(argv[1],&end,10);
        if (!end || *end || (parsed!=32 && parsed!=64 && parsed!=128 && parsed!=256)) return 1;
        threads=static_cast<unsigned int>(parsed);
    } else if (argc!=1) return 1;
    const auto start=std::chrono::steady_clock::now();
    unsigned long long *device=nullptr, counters[5]={0,~0ULL,0,0,~0ULL};
    if (!checked(cudaSetDevice(0),"device") ||
        !checked(cudaMalloc(reinterpret_cast<void **>(&device),sizeof(counters)),"allocation") ||
        !checked(cudaMemcpy(device,counters,sizeof(counters),cudaMemcpyHostToDevice),"initialise")) return 1;
    constexpr std::uint64_t total=1ULL<<31, batch=1ULL<<24;
    for (std::uint64_t begin=0;begin<total;begin+=batch) {
        compare_words<<<1024,threads>>>(begin,batch,device);
        if (!checked(cudaGetLastError(),"launch") || !checked(cudaDeviceSynchronize(),"completion")) return 1;
    }
    if (!checked(cudaMemcpy(counters,device,sizeof(counters),cudaMemcpyDeviceToHost),"download") ||
        !checked(cudaFree(device),"release")) return 1;
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("{\"threads\":%u,\"input_words\":%llu,\"mismatches\":%llu,\"first_mismatch_word\":%llu,\"guarded_words\":%llu,\"raw_quartic_mismatches\":%llu,\"raw_first_mismatch_word\":%llu,\"elapsed_s\":%.9f}\n",
        threads,static_cast<unsigned long long>(total),counters[0],counters[1],counters[2],counters[3],counters[4],elapsed);
    return counters[0] || !counters[3] ? 2 : 0;
}
