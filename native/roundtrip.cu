// A bounded, independent layout/launch check; it does not advance fluid state.
#include "probe.h"

extern "C" __global__ void roundtrip(Probe *values, unsigned int count) {
    const auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        values[i].position[0] += 1.0f;
        values[i].position[1] *= -2.0f;
        values[i].position[2] -= 0.5f;
        values[i].marker ^= 0xa5a55a5au;
    }
}
