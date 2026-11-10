// Integer identity order is independent of geometry and floating-point sums.
// Complete warps sort bounded rows in registers; dense rows retain heapsort.
#pragma once
#include <cstdint>
#include <climits>

template<unsigned int Stride>
__device__ void sift_down(std::uint32_t *row, std::uint64_t root, std::uint64_t count) {
    while (root<count/2) {
        auto child=root*2+1;
        if (child+1<count && row[child*Stride]<row[(child+1)*Stride]) ++child;
        if (row[root*Stride]>=row[child*Stride]) return;
        auto value=row[root*Stride]; row[root*Stride]=row[child*Stride]; row[child*Stride]=value;
        root=child;
    }
}
template<unsigned int Stride>
__device__ void sort_row(std::uint32_t *row, std::uint64_t count) {
    for (auto root=count/2;root>0;--root) sift_down<Stride>(row,root-1,count);
    for (auto end=count;end>1;--end) {
        auto value=row[0]; row[0]=row[(end-1)*Stride]; row[(end-1)*Stride]=value;
        sift_down<Stride>(row,0,end-1);
    }
}

template<unsigned int Width, unsigned int Span, unsigned int Stride>
__device__ __forceinline__ void row_network(std::uint32_t (&values)[Width/32], unsigned int lane) {
    std::uint32_t next[Width/32];
    #pragma unroll
    for (unsigned int slot=0;slot<Width/32;++slot) {
        const auto index=slot*32+lane;
        const auto value=values[slot];
        std::uint32_t other;
        if constexpr (Stride<32) other=__shfl_xor_sync(0xffffffffU,value,Stride);
        else other=values[slot^(Stride/32)];
        const auto low=value<other ? value : other;
        const auto high=value>other ? value : other;
        next[slot]=((index&Span)==0)==((index&Stride)==0) ? low : high;
    }
    #pragma unroll
    for (unsigned int slot=0;slot<Width/32;++slot) values[slot]=next[slot];
    if constexpr (Stride>1) row_network<Width,Span,Stride/2>(values,lane);
    else if constexpr (Span<Width) row_network<Width,Span*2,Span>(values,lane);
}

template<unsigned int Width, unsigned int Stride>
__device__ __forceinline__ void warp_sort_row(std::uint32_t *row, std::uint64_t count,
                                             unsigned int lane) {
    std::uint32_t values[Width/32];
    #pragma unroll
    for (unsigned int slot=0;slot<Width/32;++slot) {
        const auto index=slot*32+lane;
        values[slot]=index<count ? row[index*Stride] : UINT_MAX;
    }
    row_network<Width,2,1>(values,lane);
    #pragma unroll
    for (unsigned int slot=0;slot<Width/32;++slot) {
        const auto index=slot*32+lane;
        if (index<count) row[index*Stride]=values[slot];
    }
}

template<unsigned int Stride=1>
__device__ void sort_row_cooperatively(std::uint32_t *row, std::uint64_t count,
                                      unsigned int lane) {
    if (count<2) return;
    if (count<=32) warp_sort_row<32,Stride>(row,count,lane);
    else if (count<=64) warp_sort_row<64,Stride>(row,count,lane);
    else if (count<=128) warp_sort_row<128,Stride>(row,count,lane);
    else if (count<=256) warp_sort_row<256,Stride>(row,count,lane);
    else if (lane==0) sort_row<Stride>(row,count);
}
