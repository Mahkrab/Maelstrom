// Keep full signed cell coordinates and the particle-index tie-breaker. Tiles
// sort in shared memory; each later launch merges adjacent runs into disjoint
// output tiles. Ping-pong storage stays inside the context-owned cell allocation.
#pragma once

__device__ void sort_cell_tile(Cell *cells, unsigned int padded, unsigned int output, Cell *tile) {
    const unsigned int lane=threadIdx.x;
    const auto index=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+lane;
    tile[lane]=index<padded ? cells[index] : Cell{LLONG_MAX,LLONG_MAX,LLONG_MAX,UINT_MAX,0};
    __syncthreads();
    for (unsigned int span=2;span<=blockDim.x;span*=2) {
        for (unsigned int stride=span/2;stride;stride/=2) {
            const auto other=lane^stride;
            if (other>lane) {
                const Cell a=tile[lane], b=tile[other];
                if (less(b,a)==((lane&span)==0)) { tile[lane]=b;tile[other]=a; }
            }
            __syncthreads();
        }
    }
    if (index<padded) cells[static_cast<std::uint64_t>(output)*padded+index]=tile[lane];
}

// Partition the merged output at a diagonal, taking the left run first on equal
// keys. That convention also keeps duplicate padding sentinels in distinct slots.
__device__ unsigned int cell_merge_path(const Cell *left, const Cell *right,
        unsigned int length, unsigned int diagonal) {
    unsigned int lo=diagonal>length ? diagonal-length : 0;
    unsigned int hi=diagonal<length ? diagonal : length;
    while (lo<hi) {
        const auto mid=lo+(hi-lo)/2, j=diagonal-mid;
        if (j && mid<length && !less(right[j-1],left[mid])) lo=mid+1;
        else hi=mid;
    }
    return lo;
}

__device__ void merge_cell_tile(Cell *cells, unsigned int padded,
        unsigned int width, unsigned int input, Cell *tile) {
    __shared__ unsigned int left_start, right_start, left_count;
    const unsigned int lane=threadIdx.x;
    const auto start=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x;
    const auto frame=start/(2ULL*width)*(2ULL*width);
    const auto diagonal=static_cast<unsigned int>(start-frame);
    const Cell *source=cells+static_cast<std::uint64_t>(input)*padded;
    const Cell *left=source+frame, *right=left+width;
    if (!lane) {
        left_start=cell_merge_path(left,right,width,diagonal);
        const auto left_stop=cell_merge_path(left,right,width,diagonal+blockDim.x);
        right_start=diagonal-left_start;
        left_count=left_stop-left_start;
    }
    __syncthreads();
    tile[lane]=lane<left_count ? left[left_start+lane] : right[right_start+lane-left_count];
    __syncthreads();
    const Cell value=tile[lane];
    const bool from_left=lane<left_count;
    unsigned int lo=from_left ? left_count : 0;
    unsigned int hi=from_left ? blockDim.x : left_count;
    const auto offset=lo;
    while (lo<hi) {
        const auto mid=lo+(hi-lo)/2;
        // Left keys use lower_bound in the right tile, right keys upper_bound
        // in the left tile. Equal sentinels therefore never race an output slot.
        if (from_left ? less(tile[mid],value) : !less(value,tile[mid])) lo=mid+1;
        else hi=mid;
    }
    const auto own=from_left ? lane : lane-left_count;
    cells[static_cast<std::uint64_t>(input^1U)*padded+start+own+lo-offset]=value;
}
