// Driver-only compact grid. Every launch uses the owning resource's stream.
// Layout in u32 words: counts[M], offsets[M+1], cursors[M], particle_cells[N],
// indices[N], block_totals[ceil((M+1)/256)], alignment, then u64 row tile offsets.
#pragma once
__device__ std::uint32_t *grid_counts(SolverBuffers b) { return reinterpret_cast<std::uint32_t *>(b.cells); }
__device__ std::uint32_t *grid_offsets(SolverBuffers b, SolverSettings s) { return grid_counts(b)+s.grid_cells; }
__device__ std::uint32_t *grid_cursors(SolverBuffers b, SolverSettings s) { return grid_offsets(b,s)+s.grid_cells+1; }
__device__ std::uint32_t *grid_particle_cells(SolverBuffers b, SolverSettings s) { return grid_cursors(b,s)+s.grid_cells; }
__device__ std::uint32_t *grid_indices(SolverBuffers b, SolverSettings s) { return grid_particle_cells(b,s)+s.count; }
__device__ std::uint32_t *grid_totals(SolverBuffers b, SolverSettings s) { return grid_indices(b,s)+s.count; }
__device__ std::uint64_t *tile_offsets(SolverBuffers b, SolverSettings s) {
    return reinterpret_cast<std::uint64_t *>(b.cells)+(s.tile_base ? s.tile_base : static_cast<std::uint64_t>(s.padded)*4);
}

// Inclusive scan of one block, with a read barrier before each shared write.
// Only integer additions are reordered; totals are bounded by count <= 2^31.
__device__ void grid_block_scan(std::uint32_t *scan) {
    __syncthreads();
    for (unsigned int offset=1;offset<256;offset*=2) {
        auto previous=threadIdx.x>=offset ? scan[threadIdx.x-offset] : 0;
        __syncthreads();
        scan[threadIdx.x]+=previous;
        __syncthreads();
    }
}
__device__ void record_diagnostic(SolverBuffers, unsigned int, unsigned long long);
extern "C" __global__ void compact_grid(SolverBuffers b, SolverSettings s, unsigned int operation) {
    const auto lane=threadIdx.x;
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*256+lane;
    const auto m=s.grid_cells;
    auto counts=grid_counts(b), offsets=grid_offsets(b,s), cursors=grid_cursors(b,s);
    auto ids=grid_particle_cells(b,s), indices=grid_indices(b,s), totals=grid_totals(b,s);
    if (operation==27) {
        if (i<m) counts[i]=0;
    } else if (operation==28) {
        if (i>=s.count) return;
        const auto p=b.position[i];
        // Bounds originate in the prediction summary, but guard the native
        // interface too: no undefined float-to-integer conversion or OOB atomic.
        const double qx=floor(static_cast<double>(p.x)/s.width), qy=floor(static_cast<double>(p.y)/s.width), qz=floor(static_cast<double>(p.z)/s.width);
        if (!isfinite(qx) || !isfinite(qy) || !isfinite(qz) || fabs(qx)>0x1p51 || fabs(qy)>0x1p51 || fabs(qz)>0x1p51) {
            ids[i]=UINT_MAX; record_diagnostic(b,2,(i<<32)|52); return;
        }
        const auto x=static_cast<long long>(floor(static_cast<double>(p.x)/s.width))-s.grid_min[0];
        const auto y=static_cast<long long>(floor(static_cast<double>(p.y)/s.width))-s.grid_min[1];
        const auto z=static_cast<long long>(floor(static_cast<double>(p.z)/s.width))-s.grid_min[2];
        if (x<0 || y<0 || z<0 || x>=s.grid_dims[0] || y>=s.grid_dims[1] || z>=s.grid_dims[2]) {
            ids[i]=UINT_MAX; record_diagnostic(b,2,(i<<32)|52); return;
        }
        const auto id=static_cast<std::uint32_t>((static_cast<std::uint64_t>(x)*s.grid_dims[1]+y)*s.grid_dims[2]+z);
        ids[i]=id;
        atomicAdd(counts+id,1U);
    } else if (operation==29) {
        __shared__ std::uint32_t scan[256];
        const auto value=i<m ? counts[i] : 0;
        scan[lane]=value;
        grid_block_scan(scan);
        if (i<=m) offsets[i]=scan[lane]-value;
        if (lane==255) totals[blockIdx.x]=scan[lane];
    } else if (operation==30) {
        // Scan block totals using contiguous per-lane chunks. The largest
        // admitted grid has 65,537 totals: <=257 additions per lane, no serial
        // full-array scan and no fixed shared-memory ceiling on grid size.
        __shared__ std::uint32_t scan[256];
        const auto blocks=(static_cast<std::uint64_t>(m)+256)/256;
        const auto chunk=(blocks+255)/256;
        const auto first=lane*chunk, last=min(first+chunk,blocks);
        std::uint32_t sum=0;
        for (auto k=first;k<last;++k) sum+=totals[k];
        scan[lane]=sum;
        grid_block_scan(scan);
        auto carry=scan[lane]-sum;
        for (auto k=first;k<last;++k) { const auto v=totals[k]; totals[k]=carry; carry+=v; }
    } else if (operation==31) {
        if (i<=m) { const auto value=offsets[i]+totals[blockIdx.x]; offsets[i]=value; if (i<m) cursors[i]=value; }
    } else if (operation==32) {
        if (i<s.count && ids[i]!=UINT_MAX) indices[atomicAdd(cursors+ids[i],1U)]=static_cast<std::uint32_t>(i);
    }
}
