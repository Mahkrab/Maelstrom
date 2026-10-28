// Summaries replace bulk observations, while particle vectors stay in their
// working allocations. Maxima and extrema reorder no floating-point sums.
#pragma once
struct StateSummary { float scalar; Vector lower, upper; float reserved; };
static_assert(sizeof(StateSummary)==32);

__device__ void record_diagnostic(SolverBuffers b, unsigned int slot, unsigned long long value) {
    const auto previous=atomicMin(b.diagnostic+slot,value);
    if (value<previous) {
        // Successful lanes never access host memory. Winning updates alone
        // reduce PCIe traffic on failures; a second minimum tolerates their
        // different arrival order. Host access waits for stream completion.
        auto host=reinterpret_cast<unsigned long long *>(b.diagnostic[diagnostic_record_count]);
        atomicMin(host+slot,value);
    }
}

__device__ unsigned long long *plane_observation(SolverBuffers b, SolverSettings s) {
    // A Vector is 12 bytes; odd counts need one u32 before the aligned record.
    auto tail=reinterpret_cast<std::uint32_t *>(b.velocity+s.count);
    return reinterpret_cast<unsigned long long *>(tail+(s.count&1U));
}

__device__ void reset_plane_observation(SolverBuffers b, SolverSettings s) {
    auto tail=reinterpret_cast<std::uint32_t *>(b.velocity+s.count);
    for (unsigned int k=0;k<velocity_observation_bytes/sizeof(*tail);++k) tail[k]=0;
    *plane_observation(b,s)=ULLONG_MAX;
}

__device__ void observe_final_planes(SolverBuffers b, SolverSettings s, unsigned int i) {
    const auto point=b.position[i];
    for (unsigned int k=0;k<s.planes;++k) {
        const auto plane=b.planes[k];
        const float product=dot(plane.normal,point), constraint=product-plane.offset;
        if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) ||
            !isfinite(product) || !isfinite(constraint) || constraint<0.0f) {
            atomicMin(plane_observation(b,s),(static_cast<unsigned long long>(i)<<32)|k);
            break;
        }
    }
}

extern "C" __global__ void observe_state(SolverBuffers b, SolverSettings s, unsigned int operation) {
    __shared__ StateSummary values[256];
    const auto lane=threadIdx.x;
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+lane;
    StateSummary value{0,{FLT_MAX,FLT_MAX,FLT_MAX},{-FLT_MAX,-FLT_MAX,-FLT_MAX},0};
    if (i<s.count) {
        Checked c;
        if (operation==0) {
            Vector delta=c.vector(mul(s.acceleration,s.timestep),19);
            b.velocity[i]=c.vector(add(b.old_velocity[i],delta),20);
            // Acceleration errors precede every CFL error, regardless of the
            // particle index. Each stage therefore owns a separate result slot.
            Checked speed;
            value.scalar=length(b.velocity[i],speed);
            if (speed.error) {
                record_diagnostic(b,1,(static_cast<unsigned long long>(i)<<32)|speed.error);
                value.scalar=0;
            }
        } else {
            Vector delta=c.vector(mul(b.velocity[i],s.timestep),21);
            Vector p=c.vector(add(b.old_position[i],delta),22);
            b.position[i]=p;
            value.scalar=fmaxf(fabsf(p.x),fmaxf(fabsf(p.y),fabsf(p.z)));
            value.lower=p;value.upper=p;
        }
        b.errors[i]=c.error;
        if (c.error) record_diagnostic(b,0,(static_cast<unsigned long long>(i)<<32)|c.error);
    }
    values[lane]=value;
    __syncthreads();
    for (unsigned int offset=blockDim.x/2;offset;offset/=2) {
        if (lane<offset) {
            auto &a=values[lane];const auto other=values[lane+offset];
            a.scalar=fmaxf(a.scalar,other.scalar);
            if (operation==1) {
                a.lower={fminf(a.lower.x,other.lower.x),fminf(a.lower.y,other.lower.y),fminf(a.lower.z,other.lower.z)};
                a.upper={fmaxf(a.upper.x,other.upper.x),fmaxf(a.upper.y,other.upper.y),fmaxf(a.upper.z,other.upper.z)};
            }
        }
        __syncthreads();
    }
    if (!lane) reinterpret_cast<StateSummary *>(b.correction)[blockIdx.x]=values[0];
}

// Both levels scan integers in shared memory. The read barrier prevents a
// faster lane from overwriting a value its neighbour still needs this round.
__device__ void inclusive_block_prefix(std::uint64_t *scan) {
    const auto lane=threadIdx.x;
    __syncthreads();
    for (unsigned int offset=1;offset<blockDim.x;offset*=2) {
        const auto previous=lane>=offset ? scan[lane-offset] : 0;
        __syncthreads();
        scan[lane]+=previous;
        __syncthreads();
    }
}

// Counted rows have at most count-1 entries. With the checked power-of-two u32
// dimension, their complete sum fits u64. Invalid counts are diagnosed before
// later kernels may use the resulting offsets or allocate neighbour storage.
extern "C" __global__ void prefix_counts(SolverBuffers b, SolverSettings s, unsigned int operation) {
    if (b.diagnostic[0]!=ULLONG_MAX) return;
    const auto lane=threadIdx.x;
    __shared__ std::uint64_t scan[256];
    const auto n=static_cast<std::uint64_t>(s.count)+1;
    auto *totals=reinterpret_cast<std::uint64_t *>(b.correction);
    if (operation==22) {
        if (b.diagnostic[1]!=ULLONG_MAX || blockIdx.x) return;
        // Each lane scans one contiguous slice of block totals, then the block
        // scan supplies its carry. This supports any row count without a fixed
        // shared-memory ceiling or a serial pass over every total.
        const auto chunk=(s.span+blockDim.x-1)/blockDim.x;
        const auto start=lane*chunk;
        const auto stop=min(start+chunk,s.span);
        std::uint64_t sum=0;
        for (auto block=start;block<stop;++block) sum+=totals[block];
        scan[lane]=sum;
        inclusive_block_prefix(scan);
        auto carry=lane ? scan[lane-1] : 0;
        for (auto block=start;block<stop;++block) {
            const auto value=totals[block];totals[block]=carry;carry+=value;
        }
        return;
    }
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+lane;
    if (operation==23) {
        if (i<n && b.diagnostic[1]==ULLONG_MAX) b.offsets[i]+=totals[blockIdx.x];
        return;
    }
    std::uint64_t value=i<n && (s.count || !s.stride) ? b.offsets[i] : 0;
    if (i<n && ((!i && value) || (i && value>s.count-1))) {
        const auto particle=i ? i-1 : 0;
        record_diagnostic(b,1,(static_cast<unsigned long long>(particle)<<32)|52);
        value=0;
    }
    scan[lane]=value;
    inclusive_block_prefix(scan);
    if (i<n) b.offsets[i]=scan[lane];
    if (lane==blockDim.x-1) totals[blockIdx.x]=scan[lane];
}
