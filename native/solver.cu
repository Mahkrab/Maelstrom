// Each lane owns a particle row. Stream-ordered launches complete whole snapshots
// before dependent work; no floating-point atomics or block-local stage barriers.
#include "solver.h"
#include <cmath>
#include <climits>
#include <cfloat>
#include "power.cuh"

// Error numbers map to enduring checked quantities in src/cuda/solver.rs.
struct Checked {
    unsigned int error = 0;
    __device__ float scalar(float value, unsigned int quantity) {
        if (!isfinite(value) && !error) error = quantity;
        return value;
    }
    __device__ Vector vector(Vector v, unsigned int quantity) {
        if ((!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) && !error) error = quantity;
        return v;
    }
};
__device__ Vector add(Vector a, Vector b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ Vector sub(Vector a, Vector b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ Vector mul(Vector a, float b) { return {a.x*b,a.y*b,a.z*b}; }
__device__ Vector cross(Vector a, Vector b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
__device__ Vector divide(Vector a, float b) {
    return {__fdiv_rn(a.x,b),__fdiv_rn(a.y,b),__fdiv_rn(a.z,b)};
}
__device__ Vector neg(Vector a) { return {-a.x,-a.y,-a.z}; }
__device__ float dot(Vector a, Vector b) { return (a.x*b.x+a.y*b.y)+a.z*b.z; }
__device__ float length(Vector a, Checked &c) {
    return c.scalar(__fsqrt_rn(c.scalar(dot(a,a), 2)), 3);
}
__device__ float separation(Vector a, Vector b, Checked &c, unsigned int quantity = 1) {
    return length(c.vector(sub(a,b),quantity),c);
}
__device__ float poly6(float d, float h, Checked &c) {
    if (d >= h) return 0.0f;
    float h2=c.scalar(h*h,4), d2=c.scalar(d*d,5);
    float term=c.scalar(h2-d2,6);
    float h4=h2*h2, h8=h4*h4;
    float h9=c.scalar(h*h8,7);
    float denom=c.scalar((64.0f*0x1.921fb6p+1f)*h9,8);
    float coeff=c.scalar(__fdiv_rn(315.0f,denom),9);
    float cube=c.scalar(term*(term*term),10);
    return c.scalar(coeff*cube,11);
}
__device__ Vector spiky(Vector displacement, float h, Checked &c) {
    float d=length(displacement,c);
    if (d == 0.0f || d >= h) return {0,0,0};
    float h2=h*h, h4=h2*h2;
    float h6=c.scalar(h2*h4,12);
    float denom=c.scalar(0x1.921fb6p+1f*h6,13);
    float coeff=c.scalar(__fdiv_rn(-45.0f,denom),14);
    float remain=c.scalar(h-d,15);
    float scale=c.scalar(__fdiv_rn(coeff*(remain*remain),d),16);
    return c.vector(mul(displacement,scale),17);
}
__device__ bool active(SolverBuffers b, SolverSettings s, unsigned int i, unsigned int j, Checked &c) {
    // Callers enumerate the fixed row; only current-distance filtering remains.
    return separation(b.position[i],b.position[j],c,18) < s.radius;
}
#include "neighbours.cuh"
#include "spatial_sort.cuh"

// Sorting has its own entry point so its shared tile does not reduce occupancy
// for the numerical solver kernels.
extern "C" __global__ void spatial_sort(SolverBuffers b, SolverSettings s, unsigned int operation) {
    __shared__ Cell tile[256];
    if (operation==19) sort_cell_tile(b.cells,s.padded,s.stride,tile);
    else merge_cell_tile(b.cells,s.padded,s.span,s.stride,tile);
}

#include "state_observation.cuh"

extern "C" __global__ void solver(SolverBuffers b, SolverSettings s, unsigned int operation) {
    const auto lane=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (operation==3) {
        if (lane>=s.padded) return;
        unsigned int i=static_cast<unsigned int>(lane), j=i^s.stride;
        if (j>i) {
            Cell a=b.cells[i], other=b.cells[j];
            if (less(other,a) == ((i&s.span)==0)) { b.cells[i]=other; b.cells[j]=a; }
        }
        return;
    }
    if (operation==2) {
        if (lane>=s.padded) return;
        auto i=static_cast<unsigned int>(lane);
        b.cells[i]=i<s.count ? cell(b.position[i],s.width,i) : Cell{LLONG_MAX,LLONG_MAX,LLONG_MAX,UINT_MAX,0};
        return;
    }
    if (operation==18) {
        const auto row=lane/32;
        // Block sizes are multiples of a warp. A tail row therefore exits as
        // a whole warp, leaving every lane named by a shuffle participating.
        if (row>=s.count || (b.diagnostic && b.diagnostic[0]!=ULLONG_MAX)) return;
        const auto owner=static_cast<unsigned int>(row);
        const auto start=b.offsets[owner], stop=b.offsets[owner+1];
        if (b.errors[owner] || start>stop || stop>s.entries) return;
        if (start!=stop) sort_row_cooperatively(b.neighbours+start,stop-start,threadIdx.x%32);
        return;
    }
    if (lane>=s.count) return;
    // Adjacent lanes traverse nearby cells, while offsets and diagnostics stay
    // indexed by particle identity. Every particle still owns exactly one row.
    auto i=(operation==4 || operation==12) ? b.cells[lane].index
        : static_cast<unsigned int>(lane);
    const auto pass=b.first_operation ? operation-b.first_operation : 0;
    // Earlier kernels have completed on this stream. Their records are no
    // longer being written, so these reads do not race this pass's atomic min.
    // Check every earlier slot: skipped kernels leave their own sentinel alone.
    for (unsigned int previous=0;previous<pass;++previous)
        if (b.diagnostic[previous]!=ULLONG_MAX) return;
    Checked c;
    if (operation==0) {
        Vector delta=c.vector(mul(s.acceleration,s.timestep),19);
        b.velocity[i]=c.vector(add(b.old_velocity[i],delta),20);
    } else if (operation==1) {
        Vector delta=c.vector(mul(b.velocity[i],s.timestep),21);
        b.position[i]=c.vector(add(b.old_position[i],delta),22);
    } else if (operation==4 || operation==12) {
        discover_row(b,s,i,c,operation==12);
    } else if (operation==5) {
        // Complete all active-distance checks before any multiplier calculation.
        Members row(b,s,i);
        for (auto j=row.next(s.count);j<s.count;j=row.next(s.count)) {
            separation(b.position[i],b.position[j],c,18);
            if (c.error) break;
        }
    } else if (operation==6) {
        float self=poly6(0,s.radius,c), density=0;
        Vector owner{0,0,0};
        // Density includes self at its identity-ordered position.
        Members density_row(b,s,i,true);
        for (auto j=density_row.next(s.count);j<s.count;j=density_row.next(s.count)) {
            if (i==j) density=c.scalar(density+self,23);
            else if (active(b,s,i,j,c)) density=c.scalar(density+poly6(separation(b.position[i],b.position[j],c,18),s.radius,c),23);
        }
        b.density[i]=density;
        float constraint=c.scalar(c.scalar(__fdiv_rn(density,s.rest_density),24)-1.0f,25);
        b.constraint[i]=constraint;
        float reciprocal=c.scalar(__fdiv_rn(1.0f,s.rest_density),26);
        Checked prefix;
        float sum=0;
        Members owner_row(b,s,i);
        auto suffix=owner_row.stop;
        for (auto j=owner_row.next(s.count);j<s.count;j=owner_row.next(s.count)) {
            if (j>i && suffix==owner_row.stop) suffix=owner_row.cursor-1;
            if (active(b,s,i,j,c)) {
                Vector direction=spiky(sub(b.position[i],b.position[j]),s.radius,c);
                owner=c.vector(add(owner,direction),27);
                Vector dir=c.vector(mul(neg(direction),reciprocal),29);
                // These contributions precede self in the ordered sum. Keep
                // their original additions, but defer errors until all owner
                // checks finish, just as in the separate squared-row traversal.
                if (j<i) sum=prefix.scalar(sum+prefix.scalar(dot(dir,dir),30),31);
            }
        }
        owner=c.vector(mul(owner,reciprocal),28);
        if (!c.error) c.error=prefix.error;
        sum=c.scalar(sum+c.scalar(dot(owner,owner),30),31);
        Members squared_row(b,s,i);
        squared_row.cursor=suffix;
        for (auto j=squared_row.next(s.count);j<s.count;j=squared_row.next(s.count)) {
            if (active(b,s,i,j,c)) {
                Vector dir=c.vector(mul(neg(spiky(sub(b.position[i],b.position[j]),s.radius,c)),reciprocal),29);
                sum=c.scalar(sum+c.scalar(dot(dir,dir),30),31);
            }
        }
        float denom=c.scalar(sum+s.relaxation,32);
        b.multiplier[i]=c.scalar(__fdiv_rn(-constraint,denom),33);
    } else if (operation==7) {
        Vector sum{0,0,0};
        Members correction_row(b,s,i);
        for (auto j=correction_row.next(s.count);j<s.count;j=correction_row.next(s.count)) if (active(b,s,i,j,c)) {
            Vector displacement=sub(b.position[i],b.position[j]);
            float distance=separation(b.position[i],b.position[j],c,18);
            Vector direction=spiky(displacement,s.radius,c);
            float weight=poly6(distance,s.radius,c), pressure=0;
            if (weight!=0 && s.strength!=0) {
                float ratio=c.scalar(__fdiv_rn(weight,s.reference_weight),34);
                float shaped=c.scalar(ordered_power(ratio,s.exponent),35);
                pressure=c.scalar(-s.strength*shaped,36);
            }
            float pair=c.scalar(b.multiplier[i]+b.multiplier[j],37);
            float coefficient=c.scalar(pair+pressure,38);
            Vector contribution=c.vector(mul(direction,coefficient),39);
            sum=c.vector(add(sum,contribution),40);
        }
        Vector correction=c.vector(mul(sum,c.scalar(__fdiv_rn(1.0f,s.rest_density),26)),41);
        b.correction[i]=s.correction_scale==1.0f ? correction : c.vector(mul(correction,s.correction_scale),42);
    } else if (operation==8) {
        c.scalar(b.density[i]*s.mass,43);
    } else if (operation==9) {
        b.candidate[i]=c.vector(add(b.position[i],b.correction[i]),44);
    } else if (operation==10) {
        Vector point=b.candidate[i];
        for (unsigned int k=0;k<s.planes;++k) {
            auto plane=b.planes[k];
            float constraint=c.scalar(c.scalar(dot(plane.normal,point),45)-plane.offset,46);
            if (constraint<0) {
                Vector correction=c.vector(mul(plane.normal,constraint),47);
                point=c.vector(sub(point,correction),48);
                auto n=plane.normal;
                if (n.y==0 && n.z==0 && (n.x==1 || n.x==-1)) point.x=n.x==1 ? plane.offset : -plane.offset;
                else if (n.x==0 && n.z==0 && (n.y==1 || n.y==-1)) point.y=n.y==1 ? plane.offset : -plane.offset;
                else if (n.x==0 && n.y==0 && (n.z==1 || n.z==-1)) point.z=n.z==1 ? plane.offset : -plane.offset;
            }
        }
        b.position[i]=point;
    } else if (operation==11) {
        Vector displacement=c.vector(sub(b.position[i],b.old_position[i]),49);
        float reciprocal=c.scalar(__fdiv_rn(1.0f,s.timestep),50);
        b.velocity[i]=c.vector(mul(displacement,reciprocal),51);
    } else if (operation==14) {
        // Every row reads reconstructed velocity; candidate is a separate output.
        Vector sum{0,0,0};
        Members row(b,s,i);
        for (auto j=row.next(s.count);j<s.count;j=row.next(s.count)) {
            float distance=separation(b.position[i],b.position[j],c,18);
            if (distance>=s.radius) continue;
            float weight=c.scalar(__fdiv_rn(poly6(distance,s.radius,c),s.rest_density),54);
            Vector difference=c.vector(sub(b.velocity[j],b.velocity[i]),53);
            Vector contribution=c.vector(mul(difference,weight),55);
            sum=c.vector(add(sum,contribution),56);
        }
        Vector delta=c.vector(mul(sum,s.xsph),57);
        b.candidate[i]=c.vector(add(b.velocity[i],delta),58);
    } else if (operation==15) {
        // The complete velocity-effect pass has finished on this stream.
        b.velocity[i]=b.candidate[i];
    } else if (operation==16) {
        // Jacobi corrections are finished; their scratch now holds immutable curl.
        Vector omega{0,0,0};
        Members row(b,s,i);
        for (auto j=row.next(s.count);j<s.count;j=row.next(s.count)) {
            Vector displacement=c.vector(sub(b.position[i],b.position[j]),18);
            if (length(displacement,c)>=s.radius) continue;
            Vector gradient=c.vector(divide(spiky(displacement,s.radius,c),s.rest_density),59);
            Vector difference=c.vector(sub(b.velocity[j],b.velocity[i]),60);
            Vector contribution=c.vector(cross(gradient,difference),61);
            omega=c.vector(add(omega,contribution),62);
        }
        b.correction[i]=omega;
        b.density[i]=length(omega,c);
    } else if (operation==17) {
        Vector eta{0,0,0};
        Members row(b,s,i);
        for (auto j=row.next(s.count);j<s.count;j=row.next(s.count)) {
            Vector displacement=c.vector(sub(b.position[i],b.position[j]),18);
            if (length(displacement,c)>=s.radius) continue;
            Vector gradient=c.vector(divide(spiky(displacement,s.radius,c),s.rest_density),59);
            float difference=c.scalar(b.density[j]-b.density[i],63);
            Vector contribution=c.vector(mul(gradient,difference),64);
            eta=c.vector(add(eta,contribution),65);
        }
        float magnitude=length(eta,c);
        if (magnitude==0.0f) b.candidate[i]=b.velocity[i];
        else {
            Vector normal=c.vector(divide(eta,magnitude),66);
            Vector rotation=c.vector(cross(normal,b.correction[i]),67);
            Vector acceleration=c.vector(mul(rotation,s.vorticity),68);
            Vector delta=c.vector(mul(acceleration,s.timestep),69);
            b.candidate[i]=c.vector(add(b.velocity[i],delta),70);
        }
    }
    b.errors[i]=c.error;
    // Only failing lanes contend. The high word preserves particle precedence;
    // Checked already retained the first failing expression within that lane.
    if (c.error && b.diagnostic) atomicMin(b.diagnostic+pass,(static_cast<unsigned long long>(i)<<32)|c.error);
}
