// Each lane owns a particle row. Stream-ordered launches complete whole snapshots
// before dependent work; no floating-point atomics or block-local stage barriers.
#include "solver.h"
#include <cmath>
#include <climits>
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
__device__ bool member(SolverBuffers b, SolverSettings s, unsigned int i, unsigned int j) {
    return (b.membership[static_cast<std::size_t>(i)*s.words+j/32] >> (j%32)) & 1u;
}
__device__ bool active(SolverBuffers b, SolverSettings s, unsigned int i, unsigned int j, Checked &c) {
    return member(b,s,i,j) && separation(b.position[i],b.position[j],c,18) < s.radius;
}
struct Members {
    const unsigned int *row;
    unsigned int words, word=0, bits=0, base=0, owner;
    bool include_self;
    __device__ Members(SolverBuffers b,SolverSettings s,unsigned int i,bool self=false)
        :row(b.membership+static_cast<std::size_t>(i)*s.words),words(s.words),owner(i),include_self(self) {}
    __device__ unsigned int next(unsigned int end) {
        while (!bits) {
            if (word==words) return end;
            base=word*32;
            bits=row[word];
            if (include_self && word==owner/32) bits |= 1u<<(owner%32);
            ++word;
        }
        unsigned int index=static_cast<unsigned int>(__ffs(bits)-1);
        bits &= bits-1;
        return base+index;
    }
};
__device__ bool less(Cell a, Cell b) {
    if (a.x!=b.x) return a.x<b.x;
    if (a.y!=b.y) return a.y<b.y;
    if (a.z!=b.z) return a.z<b.z;
    return a.index<b.index;
}
__device__ bool same(Cell a, Cell b) { return a.x==b.x && a.y==b.y && a.z==b.z; }
__device__ Cell cell(Vector p, double width, unsigned int i) {
    return {static_cast<long long>(floor(static_cast<double>(p.x)/width)),
            static_cast<long long>(floor(static_cast<double>(p.y)/width)),
            static_cast<long long>(floor(static_cast<double>(p.z)/width)),i,0};
}

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
    if (lane>=s.count) return;
    auto i=static_cast<unsigned int>(lane);
    Checked c;
    if (operation==0) {
        Vector delta=c.vector(mul(s.acceleration,s.timestep),19);
        b.velocity[i]=c.vector(add(b.old_velocity[i],delta),20);
    } else if (operation==1) {
        Vector delta=c.vector(mul(b.velocity[i],s.timestep),21);
        b.position[i]=c.vector(add(b.old_position[i],delta),22);
    } else if (operation==4) {
        auto row=b.membership+static_cast<std::size_t>(i)*s.words;
        for (unsigned int w=0;w<s.words;++w) row[w]=0;
        // Unsafe bounding diagonals require the reference's distant-pair checks.
        if (s.validate_pairs) for (unsigned int j=0;j<s.count;++j) if (i!=j) {
            separation(b.position[i],b.position[j],c);
            if (c.error) break;
        }
        Cell origin=cell(b.position[i],s.width,i);
        for (int x=-1;x<=1;++x) for (int y=-1;y<=1;++y) for (int z=-1;z<=1;++z) {
            Cell target{origin.x+x,origin.y+y,origin.z+z,0,0};
            unsigned int lo=0, hi=s.count;
            while (lo<hi) { auto mid=lo+(hi-lo)/2; if (less(b.cells[mid],target)) lo=mid+1; else hi=mid; }
            for (auto k=lo;k<s.count && same(b.cells[k],target);++k) {
                auto j=b.cells[k].index;
                if (i!=j && separation(b.position[i],b.position[j],c)<s.radius) row[j/32] |= 1u<<(j%32);
            }
        }
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
        Members owner_row(b,s,i);
        for (auto j=owner_row.next(s.count);j<s.count;j=owner_row.next(s.count)) if (active(b,s,i,j,c)) {
            Vector direction=spiky(sub(b.position[i],b.position[j]),s.radius,c);
            owner=c.vector(add(owner,direction),27);
            c.vector(mul(neg(direction),reciprocal),29);
        }
        owner=c.vector(mul(owner,reciprocal),28);
        float sum=0;
        Members squared_row(b,s,i,true);
        for (auto j=squared_row.next(s.count);j<s.count;j=squared_row.next(s.count)) {
            if (i==j) sum=c.scalar(sum+c.scalar(dot(owner,owner),30),31);
            else if (active(b,s,i,j,c)) {
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
    }
    b.errors[i]=c.error;
}
