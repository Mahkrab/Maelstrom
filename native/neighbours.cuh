// Prediction-time rows contain only actual neighbours, sorted by identity. The
// count and fill passes share geometry; checked prefix totals determine storage.
#pragma once
#include "row_sort.cuh"

struct Members {
    const std::uint32_t *indices;
    std::uint64_t cursor, stop;
    unsigned int owner;
    bool include_self;
    __device__ Members(SolverBuffers b, SolverSettings, unsigned int i, bool self=false)
        :indices(b.neighbours),cursor(b.offsets[i]),stop(b.offsets[i+1]),owner(i),include_self(self) {}
    __device__ unsigned int next(unsigned int end) {
        if (include_self && (cursor==stop || owner<indices[cursor])) {
            include_self=false;
            return owner;
        }
        return cursor==stop ? end : indices[cursor++];
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

__device__ void discover_row(SolverBuffers b, SolverSettings s, unsigned int i,
                             Checked &c, bool fill) {
    if (!fill && s.validate_pairs) for (unsigned int j=0;j<s.count;++j) if (i!=j) {
        separation(b.position[i],b.position[j],c);
        if (c.error) break;
    }
    if (c.error) return;
    std::uint64_t cursor=fill ? b.offsets[i] : 0;
    auto start=cursor;
    std::uint64_t stop=fill ? b.offsets[i+1] : 0;
    if (fill && (start>stop || stop>s.entries)) { c.error=52; return; }
    Cell origin=cell(b.position[i],s.width,i);
    for (int x=-1;x<=1;++x) for (int y=-1;y<=1;++y) {
        // Each z-cell run keeps its own loop. The sorted cursor can advance
        // between all three without repeating the column's lower-bound search.
        Cell target{origin.x+x,origin.y+y,origin.z-1,0,0};
        unsigned int lo=0, hi=s.count;
        while (lo<hi) { auto mid=lo+(hi-lo)/2; if (less(b.cells[mid],target)) lo=mid+1; else hi=mid; }
        auto k=lo;
        for (int z=-1;z<=1;++z) {
            target.z=origin.z+z;
            for (;k<s.count && same(b.cells[k],target);++k) {
                auto j=b.cells[k].index;
                if (i!=j && separation(b.position[i],b.position[j],c)<s.radius) {
                    if (fill) {
                        if (cursor==stop) { c.error=52; return; }
                        b.neighbours[cursor]=j;
                    }
                    ++cursor;
                }
            }
        }
    }
    if (fill) {
        if (cursor!=stop) { c.error=52; return; }
    } else {
        if (i==0) b.offsets[0]=0;
        b.offsets[i+1]=cursor;
    }
}
