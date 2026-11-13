// Standalone, Driver-only attribution harness. See docs/compact-grid.md.
// Stage events are diagnostic; discovery_wall_ms includes the production bridge
// prefix, total readback, allocation and fill/sort completion boundaries.
#include "../../native/bridge.cpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

static void check(Status s) {
    if (s.code) throw std::runtime_error("CUDA code "+std::to_string(s.code)+" stage "+std::to_string(s.stage));
}
static void cuda_check(CUresult c) { if (c) throw std::runtime_error("CUDA event error "+std::to_string(c)); }
struct Events {
    decltype(&cuEventCreate) create;
    decltype(&cuEventRecord) record;
    decltype(&cuEventSynchronize) sync;
    decltype(&cuEventElapsedTime) elapsed;
    decltype(&cuEventDestroy) destroy;
    CUevent start{},stop{};
    explicit Events(Driver &d) {
        create=reinterpret_cast<decltype(create)>(dlsym(d.library,"cuEventCreate"));
        record=reinterpret_cast<decltype(record)>(dlsym(d.library,"cuEventRecord"));
        sync=reinterpret_cast<decltype(sync)>(dlsym(d.library,"cuEventSynchronize"));
        elapsed=reinterpret_cast<decltype(elapsed)>(dlsym(d.library,"cuEventElapsedTime"));
        destroy=reinterpret_cast<decltype(destroy)>(dlsym(d.library,"cuEventDestroy_v2"));
        if (!create || !record || !sync || !elapsed || !destroy) throw std::runtime_error("event symbols unavailable");
        cuda_check(create(&start,0)); cuda_check(create(&stop,0));
    }
    ~Events() { destroy(start);destroy(stop); }
    template<class F> float time(Resource &r,F f) {
        cuda_check(record(start,r.stream)); f(); cuda_check(record(stop,r.stream));
        cuda_check(sync(stop)); float ms=0;cuda_check(elapsed(&ms,start,stop)); return ms;
    }
};
int main(int argc,char **argv) try {
    if (argc!=6) throw std::runtime_error("usage: compact-grid-bench PTX grid|sort PARTICLES normal|dense|sparse REPETITIONS");
    const bool grid=std::string(argv[2])=="grid";
    const auto parsed=std::stoull(argv[3]);
    if (!parsed || parsed>(1U<<26)) throw std::runtime_error("particle count outside benchmark bound");
    const auto n=static_cast<std::uint32_t>(parsed);
    const std::string distribution=argv[4];
    const float spacing=distribution=="normal" ? 0.6f : distribution=="dense" ? 0.4f : distribution=="sparse" ? 4.0f : 0.0f;
    if (!spacing) throw std::runtime_error("unknown distribution");
    const auto repeats=std::stoi(argv[5]);
    Resource r; check(r.driver.load());
    std::ifstream file(argv[1],std::ios::binary);
    std::string ptx((std::istreambuf_iterator<char>(file)),{});
    if (ptx.empty()) throw std::runtime_error("empty PTX");
    ptx.push_back(0);
    DeviceInfo info{};check(initialise(r,0,ptx.c_str(),89,&info));
    SolverSettings s{};s.count=n;s.padded=1;while (s.padded<n) s.padded*=2;
    s.width=1;s.radius=1;s.neighbour_layout=n>=32000;
    const auto side=static_cast<unsigned int>(std::ceil(std::cbrt(static_cast<double>(n))));
    std::vector<Vector> points(n);
    std::uint32_t dims[3]={};
    for (unsigned int i=0;i<n;++i) {
        points[i]={static_cast<float>(i%side)*spacing,static_cast<float>(i/side%side)*spacing,static_cast<float>(i/(side*side))*spacing};
        const float coordinates[]={points[i].x,points[i].y,points[i].z};
        for (unsigned int a=0;a<3;++a) dims[a]=std::max(dims[a],static_cast<unsigned int>(std::floor(coordinates[a]))+1);
    }
    const auto cells=static_cast<std::uint64_t>(dims[0])*dims[1]*dims[2];
    if (grid && cells<=16777216 && cells<=std::max<std::uint64_t>(4096,8ULL*n)) {
        s.grid_cells=static_cast<unsigned int>(cells);
        for (unsigned int a=0;a<3;++a) s.grid_dims[a]=dims[a];
        const auto words=3*cells+1+2ULL*n+(cells+256)/256;
        s.tile_base=(words+1)/2;
    }
    const std::uint64_t index_bytes=s.grid_cells ? (s.tile_base+(n+31ULL)/32+1)*8 : static_cast<std::uint64_t>(s.padded)*64;
    using MemInfo=decltype(&cuMemGetInfo);
    const auto mem_info=reinterpret_cast<MemInfo>(dlsym(r.driver.library,"cuMemGetInfo_v2"));
    if (!mem_info) throw std::runtime_error("memory info symbol unavailable");
    auto free_memory=[&] { std::size_t free=0,total=0; cuda_check(r.driver.cuCtxPushCurrent_fn(r.context)); cuda_check(mem_info(&free,&total)); CUcontext previous;cuda_check(r.driver.cuCtxPopCurrent_fn(&previous)); return free; };
    const auto initial_free=free_memory();
    const std::uint64_t sizes[]={n*12ULL,n*12ULL+16,n*12ULL,n*12ULL+16,n*4ULL,n*4ULL,n*4ULL,std::max(32ULL,n*12ULL),n*16ULL,0,index_bytes,0,n*4ULL,(n+1ULL)*8};
    for (unsigned int slot=0;slot<14;++slot) check(maelstrom_cuda_storage(&r,slot,nullptr,sizes[slot],0));
    check(maelstrom_cuda_storage(&r,2,points.data(),n*12ULL,1));
    check(maelstrom_cuda_begin_attempt(&r));
    auto build=[&] { if (s.grid_cells) check(solver(r,s,26,128)); else {check(solver(r,s,2,128));check(solver(r,s,13,128));} };
    auto discover=[&] {
        s.entries=0;build();std::uint64_t diagnostic;
        check(solver(r,s,4,128,&diagnostic,false,Observation::CountPrefix));
        if (diagnostic!=UINT64_MAX) throw std::runtime_error("count diagnostic "+std::to_string(diagnostic));
        const auto offset=s.neighbour_layout ? (s.tile_base ? s.tile_base : static_cast<std::uint64_t>(s.padded)*4)+(n+31ULL)/32 : n;
        check(maelstrom_cuda_storage(&r,s.neighbour_layout ? 10 : 13,&s.entries,8,2,offset*8));
        check(maelstrom_cuda_storage(&r,11,nullptr,s.entries*4,0));
        check(solver(r,s,12,128,&diagnostic));
        if (diagnostic!=UINT64_MAX) throw std::runtime_error("fill diagnostic "+std::to_string(diagnostic));
    };
    // One untimed complete discovery warms JIT, allocations and all kernels.
    discover();
    check(solver_call(r,Boundary::HostAccess,[]()->Status{return {0,0};}));
    cuda_check(r.driver.cuCtxPushCurrent_fn(r.context));
    {
    Events events(r.driver);
    for (int sample=0;sample<repeats;++sample) {
        // Context remains current; nested bridge calls restore it correctly.
        const auto start=std::chrono::steady_clock::now();discover();
        const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const float index=events.time(r,build);
        SolverBuffers b{};__builtin_memcpy(&b,r.slots,sizeof(r.slots));
        __builtin_memcpy(&b.diagnostic,&r.diagnostic,sizeof(r.diagnostic));
        auto direct=[&](unsigned int op) {
            void *args[]={&b,&s,&op};
            const auto lanes=op==18 ? 32ULL*n : n;
            cuda_check(r.driver.cuLaunchKernel_fn(r.solver,static_cast<unsigned int>((lanes+127)/128),1,1,128,1,1,0,r.stream,args,nullptr));
        };
        const auto count=events.time(r,[&]{direct(4);});
        std::uint64_t diagnostic;
        check(solver(r,s,4,128,&diagnostic,false,Observation::Prefix));
        if (diagnostic!=UINT64_MAX) throw std::runtime_error("prefix diagnostic");
        const auto fill=events.time(r,[&]{direct(12);});
        const auto sort=events.time(r,[&]{direct(18);});
        // Exact high-water reservation and driver memory observation. The latter
        // includes context/module overhead and is process/device dependent.
        std::size_t reserved=diagnostic_allocation_bytes;
        for (const auto capacity:r.capacities) reserved+=capacity;
        std::cout<<"{\"policy\":\""<<argv[2]<<"\",\"selected\":\""<<(s.grid_cells ? "grid" : "sort")<<"\",\"distribution\":\""<<distribution<<"\",\"particles\":"<<n<<",\"sample\":"<<sample<<",\"index_ms\":"<<index<<",\"count_ms\":"<<count<<",\"fill_ms\":"<<fill<<",\"row_sort_ms\":"<<sort<<",\"discovery_wall_ms\":"<<wall<<",\"reserved_device_bytes\":"<<reserved<<",\"observed_device_growth_bytes\":"<<(initial_free-free_memory())<<",\"index_bytes\":"<<index_bytes<<",\"entries\":"<<s.entries<<"}\n";
    }
    }
    CUcontext previous;cuda_check(r.driver.cuCtxPopCurrent_fn(&previous));
    return 0;
} catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
