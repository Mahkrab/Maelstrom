// Inspect JIT resource costs for matched module exports; no kernels are run.
#include "../../native/bridge.cpp"
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
static std::string read(const char *path) {
    std::ifstream file(path,std::ios::binary);
    std::string value((std::istreambuf_iterator<char>(file)),{});
    if (value.empty()) throw std::runtime_error("empty PTX");
    value.push_back(0);return value;
}
int main(int argc,char **argv) try {
    if (argc!=3) throw std::runtime_error("usage: resources CANDIDATE_PTX CONTROL_PTX");
    Resource r;DeviceInfo info{};
    if (r.driver.load().code) throw std::runtime_error("load driver");
    const auto current=read(argv[1]);
    if (initialise(r,0,current.c_str(),89,&info).code) throw std::runtime_error("initialise");
    if (r.driver.cuCtxPushCurrent_fn(r.context)) throw std::runtime_error("bind context");
    const auto attribute=reinterpret_cast<decltype(&cuFuncGetAttribute)>(dlsym(r.driver.library,"cuFuncGetAttribute"));
    if (!attribute) throw std::runtime_error("attribute symbol");
    const auto old=read(argv[2]);CUmodule module{};
    if (r.driver.cuModuleLoadData_fn(&module,old.c_str())) throw std::runtime_error("control module");
    for (auto source : {std::make_pair("candidate",r.module),std::make_pair("original",module)}) {
        for (auto name : {"solver","discover_neighbours","compact_grid","prefix_counts","spatial_sort"}) {
            CUfunction function{};
            if (r.driver.cuModuleGetFunction_fn(&function,source.second,name)) continue;
            int registers=0,local=0,shared=0;
            if (attribute(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,function) || attribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function) || attribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function)) throw std::runtime_error("attributes");
            std::cout<<"{\"source\":\""<<source.first<<"\",\"kernel\":\""<<name<<"\",\"registers\":"<<registers<<",\"local_bytes\":"<<local<<",\"shared_bytes\":"<<shared<<"}\n";
        }
    }
    CUcontext previous;if (r.driver.cuCtxPopCurrent_fn(&previous)) throw std::runtime_error("unbind");
    return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
