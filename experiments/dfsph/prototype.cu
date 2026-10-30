// Isolated single-phase DFSPH, following BK17 and the SPlisHSPlasH scalar solver.
// The production CUDA neighbour search is included unchanged; no PBF entry point
// or interface is modified. Reference attribution is in reference-LICENSE.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../native/kernels.cu"

static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T>static T* alloc(std::size_t n){T*p=nullptr;if(n)check(cudaMalloc(&p,n*sizeof(T)));return p;}
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
struct Dfsph {
    SolverBuffers rows{};unsigned fluid,total;float h,volume,dt;bool wendland;float jacobi_weight;
    float *rho,*factor,*source,*density_pressure,*divergence_pressure;
    Vector *acceleration;
    double *blocks;
    unsigned *failure;
};

__device__ float cubic_weight(float r,float h){
    const float q=r/h;if(q>=1)return 0;
    const float k=8.0f/(0x1.921fb6p+1f*h*h*h);
    return k*(q<=.5f?6*q*q*q-6*q*q+1:2*(1-q)*(1-q)*(1-q));
}
__device__ Vector cubic_gradient(Vector r,float h){
    float d=sqrtf(dot(r,r)),q=d/h;if(d<=1e-9f||q>=1)return {0,0,0};
    float l=48.0f/(0x1.921fb6p+1f*h*h*h);
    float value=q<=.5f?l*q*(3*q-2):l*(-(1-q)*(1-q));
    return mul(r,value/(d*h));
}
__device__ float weight(Dfsph d,float r){
    if(!d.wendland)return cubic_weight(r,d.h);float q=r/d.h;if(q>=1)return 0;
    float a=1-q;return 21/(2*0x1.921fb6p+1f*d.h*d.h*d.h)*a*a*a*a*(1+4*q);
}
__device__ Vector gradient(Dfsph d,Vector r){
    if(!d.wendland)return cubic_gradient(r,d.h);float distance=sqrtf(dot(r,r)),q=distance/d.h;
    if(distance<=1e-9f||q>=1)return {0,0,0};float a=1-q;
    return mul(r,-210/(0x1.921fb6p+1f*d.h*d.h*d.h)*q*a*a*a/(d.h*distance));
}
__device__ void finite_value(Dfsph d,float value){if(!isfinite(value))atomicOr(d.failure,1u);}
__global__ void density_factor(Dfsph d){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.fluid)return;
    Vector x=d.rows.position[i],owner{0,0,0};float density=weight(d,0),denominator=0;
    for(auto k=d.rows.offsets[i];k<d.rows.offsets[i+1];k++){
        unsigned j=d.rows.neighbours[k];Vector r=sub(x,d.rows.position[j]);float distance=sqrtf(dot(r,r));
        density+=weight(d,distance);Vector g=mul(gradient(d,r),d.volume);owner=add(owner,g);
        if(j<d.fluid)denominator+=dot(g,g);
    }
    denominator+=dot(owner,owner);d.rho[i]=density*d.volume;d.factor[i]=denominator>1e-5f?1/denominator:0;
    finite_value(d,d.rho[i]);finite_value(d,d.factor[i]);
}
__global__ void initialise_pressure(Dfsph d,bool density){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.fluid)return;
    float delta=0;Vector x=d.rows.position[i],v=d.rows.velocity[i];
    for(auto k=d.rows.offsets[i];k<d.rows.offsets[i+1];k++){
        unsigned j=d.rows.neighbours[k];Vector vj=j<d.fluid?d.rows.velocity[j]:Vector{0,0,0};
        delta+=dot(sub(v,vj),gradient(d,sub(x,d.rows.position[j])));
    }
    delta*=d.volume;float*pressure=density?d.density_pressure:d.divergence_pressure;
    float adv=d.rho[i]+d.dt*delta;
    d.source[i]=density?1-adv:-delta;
    bool active=density?adv>1:(delta>0&&d.rows.offsets[i+1]-d.rows.offsets[i]>=20);
    pressure[i]=active?.5f*fminf(pressure[i],density?.00025f:.5f)/(density?d.dt*d.dt:d.dt):0;
    finite_value(d,d.source[i]);finite_value(d,pressure[i]);
}
__global__ void pressure_acceleration(Dfsph d,bool density){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.fluid)return;
    float*p=density?d.density_pressure:d.divergence_pressure;float pi=p[i];Vector x=d.rows.position[i],a{0,0,0};
    for(auto k=d.rows.offsets[i];k<d.rows.offsets[i+1];k++){
        unsigned j=d.rows.neighbours[k];float psum=pi+(j<d.fluid?p[j]:0);
        if(fabsf(psum)>1e-5f)a=sub(a,mul(gradient(d,sub(x,d.rows.position[j])),d.volume*psum));
    }
    d.acceleration[i]=a;finite_value(d,a.x);finite_value(d,a.y);finite_value(d,a.z);
}
__global__ void update_pressure(Dfsph d,bool density){
    __shared__ double values[128];unsigned lane=threadIdx.x,i=blockIdx.x*blockDim.x+lane;double error=0;
    if(i<d.fluid){
        float ap=0;Vector ai=d.acceleration[i],x=d.rows.position[i];
        for(auto k=d.rows.offsets[i];k<d.rows.offsets[i+1];k++){
            unsigned j=d.rows.neighbours[k];Vector aj=j<d.fluid?d.acceleration[j]:Vector{0,0,0};
            ap+=dot(sub(ai,aj),gradient(d,sub(x,d.rows.position[j])));
        }
        float timestep=density?d.dt*d.dt:d.dt;float residual=d.source[i]-timestep*d.volume*ap;
        float*p=density?d.density_pressure:d.divergence_pressure;
        p[i]=fmaxf(p[i]-d.jacobi_weight*residual*d.factor[i]/timestep,0);
        // The author reference suppresses the convergence residual for deficient
        // divergence rows; it still applies its pressure update to those rows.
        error=(!density&&d.rows.offsets[i+1]-d.rows.offsets[i]<20)?0:fmaxf(-residual,0);
        finite_value(d,residual);finite_value(d,p[i]);
    }
    values[lane]=error;__syncthreads();
    for(unsigned stride=64;stride;stride/=2){if(lane<stride)values[lane]+=values[lane+stride];__syncthreads();}
    if(!lane)d.blocks[blockIdx.x]=values[0];
}
__global__ void apply_pressure(Dfsph d,bool density){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.fluid)return;
    d.rows.velocity[i]=add(d.rows.velocity[i],mul(d.acceleration[i],d.dt));
    (density?d.density_pressure:d.divergence_pressure)[i]*=density?d.dt*d.dt:d.dt;
}
__global__ void add_gravity(Dfsph d,Vector gravity){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<d.fluid)d.rows.velocity[i]=add(d.rows.velocity[i],mul(gravity,d.dt));
}
__global__ void integrate(Dfsph d,Vector upper,float inset,Vector gravity){
    __shared__ double block[128][8];unsigned lane=threadIdx.x,i=blockIdx.x*blockDim.x+lane;double value[8]{};
    if(i<d.fluid){
        Vector v=d.rows.velocity[i],old=d.rows.old_velocity[i],p=add(d.rows.position[i],mul(v,d.dt));
        Vector impulse=sub(sub(v,old),mul(gravity,d.dt));
        value[0]=impulse.x;value[1]=impulse.y;value[2]=impulse.z;
        if(inset>0){
            float point[3]{p.x,p.y,p.z},vel[3]{v.x,v.y,v.z},hi[3]{upper.x,upper.y,upper.z};
            for(unsigned a=0;a<3;a++)if(point[a]<inset||point[a]>hi[a]-inset){
                bool lower=point[a]<inset;float before=vel[a];point[a]=lower?inset:hi[a]-inset;
                if(lower?vel[a]<0:vel[a]>0)vel[a]=0;
                value[4]+=.5*(double(before)*before-double(vel[a])*vel[a]);value[5]++;
            }
            p={point[0],point[1],point[2]};v={vel[0],vel[1],vel[2]};
        }
        d.rows.velocity[i]=v;d.rows.position[i]=p;
        value[3]=.5*(double(v.x)*v.x+double(v.y)*v.y+double(v.z)*v.z);value[6]=sqrt(2*value[3]);
        for(float f:{p.x,p.y,p.z,v.x,v.y,v.z})if(!isfinite(f)){value[7]++;atomicOr(d.failure,1u);}
    }
    for(unsigned k=0;k<8;k++)block[lane][k]=value[k];__syncthreads();
    for(unsigned stride=64;stride;stride/=2){if(lane<stride)for(unsigned k=0;k<8;k++)block[lane][k]=k==6?fmax(block[lane][k],block[lane+stride][k]):block[lane][k]+block[lane+stride][k];__syncthreads();}
    if(!lane)for(unsigned k=0;k<8;k++)d.blocks[8*blockIdx.x+k]=block[0][k];
}

int main(int argc,char**argv)try{
    if(argc!=12&&argc!=14&&argc!=15)throw std::runtime_error("input dt steps upperCSV gravityCSV densityPercent divergencePercent prefix observationPeriod inset solid|none [wendland support [JacobiWeight]]");
    std::ifstream in(argv[1],std::ios::binary);SolverSettings s{};in.read(reinterpret_cast<char*>(&s),96);
    if(!in||!s.count||s.count>200000||s.radius!=.1f)throw std::runtime_error("prototype requires bounded particles and 0.1 m support");
    unsigned fluid=s.count,steps=std::stoul(argv[3]);float dt=std::stof(argv[2]),density_tol=std::stof(argv[6])*.01f,divergence_tol=std::stof(argv[7])*.01f;
    if(!steps||steps>10000||!std::isfinite(dt)||!std::isfinite(density_tol)||!std::isfinite(divergence_tol)||dt<=0||dt>.02f||density_tol<=0||divergence_tol<=0)throw std::runtime_error("invalid numerical policy");
    Vector upper{},gravity{};if(std::sscanf(argv[4],"%f,%f,%f",&upper.x,&upper.y,&upper.z)!=3||std::sscanf(argv[5],"%f,%f,%f",&gravity.x,&gravity.y,&gravity.z)!=3)throw std::runtime_error("three-component geometry/gravity required");
    for(float a:{upper.x,upper.y,upper.z})if(!std::isfinite(a)||a<=0||a>100)throw std::runtime_error("box outside bounded prototype domain");
    std::string prefix=argv[8],wall=argv[11];if(wall!="solid"&&wall!="none")throw std::runtime_error("unknown wall model");
    for(float g:{gravity.x,gravity.y,gravity.z})if(!std::isfinite(g))throw std::runtime_error("nonfinite gravity");
    bool wendland=argc>=14;if(wendland){if(std::string(argv[12])!="wendland"||std::stof(argv[13])!=.15f)throw std::runtime_error("unsupported kernel/support control");s.radius=.15f;}
    float jacobi_weight=argc==15?std::stof(argv[14]):.5f;if(jacobi_weight!=.5f&&jacobi_weight!=.1f)throw std::runtime_error("unsupported bounded Jacobi-weight control");
    float dx=.05f,inset=std::stof(argv[10]);if(!std::isfinite(inset)||inset<0||inset>dx/2)throw std::runtime_error("invalid collision extent");
    std::vector<Vector> x(fluid),v(fluid);in.read(reinterpret_cast<char*>(x.data()),fluid*sizeof(Vector));in.read(reinterpret_cast<char*>(v.data()),fluid*sizeof(Vector));
    if(!in)throw std::runtime_error("particle state incomplete");
    in.seekg(0,std::ios::end);if(in.tellg()!=std::streamoff(96+std::uint64_t(fluid)*24+std::uint64_t(s.planes)*16))throw std::runtime_error("input byte count differs");
    for(auto*a:{&x,&v})for(auto p:*a)for(float f:{p.x,p.y,p.z})if(!std::isfinite(f))throw std::runtime_error("nonfinite input");
    unsigned nx=std::lround(upper.x/dx),ny=std::lround(upper.y/dx),nz=std::lround(upper.z/dx);
    int layers=std::lround(s.radius/dx);
    if(wall=="solid"&&(std::uint64_t(nx+2*layers)*(ny+2*layers)*(nz+2*layers)-std::uint64_t(nx)*ny*nz+fluid>512000))throw std::runtime_error("solid sample count exceeds bound");
    if(wall=="solid")for(int a=-layers;a<int(nx)+layers;a++)for(int b=-layers;b<int(ny)+layers;b++)for(int c=-layers;c<int(nz)+layers;c++)
        if(a<0||b<0||c<0||a>=int(nx)||b>=int(ny)||c>=int(nz))x.push_back({(a+.5f)*dx,(b+.5f)*dx,(c+.5f)*dx});
    if(x.size()>512000)throw std::runtime_error("total boundary/fluid count exceeds bound");
    s.count=x.size();s.padded=1;while(s.padded<s.count)s.padded*=2;s.planes=0;s.validate_pairs=0;s.width=dx;while(s.width<s.radius)s.width*=2;
    v.resize(s.count,Vector{0,0,0});check(cudaSetDevice(0));Dfsph d{};d.fluid=fluid;d.total=s.count;d.h=s.radius;d.dt=dt;d.volume=dx*dx*dx;d.wendland=wendland;d.jacobi_weight=jacobi_weight;
    auto&b=d.rows;b.position=alloc<Vector>(s.count);b.velocity=alloc<Vector>(s.count);b.old_velocity=alloc<Vector>(fluid);
    b.cells=alloc<Cell>(2*s.padded);b.offsets=alloc<std::uint64_t>(s.count+1);b.errors=alloc<unsigned>(s.count);b.diagnostic=alloc<unsigned long long>(5);b.correction=alloc<Vector>(s.count);
    unsigned long long*mirror=nullptr,*mapped=nullptr;check(cudaHostAlloc(&mirror,32,cudaHostAllocMapped));check(cudaHostGetDevicePointer(&mapped,mirror,0));check(cudaMemcpy(b.diagnostic+4,&mapped,sizeof(mapped),cudaMemcpyHostToDevice));
    d.rho=alloc<float>(fluid);d.factor=alloc<float>(fluid);d.source=alloc<float>(fluid);d.density_pressure=alloc<float>(fluid);d.divergence_pressure=alloc<float>(fluid);d.acceleration=alloc<Vector>(fluid);d.failure=alloc<unsigned>(1);
    unsigned blocks=(fluid+127)/128;d.blocks=alloc<double>(blocks*8);std::vector<double> block_values(blocks*8);
    check(cudaMemcpy(b.position,x.data(),x.size()*sizeof(Vector),cudaMemcpyHostToDevice));check(cudaMemcpy(b.velocity,v.data(),v.size()*sizeof(Vector),cudaMemcpyHostToDevice));
    check(cudaMemset(d.density_pressure,0,fluid*4));check(cudaMemset(d.divergence_pressure,0,fluid*4));check(cudaMemset(d.failure,0,4));
    std::ofstream trace(prefix+".jsonl");trace.precision(17);double solver_time=0,observation_time=0;std::uint64_t capacity=0,pressure_total=0,divergence_total=0,logical_visits=0,launches=0,contacts=0;unsigned cap_steps=0;
    bool benchmark=std::string(argv[9])=="benchmark";
    float observation_period=benchmark?1.0f/60:std::stof(argv[9]);if(!std::isfinite(observation_period)||observation_period<=0)throw std::runtime_error("invalid observation period");
    unsigned cadence=std::max(1u,unsigned(std::lround(observation_period/dt)));std::ofstream fingerprints(prefix+"-observations.jsonl");
    auto save=[&](unsigned step){auto start=Clock::now();std::vector<Vector> p(fluid),velocity(fluid);check(cudaMemcpy(p.data(),b.position,fluid*sizeof(Vector),cudaMemcpyDeviceToHost));check(cudaMemcpy(velocity.data(),b.velocity,fluid*sizeof(Vector),cudaMemcpyDeviceToHost));observation_time+=elapsed(start);
        // Qualification keeps every sampled state. Timing repeats retain three
        // full states and fingerprints of every60Hz host observation, avoiding
        // repeated multi-gigabyte disk output outside the computation clock.
        std::uint64_t hash=14695981039346656037ULL;for(auto*a:{&p,&velocity}){auto*bytes=reinterpret_cast<const unsigned char*>(a->data());for(std::size_t k=0;k<a->size()*sizeof(Vector);k++){hash^=bytes[k];hash*=1099511628211ULL;}}
        fingerprints<<"{\"step\":"<<step<<",\"fnv1a64\":\""<<hash<<"\"}\n";
        if(benchmark&&step!=0&&step!=1&&step!=steps)return;
        std::ofstream out(prefix+"-"+std::to_string(step)+".bin",std::ios::binary);out.write(reinterpret_cast<char*>(p.data()),p.size()*sizeof(Vector));out.write(reinterpret_cast<char*>(velocity.data()),velocity.size()*sizeof(Vector));if(!out)throw std::runtime_error("snapshot write failed");};
    {std::ofstream out(prefix+"-boundary.bin",std::ios::binary);unsigned count=s.count-fluid;out.write(reinterpret_cast<char*>(&count),4);for(unsigned j=fluid;j<s.count;j++){out.write(reinterpret_cast<char*>(&x[j]),12);out.write(reinterpret_cast<char*>(&d.volume),4);}}
    save(0);
    for(unsigned step=1;step<=steps;step++){
        auto start=Clock::now();std::fill_n(mirror,4,ULLONG_MAX);check(cudaMemset(b.diagnostic,0xff,32));
        check(cudaMemcpy(b.old_velocity,b.velocity,fluid*sizeof(Vector),cudaMemcpyDeviceToDevice));
        solver<<<(s.padded+127)/128,128>>>(b,s,2);launches++;
        unsigned merges=0;for(unsigned width=128;width<s.padded;width*=2)merges++;
        s.stride=merges&1;s.span=0;spatial_sort<<<(s.padded+127)/128,128>>>(b,s,19);launches++;
        for(unsigned width=128;width<s.padded;width*=2){s.span=width;spatial_sort<<<(s.padded+127)/128,128>>>(b,s,20);launches++;s.stride^=1;}
        solver<<<(s.count+127)/128,128>>>(b,s,4);launches++;
        s.stride=1;s.span=(s.count+128)/128;for(unsigned op=21;op<=23;op++){prefix_counts<<<(op==22?1:(s.count+128)/128),128>>>(b,s,op);launches++;}
        check(cudaMemcpy(&s.entries,b.offsets+s.count,8,cudaMemcpyDeviceToHost));std::uint64_t entries=0;check(cudaMemcpy(&entries,b.offsets+fluid,8,cudaMemcpyDeviceToHost));
        if(mirror[0]!=ULLONG_MAX||mirror[1]!=ULLONG_MAX)throw std::runtime_error("native neighbour diagnostic");
        if(s.entries>std::uint64_t(s.count)*(s.count-1)||s.entries>std::numeric_limits<std::size_t>::max()/4)throw std::runtime_error("invalid neighbour total");
        if(s.entries>capacity){std::uint64_t wanted=std::max<std::uint64_t>(4096,capacity);while(wanted<s.entries)wanted*=2;auto*next=alloc<unsigned>(wanted);if(b.neighbours)check(cudaFree(b.neighbours));b.neighbours=next;capacity=wanted;}
        solver<<<(s.count+127)/128,128>>>(b,s,12);solver<<<(std::uint64_t(s.count)*32+127)/128,128>>>(b,s,18);launches+=2;
        density_factor<<<blocks,128>>>(d);launches++;unsigned iterations[2]{};double errors[2]{};
        for(unsigned phase=0;phase<2;phase++){
            bool density=phase==1;if(density){add_gravity<<<blocks,128>>>(d,gravity);launches++;}
            initialise_pressure<<<blocks,128>>>(d,density);launches++;
            double error=0;unsigned count=0;
            do{
                pressure_acceleration<<<blocks,128>>>(d,density);update_pressure<<<blocks,128>>>(d,density);launches+=2;
                check(cudaMemcpy(block_values.data(),d.blocks,blocks*sizeof(double),cudaMemcpyDeviceToHost));error=0;for(unsigned k=0;k<blocks;k++)error+=block_values[k];error/=fluid;count++;
                if(!std::isfinite(error)||mirror[0]!=ULLONG_MAX)throw std::runtime_error("nonfinite pressure or neighbour diagnostic");
            }while(count<100&&(count<(density?2u:1u)||error>(density?density_tol:divergence_tol/dt)));
            pressure_acceleration<<<blocks,128>>>(d,density);apply_pressure<<<blocks,128>>>(d,density);launches+=2;
            iterations[phase]=count;errors[phase]=error;logical_visits+=entries*(2*count+2);
        }
        integrate<<<blocks,128>>>(d,upper,inset,gravity);launches++;check(cudaMemcpy(block_values.data(),d.blocks,blocks*8*sizeof(double),cudaMemcpyDeviceToHost));
        unsigned failure=0;check(cudaMemcpy(&failure,d.failure,4,cudaMemcpyDeviceToHost));if(failure)throw std::runtime_error("nonfinite candidate");
        double statistics[8]{};for(unsigned k=0;k<blocks;k++)for(unsigned j=0;j<8;j++)statistics[j]=j==6?std::max(statistics[j],block_values[8*k+j]):statistics[j]+block_values[8*k+j];
        double time=elapsed(start);solver_time+=time;logical_visits+=entries;pressure_total+=iterations[1];divergence_total+=iterations[0];contacts+=statistics[5];if(iterations[0]==100||iterations[1]==100)cap_steps++;
        trace<<"{\"step\":"<<step<<",\"dt\":"<<dt<<",\"solver_s\":"<<time<<",\"density_iterations\":"<<iterations[1]<<",\"divergence_iterations\":"<<iterations[0]<<",\"density_residual_mean\":"<<errors[1]<<",\"divergence_residual_mean\":"<<errors[0]<<",\"fluid_neighbour_entries\":"<<entries<<",\"all_neighbour_entries\":"<<s.entries<<",\"row_sweeps\":"<<(5+2*(iterations[0]+iterations[1]))<<",\"kinetic_J_kg\":"<<statistics[3]/fluid<<",\"max_speed\":"<<statistics[6]<<",\"pressure_impulse_per_mass\":["<<statistics[0]/fluid<<","<<statistics[1]/fluid<<","<<statistics[2]/fluid<<"],\"wall_contacts\":"<<statistics[5]<<",\"wall_energy_loss_J_kg\":"<<statistics[4]/fluid<<"}\n";
        if(step==1||step%cadence==0||step==steps)save(step);
    }
    std::cout.precision(17);std::cout<<"{\"particles\":"<<fluid<<",\"boundary_particles\":"<<s.count-fluid<<",\"steps\":"<<steps<<",\"dt\":"<<dt<<",\"simulated_s\":"<<double(dt)*steps<<",\"solver_s\":"<<solver_time<<",\"observation_s\":"<<observation_time<<",\"density_iterations\":"<<pressure_total<<",\"divergence_iterations\":"<<divergence_total<<",\"fluid_row_entry_visits\":"<<logical_visits<<",\"search_count_fill_passes\":"<<2*steps<<",\"launches\":"<<launches<<",\"cap_steps\":"<<cap_steps<<",\"wall_contacts\":"<<contacts<<",\"neighbour_capacity_entries\":"<<capacity<<"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
