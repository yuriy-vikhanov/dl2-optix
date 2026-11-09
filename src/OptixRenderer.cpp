#include "OptixRenderer.h"
#include "NurbsModel.h"
#include <optix_function_table_definition.h> // Exactly one definition in the executable.
#include "ErrorChecks.h"
#include "DeviceShared.h"
#include <optix_stack_size.h>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <utility>

namespace nurbs {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
void timing(const std::string& name,double milliseconds) { std::cout << std::left << std::setw(26) << name << std::fixed << std::setprecision(3) << milliseconds << " ms\n"; }
class Buffer {
public:
    Buffer()=default;
    ~Buffer() { cleanup([&] { if(pointer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pointer))); }); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& b) noexcept: pointer(std::exchange(b.pointer,0)),size(std::exchange(b.size,0)) {}
    Buffer& operator=(Buffer&& b) noexcept {
        if(this!=&b) { cleanup([&] { if(pointer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pointer))); }); pointer=std::exchange(b.pointer,0); size=std::exchange(b.size,0); }
        return *this;
    }
    void resize(std::size_t bytes) {
        if(bytes==size) return;
        if(pointer) { CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pointer))); pointer=0; size=0; }
        if(bytes) { void* p=nullptr; CUDA_CHECK(cudaMalloc(&p,bytes)); pointer=reinterpret_cast<CUdeviceptr>(p); size=bytes; }
    }
    void upload(const void* source,std::size_t bytes) { resize(bytes); if(bytes) CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pointer),source,bytes,cudaMemcpyHostToDevice)); }
    template<class T> void upload(const std::vector<T>& values) { upload(values.data(),values.size()*sizeof(T)); }
    template<class T> T* as() const { return reinterpret_cast<T*>(pointer); }
    CUdeviceptr address() const { return pointer; }
    std::size_t bytes() const { return size; }
private:
    CUdeviceptr pointer=0;
    std::size_t size=0;
};
template<class T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct EmptyData { unsigned unused[OPTIX_SBT_RECORD_ALIGNMENT/sizeof(unsigned)]{}; };
using EmptyRecord=Record<EmptyData>;
using HitRecord=Record<HitData>;
static_assert(sizeof(HitRecord)%OPTIX_SBT_RECORD_ALIGNMENT==0);
struct Gas { Buffer output; OptixTraversableHandle handle=0; };
void contextLog(unsigned level,const char* tag,const char* message,void*) {
    std::cerr << "[OptiX " << level << "][" << tag << "] " << message << '\n';
}
float checkedFloat(double x) {
    const auto value=static_cast<float>(x);
    if(!std::isfinite(value)) throw std::runtime_error("Scene coordinates exceed GPU float range");
    return value;
}
float3 gpu(Vec3 v) { return {checkedFloat(v.x),checkedFloat(v.y),checkedFloat(v.z)}; }
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        try { CUDA_CHECK(cudaEventCreate(&stop)); }
        catch(...) { cleanup([&] { CUDA_CHECK(cudaEventDestroy(start)); }); throw; }
    }
    ~EventTimer() {
        cleanup([&] { CUDA_CHECK(cudaEventDestroy(stop)); });
        cleanup([&] { CUDA_CHECK(cudaEventDestroy(start)); });
    }
    void begin(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start,stream)); }
    double end(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(stop,stream)); CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop)); return ms;
    }
private:
    cudaEvent_t start=nullptr,stop=nullptr;
};
}

struct Renderer {
    cudaStream_t stream=nullptr;
    OptixDeviceContext context=nullptr;
    OptixModule module=nullptr;
    OptixProgramGroup raygen=nullptr,miss=nullptr,hitgroup=nullptr;
    OptixPipeline pipeline=nullptr;
    OptixShaderBindingTable sbt{};
    Bounds bounds;
    Vec3 renderOrigin;
    Gas gas;
    Buffer raygenRecord,missRecord,hitRecord,launchBuffer,pixels;
    Buffer nurbsPatches,nurbsTrims,nurbsAabbs,surfaces,solverFailures;

    ~Renderer() {
        if(stream) cleanup([&] { CUDA_CHECK(cudaStreamSynchronize(stream)); });
        if(pipeline) cleanup([&] { OPTIX_CHECK(optixPipelineDestroy(pipeline)); });
        if(hitgroup) cleanup([&] { OPTIX_CHECK(optixProgramGroupDestroy(hitgroup)); });
        if(miss) cleanup([&] { OPTIX_CHECK(optixProgramGroupDestroy(miss)); });
        if(raygen) cleanup([&] { OPTIX_CHECK(optixProgramGroupDestroy(raygen)); });
        if(module) cleanup([&] { OPTIX_CHECK(optixModuleDestroy(module)); });
        if(context) cleanup([&] { OPTIX_CHECK(optixDeviceContextDestroy(context)); });
        if(stream) cleanup([&] { CUDA_CHECK(cudaStreamDestroy(stream)); });
        // Buffer members release their CUDA allocations after OptiX destruction.
    }
    void initialize(const std::filesystem::path& path) {
        int count=0; CUDA_CHECK(cudaGetDeviceCount(&count));
        if(count==0) throw std::runtime_error("No NVIDIA CUDA device found; this renderer requires an OptiX-compatible NVIDIA GPU");
        CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
        std::cout << "CUDA device: " << prop.name << " (compute " << prop.major << '.' << prop.minor << ")\n";
        CUDA_CHECK(cudaStreamCreate(&stream));
        OPTIX_CHECK(optixInit());
        CUcontext cudaContext=nullptr; DRIVER_CHECK(cuCtxGetCurrent(&cudaContext));
        if(!cudaContext) throw std::runtime_error("CUDA primary context was not initialized");
        OptixDeviceContextOptions options{};
        options.logCallbackFunction=contextLog; options.logCallbackLevel=3;
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext,&options,&context));
        buildPipeline(path);
    }
    void buildPipeline(const std::filesystem::path& path) {
        const auto start=Clock::now();
        std::ifstream file(path,std::ios::binary);
        if(!file) throw std::runtime_error("Cannot open generated device program: "+path.string()+". Build NurbsOptixRenderer to generate/copy DevicePrograms.ptx.");
        const std::string ptx((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        if(ptx.empty() || file.bad()) throw std::runtime_error("Generated device program is empty or unreadable");
        OptixModuleCompileOptions moduleOptions{};
#ifndef NDEBUG
        moduleOptions.optLevel=OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleOptions.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        moduleOptions.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        OptixPipelineCompileOptions compile{};
        compile.usesMotionBlur=false;
        compile.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile.numPayloadValues=2; compile.numAttributeValues=3;
        compile.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
        compile.pipelineLaunchParamsVariableName="params";
        compile.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        std::array<char,16384> log{}; std::size_t logSize=log.size();
        OPTIX_CHECK_LOG(optixModuleCreate(context,&moduleOptions,&compile,ptx.data(),ptx.size(),log.data(),&logSize,&module),log.data());
        if(log[0]) std::cout << "OptiX module: " << log.data() << '\n';
        auto makeGroup=[&](OptixProgramGroupDesc description,OptixProgramGroup& group) {
            OptixProgramGroupOptions options{}; log.fill(0); logSize=log.size();
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context,&description,1,&options,log.data(),&logSize,&group),log.data());
            if(log[0]) std::cout << "OptiX program: " << log.data() << '\n';
        };
        OptixProgramGroupDesc description{};
        description.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        description.raygen.module=module; description.raygen.entryFunctionName="__raygen__render";
        makeGroup(description,raygen);
        description={}; description.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
        description.miss.module=module; description.miss.entryFunctionName="__miss__background";
        makeGroup(description,miss);
        description={}; description.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        description.hitgroup.moduleIS=module;
        description.hitgroup.entryFunctionNameIS="__intersection__nurbs";
        description.hitgroup.moduleCH=module;
        description.hitgroup.entryFunctionNameCH="__closesthit__surface";
        makeGroup(description,hitgroup);
        const std::array groups{raygen,miss,hitgroup};
        OptixPipelineLinkOptions link{}; link.maxTraceDepth=1;
        log.fill(0); logSize=log.size();
        OPTIX_CHECK_LOG(optixPipelineCreate(context,&compile,&link,groups.data(),static_cast<unsigned>(groups.size()),log.data(),&logSize,&pipeline),log.data());
        if(log[0]) std::cout << "OptiX pipeline: " << log.data() << '\n';
        OptixStackSizes sizes{};
        for(const auto group:groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(group,&sizes,pipeline));
        unsigned fromTraversal=0,fromState=0,continuation=0;
        OPTIX_CHECK(optixUtilComputeStackSizes(&sizes,1,0,0,&fromTraversal,&fromState,&continuation));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline,fromTraversal,fromState,continuation,1)); // Single GAS
        EmptyRecord rg{},ms{};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen,&rg)); OPTIX_CHECK(optixSbtRecordPackHeader(miss,&ms));
        raygenRecord.upload(&rg,sizeof(rg)); missRecord.upload(&ms,sizeof(ms));
        sbt.raygenRecord=raygenRecord.address(); sbt.missRecordBase=missRecord.address();
        sbt.missRecordStrideInBytes=sizeof(EmptyRecord); sbt.missRecordCount=1;
        timing("OptiX pipeline build:",elapsed(start));
    }
    void buildNurbsGas(const NurbsModel& source) {
        const auto start=Clock::now();
        auto patches=source.patches;
        std::vector<OptixAabb> aabbs;
        for(auto& patch:patches) {
            Bounds patchBounds;
            for(unsigned i=0;i<(patch.degreeU+1)*(patch.degreeV+1);++i) {
                auto& p=patch.poles[i];
                p.x-=renderOrigin.x*p.w; p.y-=renderOrigin.y*p.w; p.z-=renderOrigin.z*p.w;
                patchBounds.add({p.x/p.w,p.y/p.w,p.z/p.w});
            }
            // Positive weights keep the exact surface in its control hull.
            // Expand before converting to float, including planar AABBs.
            const double pad=std::max(1e-6,source.bounds.radius()*2e-6);
            aabbs.push_back({checkedFloat(patchBounds.lo.x-pad),checkedFloat(patchBounds.lo.y-pad),checkedFloat(patchBounds.lo.z-pad),
                checkedFloat(patchBounds.hi.x+pad),checkedFloat(patchBounds.hi.y+pad),checkedFloat(patchBounds.hi.z+pad)});
        }
        nurbsPatches.upload(patches); nurbsTrims.upload(source.trims); nurbsAabbs.upload(aabbs);
        const CUdeviceptr aabbBuffer=nurbsAabbs.address();
        const unsigned flags=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput input{}; input.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers=&aabbBuffer;
        input.customPrimitiveArray.numPrimitives=static_cast<unsigned>(aabbs.size());
        input.customPrimitiveArray.strideInBytes=sizeof(OptixAabb);
        input.customPrimitiveArray.flags=&flags; input.customPrimitiveArray.numSbtRecords=1;
        OptixAccelBuildOptions options{}; options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation=OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{}; OPTIX_CHECK(optixAccelComputeMemoryUsage(context,&options,&input,1,&sizes));
        Buffer scratch;
        scratch.resize(sizes.tempSizeInBytes); gas.output.resize(sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(context,stream,&options,&input,1,scratch.address(),scratch.bytes(),gas.output.address(),gas.output.bytes(),&gas.handle,nullptr,0));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timing("GAS rational patches:",elapsed(start));
    }
    void uploadSbt() {
        HitRecord record{};
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup,&record));
        record.data={nurbsPatches.as<NurbsPatch>(),nurbsTrims.as<NurbsTrim>()};
        hitRecord.upload(&record,sizeof(record));
        sbt.hitgroupRecordBase=hitRecord.address();
        sbt.hitgroupRecordStrideInBytes=sizeof(HitRecord); sbt.hitgroupRecordCount=1;
    }
    RenderResult render(unsigned width,unsigned height,double cameraElevation,unsigned benchmarkFrames,bool profile,unsigned samplesPerPixel) {
        const double radius=bounds.radius();
        const Vec3 center=bounds.center();
        const double tangent=std::tan(40.0*std::numbers::pi/360.0);
        const double aspect=static_cast<double>(width)/height;
        const double fitHalfAngle=std::atan(tangent*std::min(1.0,aspect));
        const double angle=cameraElevation*std::numbers::pi/180.0;
        const double horizontal=std::cos(angle)/std::sqrt(2.0);
        const Vec3 offset{horizontal,-horizontal,std::sin(angle)};
        const Vec3 eye=center+normalized(offset)*(1.1*radius/std::sin(fitHalfAngle));
        const Vec3 forward=normalized(center-eye);
        const Vec3 right=normalized(cross(forward,Vec3{0,0,1}));
        const Vec3 up=normalized(cross(right,forward));
        const std::size_t count=static_cast<std::size_t>(width)*height;
        pixels.resize(count*sizeof(uchar4)); surfaces.resize(count*sizeof(float4));
        solverFailures.resize(sizeof(unsigned long long));
        CUDA_CHECK(cudaMemsetAsync(solverFailures.as<void>(),0,solverFailures.bytes(),stream));
        LaunchParams launch{};
        launch.traversable=gas.handle; launch.pixels=pixels.as<uchar4>();
        launch.width=width; launch.height=height;
        launch.eye=gpu(eye-renderOrigin); launch.forward=gpu(forward); launch.right=gpu(right); launch.up=gpu(up);
        launch.tanHalfFov=checkedFloat(tangent); launch.aspect=checkedFloat(aspect);
        launch.surfaces=surfaces.as<float4>(); launch.solverFailures=solverFailures.as<unsigned long long>();
        launch.shade=1;
        launch.samplesPerPixel=samplesPerPixel;
        launchBuffer.upload(&launch,sizeof(launch));
        EventTimer timer;
        if(profile) {
            launch.shade=0; launchBuffer.upload(&launch,sizeof(launch)); timer.begin(stream);
            OPTIX_CHECK(optixLaunch(pipeline,stream,launchBuffer.address(),launchBuffer.bytes(),&sbt,width,height,1));
            timing("geometry only:",timer.end(stream));
            launch.shade=1; launchBuffer.upload(&launch,sizeof(launch));
        }
        if(benchmarkFrames>1) {
            OPTIX_CHECK(optixLaunch(pipeline,stream,launchBuffer.address(),launchBuffer.bytes(),&sbt,width,height,1));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        std::vector<double> frameTimes;
        for(unsigned frame=0;frame<benchmarkFrames;++frame) {
            timer.begin(stream);
            OPTIX_CHECK(optixLaunch(pipeline,stream,launchBuffer.address(),launchBuffer.bytes(),&sbt,width,height,1));
            frameTimes.push_back(timer.end(stream));
        }
        std::sort(frameTimes.begin(),frameTimes.end());
        RenderResult result;
        result.milliseconds=0.5*(frameTimes[frameTimes.size()/2]+frameTimes[(frameTimes.size()-1)/2]);
        result.samplesPerPixel=samplesPerPixel;
        if(benchmarkFrames>1) std::cout << "Cached-scene GPU benchmark: " << benchmarkFrames << " frames; median "
            << result.milliseconds << " ms; min " << frameTimes.front() << " ms; max " << frameTimes.back() << " ms\n";
        result.pixels.resize(count); result.surfaces.resize(count);
        CUDA_CHECK(cudaMemcpy(result.pixels.data(),pixels.as<void>(),pixels.bytes(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.surfaces.data(),surfaces.as<void>(),surfaces.bytes(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&result.solverFailures,solverFailures.as<void>(),solverFailures.bytes(),cudaMemcpyDeviceToHost));
        result.camera={launch.eye,launch.forward,launch.right,launch.up,launch.tanHalfFov,launch.aspect};
        result.renderOrigin=renderOrigin;
        timing("render:",result.milliseconds);
        std::cout << "rays: " << static_cast<unsigned long long>(count)*samplesPerPixel
            << "\nNURBS solver budget failures: " << result.solverFailures << '\n';
        return result;
    }
};

RenderResult renderNurbsOptix(const NurbsModel& source,unsigned width,unsigned height,
    const std::filesystem::path& ptx,double cameraElevation,unsigned benchmarkFrames,bool profile,unsigned samplesPerPixel) {
    if(benchmarkFrames<1 || benchmarkFrames>1000) throw std::runtime_error("Benchmark frame count must be 1..1000");
    if(samplesPerPixel!=1 && samplesPerPixel!=4) throw std::runtime_error("Samples per pixel must be 1 or 4");
    if(!std::isfinite(cameraElevation) || cameraElevation<-89 || cameraElevation>89)
        throw std::runtime_error("Camera elevation must be a finite angle in -89..89 degrees");
    if(width==0 || height==0 || width>16384 || height>16384)
        throw std::runtime_error("Image dimension must be 1..16384");
    if(source.patches.empty() || source.patches.size()>std::numeric_limits<unsigned>::max())
        throw std::runtime_error("Invalid/empty rational patch set");
    const double radius=source.bounds.radius();
    if(!(radius>0) || !std::isfinite(radius) || radius>1e20)
        throw std::runtime_error("Scene bounds exceed supported numeric range");
    Renderer renderer;
    renderer.bounds=source.bounds; renderer.renderOrigin=source.bounds.center();
    renderer.initialize(ptx); renderer.buildNurbsGas(source); renderer.uploadSbt();
    std::cout << "Material: silver conductor, metallic=1, roughness=0.24, GGX/Smith/Schlick, 3 studio lights\n"
        << "Primary samples/pixel: " << samplesPerPixel << '\n'
        << "Camera elevation: " << cameraElevation << " degrees above horizontal\n";
    return renderer.render(width,height,cameraElevation,benchmarkFrames,profile,samplesPerPixel);
}
}
