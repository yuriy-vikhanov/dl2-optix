#include "OptixRenderer.h"
#include <optix_function_table_definition.h> // Exactly one definition in the executable.
#include "ErrorChecks.h"
#include "DeviceShared.h"
#include <optix_stack_size.h>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <utility>

namespace csg {
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
struct Gas { Buffer vertices,indices,normals,output; OptixTraversableHandle handle=0; };
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
    CsgModel model;
    Vec3 renderOrigin;
    std::vector<Gas> gases;
    Buffer instances,iasOutput,tree,types,raygenRecord,missRecord,hitRecords;
    Buffer launchBuffer,pixels,overflow;
    OptixTraversableHandle ias=0;

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
        if(!file) throw std::runtime_error("Cannot open generated device program: "+path.string()+". Build CsgOptixRenderer to generate/copy DevicePrograms.ptx.");
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
        compile.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        compile.numPayloadValues=2; compile.numAttributeValues=2;
        compile.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
        compile.pipelineLaunchParamsVariableName="params";
        compile.usesPrimitiveTypeFlags=static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
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
        description.miss.module=module; description.miss.entryFunctionName="__miss__all";
        makeGroup(description,miss);
        description={}; description.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        description.hitgroup.moduleAH=module; description.hitgroup.entryFunctionNameAH="__anyhit__collect";
        // Built-in triangle intersection; closest-hit is deliberately absent.
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
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline,fromTraversal,fromState,continuation,2)); // IAS -> GAS
        EmptyRecord rg{},ms{};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen,&rg)); OPTIX_CHECK(optixSbtRecordPackHeader(miss,&ms));
        raygenRecord.upload(&rg,sizeof(rg)); missRecord.upload(&ms,sizeof(ms));
        sbt.raygenRecord=raygenRecord.address(); sbt.missRecordBase=missRecord.address();
        sbt.missRecordStrideInBytes=sizeof(EmptyRecord); sbt.missRecordCount=1;
        timing("OptiX pipeline build:",elapsed(start));
    }
    void buildGas(std::size_t index,const Mesh& mesh) {
        if(mesh.vertices.empty() || mesh.indices.empty() || mesh.vertices.size()>std::numeric_limits<unsigned>::max() || mesh.indices.size()>std::numeric_limits<unsigned>::max()) throw std::runtime_error("Invalid/empty operand mesh");
        if(!mesh.normals.empty() && mesh.normals.size()!=mesh.vertices.size()) throw std::runtime_error("One shading normal is required per mesh vertex");
        for(const auto t:mesh.indices) if(t.x>=mesh.vertices.size() || t.y>=mesh.vertices.size() || t.z>=mesh.vertices.size()) throw std::runtime_error("Triangle index is out of bounds");
        const auto start=Clock::now(); auto& gas=gases.at(index);
        gas.vertices.upload(mesh.vertices); gas.indices.upload(mesh.indices);
        gas.normals.upload(mesh.normals);
        const CUdeviceptr vertexBuffer=gas.vertices.address();
        const unsigned flags=OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
        OptixBuildInput input{}; input.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& triangles=input.triangleArray;
        triangles.vertexBuffers=&vertexBuffer; triangles.numVertices=static_cast<unsigned>(mesh.vertices.size());
        triangles.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3; triangles.vertexStrideInBytes=sizeof(float3);
        triangles.indexBuffer=gas.indices.address(); triangles.numIndexTriplets=static_cast<unsigned>(mesh.indices.size());
        triangles.indexFormat=OPTIX_INDICES_FORMAT_UNSIGNED_INT3; triangles.indexStrideInBytes=sizeof(uint3);
        triangles.flags=&flags; triangles.numSbtRecords=1;
        OptixAccelBuildOptions options{}; options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation=OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{}; OPTIX_CHECK(optixAccelComputeMemoryUsage(context,&options,&input,1,&sizes));
        Buffer scratch; scratch.resize(sizes.tempSizeInBytes); gas.output.resize(sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(context,stream,&options,&input,1,scratch.address(),scratch.bytes(),gas.output.address(),gas.output.bytes(),&gas.handle,nullptr,0));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(!gas.handle) throw std::runtime_error("OptiX GAS returned a null handle");
        timing(std::string("GAS ")+primitiveName(model.operands.at(index).type)+" build:",elapsed(start));
    }
    void buildIas() {
        const auto start=Clock::now();
        std::vector<OptixInstance> host(gases.size());
        for(std::size_t i=0;i<host.size();++i) {
            const auto& p=model.operands[i].placement;
            const Vec3 origin=p.origin-renderOrigin;
            const std::array<float,12> transform{
                checkedFloat(p.x.x),checkedFloat(p.y.x),checkedFloat(p.z.x),checkedFloat(origin.x),
                checkedFloat(p.x.y),checkedFloat(p.y.y),checkedFloat(p.z.y),checkedFloat(origin.y),
                checkedFloat(p.x.z),checkedFloat(p.y.z),checkedFloat(p.z.z),checkedFloat(origin.z)};
            auto& instance=host[i]; std::memcpy(instance.transform,transform.data(),sizeof(instance.transform));
            instance.instanceId=static_cast<unsigned>(i); instance.sbtOffset=static_cast<unsigned>(i);
            instance.visibilityMask=255; instance.flags=OPTIX_INSTANCE_FLAG_NONE; instance.traversableHandle=gases[i].handle;
        }
        instances.upload(host);
        OptixBuildInput input{}; input.type=OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        input.instanceArray.instances=instances.address(); input.instanceArray.numInstances=static_cast<unsigned>(host.size());
        OptixAccelBuildOptions options{};
        options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation=OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{}; OPTIX_CHECK(optixAccelComputeMemoryUsage(context,&options,&input,1,&sizes));
        iasOutput.resize(sizes.outputSizeInBytes);
        Buffer scratch; scratch.resize(sizes.tempSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(context,stream,&options,&input,1,scratch.address(),scratch.bytes(),iasOutput.address(),iasOutput.bytes(),&ias,nullptr,0));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(!ias) throw std::runtime_error("OptiX IAS returned a null handle");
        timing("IAS build:",elapsed(start));
    }
    void uploadTree() {
        std::vector<GpuCsgNode> nodes;
        for(const auto& n:model.nodes) nodes.push_back(n.gpu);
        tree.upload(nodes);
    }
    void uploadSbt() {
        std::vector<HitRecord> records(gases.size()); std::vector<std::uint32_t> operandTypes;
        for(std::size_t i=0;i<gases.size();++i) {
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup,&records[i]));
            records[i].data={gases[i].vertices.as<float3>(),gases[i].indices.as<uint3>(),gases[i].normals.as<float3>()};
            operandTypes.push_back(static_cast<std::uint32_t>(model.operands[i].type));
        }
        hitRecords.upload(records); types.upload(operandTypes);
        sbt.hitgroupRecordBase=hitRecords.address(); sbt.hitgroupRecordStrideInBytes=sizeof(HitRecord); sbt.hitgroupRecordCount=static_cast<unsigned>(records.size());
    }
    RenderResult render(unsigned width,unsigned height) {
        const Bounds bounds=worldBounds(model); const double radius=bounds.radius();
        if(!(radius>0) || !std::isfinite(radius) || radius>1e20) throw std::runtime_error("Scene bounds exceed supported numeric range");
        const Vec3 center=bounds.center();
        const double tangent=std::tan(40.0*std::numbers::pi/360.0);
        const double aspect=static_cast<double>(width)/height;
        const double fitHalfAngle=std::atan(tangent*std::min(1.0,aspect));
        const Vec3 eye=center+normalized(Vec3{1,-1,0.7})*(1.1*radius/std::sin(fitHalfAngle));
        const Vec3 forward=normalized(center-eye);
        const Vec3 referenceUp=std::abs(forward.z)>0.99 ? Vec3{0,1,0} : Vec3{0,0,1};
        const Vec3 right=normalized(cross(forward,referenceUp)); const Vec3 up=normalized(cross(right,forward));
        const std::size_t count=static_cast<std::size_t>(width)*height;
        pixels.resize(count*sizeof(uchar4)); overflow.resize(sizeof(unsigned long long));
        CUDA_CHECK(cudaMemsetAsync(overflow.as<void>(),0,overflow.bytes(),stream));
        LaunchParams launch{};
        launch.traversable=ias; launch.pixels=pixels.as<uchar4>(); launch.width=width; launch.height=height;
        launch.nodes=tree.as<GpuCsgNode>(); launch.operandTypes=types.as<std::uint32_t>();
        launch.nodeCount=static_cast<unsigned>(model.nodes.size()); launch.operandCount=static_cast<unsigned>(model.operands.size()); launch.root=model.root;
        launch.eye=gpu(eye-renderOrigin); launch.forward=gpu(forward); launch.right=gpu(right); launch.up=gpu(up);
        launch.tanHalfFov=checkedFloat(tangent); launch.aspect=checkedFloat(aspect);
        launch.sceneEpsilon=checkedFloat(radius*8.0*std::numeric_limits<float>::epsilon()); launch.overflowRays=overflow.as<unsigned long long>();
        launchBuffer.upload(&launch,sizeof(launch));
        EventTimer timer; timer.begin(stream);
        OPTIX_CHECK(optixLaunch(pipeline,stream,launchBuffer.address(),launchBuffer.bytes(),&sbt,width,height,1));
        RenderResult result; result.milliseconds=timer.end(stream); result.pixels.resize(count);
        CUDA_CHECK(cudaMemcpy(result.pixels.data(),pixels.as<void>(),pixels.bytes(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&result.overflowRays,overflow.as<void>(),overflow.bytes(),cudaMemcpyDeviceToHost));
        timing("render:",result.milliseconds);
        std::cout << "rays: " << count << "\nhit overflow: " << result.overflowRays << '\n';
        return result;
    }
};

RenderResult renderOptix(const CsgModel& model,const std::vector<Mesh>& meshes,
    unsigned width,unsigned height,const std::filesystem::path& ptx) {
    if(meshes.size()!=model.operands.size()) throw std::runtime_error("One separate mesh is required for each operand");
    Renderer renderer;
    renderer.initialize(ptx);
    renderer.model=model; renderer.renderOrigin=worldBounds(model).center(); renderer.gases.resize(meshes.size());
    for(std::size_t i=0;i<meshes.size();++i) renderer.buildGas(i,meshes[i]);
    renderer.buildIas(); renderer.uploadTree(); renderer.uploadSbt();
    return renderer.render(width,height);
}
}
