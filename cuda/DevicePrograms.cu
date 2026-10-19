#include "DeviceShared.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include <float.h>

extern "C" {
__constant__ csg::LaunchParams params;
}

namespace {
    using namespace csg;

    __device__ float3 add(float3 a,float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
    __device__ float3 sub(float3 a,float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
    __device__ float3 mul(float3 a,float s) { return make_float3(a.x*s,a.y*s,a.z*s); }
    __device__ float dot3(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
    __device__ float3 cross3(float3 a,float3 b) { return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
    __device__ float3 unit(float3 a) {
        const float scale=fmaxf(fabsf(a.x),fmaxf(fabsf(a.y),fabsf(a.z)));
        if(scale==0.0f) return make_float3(0,0,0);
        a=make_float3(a.x/scale,a.y/scale,a.z/scale);
        return mul(a,rsqrtf(dot3(a,a)));
    }

    struct RayEvents { HitEvent hits[MaxHits]; unsigned count; bool overflow; };

    __device__ RayEvents* eventsPointer() {
        const auto p=(static_cast<unsigned long long>(optixGetPayload_1())<<32)|optixGetPayload_0();
        return reinterpret_cast<RayEvents*>(p);
    }

    __device__ bool before(const HitEvent& a,const HitEvent& b) {
        if(a.t!=b.t) return a.t<b.t;
        if(a.operandId!=b.operandId) return a.operandId<b.operandId;
        return a.primitiveId<b.primitiveId;
    }

    __device__ void sortEvents(RayEvents& events) {
        for(unsigned i=1;i<events.count;++i) {
            const auto event=events.hits[i]; unsigned j=i;
            while(j && before(event,events.hits[j-1])) { events.hits[j]=events.hits[j-1]; --j; }
            events.hits[j]=event;
        }
    }

    __device__ bool evaluate(const bool* inside) {
        bool values[MaxNodes];
        for(unsigned i=0;i<params.nodeCount;++i) {
            const auto n=params.nodes[i];
            switch(n.type) {
            case NodeType::Leaf: values[i]=inside[n.operandId]; break;
            case NodeType::Union: values[i]=values[n.left]||values[n.right]; break;
            case NodeType::Intersection: values[i]=values[n.left]&&values[n.right]; break;
            case NodeType::Difference: values[i]=values[n.left]&&!values[n.right]; break;
            }
        }
        return values[params.root];
    }
    // 1 = entering, 2 = leaving, 0 = numerically grazing.
    __device__ unsigned crossing(const HitEvent& h,float3 direction) {
        const float d=dot3(direction,h.geometricNormal);
        if(fabsf(d)<=8.0f*FLT_EPSILON) return 0;
        return d<0.0f ? 1u : 2u;
    }

    __device__ unsigned groupEnd(const RayEvents& e,unsigned begin) {
        const float anchor=e.hits[begin].t;
        const float epsilon=fmaxf(params.sceneEpsilon,4.0f*FLT_EPSILON*fabsf(anchor));
        unsigned end=begin+1;
        while(end<e.count && e.hits[end].t-anchor<=epsilon) ++end;
        return end;
    }

    __device__ void groupSigns(const RayEvents& e,unsigned begin,unsigned end,float3 direction,unsigned* signs,int* representatives) {
        for(unsigned operand=0;operand<params.operandCount;++operand) { signs[operand]=0; representatives[operand]=-1; }
        for(unsigned i=begin;i<end;++i) {
            const auto& h=e.hits[i]; const unsigned sign=crossing(h,direction);
            if(!sign) continue;
            signs[h.operandId]|=sign; // Same-operand duplicate triangles collapse here.
            int& best=representatives[h.operandId];
            if(best<0 || fabsf(dot3(direction,h.geometricNormal))>fabsf(dot3(direction,e.hits[best].geometricNormal)) ||
            (fabsf(dot3(direction,h.geometricNormal))==fabsf(dot3(direction,e.hits[best].geometricNormal)) && h.primitiveId<e.hits[best].primitiveId)) best=static_cast<int>(i);
        }
    }

    __device__ bool visibleBoundary(const RayEvents& e,float3 direction,HitEvent& surface) {
        bool inside[MaxOperands]={};
        bool known[MaxOperands]={};
        unsigned signs[MaxOperands]; int representatives[MaxOperands];
        for(unsigned begin=0;begin<e.count;) {
            const unsigned end=groupEnd(e,begin);
            groupSigns(e,begin,end,direction,signs,representatives);
            for(unsigned op=0;op<params.operandCount;++op) if(!known[op] && (signs[op]==1 || signs[op]==2)) {
                inside[op]=signs[op]==2; known[op]=true;
            }
            begin=end;
        }
        for(unsigned begin=0;begin<e.count;) {
            const unsigned end=groupEnd(e,begin);
            groupSigns(e,begin,end,direction,signs,representatives);
            const bool oldResult=evaluate(inside);
            bool currentResult=oldResult; int contributor=-1;
            for(unsigned op=0;op<params.operandCount;++op) {
                if(signs[op]!=1 && signs[op]!=2) continue;
                inside[op]=signs[op]==1;
                const bool nextResult=evaluate(inside);
                if(nextResult!=currentResult) contributor=representatives[op];
                currentResult=nextResult;
            }
            if(oldResult!=currentResult && contributor>=0) {
                surface=e.hits[contributor];
                const bool enteringResult=!oldResult && currentResult;
                const float d=dot3(direction,surface.geometricNormal);
                if((enteringResult && d>0) || (!enteringResult && d<0)) {
                    surface.geometricNormal=mul(surface.geometricNormal,-1.0f);
                    surface.shadingNormal=mul(surface.shadingNormal,-1.0f);
                }
                return true; // Also returns the first exit for an inside camera.
            }
            begin=end;
        }
        return false;
    }

    __device__ unsigned char channel(float v) {
        const float linear=fminf(1.0f,fmaxf(0.0f,v));
        const float srgb=linear<=0.0031308f ? 12.92f*linear : 1.055f*powf(linear,1.0f/2.4f)-0.055f;
        return static_cast<unsigned char>(srgb*255.0f+0.5f);
    }
}

extern "C" __global__ void __anyhit__collect() {
    auto* events=eventsPointer();
    if(events->count==csg::MaxHits) {
        if(!events->overflow) atomicAdd(params.overflowRays,1ull);
        events->overflow=true;
        optixTerminateRay(); // Invalid ray is visibly marked, never silently shaded.
        return;
    }
    const auto* data=reinterpret_cast<const csg::HitData*>(optixGetSbtDataPointer());
    const unsigned primitive=optixGetPrimitiveIndex();
    const uint3 tri=data->indices[primitive];
    const float3 localNormal=cross3(unit(sub(data->vertices[tri.y],data->vertices[tri.x])),unit(sub(data->vertices[tri.z],data->vertices[tri.x])));
    auto& hit=events->hits[events->count++];
    hit.t=optixGetRayTmax(); hit.operandId=optixGetInstanceId(); hit.primitiveId=primitive;
    hit.geometricNormal=unit(optixTransformNormalFromObjectToWorldSpace(localNormal));
    float3 shadingNormal=localNormal;
    if(data->normals) {
        const float2 uv=optixGetTriangleBarycentrics();
        shadingNormal=add(mul(data->normals[tri.x],1.0f-uv.x-uv.y),
            add(mul(data->normals[tri.y],uv.x),mul(data->normals[tri.z],uv.y)));
    }
    hit.shadingNormal=unit(optixTransformNormalFromObjectToWorldSpace(shadingNormal));
    optixIgnoreIntersection(); // Do not reduce tmax: keep collecting all operands.
}

extern "C" __global__ void __miss__all() {}

extern "C" __global__ void __raygen__render() {
    const uint3 pixel=optixGetLaunchIndex();
    const float x=(2.0f*(static_cast<float>(pixel.x)+0.5f)/params.width-1.0f)*params.aspect*params.tanHalfFov;
    const float y=(1.0f-2.0f*(static_cast<float>(pixel.y)+0.5f)/params.height)*params.tanHalfFov;
    const float3 direction=unit(add(params.forward,add(mul(params.right,x),mul(params.up,y))));
    RayEvents events; events.count=0; events.overflow=false;
    const auto pointer=reinterpret_cast<unsigned long long>(&events);
    unsigned low=static_cast<unsigned>(pointer),high=static_cast<unsigned>(pointer>>32);

    optixTrace(params.traversable,params.eye,direction,0.0f,1.0e30f,0.0f,OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,0,1,0,low,high);
    
    float3 color=make_float3(0.025f,0.028f,0.035f);

    if (events.overflow) color=make_float3(1.0f,0.0f,1.0f);
    else {
        sortEvents(events);
        csg::HitEvent surface;

        if(visibleBoundary(events,direction,surface)) {
            const float3 base=params.operandTypes[surface.operandId]==static_cast<unsigned>(csg::PrimitiveType::Box)
                ? make_float3(0.85f,0.30f,0.075f) : make_float3(0.08f,0.39f,0.82f);

            // Smooth normals affect lighting only. Triangle normals above
            // remain responsible for CSG enter/exit classification.
            float3 normal=surface.shadingNormal;
            // Two-sided lighting makes an inside-camera view readable; the CSG
            // outward normal above remains the classification/orientation rule.
            if(dot3(normal,direction)>0) normal=mul(normal,-1.0f);
            const float3 light=unit(add(mul(params.forward,-1.0f),add(mul(params.up,1.4f),mul(params.right,-0.5f))));
            const float diffuse=fmaxf(0.0f,dot3(normal,light));
            const float3 halfway=unit(sub(light,direction));
            const float specular=diffuse>0 ? 0.16f*powf(fmaxf(0.0f,dot3(normal,halfway)),40.0f) : 0.0f;
            color=add(mul(base,0.16f+0.78f*diffuse),make_float3(specular,specular,specular));
        }
    }

    const auto index=static_cast<unsigned long long>(pixel.y)*params.width+pixel.x;
    params.pixels[index]=make_uchar4(channel(color.x),channel(color.y),channel(color.z),255);
}
