#include "DeviceShared.h"
#include <optix_device.h>
#include <cuda_runtime.h>

extern "C" {
__constant__ nurbs::LaunchParams params;
}

namespace {
using namespace nurbs;
__device__ float3 add(float3 a,float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ float3 sub(float3 a,float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ float3 mul(float3 a,float s) { return make_float3(a.x*s,a.y*s,a.z*s); }
__device__ float dot3(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ float3 cross3(float3 a,float3 b) { return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
__device__ float3 unit(float3 a) {
    const float scale=fmaxf(fabsf(a.x),fmaxf(fabsf(a.y),fabsf(a.z)));
    if(scale==0) return make_float3(0,0,0);
    a=make_float3(a.x/scale,a.y/scale,a.z/scale);
    return mul(a,rsqrtf(dot3(a,a)));
}

#include "NurbsDevice.cuh"

struct SurfaceHit { float t; float3 normal; };
__device__ SurfaceHit* hitPointer() {
    const auto p=(static_cast<unsigned long long>(optixGetPayload_1())<<32)|optixGetPayload_0();
    return reinterpret_cast<SurfaceHit*>(p);
}
__device__ unsigned char channel(float v) {
    const float linear=fminf(1.0f,fmaxf(0.0f,v));
    const float srgb=linear<=0.0031308f ? 12.92f*linear : 1.055f*powf(linear,1.0f/2.4f)-0.055f;
    return static_cast<unsigned char>(srgb*255.0f+0.5f);
}
}

extern "C" __global__ void __miss__background() {}
extern "C" __global__ void __intersection__nurbs() { intersectNurbs(); }
extern "C" __global__ void __closesthit__surface() {
    auto* hit=hitPointer();
    hit->t=optixGetRayTmax();
    // Patches are already transformed to centred world coordinates.
    hit->normal=unit(make_float3(__uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),__uint_as_float(optixGetAttribute_2())));
}

extern "C" __global__ void __raygen__render() {
    const uint3 pixel=optixGetLaunchIndex();
    const auto index=static_cast<unsigned long long>(pixel.y)*params.width+pixel.x;
    const unsigned samples=params.samplesPerPixel;
    float3 accumulated=make_float3(0,0,0);
    for(unsigned sample=0;sample<samples;++sample) {
        const float sx=samples==1 ? 0.5f : 0.25f+0.5f*(sample&1);
        const float sy=samples==1 ? 0.5f : 0.25f+0.5f*(sample>>1);
        const float x=(2.0f*(static_cast<float>(pixel.x)+sx)/params.width-1.0f)*params.aspect*params.tanHalfFov;
        const float y=(1.0f-2.0f*(static_cast<float>(pixel.y)+sy)/params.height)*params.tanHalfFov;
        const float3 direction=unit(add(params.forward,add(mul(params.right,x),mul(params.up,y))));
        SurfaceHit hit{-1,make_float3(0,0,0)};
        const auto pointer=reinterpret_cast<unsigned long long>(&hit);
        unsigned low=static_cast<unsigned>(pointer),high=static_cast<unsigned>(pointer>>32);
        optixTrace(params.traversable,params.eye,direction,0.0f,1.0e30f,0.0f,
            OptixVisibilityMask(255),OPTIX_RAY_FLAG_NONE,0,1,0,low,high);
        if(sample==0) params.surfaces[index]=make_float4(hit.normal.x,hit.normal.y,hit.normal.z,hit.t);

        float3 color=make_float3(0.025f,0.028f,0.035f);
        if(hit.t>=0) {
            float3 normal=hit.normal;
            if(dot3(normal,direction)>0) normal=mul(normal,-1);
            color=params.shade ? metallicPbr(normal,mul(direction,-1)) : add(mul(normal,0.5f),make_float3(0.5f,0.5f,0.5f));
        }
        accumulated=add(accumulated,color);
    }
    const float3 color=mul(accumulated,1.0f/samples);
    params.pixels[index]=make_uchar4(channel(filmic(color.x)),channel(filmic(color.y)),channel(filmic(color.z)),255);
}
