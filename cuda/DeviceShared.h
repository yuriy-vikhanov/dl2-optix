#pragma once
#include "CsgTypes.h"
#include <optix.h>
#include <vector_types.h>
#include <cstdint>

namespace csg {
inline constexpr unsigned MaxHits = 32;
struct HitEvent {
    float t;
    std::uint32_t operandId;
    std::uint32_t primitiveId;
    float3 geometricNormal;
    float3 shadingNormal;
};
struct HitData {
    const float3* vertices;
    const uint3* indices;
    const float3* normals;
    std::uint64_t reserved=0; // Explicit padding to keep SBT records aligned.
};
struct LaunchParams {
    OptixTraversableHandle traversable;
    uchar4* pixels;
    unsigned width,height;
    const GpuCsgNode* nodes;
    const std::uint32_t* operandTypes;
    unsigned nodeCount,operandCount,root;
    float3 eye,forward,right,up;
    float tanHalfFov,aspect,sceneEpsilon;
    unsigned long long* overflowRays;
};
}
