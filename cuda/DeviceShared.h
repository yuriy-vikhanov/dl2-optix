#pragma once
#include "NurbsShared.h"
#include <optix.h>
#include <vector_types.h>

namespace nurbs {
struct HitData {
    const NurbsPatch* patches;
    const NurbsTrim* trims;
};
struct LaunchParams {
    OptixTraversableHandle traversable;
    uchar4* pixels;
    unsigned width,height;
    float3 eye,forward,right,up;
    float tanHalfFov,aspect;
    float4* surfaces;
    unsigned long long* solverFailures;
    unsigned shade;
    unsigned samplesPerPixel;
};
}
