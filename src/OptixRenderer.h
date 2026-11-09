#pragma once
#include "Geometry.h"
#include <vector_types.h>
#include <filesystem>
#include <vector>

namespace nurbs {
struct NurbsModel;
struct RenderCamera {
    float3 eye,forward,right,up;
    float tanHalfFov=0,aspect=0;
};
struct RenderResult {
    std::vector<uchar4> pixels;
    double milliseconds=0;
    unsigned long long solverFailures=0;
    unsigned samplesPerPixel=4;
    std::vector<float4> surfaces;
    RenderCamera camera{};
    Vec3 renderOrigin;
};
RenderResult renderNurbsOptix(const NurbsModel& model,unsigned width,unsigned height,
    const std::filesystem::path& ptx,double cameraElevation=35,unsigned benchmarkFrames=1,bool profile=false,unsigned samplesPerPixel=4);
}
