#pragma once
#include "CsgModel.h"
#include "OcctMesher.h"
#include <filesystem>

namespace csg {
struct RenderResult {
    std::vector<uchar4> pixels;
    double milliseconds=0;
    unsigned long long overflowRays=0;
};
RenderResult renderOptix(const CsgModel& model,const std::vector<Mesh>& meshes,
    unsigned width,unsigned height,const std::filesystem::path& ptx);
}
