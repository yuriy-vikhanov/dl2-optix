#pragma once
#include "Geometry.h"
#include "NurbsShared.h"
#include <TopoDS_Shape.hxx>
#include <filesystem>
#include <vector>

namespace nurbs {
struct NurbsModel {
    TopoDS_Shape shape;
    std::vector<NurbsPatch> patches;
    std::vector<NurbsTrim> trims;
    Bounds bounds;
    unsigned faceCount=0,rationalFaces=0;
};
NurbsModel loadStepNurbs(const std::filesystem::path& path);
struct RenderResult;
// Independent OCCT analytic intersections, including the original face trims.
void validateNurbsRender(const NurbsModel& model,const RenderResult& result,unsigned width,unsigned height);
}
