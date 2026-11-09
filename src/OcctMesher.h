#pragma once
#include "CsgModel.h"
#include <vector_types.h>
#include <vector>

namespace csg {
struct Mesh {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    // Optional smooth outward normals. Empty for operands with sharp faces.
    std::vector<float3> normals;
};
Mesh tessellateOperand(const Primitive& primitive);
}
