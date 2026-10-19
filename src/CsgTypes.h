#pragma once
#include <cstdint>

namespace csg {
inline constexpr std::uint32_t MaxOperands = 16;
inline constexpr std::uint32_t MaxNodes = 63;
enum class CsgOperator : std::uint32_t { Union = 1, Intersection = 2, Difference = 3 };
enum class PrimitiveType : std::uint32_t { Box, Sphere };
enum class NodeType : std::uint32_t { Leaf = 0, Union = 1, Intersection = 2, Difference = 3 };

// Postorder: children precede parents. This layout is shared verbatim with CUDA.
struct GpuCsgNode {
    NodeType type = NodeType::Leaf;
    std::uint32_t left = 0;
    std::uint32_t right = 0;
    std::uint32_t operandId = 0;
};
static_assert(sizeof(GpuCsgNode) == 16);
}
