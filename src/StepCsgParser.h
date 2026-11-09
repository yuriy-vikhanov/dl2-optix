#pragma once
#include "CsgModel.h"
#include <filesystem>
#include <string_view>

namespace csg {
// Reads a Part 21 document. Unsupported entities can be present outside the
// reachable CSG graph. Multiple distinct represented solids are rejected.
CsgModel parseStepCsg(std::string_view text);
CsgModel loadStepCsg(const std::filesystem::path& path);
}
