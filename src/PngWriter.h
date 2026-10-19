#pragma once
#include <filesystem>
#include <span>
#include <vector_types.h>

namespace csg {
void writePng(const std::filesystem::path& output,unsigned width,unsigned height,std::span<const uchar4> pixels);
}
