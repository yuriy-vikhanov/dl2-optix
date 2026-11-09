#pragma once
#include "CsgTypes.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace csg {
struct Vec3 { double x = 0, y = 0, z = 0; };
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
inline double length(Vec3 a) { return std::hypot(a.x,a.y,a.z); }
inline Vec3 normalized(Vec3 a) {
    const double n = length(a);
    if (!(n > 0) || !std::isfinite(n)) throw std::runtime_error("Invalid zero/non-finite direction");
    return a*(1/n);
}
inline std::ostream& operator<<(std::ostream& s, Vec3 a) { return s << '(' << a.x << ", " << a.y << ", " << a.z << ')'; }
struct Placement {
    Vec3 origin;
    Vec3 x{1,0,0}, y{0,1,0}, z{0,0,1};
    Vec3 transform(Vec3 p) const { return origin+x*p.x+y*p.y+z*p.z; }
};
struct Primitive {
    PrimitiveType type = PrimitiveType::Box;
    std::uint64_t stepId = 0;
    Placement placement;
    Vec3 size;
    double radius = 0;
};
struct CsgNode {
    GpuCsgNode gpu;
    std::uint64_t stepId = 0;
};
struct CsgModel {
    std::vector<Primitive> operands;
    std::vector<CsgNode> nodes;
    std::uint32_t root = 0;
    std::uint64_t representationId = 0, solidId = 0, rootExpressionId = 0;
};
inline const char* primitiveName(PrimitiveType t) { return t == PrimitiveType::Box ? "BOX" : "SPHERE"; }
inline const char* nodeName(NodeType t) {
    switch(t) {
    case NodeType::Leaf: return "LEAF";
    case NodeType::Union: return "UNION";
    case NodeType::Intersection: return "INTERSECTION";
    case NodeType::Difference: return "DIFFERENCE";
    }
    return "INVALID";
}
struct Bounds {
    Vec3 lo{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    Vec3 hi{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
    void add(Vec3 p) {
        lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
        hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
    }
    Vec3 center() const { return lo+(hi-lo)*0.5; }
    double radius() const { return length(hi-lo)*0.5; }
};
inline Bounds worldBounds(const CsgModel& model) {
    Bounds b;
    for (const auto& p:model.operands) {
        const Vec3 lo=p.type==PrimitiveType::Box ? Vec3{} : Vec3{-p.radius,-p.radius,-p.radius};
        const Vec3 hi=p.type==PrimitiveType::Box ? p.size : Vec3{p.radius,p.radius,p.radius};
        for (unsigned i=0;i<8;++i) b.add(p.placement.transform({(i&1)?hi.x:lo.x,(i&2)?hi.y:lo.y,(i&4)?hi.z:lo.z}));
    }
    return b;
}
void printTree(const CsgModel& model, std::ostream& out);
}
