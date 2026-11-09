#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace nurbs {
struct Vec3 { double x=0,y=0,z=0; };
inline Vec3 operator+(Vec3 a,Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(Vec3 a,Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator*(Vec3 a,double s) { return {a.x*s,a.y*s,a.z*s}; }
inline double length(Vec3 a) { return std::hypot(a.x,a.y,a.z); }
inline Vec3 cross(Vec3 a,Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
inline Vec3 normalized(Vec3 a) {
    const double n=length(a);
    if(!(n>0) || !std::isfinite(n)) throw std::runtime_error("Invalid zero/non-finite direction");
    return a*(1/n);
}
inline std::ostream& operator<<(std::ostream& s,Vec3 a) { return s << '(' << a.x << ", " << a.y << ", " << a.z << ')'; }
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
}
