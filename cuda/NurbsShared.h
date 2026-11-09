#pragma once
#include <vector_types.h>

namespace nurbs {
inline constexpr int MaxNurbsDegree=3;
inline constexpr int MaxNurbsPoles=(MaxNurbsDegree+1)*(MaxNurbsDegree+1);
inline constexpr int MaxTrimDegree=8;
struct NurbsPoint { double x,y,z,w; };

// Exact homogeneous rational Bezier data, obtained by knot insertion.
// Neither the patches nor their UV boundary curves contain triangles.
struct NurbsPatch {
    NurbsPoint poles[MaxNurbsPoles]{};
    double u0=0,u1=1,v0=0,v1=1;
    unsigned degreeU=0,degreeV=0,face=0,trimOffset=0,trimCount=0;
    double orientation=1;
    unsigned affine=0;
};
struct NurbsTrim {
    double3 poles[MaxTrimDegree+1]{}; // (u*w,v*w,w), monotone in v.
    unsigned degree=0;
    double uMin=0,uMax=0;
};
}
