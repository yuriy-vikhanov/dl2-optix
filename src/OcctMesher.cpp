#include "OcctMesher.h"
#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Poly_Triangulation.hxx>
#include <Precision.hxx>
#include <Standard_Failure.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Trsf.hxx>
#include <string>

namespace csg {
Mesh tessellateOperand(const Primitive& p) {
    try {
        TopoDS_Shape shape;
        if(p.type==PrimitiveType::Box) shape=BRepPrimAPI_MakeBox(p.size.x,p.size.y,p.size.z).Shape();
        else shape=BRepPrimAPI_MakeSphere(p.radius).Shape();
        const double scale=p.type==PrimitiveType::Box ? length(p.size) : 2*p.radius;
        // Set face-interior controls explicitly: the short constructor uses
        // twice the boundary angle and skips analytical-surface deflection checks.
        const bool sphere=p.type==PrimitiveType::Sphere;
        const double deflection=scale*(sphere ? 0.0002 : 0.002);
        const double angle=sphere ? 0.08 : 0.25;
        IMeshTools_Parameters parameters;
        parameters.Deflection=std::max(deflection,Precision::Confusion());
        parameters.Angle=angle;
        if(sphere) {
            // Budget tighter face-interior error than boundary discretization;
            // a cut can pass through a triangle's interior, away from STEP edges.
            parameters.DeflectionInterior=parameters.Deflection*0.5;
            parameters.AngleInterior=angle*0.5;
            parameters.EnableControlSurfaceDeflectionAllSurfaces=true;
        }
        parameters.Relative=false;
        parameters.InParallel=true;
        BRepMesh_IncrementalMesh mesher(shape,parameters);
        if(!mesher.IsDone()) throw std::runtime_error("OCCT tessellation failed");
        Mesh mesh;
        for(TopExp_Explorer faces(shape,TopAbs_FACE);faces.More();faces.Next()) {
            const auto face=TopoDS::Face(faces.Current());
            TopLoc_Location location;
            const Handle(Poly_Triangulation) tri=BRep_Tool::Triangulation(face,location);
            if(tri.IsNull()) throw std::runtime_error("OCCT face has no triangulation");
            const auto base=static_cast<unsigned>(mesh.vertices.size());
            const auto transform=location.Transformation();
            for(int i=1;i<=tri->NbNodes();++i) {
                const auto v=tri->Node(i).Transformed(transform);
                mesh.vertices.push_back({static_cast<float>(v.X()),static_cast<float>(v.Y()),static_cast<float>(v.Z())});
                if(p.type==PrimitiveType::Sphere) {
                    // The local sphere is centred at the origin. Radial normals
                    // agree across its seam and poles without averaging facets.
                    const auto n=normalized(Vec3{v.X(),v.Y(),v.Z()});
                    mesh.normals.push_back({static_cast<float>(n.x),static_cast<float>(n.y),static_cast<float>(n.z)});
                }
            }
            const bool reverse=(face.Orientation()==TopAbs_REVERSED)!=transform.IsNegative();
            for(int i=1;i<=tri->NbTriangles();++i) {
                int a,b,c; tri->Triangle(i).Get(a,b,c); if(reverse) std::swap(b,c);
                const uint3 triangle{base+static_cast<unsigned>(a-1),base+static_cast<unsigned>(b-1),base+static_cast<unsigned>(c-1)};
                const auto point=[&](unsigned index) {
                    const auto v=mesh.vertices[index]; return Vec3{v.x,v.y,v.z};
                };
                // OCCT's spherical seam duplicates pole nodes. Discard its
                // zero-area pole triangles using the actual uploaded vertices.
                if(length(cross(point(triangle.y)-point(triangle.x),point(triangle.z)-point(triangle.x)))==0) continue;
                mesh.indices.push_back(triangle);
            }
        }
        if(mesh.vertices.empty() || mesh.indices.empty()) throw std::runtime_error("Empty operand mesh");
        return mesh;
    } catch(const Standard_Failure& e) {
        throw std::runtime_error(std::string("OCCT: ")+(e.GetMessageString()?e.GetMessageString():"primitive/mesh failure"));
    }
}
}
