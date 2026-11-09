#include "NurbsModel.h"
#include "OptixRenderer.h"
#include <STEPControl_Reader.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierSurface.hxx>
#include <GeomConvert_BSplineSurfaceToBezierSurface.hxx>
#include <Geom2d_BSplineCurve.hxx>
#include <Geom2d_BezierCurve.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <Geom2dConvert.hxx>
#include <Geom2dConvert_BSplineCurveToBezierCurve.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <Poly_Triangulation.hxx>
#include <Precision.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <gp_Lin.hxx>
#include <iostream>
#include <iomanip>

namespace nurbs {
namespace {
void addTrim(const Handle(Geom2d_BezierCurve)& curve,std::vector<NurbsTrim>& trims,int depth=0) {
    if(curve->Degree()>MaxTrimDegree) throw std::runtime_error("NURBS trim degree exceeds 8");
    bool increasing=true,decreasing=true;
    for(int i=2;i<=curve->NbPoles();++i) {
        const double delta=curve->Pole(i).Y()-curve->Pole(i-1).Y();
        increasing &= delta>=-1e-12; decreasing &= delta<=1e-12;
    }
    if(!increasing && !decreasing) {
        if(depth==24) throw std::runtime_error("Cannot split trim into monotone rational curves");
        auto left=Handle(Geom2d_BezierCurve)::DownCast(curve->Copy());
        auto right=Handle(Geom2d_BezierCurve)::DownCast(curve->Copy());
        left->Segment(0,0.5); right->Segment(0.5,1);
        addTrim(left,trims,depth+1); addTrim(right,trims,depth+1); return;
    }
    NurbsTrim trim; trim.degree=curve->Degree();
    trim.uMin=std::numeric_limits<double>::infinity();
    trim.uMax=-std::numeric_limits<double>::infinity();
    for(int i=1;i<=curve->NbPoles();++i) {
        const auto p=curve->Pole(i); const double w=curve->Weight(i);
        if(!(w>0) || !std::isfinite(w)) throw std::runtime_error("Positive finite trim weights required");
        trim.poles[i-1]={p.X()*w,p.Y()*w,w};
        trim.uMin=std::min(trim.uMin,p.X()); trim.uMax=std::max(trim.uMax,p.X());
    }
    trims.push_back(trim);
}
}

NurbsModel loadStepNurbs(const std::filesystem::path& path) {
    STEPControl_Reader reader;
    if(reader.ReadFile(path.string().c_str())!=IFSelect_RetDone || reader.TransferRoots()==0)
        throw std::runtime_error("OCCT could not read/transfer the STEP BRep");
    NurbsModel model; model.shape=reader.OneShape();
    if(model.shape.IsNull() || !BRepCheck_Analyzer(model.shape).IsValid())
        throw std::runtime_error("STEP BRep is empty or invalid");
    for(TopExp_Explorer faces(model.shape,TopAbs_FACE);faces.More();faces.Next()) {
        const auto face=TopoDS::Face(faces.Current());
        TopLoc_Location location;
        const auto surface=Handle(Geom_BSplineSurface)::DownCast(BRep_Tool::Surface(face,location));
        if(surface.IsNull()) throw std::runtime_error("Direct NURBS mode currently requires B-spline surfaces on every face");
        if(surface->UDegree()>MaxNurbsDegree || surface->VDegree()>MaxNurbsDegree)
            throw std::runtime_error("Direct NURBS surface degree exceeds 3; no mesh fallback is used");
        if(surface->IsUPeriodic() || surface->IsVPeriodic())
            throw std::runtime_error("Periodic NURBS surfaces need seam unwrapping; unsupported in this mode");
        double u0,u1,v0,v1; BRepTools::UVBounds(face,u0,u1,v0,v1);
        const unsigned trimOffset=static_cast<unsigned>(model.trims.size());
        for(TopExp_Explorer edges(face,TopAbs_EDGE);edges.More();edges.Next()) {
            double first,last;
            const auto curve=BRep_Tool::CurveOnSurface(TopoDS::Edge(edges.Current()),face,first,last);
            if(curve.IsNull()) throw std::runtime_error("Face boundary has no UV pcurve");
            auto trimmed=new Geom2d_TrimmedCurve(curve,first,last);
            auto bspline=Geom2dConvert::CurveToBSplineCurve(trimmed);
            Geom2dConvert_BSplineCurveToBezierCurve converter(bspline);
            for(int i=1;i<=converter.NbArcs();++i) addTrim(converter.Arc(i),model.trims);
        }
        const unsigned trimCount=static_cast<unsigned>(model.trims.size())-trimOffset;
        if(!trimCount) throw std::runtime_error("NURBS face has no trimming boundary");
        GeomConvert_BSplineSurfaceToBezierSurface converter(surface,u0,u1,v0,v1,Precision::PConfusion());
        TColStd_Array1OfReal us(1,converter.NbUPatches()+1),vs(1,converter.NbVPatches()+1);
        converter.UKnots(us); converter.VKnots(vs);
        for(int iu=1;iu<=converter.NbUPatches();++iu) for(int iv=1;iv<=converter.NbVPatches();++iv) {
            const auto original=converter.Patch(iu,iv);
            // Exact subdivision tightens the convex-hull AABBs. These remain
            // rational surfaces, not planar facets or sampled approximations.
            constexpr int divisions=8;
            for(int su=0;su<divisions;++su) for(int sv=0;sv<divisions;++sv) {
                const double a=double(su)/divisions,b=double(su+1)/divisions;
                const double c=double(sv)/divisions,d=double(sv+1)/divisions;
                auto bezier=Handle(Geom_BezierSurface)::DownCast(original->Copy());
                bezier->Segment(a,b,c,d);
                NurbsPatch patch;
                patch.degreeU=bezier->UDegree(); patch.degreeV=bezier->VDegree();
                patch.face=model.faceCount; patch.trimOffset=trimOffset; patch.trimCount=trimCount;
                patch.orientation=face.Orientation()==TopAbs_REVERSED ? -1 : 1;
                patch.u0=us(iu)+(us(iu+1)-us(iu))*a; patch.u1=us(iu)+(us(iu+1)-us(iu))*b;
                patch.v0=vs(iv)+(vs(iv+1)-vs(iv))*c; patch.v1=vs(iv)+(vs(iv+1)-vs(iv))*d;
                for(unsigned i=0;i<=patch.degreeU;++i) for(unsigned j=0;j<=patch.degreeV;++j) {
                    const auto p=bezier->Pole(i+1,j+1).Transformed(location.Transformation());
                    const double w=bezier->Weight(i+1,j+1);
                    if(!(w>0) || !std::isfinite(w)) throw std::runtime_error("Positive finite surface weights required");
                    patch.poles[i*(patch.degreeV+1)+j]={p.X()*w,p.Y()*w,p.Z()*w,w};
                    model.bounds.add({p.X(),p.Y(),p.Z()});
                }
                if(patch.degreeU==1 && patch.degreeV==1) {
                    const auto p00=patch.poles[0],p01=patch.poles[1],p10=patch.poles[2],p11=patch.poles[3];
                    // An affine degree-one NURBS patch has a linear ray solve.
                    // Test exact equalities; curved/bilinear patches use the
                    // general rational solver, with no fitted surface proxy.
                    patch.affine=p00.w==p01.w && p00.w==p10.w && p00.w==p11.w
                        && p11.x-p01.x==p10.x-p00.x && p11.y-p01.y==p10.y-p00.y && p11.z-p01.z==p10.z-p00.z;
                }
                model.patches.push_back(patch);
            }
        }
        if(surface->IsURational() || surface->IsVRational()) ++model.rationalFaces;
        ++model.faceCount;
        if(!BRep_Tool::Triangulation(face,location).IsNull())
            throw std::runtime_error("Unexpected triangulation in direct NURBS import");
    }
    if(!model.faceCount) throw std::runtime_error("STEP contains no BRep faces");
    std::cout << "Direct NURBS: " << model.faceCount << " BRep faces (" << model.rationalFaces
        << " rational), " << model.patches.size() << " exact Bezier subpatches, " << model.trims.size()
        << " rational UV trim arcs, 0 triangles\nBounds: " << model.bounds.lo << " .. " << model.bounds.hi << '\n';
    return model;
}

void validateNurbsRender(const NurbsModel& model,const RenderResult& result,unsigned width,unsigned height) {
    IntCurvesFace_ShapeIntersector reference; reference.Load(model.shape,1e-8);
    unsigned rays=0,hits=0,maskErrors=0,depthErrors=0,normalErrors=0;
    double maxDepthError=0,minNormalDot=1;
    const auto vec=[](float3 p) { return Vec3{p.x,p.y,p.z}; };
    const auto& camera=result.camera;
    const double tolerance=model.bounds.radius()*2e-5;
    const float subpixel=result.samplesPerPixel==1 ? 0.5f : 0.25f;
    // Match the first subpixel and the actual float GPU camera.
    for(unsigned y=3;y<height;y+=std::max(1u,height/24)) for(unsigned x=5;x<width;x+=std::max(1u,width/40)) {
        const float px=(2.0f*(x+subpixel)/width-1.0f)*camera.aspect*camera.tanHalfFov;
        const float py=(1.0f-2.0f*(y+subpixel)/height)*camera.tanHalfFov;
        const Vec3 direction=normalized(vec(camera.forward)+vec(camera.right)*px+vec(camera.up)*py);
        const Vec3 origin=vec(camera.eye)+result.renderOrigin;
        reference.Perform(gp_Lin(gp_Pnt(origin.x,origin.y,origin.z),gp_Dir(direction.x,direction.y,direction.z)),0,1e10);
        if(!reference.IsDone()) throw std::runtime_error("OCCT reference intersection failed");
        const auto gpu=result.surfaces[static_cast<std::size_t>(y)*width+x];
        const bool cpuHit=reference.NbPnt()>0,gpuHit=gpu.w>=0;
        ++rays;
        if(cpuHit!=gpuHit) { ++maskErrors; continue; }
        if(!cpuHit) continue;
        ++hits;
        const double error=std::abs(reference.WParameter(1)-gpu.w);
        maxDepthError=std::max(maxDepthError,error); if(error>tolerance) ++depthErrors;
        TopLoc_Location location;
        const auto surface=BRep_Tool::Surface(reference.Face(1),location);
        gp_Pnt p; gp_Vec du,dv;
        surface->D1(reference.UParameter(1),reference.VParameter(1),p,du,dv);
        gp_Vec n=du.Crossed(dv).Transformed(location.Transformation());
        if(reference.Face(1).Orientation()==TopAbs_REVERSED) n.Reverse();
        if(n.Magnitude()>1e-10) {
            n.Normalize(); const double agreement=n.X()*gpu.x+n.Y()*gpu.y+n.Z()*gpu.z;
            minNormalDot=std::min(minNormalDot,agreement); if(agreement<0.999) ++normalErrors;
        }
    }
    std::cout << "OCCT analytic validation: " << rays << " rays, " << hits << " surface hits, " << maskErrors
        << " mask errors, " << depthErrors << " depth errors, " << normalErrors << " normal errors\n"
        << std::scientific << std::setprecision(6) << "Max depth error: " << maxDepthError
        << "; min normal dot: " << minNormalDot << '\n' << std::fixed << std::setprecision(3);
    if(maskErrors || depthErrors || normalErrors || hits<50)
        throw std::runtime_error("Direct NURBS render failed independent analytic validation");
}
}
