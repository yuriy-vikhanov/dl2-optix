// Included inside DevicePrograms.cu's anonymous namespace, after vector helpers.
__device__ __forceinline__ float reciprocal(float x) { return 1/x; }
__device__ __forceinline__ double reciprocal(double x) {
    return 1/x;
}
template<class Pair,class Real>
__device__ __forceinline__ Pair mix2(Pair a,Pair b,Real t) {
    return {a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t};
}
template<class Pair,class Real>
__device__ __forceinline__ void splitCurve(Pair* p,unsigned degree,Real t,bool right) {
    Pair output[MaxNurbsDegree+1];
    output[right ? degree : 0]=p[right ? degree : 0];
    for(unsigned level=1;level<=degree;++level) {
        for(unsigned i=0;i<=degree-level;++i) p[i]=mix2(p[i],p[i+1],t);
        output[right ? degree-level : level]=p[right ? degree-level : 0];
    }
    for(unsigned i=0;i<=degree;++i) p[i]=output[i];
}
template<class Pair,class Real>
__device__ __forceinline__ void restrictCurve(Pair* p,unsigned degree,Real low,Real high) {
    if(low>0) splitCurve(p,degree,low,true);
    if(high<1) splitCurve(p,degree,(high-low)*reciprocal(Real(1)-low),false);
}
// All bounds are dyadic subdivision points (at most 12 splits per axis).
// Float stores these exactly and halves the traversal stack's memory traffic.
struct ParameterBox { float u0,u1,v0,v1; unsigned depth; };
template<class Real> struct ClipTypes;
template<> struct ClipTypes<double> { using Pair=double2; };
template<> struct ClipTypes<float> { using Pair=float2; };
template<class Real> struct Interval { Real low,high; };
template<class Real>
__device__ __forceinline__ Interval<Real> intervalScale(Interval<Real> a,Real s) {
    return s>=0 ? Interval<Real>{a.low*s,a.high*s} : Interval<Real>{a.high*s,a.low*s};
}
template<class Real>
__device__ __forceinline__ Interval<Real> intervalSubtract(Interval<Real> a,Interval<Real> b) { return {a.low-b.high,a.high-b.low}; }
template<class Real>
__device__ __forceinline__ Real intervalMagnitude(Interval<Real> a) { return fmax(fabs(a.low),fabs(a.high)); }
__device__ double middleBernstein(unsigned degree,unsigned i) {
    if(degree==1) return 0.5;
    if(degree==2) return i==1 ? 0.5 : 0.25;
    if(degree==3) return (i==1 || i==2) ? 0.375 : 0.125;
    return 1;
}
// 0: exclude; 1: subdivision required; 2: one isolated root in this box.
// The Krawczyk enclosure uses Bernstein derivative bounds, not a guessed
// Newton seed. Padding and an interior margin keep uncertain cases on the
// existing subdivision path, including roots at parameter seams.
template<unsigned DU=0,unsigned DV=0,class Real=double>
__device__ __forceinline__ unsigned analyzeProjectedHull(const NurbsPatch& patch,const typename ClipTypes<Real>::Pair* projected,const ParameterBox& box,Real epsilon) {
    using Pair=typename ClipTypes<Real>::Pair;
    const unsigned degreeU=DU ? DU : patch.degreeU,degreeV=DV ? DV : patch.degreeV;
    Pair net[DU ? (DU+1)*(DV+1) : MaxNurbsPoles],line[MaxNurbsDegree+1];
    const unsigned nu=degreeU+1,nv=degreeV+1;
    for(unsigned j=0;j<nv;++j) {
        for(unsigned i=0;i<nu;++i) line[i]=projected[i*nv+j];
        restrictCurve(line,degreeU,Real(box.u0),Real(box.u1));
        for(unsigned i=0;i<nu;++i) net[i*nv+j]=line[i];
    }
    constexpr Real huge=sizeof(Real)==4 ? 1e30 : 1e300;
    Real minx=huge,miny=huge,maxx=-huge,maxy=-huge;
    for(unsigned i=0;i<nu;++i) {
        for(unsigned j=0;j<nv;++j) line[j]=net[i*nv+j];
        restrictCurve(line,degreeV,Real(box.v0),Real(box.v1));
        for(unsigned j=0;j<nv;++j) {
            net[i*nv+j]=line[j];
            minx=fmin(minx,line[j].x); maxx=fmax(maxx,line[j].x);
            miny=fmin(miny,line[j].y); maxy=fmax(maxy,line[j].y);
        }
    }
    if(minx>epsilon || maxx<-epsilon || miny>epsilon || maxy<-epsilon) return 0;
    if(box.depth&1) return 1;

    Pair f=Pair{Real(0),Real(0)},ju=f,jv=f;
    Interval<Real> a{huge,-huge},b=a,c=a,d=a;
    for(unsigned i=0;i<nu;++i) for(unsigned j=0;j<nv;++j) {
        const auto q=net[i*nv+j];
        const Real weight=Real(middleBernstein(degreeU,i))*Real(middleBernstein(degreeV,j));
        f.x+=q.x*weight; f.y+=q.y*weight;
        if(i+1<nu) {
            const auto next=net[(i+1)*nv+j];
            const Real dx=degreeU*(next.x-q.x),dy=degreeU*(next.y-q.y);
            a.low=fmin(a.low,dx); a.high=fmax(a.high,dx);
            c.low=fmin(c.low,dy); c.high=fmax(c.high,dy);
            const Real w=Real(middleBernstein(degreeU-1,i))*Real(middleBernstein(degreeV,j));
            ju.x+=dx*w; ju.y+=dy*w;
        }
        if(j+1<nv) {
            const auto next=net[i*nv+j+1];
            const Real dx=degreeV*(next.x-q.x),dy=degreeV*(next.y-q.y);
            b.low=fmin(b.low,dx); b.high=fmax(b.high,dx);
            d.low=fmin(d.low,dy); d.high=fmax(d.high,dy);
            const Real w=Real(middleBernstein(degreeU,i))*Real(middleBernstein(degreeV-1,j));
            jv.x+=dx*w; jv.y+=dy*w;
        }
    }
    const Real determinant=ju.x*jv.y-jv.x*ju.y;
    if(fabs(determinant)<1e-24) return 1;
    const Real inverseDeterminant=reciprocal(determinant);
    const Real c00=jv.y*inverseDeterminant,c01=-jv.x*inverseDeterminant;
    const Real c10=-ju.y*inverseDeterminant,c11=ju.x*inverseDeterminant;
    const Real pad=(sizeof(Real)==4 ? Real(8) : Real(64))*epsilon;
    a.low-=pad; a.high+=pad; b.low-=pad; b.high+=pad;
    c.low-=pad; c.high+=pad; d.low-=pad; d.high+=pad;
    const Interval<Real> m00=intervalSubtract(intervalSubtract({Real(1),Real(1)},intervalScale(a,c00)),intervalScale(c,c01));
    const Interval<Real> m01=intervalSubtract(intervalSubtract({Real(0),Real(0)},intervalScale(b,c00)),intervalScale(d,c01));
    const Interval<Real> m10=intervalSubtract(intervalSubtract({Real(0),Real(0)},intervalScale(a,c10)),intervalScale(c,c11));
    const Interval<Real> m11=intervalSubtract(intervalSubtract({Real(1),Real(1)},intervalScale(b,c10)),intervalScale(d,c11));
    const Real contractionX=intervalMagnitude(m00)+intervalMagnitude(m01);
    const Real contractionY=intervalMagnitude(m10)+intervalMagnitude(m11);
    const Real centerX=Real(0.5)-c00*f.x-c01*f.y,centerY=Real(0.5)-c10*f.x-c11*f.y;
    const Real radiusX=Real(0.5)*contractionX+pad*(Real(1)+fabs(c00)+fabs(c01));
    const Real radiusY=Real(0.5)*contractionY+pad*(Real(1)+fabs(c10)+fabs(c11));
    if(centerX-radiusX>1 || centerX+radiusX<0 || centerY-radiusY>1 || centerY+radiusY<0) return 0;
    constexpr Real margin=1e-7;
    if(contractionX<0.95 && contractionY<0.95 && centerX-radiusX>margin && centerX+radiusX<1-margin
        && centerY-radiusY>margin && centerY+radiusY<1-margin) return 2;
    return 1;
}
__device__ __forceinline__ void bernstein(unsigned degree,double t,double* b,double* derivative) {
    double lower[MaxNurbsDegree+1]={1,0,0,0};
    b[0]=1;
    for(unsigned n=1;n<=degree;++n) {
        for(unsigned i=0;i<n;++i) lower[i]=b[i];
        b[n]=t*b[n-1];
        for(unsigned i=n-1;i>0;--i) b[i]=(1-t)*b[i]+t*b[i-1];
        b[0]*=1-t;
    }
    for(unsigned i=0;i<=degree;++i)
        derivative[i]=degree*((i ? lower[i-1] : 0)-(i<degree ? lower[i] : 0));
}
template<unsigned DU=0,unsigned DV=0>
__device__ __forceinline__ void surfaceValue(const NurbsPatch& patch,double u,double v,double3& p,double3& du,double3& dv) {
    const unsigned degreeU=DU ? DU : patch.degreeU,degreeV=DV ? DV : patch.degreeV;
    double bu[MaxNurbsDegree+1],bv[MaxNurbsDegree+1],dbu[MaxNurbsDegree+1],dbv[MaxNurbsDegree+1];
    bernstein(degreeU,u,bu,dbu); bernstein(degreeV,v,bv,dbv);
    NurbsPoint h{0,0,0,0},hu=h,hv=h;
    for(unsigned i=0;i<=degreeU;++i) for(unsigned j=0;j<=degreeV;++j) {
        const auto q=patch.poles[i*(degreeV+1)+j];
        const double a=bu[i]*bv[j],b=dbu[i]*bv[j],c=bu[i]*dbv[j];
        h.x+=q.x*a; h.y+=q.y*a; h.z+=q.z*a; h.w+=q.w*a;
        hu.x+=q.x*b; hu.y+=q.y*b; hu.z+=q.z*b; hu.w+=q.w*b;
        hv.x+=q.x*c; hv.y+=q.y*c; hv.z+=q.z*c; hv.w+=q.w*c;
    }
    const double inverseWeight=reciprocal(h.w);
    p=make_double3(h.x*inverseWeight,h.y*inverseWeight,h.z*inverseWeight);
    du=make_double3((hu.x-p.x*hu.w)*inverseWeight,(hu.y-p.y*hu.w)*inverseWeight,(hu.z-p.z*hu.w)*inverseWeight);
    dv=make_double3((hv.x-p.x*hv.w)*inverseWeight,(hv.y-p.y*hv.w)*inverseWeight,(hv.z-p.z*hv.w)*inverseWeight);
}
__device__ double dotDouble(double3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ double3 trimValue(const NurbsTrim& trim,double t) {
    double3 p[MaxTrimDegree+1];
    for(unsigned i=0;i<=trim.degree;++i) p[i]=trim.poles[i];
    for(unsigned n=trim.degree;n>0;--n) for(unsigned i=0;i<n;++i) {
        p[i].x+=(p[i+1].x-p[i].x)*t; p[i].y+=(p[i+1].y-p[i].y)*t; p[i].z+=(p[i+1].z-p[i].z)*t;
    }
    const double inverse=reciprocal(p[0].z);
    return make_double3(p[0].x*inverse,p[0].y*inverse,1);
}
__device__ bool insideTrims(const NurbsPatch& patch,const NurbsTrim* trims,double u,double v) {
    const double x=patch.u0+(patch.u1-patch.u0)*u,y=patch.v0+(patch.v1-patch.v0)*v;
    bool inside=false;
    for(unsigned k=0;k<patch.trimCount;++k) {
        const auto& trim=trims[patch.trimOffset+k];
        const double y0=trim.poles[0].y/trim.poles[0].z;
        const double y1=trim.poles[trim.degree].y/trim.poles[trim.degree].z;
        // Half-open crossing rule avoids double counting shared UV vertices.
        if((y0>y)==(y1>y)) continue;
        // Positive weights bound the whole trim curve by its control hull.
        // Most CAD pcurves are iso-parameter edges: classify their crossings
        // without evaluating/bisecting the same rational line forty times.
        if(trim.uMax<=x) continue;
        if(trim.uMin>x) { inside=!inside; continue; }
        if(trim.degree==1) {
            const auto a=trim.poles[0],b=trim.poles[1];
            const double t=(y*a.z-a.y)*reciprocal((b.y-a.y)-y*(b.z-a.z));
            const double crossingX=(a.x+(b.x-a.x)*t)*reciprocal(a.z+(b.z-a.z)*t);
            if(crossingX>x) inside=!inside;
            continue;
        }
        double low=0,high=1;
        for(unsigned i=0;i<40;++i) {
            const double middle=(low+high)*0.5;
            if((trimValue(trim,middle).y>y)==(y0>y)) low=middle; else high=middle;
        }
        if(trimValue(trim,(low+high)*0.5).x>x) inside=!inside;
    }
    return inside;
}
template<unsigned DU=0,unsigned DV=0>
__device__ __forceinline__ bool newtonRoot(const NurbsPatch& patch,const ParameterBox& box,float3 origin,float3 direction,float3 axisX,float3 axisY,
                          double& u,double& v,double& t,float3& normal) {
    u=(box.u0+box.u1)*0.5; v=(box.v0+box.v1)*0.5;
    double3 p,du,dv;
    for(unsigned iteration=0;iteration<25;++iteration) {
        surfaceValue<DU,DV>(patch,u,v,p,du,dv);
        const double3 delta=make_double3(p.x-origin.x,p.y-origin.y,p.z-origin.z);
        const double fx=dotDouble(delta,axisX),fy=dotDouble(delta,axisY);
        if(fabs(fx)+fabs(fy)<1e-10) break;
        const double a=dotDouble(du,axisX),b=dotDouble(dv,axisX),c=dotDouble(du,axisY),d=dotDouble(dv,axisY);
        const double determinant=a*d-b*c;
        if(fabs(determinant)<1e-24) return false;
        const double inverseDeterminant=reciprocal(determinant);
        double stepU=(d*fx-b*fy)*inverseDeterminant,stepV=(-c*fx+a*fy)*inverseDeterminant;
        double damping=1;
        while(damping>1e-8 && (u-stepU*damping < -1e-7 || u-stepU*damping>1+1e-7 || v-stepV*damping < -1e-7 || v-stepV*damping>1+1e-7)) damping*=0.5;
        if(damping<=1e-8) return false;
        u-=stepU*damping; v-=stepV*damping;
    }
    if(u < -1e-7 || u>1+1e-7 || v < -1e-7 || v>1+1e-7) return false;
    u=fmin(1.0,fmax(0.0,u)); v=fmin(1.0,fmax(0.0,v));
    surfaceValue<DU,DV>(patch,u,v,p,du,dv);
    const double3 delta=make_double3(p.x-origin.x,p.y-origin.y,p.z-origin.z);
    if(fabs(dotDouble(delta,axisX))+fabs(dotDouble(delta,axisY))>1e-8) return false;
    t=dotDouble(delta,direction)*reciprocal(dot3(direction,direction));
    const double3 n=make_double3(du.y*dv.z-du.z*dv.y,du.z*dv.x-du.x*dv.z,du.x*dv.y-du.y*dv.x);
    const double magnitude=sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    if(!(magnitude>1e-20)) return false;
    const double inverseMagnitude=reciprocal(magnitude)*patch.orientation;
    normal=make_float3(float(n.x*inverseMagnitude),float(n.y*inverseMagnitude),float(n.z*inverseMagnitude));
    return true;
}
template<unsigned DU=0,unsigned DV=0>
__device__ __forceinline__ void intersectPatch(const NurbsPatch& patch,const NurbsTrim* trims) {
    const unsigned degreeU=DU ? DU : patch.degreeU,degreeV=DV ? DV : patch.degreeV;
    const auto origin=optixGetObjectRayOrigin(),direction=optixGetObjectRayDirection();
    if((DU==0 || (DU==1 && DV==1)) && patch.affine) {
        const auto a=patch.poles[0],b=patch.poles[1],c=patch.poles[2];
        const double inverse=reciprocal(a.w);
        const double3 du=make_double3((c.x-a.x)*inverse,(c.y-a.y)*inverse,(c.z-a.z)*inverse);
        const double3 dv=make_double3((b.x-a.x)*inverse,(b.y-a.y)*inverse,(b.z-a.z)*inverse);
        const double3 n=make_double3(du.y*dv.z-du.z*dv.y,du.z*dv.x-du.x*dv.z,du.x*dv.y-du.y*dv.x);
        const double denominator=dotDouble(n,direction);
        if(fabs(denominator)<1e-24) return;
        const double3 delta=make_double3(a.x*inverse-origin.x,a.y*inverse-origin.y,a.z*inverse-origin.z);
        const double t=(delta.x*n.x+delta.y*n.y+delta.z*n.z)*reciprocal(denominator);
        if(t<optixGetRayTmin() || t>optixGetRayTmax()) return;
        const double3 p=make_double3(direction.x*t-delta.x,direction.y*t-delta.y,direction.z*t-delta.z);
        const double uu=du.x*du.x+du.y*du.y+du.z*du.z,vv=dv.x*dv.x+dv.y*dv.y+dv.z*dv.z;
        const double uv=du.x*dv.x+du.y*dv.y+du.z*dv.z;
        const double pu=p.x*du.x+p.y*du.y+p.z*du.z,pv=p.x*dv.x+p.y*dv.y+p.z*dv.z;
        const double det=uu*vv-uv*uv;
        if(!(det>1e-24)) return;
        const double inverseDeterminant=reciprocal(det);
        double u=(pu*vv-pv*uv)*inverseDeterminant,v=(pv*uu-pu*uv)*inverseDeterminant;
        if(u < -1e-7 || u>1+1e-7 || v < -1e-7 || v>1+1e-7) return;
        u=fmin(1.0,fmax(0.0,u)); v=fmin(1.0,fmax(0.0,v));
        if(!insideTrims(patch,trims,u,v)) return;
        const double length=sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
        const double inverseLength=reciprocal(length)*patch.orientation;
        const float3 normal=make_float3(float(n.x*inverseLength),float(n.y*inverseLength),float(n.z*inverseLength));
        optixReportIntersection(float(t),0,__float_as_uint(normal.x),__float_as_uint(normal.y),__float_as_uint(normal.z));
        return;
    }
    const float3 axisX=unit(cross3(direction,fabsf(direction.z)<0.9f ? make_float3(0,0,1) : make_float3(0,1,0)));
    const float3 axisY=unit(cross3(direction,axisX));
    double2 projected[DU ? (DU+1)*(DV+1) : MaxNurbsPoles]; double scale=1;
    for(unsigned i=0;i<(degreeU+1)*(degreeV+1);++i) {
        const auto p=patch.poles[i];
        const double3 relative=make_double3(p.x-origin.x*p.w,p.y-origin.y*p.w,p.z-origin.z*p.w);
        projected[i]=make_double2(dotDouble(relative,axisX),dotDouble(relative,axisY));
        scale=fmax(scale,fmax(fabs(projected[i].x),fabs(projected[i].y)));
    }
    float2 projectedFast[DU ? (DU+1)*(DV+1) : MaxNurbsPoles];
    for(unsigned i=0;i<(degreeU+1)*(degreeV+1);++i) projectedFast[i]=make_float2(float(projected[i].x),float(projected[i].y));
    ParameterBox stack[32]; unsigned size=1,visits=0,rootCount=0; double roots[8];
    stack[0]={0,1,0,1,0};
    while(size) {
        if(++visits>8192) { atomicAdd(params.solverFailures,1ull); return; }
        const auto box=stack[--size];
        unsigned classification=1;
        // Coarse FP32 enclosures include a control-net rounding allowance.
        // Ambiguous boxes and unusual exponents use the FP64 enclosure.
        if(box.depth<=6 && scale<=1e20)
            classification=analyzeProjectedHull<DU,DV,float>(patch,projectedFast,box,float(scale)*32*1.1920928955078125e-7f);
        if(classification==1) classification=analyzeProjectedHull<DU,DV>(patch,projected,box,scale*1e-13);
        if(!classification) continue;
        if(classification==2 || box.depth==24) {
            double u,v,t; float3 normal;
            const bool root=newtonRoot<DU,DV>(patch,box,origin,direction,axisX,axisY,u,v,t,normal)
                && u>=box.u0-1e-9 && u<=box.u1+1e-9 && v>=box.v0-1e-9 && v<=box.v1+1e-9;
            if(root) {
                if(t<optixGetRayTmin() || t>optixGetRayTmax() || !insideTrims(patch,trims,u,v)) continue;
                bool duplicate=false;
                for(unsigned i=0;i<rootCount;++i) duplicate |= fabs(roots[i]-t)<1e-7;
                if(duplicate) continue;
                if(rootCount==8) { atomicAdd(params.solverFailures,1ull); return; }
                roots[rootCount++]=t;
                optixReportIntersection(float(t),0,__float_as_uint(normal.x),__float_as_uint(normal.y),__float_as_uint(normal.z));
                continue;
            }
            if(box.depth==24) continue;
        }
        auto left=box,right=box; ++left.depth; ++right.depth;
        if((box.depth&1)==0) { const double mid=(box.u0+box.u1)*0.5; left.u1=mid; right.u0=mid; }
        else { const double mid=(box.v0+box.v1)*0.5; left.v1=mid; right.v0=mid; }
        stack[size++]=right; stack[size++]=left;
    }
}

__device__ void intersectNurbs() {
    const auto* data=reinterpret_cast<const HitData*>(optixGetSbtDataPointer());
    const auto& patch=data->patches[optixGetPrimitiveIndex()];
    if(patch.degreeU==1 && patch.degreeV==1) intersectPatch<1,1>(patch,data->trims);
    else if(patch.degreeU==2 && patch.degreeV==1) intersectPatch<2,1>(patch,data->trims);
    else if(patch.degreeU==1 && patch.degreeV==2) intersectPatch<1,2>(patch,data->trims);
    else if(patch.degreeU==2 && patch.degreeV==2) intersectPatch<2,2>(patch,data->trims);
    else intersectPatch<>(patch,data->trims);
}
__device__ float3 product(float3 a,float3 b) { return make_float3(a.x*b.x,a.y*b.y,a.z*b.z); }
__device__ float3 studioEnvironment(float3 d) {
    return mul(make_float3(0.32f,0.38f,0.48f),0.28f+0.72f*fmaxf(0,d.z));
}
__device__ float3 conductorFresnel(float cosine) {
    const float3 f0=make_float3(0.91f,0.92f,0.92f);
    const float x=1-fminf(1,fmaxf(0,cosine)),x2=x*x;
    return add(f0,mul(sub(make_float3(1,1,1),f0),x2*x2*x));
}
__device__ float3 ggxStudioLight(float3 normal,float3 view,float3 light,float3 radiance,float radius) {
    constexpr float roughness=0.24f;
    const float nv=fmaxf(1e-5f,dot3(normal,view));
    const float nl=dot3(normal,light);
    if(nl<=0) return make_float3(0,0,0);
    const float3 halfway=unit(add(light,view));
    const float nh=fmaxf(0,dot3(normal,halfway)),vh=fmaxf(0,dot3(view,halfway));
    const float alpha=roughness*roughness,alpha2=alpha*alpha;
    // Broaden the lobe to approximate a finite studio source in one evaluation.
    const float a=alpha+radius,a2=a*a;
    const float denominator=nh*nh*(a2-1)+1;
    const float distribution=a2/(3.14159265359f*denominator*denominator);
    const float gv=2*nv/(nv+sqrtf(alpha2+(1-alpha2)*nv*nv));
    const float gl=2*nl/(nl+sqrtf(alpha2+(1-alpha2)*nl*nl));
    return mul(product(conductorFresnel(vh),radiance),distribution*gv*gl/(4*nv));
}
__device__ float3 metallicPbr(float3 normal,float3 view) {
    const float nv=fmaxf(0,dot3(normal,view));
    const float3 reflected=sub(mul(normal,2*nv),view);
    float3 result=product(studioEnvironment(reflected),conductorFresnel(nv));
    result=add(result,ggxStudioLight(normal,view,unit(make_float3(-0.5f,-0.4f,1)),make_float3(4.0f,3.7f,3.2f),0.16f));
    result=add(result,ggxStudioLight(normal,view,unit(make_float3(0.8f,0.4f,0.5f)),make_float3(2.1f,2.5f,3.0f),0.10f));
    return add(result,ggxStudioLight(normal,view,unit(make_float3(-0.7f,0.8f,0.2f)),make_float3(2,2,2),0.20f));
}
__device__ float filmic(float x) {
    x*=0.9f;
    return fminf(1,fmaxf(0,(x*(2.51f*x+0.03f))/(x*(2.43f*x+0.59f)+0.14f)));
}
