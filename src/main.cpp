#include "OptixRenderer.h"
#include "PngWriter.h"
#include "NurbsModel.h"
#include <Standard_Failure.hxx>
#include <Windows.h>
#include <cmath>
#include <cwctype>
#include <iostream>
#include <string>

namespace {
std::filesystem::path executableDirectory() {
    std::wstring buffer(32768,L'\0');
    const DWORD size=GetModuleFileNameW(nullptr,buffer.data(),static_cast<DWORD>(buffer.size()));
    if(!size || size>=buffer.size()) throw std::runtime_error("Cannot locate executable directory");
    buffer.resize(size); return std::filesystem::path(buffer).parent_path();
}
unsigned pixels(const std::wstring& text) {
    std::size_t end=0; const auto n=std::stoul(text,&end);
    if(end!=text.size() || n==0 || n>16384) throw std::runtime_error("Image dimension must be 1..16384");
    return static_cast<unsigned>(n);
}
double cameraAngle(const std::wstring& text) {
    std::size_t end=0; const double angle=std::stod(text,&end);
    if(end!=text.size() || !std::isfinite(angle) || angle<-89 || angle>89)
        throw std::runtime_error("Camera elevation must be a finite angle in -89..89 degrees");
    return angle;
}
std::filesystem::path inputFile(std::filesystem::path path) {
    if(std::filesystem::is_directory(path)) {
        std::vector<std::filesystem::path> files;
        for(const auto& entry:std::filesystem::directory_iterator(path)) {
            auto extension=entry.path().extension().wstring();
            for(auto& c:extension) c=static_cast<wchar_t>(std::towlower(c));
            if(entry.is_regular_file() && (extension==L".step" || extension==L".stp")) files.push_back(entry.path());
        }
        if(files.size()!=1) throw std::runtime_error("Input directory must contain exactly one STEP file");
        path=files.front();
    }
    if(!std::filesystem::is_regular_file(path)) throw std::runtime_error("STEP file does not exist: "+path.string());
    return path;
}
}

int wmain(int argc,wchar_t** argv) {
    try {
        const auto exe=executableDirectory();
        std::filesystem::path input=exe/"rounded_box_nurbs.step",output="output.png";
        unsigned width=1920,height=1080;
        double cameraElevation=35;
        bool validateNurbs=false;
        bool profile=false;
        unsigned benchmarkFrames=1;
        unsigned samplesPerPixel=4;
        for(int i=1;i<argc;++i) {
            const std::wstring flag=argv[i];
            if(flag==L"--help") {
                std::cout << "NurbsOptixRenderer [--input file-or-directory] [--output output.png] [--width 1920] [--height 1080] [--camera-elevation 35] [--validate-nurbs] [--benchmark frames] [--profile] [--samples 1|4]\n";
                return 0;
            }
            if(flag==L"--validate-nurbs") { validateNurbs=true; continue; }
            if(flag==L"--profile") { profile=true; continue; }
            if(i+1>=argc) throw std::runtime_error("Missing command-line value");
            const std::wstring value=argv[++i];
            if(flag==L"--input") input=value;
            else if(flag==L"--output") output=value;
            else if(flag==L"--width") width=pixels(value);
            else if(flag==L"--height") height=pixels(value);
            else if(flag==L"--camera-elevation") cameraElevation=cameraAngle(value);
            else if(flag==L"--benchmark") {
                benchmarkFrames=pixels(value);
                if(benchmarkFrames>1000) throw std::runtime_error("Benchmark frame count must be 1..1000");
            }
            else if(flag==L"--samples") {
                samplesPerPixel=pixels(value);
                if(samplesPerPixel!=1 && samplesPerPixel!=4) throw std::runtime_error("Samples per pixel must be 1 or 4");
            }
            else throw std::runtime_error("Unknown command-line option");
        }

        input=inputFile(input);
        const auto model=nurbs::loadStepNurbs(input);
        const auto result=nurbs::renderNurbsOptix(model,width,height,exe/"DevicePrograms.ptx",cameraElevation,benchmarkFrames,profile,samplesPerPixel);
        if(result.solverFailures) { std::cerr << "NURBS solver exhausted its subdivision/root budget.\n"; return 2; }
        if(validateNurbs) {
            nurbs::validateNurbsRender(model,result,width,height);
            std::vector<uchar4> normals; normals.reserve(result.surfaces.size());
            for(const auto p:result.surfaces) {
                const auto channel=[](float x) { return static_cast<unsigned char>(std::clamp(127.5f*(x+1),0.0f,255.0f)); };
                normals.push_back(p.w<0 ? uchar4{0,0,0,255} : uchar4{channel(p.x),channel(p.y),channel(p.z),255});
            }
            const auto diagnostic=output.parent_path()/(output.stem().wstring()+L"-normals.png");
            nurbs::writePng(diagnostic,width,height,normals);
        }
        nurbs::writePng(output,width,height,result.pixels);
        std::cout << "Rendered: " << std::filesystem::absolute(output).string() << "\nResolution: " << width << 'x' << height
            << "\nRender time: " << result.milliseconds << " ms\n";

        return 0;

    } catch(const Standard_Failure& e) {
        std::cerr << "OCCT error: " << (e.GetMessageString() ? e.GetMessageString() : "geometry failure") << '\n'; return 1;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
