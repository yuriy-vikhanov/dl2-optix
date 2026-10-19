#include "StepCsgParser.h"
#include "OcctMesher.h"
#include "OptixRenderer.h"
#include "PngWriter.h"
#include <Windows.h>
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
        std::filesystem::path input=exe/"box_sphere_intersection.STEP",output="output.png";
        unsigned width=1280,height=720;
        for(int i=1;i<argc;++i) {
            const std::wstring flag=argv[i];
            if(flag==L"--help") {
                std::cout << "CsgOptixRenderer [--input file-or-directory] [--output output.png] [--width 1280] [--height 720]\n";
                return 0;
            }
            if(i+1>=argc) throw std::runtime_error("Missing command-line value");
            const std::wstring value=argv[++i];
            if(flag==L"--input") input=value;
            else if(flag==L"--output") output=value;
            else if(flag==L"--width") width=pixels(value);
            else if(flag==L"--height") height=pixels(value);
            else throw std::runtime_error("Unknown command-line option");
        }

        input=inputFile(input);

        const auto model=csg::loadStepCsg(input);
        csg::printTree(model,std::cout);
        std::vector<csg::Mesh> meshes;

        for (const auto& p:model.operands) {
            meshes.push_back(csg::tessellateOperand(p));
            std::cout << csg::primitiveName(p.type) << " mesh: " << meshes.back().vertices.size()
                << " vertices, " << meshes.back().indices.size() << " triangles\n";
        }

        const auto result=csg::renderOptix(model,meshes,width,height,exe/"DevicePrograms.ptx");
        csg::writePng(output,width,height,result.pixels);
        std::cout << "Rendered: " << std::filesystem::absolute(output).string() << "\nResolution: " << width << 'x' << height
            << "\nRender time: " << result.milliseconds << " ms\n";
        if(result.overflowRays) { std::cerr << "Hit overflow: affected pixels are magenta.\n"; return 2; }

        return 0;

    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
