#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace csg {
inline std::string apiFailure(const char* api,int code,const char* name,const char* detail,const char* expression,const char* file,int line) {
    std::ostringstream message;
    message << api << " error " << code << " (" << (name?name:"unknown") << ") at " << file << ':' << line
            << "\n  " << expression << "\n  " << (detail?detail:"No additional description");
    return message.str();
}
inline void checkCuda(cudaError_t result,const char* expression,const char* file,int line) {
    if(result!=cudaSuccess) throw std::runtime_error(apiFailure("CUDA runtime",static_cast<int>(result),cudaGetErrorName(result),cudaGetErrorString(result),expression,file,line));
}
inline void checkDriver(CUresult result,const char* expression,const char* file,int line) {
    if(result==CUDA_SUCCESS) return;
    const char* name=nullptr; const char* detail=nullptr;
    (void)cuGetErrorName(result,&name); (void)cuGetErrorString(result,&detail);
    throw std::runtime_error(apiFailure("CUDA driver",static_cast<int>(result),name,detail,expression,file,line));
}
inline void checkOptix(OptixResult result,const char* expression,const char* file,int line,const char* log=nullptr) {
    if(result!=OPTIX_SUCCESS) {
        auto message=apiFailure("OptiX",static_cast<int>(result),optixGetErrorName(result),optixGetErrorString(result),expression,file,line);
        if(log && *log) message+="\n  Compiler log: "+std::string(log);
        throw std::runtime_error(message);
    }
}
// Destructors cannot throw, but cleanup failures retain API/location details.
template<class F> void cleanup(F&& action) noexcept {
    try { action(); } catch(const std::exception& e) { std::cerr << "Cleanup: " << e.what() << '\n'; }
}
}
#define CUDA_CHECK(call) ::csg::checkCuda((call),#call,__FILE__,__LINE__)
#define DRIVER_CHECK(call) ::csg::checkDriver((call),#call,__FILE__,__LINE__)
#define OPTIX_CHECK(call) ::csg::checkOptix((call),#call,__FILE__,__LINE__)
#define OPTIX_CHECK_LOG(call,log) ::csg::checkOptix((call),#call,__FILE__,__LINE__,(log))
