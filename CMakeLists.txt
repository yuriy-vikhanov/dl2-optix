cmake_minimum_required(VERSION 4.2)
project(CsgOptixRenderer LANGUAGES CXX)
if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Windows x64 and MSVC are required.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "OptiX 9.1 SDK root")
set(OCCT_ROOT "$ENV{OCCT_ROOT}" CACHE PATH "OCCT installation root")
find_package(CUDAToolkit 13.2 REQUIRED)
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" _version REGEX "^#define[ \t]+OPTIX_VERSION[ \t]+[0-9]+")
if(NOT _version MATCHES "OPTIX_VERSION[ \t]+90100")
    message(FATAL_ERROR "Use the OptiX 9.1 SDK.")
endif()
find_package(OpenCASCADE CONFIG REQUIRED COMPONENTS FoundationClasses ModelingData ModelingAlgorithms
    HINTS "${OCCT_ROOT}" "${OCCT_ROOT}/cmake" "${OCCT_ROOT}/lib/cmake/opencascade")

add_executable(CsgOptixRenderer src/main.cpp src/StepCsgParser.cpp src/OcctMesher.cpp
    src/OptixRenderer.cpp src/PngWriter.cpp cuda/DevicePrograms.cu cuda/DeviceShared.h)
set_source_files_properties(cuda/DevicePrograms.cu PROPERTIES HEADER_FILE_ONLY TRUE)
target_compile_definitions(CsgOptixRenderer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_compile_options(CsgOptixRenderer PRIVATE /W4 /permissive- /EHsc /utf-8 /Zc:__cplusplus)
target_include_directories(CsgOptixRenderer PRIVATE src cuda "${OPTIX_INCLUDE_DIR}" "${OpenCASCADE_INCLUDE_DIR}")
target_link_libraries(CsgOptixRenderer PRIVATE CUDA::cudart_static CUDA::cuda_driver windowscodecs ole32)
# CUDA's static runtime requests LIBCMT; resolve its CRT calls through the
# application's DLL runtime, matching OCCT and CMAKE_MSVC_RUNTIME_LIBRARY.
target_link_options(CsgOptixRenderer PRIVATE /NODEFAULTLIB:LIBCMT)
foreach(lib TKernel TKMath TKG2d TKG3d TKGeomBase TKBRep TKGeomAlgo TKTopAlgo TKPrim TKMesh)
    if(TARGET OpenCASCADE::${lib})
        target_link_libraries(CsgOptixRenderer PRIVATE OpenCASCADE::${lib})
    elseif(TARGET ${lib})
        target_link_libraries(CsgOptixRenderer PRIVATE ${lib})
    else()
        message(FATAL_ERROR "OCCT imported target ${lib} is missing; check OpenCASCADE_DIR.")
    endif()
endforeach()
if(TARGET OpenCASCADE::TKernel)
    set(_occt_runtime_dir "$<TARGET_FILE_DIR:OpenCASCADE::TKernel>")
else()
    set(_occt_runtime_dir "$<TARGET_FILE_DIR:TKernel>")
endif()

# A normal VS custom build step produces PTX directly next to each configuration's
# executable. No manually invoked nvcc or CUDA MSBuild integration is needed.
set(ptx "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/DevicePrograms.ptx")
add_custom_command(OUTPUT "${ptx}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++20 --machine=64 --gpu-architecture=compute_75
        --compiler-bindir "${CMAKE_CXX_COMPILER}" --generate-line-info
        "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/DevicePrograms.cu" -o "${ptx}"
    DEPENDS cuda/DevicePrograms.cu cuda/DeviceShared.h src/CsgTypes.h
    VERBATIM COMMENT "Compiling OptiX PTX")
add_custom_target(CsgDevicePrograms DEPENDS "${ptx}")
add_dependencies(CsgOptixRenderer CsgDevicePrograms)
# Stage only the user's original input; there is no generated sample.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/box_sphere_intersection.STEP"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/INPUT/box_sphere_intersection.STEP")
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT CsgOptixRenderer)
set_target_properties(CsgOptixRenderer PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:CsgOptixRenderer>"
    VS_DEBUGGER_ENVIRONMENT "PATH=${_occt_runtime_dir};$ENV{PATH}")
