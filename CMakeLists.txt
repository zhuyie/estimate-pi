cmake_minimum_required(VERSION 3.16)

project(estimate-pi)
enable_language(OBJC)

set(CMAKE_CXX_STANDARD 11)

add_executable(
    estimate_pi_cpu 
    estimate_pi_cpu.cpp)

add_executable(
    estimate_pi_opencl 
    estimate_pi_opencl.cpp)
if (MSVC)
    target_include_directories(
        estimate_pi_opencl 
        PUBLIC "$ENV{OCL_ROOT}/include"
        PUBLIC "3rdparty/GPUPerfAPI/include")
    if ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_link_directories(estimate_pi_opencl PUBLIC "$ENV{OCL_ROOT}/lib/x86")
    else()
        target_link_directories(estimate_pi_opencl PUBLIC "$ENV{OCL_ROOT}/lib/x86_64")
    endif()
    target_link_libraries(estimate_pi_opencl "opencl.lib")
endif()
if (APPLE)
    target_link_libraries(estimate_pi_opencl "-framework OpenCL")
endif()

if (APPLE)
    add_executable(
        estimate_pi_metal
        estimate_pi_metal.m)
    # Metal API will failed if we don't link with CoreGraphics.
    target_link_libraries(
        estimate_pi_metal
        "-framework Foundation"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework CoreGraphics")
endif()