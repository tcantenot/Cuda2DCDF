cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(CUDA_CDF LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

add_subdirectory(external)

set(target_name "CUDA_CDF")
add_executable(${target_name} main.cu cdf.cuh)

set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CUDA_CDF)
    set_property(TARGET ${target_name} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
endif()

set_target_properties(${target_name} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(${target_name} PUBLIC CUDA::cudart ext::tinyexr)
