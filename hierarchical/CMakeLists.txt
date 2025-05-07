cmake_minimum_required(VERSION 3.15)

project(HierarchicalExample
        VERSION 1.0
        LANGUAGES CXX)

# create executable
add_executable(hierarchical_reverse main.cpp)

# set C++ standard to use
target_compile_features(hierarchical_reverse PUBLIC cxx_std_17)

# check which SYCL implementation is requested
set(EXAMPLE_SYCL_IMPLEMENTATION "" CACHE STRING "Set the SYCL implementation that should be used.")
if (EXAMPLE_SYCL_IMPLEMENTATION STREQUAL "icpx" OR EXAMPLE_SYCL_IMPLEMENTATION STREQUAL "dpcpp")
    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "IntelLLVM")
        message(FATAL_ERROR "Please set CMAKE_CXX_COMPILER to icpx/dpcpp in order to use it as SYCL implementation.")
    endif ()
    # use Intel icpx
    message(STATUS "Using Intel's DPC++/icpx as SYCL compiler.")
    target_compile_options(hierarchical_reverse PRIVATE -sycl-std=2020 -fsycl)
    target_link_options(hierarchical_reverse PRIVATE -fsycl)
elseif (EXAMPLE_SYCL_IMPLEMENTATION STREQUAL "acpp")
    # AdaptiveCpp targets must be manually set
    if (NOT ACPP_TARGETS)
        message(FATAL_ERROR "Please manually set the ACPP_TARGETS!")
    elseif (ACPP_TARGETS MATCHES "generic")
        message(FATAL_ERROR "ACPP_TARGETS=\"generic\" does not support the hierarchical data parallel kernels!")
    endif ()
    find_package(AdaptiveCpp CONFIG REQUIRED)
    # use AdaptiveCpp
    message(STATUS "Using AdaptiveCpp as SYCL compiler.")
    add_sycl_to_target(TARGET hierarchical_reverse SOURCES main.cpp)
else ()
    message(FATAL_ERROR "Unrecognized SYCL implementation: \"${EXAMPLE_SYCL_IMPLEMENTATION}\". Please set EXAMPLE_SYCL_IMPLEMENTATION.")
endif ()

# set the offload device (CPU or GPU)
set(EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE "gpu" CACHE STRING "Set the SYCL offload device type.")
if (NOT (EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE STREQUAL "gpu" OR EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE STREQUAL "cpu"))
    message(FATAL_ERROR "Invalid offload device type: \"${EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE}\". Must be one of: gpu, cpu.")
endif ()
message(STATUS "Using ${EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE} as target device.")

# add compile definition
if (EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE STREQUAL "gpu")
    target_compile_definitions(hierarchical_reverse PUBLIC EXAMPLE_SYCL_GPU_DEVICE)
endif ()
