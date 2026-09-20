#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#

cmake_minimum_required(VERSION 3.14)
project(kurtm_benchmark VERSION 1.0.0)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=linxicore6100 --rtlib=compiler-rt")
    set(KURTM_COMPILER "bisheng")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.4-a+sve+sme-f64f64 -ffast-math -fomit-frame-pointer")
    set(KURTM_COMPILER "gcc")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pipe -fopenmp")

include_directories("$ENV{HPCKIT_PATH}/latest/kugeo/${KURTM_COMPILER}/include")
link_directories($ENV{HPCKIT_PATH}/latest/kugeo/${KURTM_COMPILER}/lib)

add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark
        -Wl,--start-group
        kurtm
        -Wl,--end-group
)

