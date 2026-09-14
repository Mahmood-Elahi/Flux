cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
endif()

project(FluxNative LANGUAGES CXX CUDA)

option(FLUX_BUILD_NATIVE_TESTS "Build standalone CUDA correctness tests" ON)
option(FLUX_BUILD_NATIVE_BENCHMARKS "Build standalone CUDA microbenchmarks" OFF)
option(FLUX_BUILD_NATIVE_RUNTIME_TESTS
    "Build ATen-backed native prefill/decode runtime tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(flux_cuda_kernels STATIC
    csrc/rmsnorm/rmsnorm_cuda.cu
    csrc/residual_rmsnorm/residual_rmsnorm_cuda.cu
    csrc/softmax/softmax_cuda.cu
    csrc/rope/rope_cuda.cu
    csrc/packed_swiglu/packed_swiglu_cuda.cu
    csrc/packed_qkv_rope_cache/packed_qkv_rope_cache_cuda.cu
    csrc/gqa_decode_attention/gqa_decode_attention_cuda.cu
    csrc/streaming_prefill_gqa/streaming_prefill_gqa_cuda.cu
    csrc/packed_gate_up_gemv/packed_gate_up_gemv_cuda.cu)
target_include_directories(flux_cuda_kernels PUBLIC
    csrc/rmsnorm
    csrc/residual_rmsnorm
    csrc/softmax
    csrc/attention_score_softmax
    csrc/rope
    csrc/packed_swiglu
    csrc/packed_qkv_rope_cache
    csrc/gqa_decode_attention
    csrc/streaming_prefill_gqa
    csrc/packed_gate_up_gemv)
set_target_properties(flux_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(flux_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Xcompiler=/EHsc>)
else()
    target_compile_options(flux_cuda_kernels PRIVATE -Wall -Wextra)
endif()

add_library(flux_cpu_ops STATIC
    csrc/rmsnorm/rmsnorm.cpp
    csrc/residual_rmsnorm/residual_rmsnorm.cpp
    csrc/softmax/softmax.cpp)
target_include_directories(flux_cpu_ops PUBLIC
    csrc/rmsnorm
    csrc/residual_rmsnorm
    csrc/softmax)
if(MSVC)
    target_compile_options(flux_cpu_ops PRIVATE /W4 /permissive-)
else()
    target_compile_options(flux_cpu_ops PRIVATE -Wall -Wextra)
endif()

if(FLUX_BUILD_NATIVE_TESTS)
    enable_testing()
    add_subdirectory(csrc/tests)
endif()

if(FLUX_BUILD_NATIVE_BENCHMARKS)
    add_subdirectory(csrc/benchmarks)
endif()
