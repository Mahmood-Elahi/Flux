function(flux_add_cuda_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE flux_cuda_kernels)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "cuda;native")
endfunction()

function(flux_add_cpu_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE flux_cpu_ops)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "cpu;native")
endfunction()

flux_add_cpu_test(flux_test_rmsnorm_cpu ../rmsnorm/test_rmsnorm.cpp)
flux_add_cpu_test(
    flux_test_residual_rmsnorm_cpu
    ../residual_rmsnorm/test_residual_rmsnorm.cpp)
flux_add_cpu_test(flux_test_softmax_cpu ../softmax/test_softmax.cpp)

flux_add_cuda_test(flux_test_rmsnorm ../rmsnorm/test_rmsnorm_cuda.cu)
flux_add_cuda_test(
    flux_test_residual_rmsnorm
    ../residual_rmsnorm/test_residual_rmsnorm_cuda.cu)
flux_add_cuda_test(flux_test_softmax ../softmax/test_softmax_cuda.cu)
flux_add_cuda_test(flux_test_transformer_ops test_transformer_ops.cu)
flux_add_cuda_test(flux_test_decode_ops test_decode_ops.cu)
flux_add_cuda_test(
    flux_test_streaming_prefill_gqa
    test_streaming_prefill_gqa.cu)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(flux_test_streaming_prefill_gqa PRIVATE CUDA::cublas)

if(FLUX_BUILD_NATIVE_RUNTIME_TESTS)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        find_package(CUDAToolkit REQUIRED)
        add_executable(flux_test_native_runtimes
            test_native_runtimes.cu
            ../native_decode_runtime/native_decode_runtime_cuda.cu
            ../native_prefill_runtime/native_prefill_runtime_cuda.cu
            ../cublaslt_linear/cublaslt_linear_torch.cpp)
        target_link_libraries(flux_test_native_runtimes PRIVATE
            flux_cuda_kernels
            ${TORCH_LIBRARIES}
            CUDA::cublas
            CUDA::cublasLt)
        target_include_directories(flux_test_native_runtimes PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ../native_decode_runtime
            ../native_prefill_runtime
            ../cublaslt_linear)
        target_compile_options(flux_test_native_runtimes PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj -Xcompiler=/Zc:preprocessor>)
        set_property(TARGET flux_test_native_runtimes PROPERTY CXX_STANDARD 20)
        set_property(TARGET flux_test_native_runtimes PROPERTY CUDA_STANDARD 20)
        add_test(NAME flux_test_native_runtimes COMMAND flux_test_native_runtimes)
        set_tests_properties(flux_test_native_runtimes PROPERTIES
            LABELS "cuda;native;runtime")
    else()
        message(WARNING
            "PyTorch CMake package was not found; native runtime test disabled")
    endif()
endif()
