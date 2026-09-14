add_executable(flux_cuda_microbenchmarks cuda_microbenchmarks.cu)
target_link_libraries(flux_cuda_microbenchmarks PRIVATE flux_cuda_kernels)
target_include_directories(flux_cuda_microbenchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
