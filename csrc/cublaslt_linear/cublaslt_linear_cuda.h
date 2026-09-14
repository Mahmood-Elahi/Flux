#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace flux {

struct CublasLtAlgorithmConfig {
    std::int32_t algorithm_id;
    std::uint32_t tile_id;
    std::int32_t split_k;
    std::uint32_t reduction_scheme;
    std::uint32_t cta_swizzle;
    std::uint32_t custom_option;
    std::uint32_t stages_id;
};

// Launches row-major FP32 input @ weight.T with a previously validated exact
// cuBLASLt configuration. Plan and handle initialization must be warmed before
// CUDA Graph capture. The call uses the supplied stream and does not allocate.
void cublaslt_linear_config_cuda_fp32(
    const float* input,
    const float* weight,
    float* output,
    std::int64_t input_width,
    std::int64_t output_width,
    void* workspace,
    std::size_t workspace_bytes,
    int device,
    CublasLtAlgorithmConfig config,
    cudaStream_t stream);

}  // namespace flux
