#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace flux {

// Launches FP32 RMSNorm on the provided CUDA stream. Input consists of
// num_rows contiguous rows; weight is shared across rows. The call is
// asynchronous and returns only parameter or kernel-launch errors.
cudaError_t rmsnorm_cuda_fp32(
    const float* input,
    const float* weight,
    float* output,
    std::size_t num_rows,
    std::size_t hidden_size,
    float epsilon,
    cudaStream_t stream);

}  // namespace flux
