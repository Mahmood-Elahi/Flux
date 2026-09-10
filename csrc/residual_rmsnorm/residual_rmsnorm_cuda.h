#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace flux {

// Launches fused FP32 residual addition and RMSNorm on the provided CUDA
// stream. Inputs contain num_rows contiguous rows; weight is shared across
// rows. norm_out is written first in the public output order, followed by the
// preserved pre-normalization residual_out. All buffers must be distinct. The
// call is asynchronous and returns only parameter or launch errors.
cudaError_t residual_rmsnorm_cuda_fp32(
    const float* hidden,
    const float* residual,
    const float* weight,
    float* norm_out,
    float* residual_out,
    std::size_t num_rows,
    std::size_t hidden_size,
    float epsilon,
    cudaStream_t stream);

}  // namespace flux
