#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace flux {

// Launches numerically stable FP32 softmax independently for each contiguous
// row on the provided CUDA stream. Input and output each contain
// num_rows * row_width elements. The call is asynchronous and returns only
// parameter or kernel-launch errors.
cudaError_t softmax_cuda_fp32(
    const float* input,
    float* output,
    std::size_t num_rows,
    std::size_t row_width,
    cudaStream_t stream);

}  // namespace flux
