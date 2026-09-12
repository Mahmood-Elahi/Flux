#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace flux {

// Launches packed FP32 SwiGLU asynchronously on the provided stream.
cudaError_t packed_swiglu_cuda_fp32(
    const float* packed,
    float* output,
    std::size_t num_rows,
    std::size_t intermediate_size,
    cudaStream_t stream);

}  // namespace flux
