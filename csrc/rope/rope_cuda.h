#pragma once

#include "rope.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace flux {

cudaError_t rope_cuda_fp32(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_output,
    float* key_output,
    std::size_t batch_size,
    std::size_t query_heads,
    std::size_t key_heads,
    std::size_t sequence_length,
    std::size_t head_dim,
    std::size_t rope_batch_size,
    RopeStrides query_strides,
    RopeStrides key_strides,
    RopeEmbeddingStrides cos_strides,
    RopeEmbeddingStrides sin_strides,
    cudaStream_t stream);

}  // namespace flux
