#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace flux {

// Asynchronously applies softmax(scores * scale + additive_mask) on the
// supplied CUDA stream. The mask strides are in float elements.
cudaError_t attention_score_softmax_cuda_fp32(
    const float* scores,
    const float* additive_attention_mask,
    float* output,
    float scale,
    std::size_t batch_size,
    std::size_t heads,
    std::size_t query_length,
    std::size_t key_length,
    std::size_t mask_batch_size,
    std::size_t mask_head_size,
    std::size_t mask_query_size,
    std::size_t mask_key_size,
    std::size_t mask_stride_batch,
    std::size_t mask_stride_head,
    std::size_t mask_stride_query,
    cudaStream_t stream);

}  // namespace flux
