#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace flux {

cudaError_t gqa_decode_attention_cuda_fp32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* additive_attention_mask,
    const std::int64_t* cache_length,
    float* output,
    float* workspace,
    float scale,
    std::size_t batch_size,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t cache_capacity,
    std::size_t head_dim,
    std::size_t num_chunks,
    const std::int64_t* query_strides,
    const std::int64_t* key_strides,
    const std::int64_t* value_strides,
    const std::int64_t* mask_sizes,
    const std::int64_t* mask_strides,
    cudaStream_t stream);

}  // namespace flux
