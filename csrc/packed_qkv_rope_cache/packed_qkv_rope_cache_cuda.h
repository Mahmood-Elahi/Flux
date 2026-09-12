#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flux {

struct PackedQKVStrides {
    std::ptrdiff_t batch;
    std::ptrdiff_t sequence;
    std::ptrdiff_t dimension;
};

struct PackedQKVEmbeddingStrides {
    std::ptrdiff_t batch;
    std::ptrdiff_t sequence;
    std::ptrdiff_t dimension;
};

struct PackedQKVCacheStrides {
    std::ptrdiff_t batch;
    std::ptrdiff_t head;
    std::ptrdiff_t sequence;
    std::ptrdiff_t dimension;
};

cudaError_t packed_qkv_rope_cache_cuda_fp32(
    const float* packed_qkv,
    const float* cos,
    const float* sin,
    float* key_cache,
    float* value_cache,
    std::int64_t* cache_length,
    float* query_output,
    std::size_t cache_capacity,
    PackedQKVStrides packed_strides,
    PackedQKVEmbeddingStrides cos_strides,
    PackedQKVEmbeddingStrides sin_strides,
    PackedQKVCacheStrides key_cache_strides,
    PackedQKVCacheStrides value_cache_strides,
    cudaStream_t stream);

}  // namespace flux
