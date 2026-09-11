#pragma once

#include <cstddef>

namespace flux {

struct RopeStrides {
    std::ptrdiff_t batch;
    std::ptrdiff_t head;
    std::ptrdiff_t sequence;
    std::ptrdiff_t dimension;
};

struct RopeEmbeddingStrides {
    std::ptrdiff_t batch;
    std::ptrdiff_t sequence;
    std::ptrdiff_t dimension;
};

void rope_fp32(
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
    RopeEmbeddingStrides sin_strides);

}  // namespace flux
