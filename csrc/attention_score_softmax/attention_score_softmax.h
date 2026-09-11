#pragma once

#include <cstddef>

namespace flux {

// Applies softmax(scores * scale + additive_mask) over each final-dimension
// row. Scores and output use contiguous [B, H, Q, K] storage. The contiguous
// mask has broadcastable B/H/Q dimensions and a non-broadcast K dimension.
void attention_score_softmax_fp32(
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
    std::size_t mask_key_size);

}  // namespace flux
