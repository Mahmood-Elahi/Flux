#include "attention_score_softmax.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

namespace flux {

void attention_score_softmax_fp32(
    const float* scores,
    const float* additive_attention_mask,
    float* output,
    const float scale,
    const std::size_t batch_size,
    const std::size_t heads,
    const std::size_t query_length,
    const std::size_t key_length,
    const std::size_t mask_batch_size,
    const std::size_t mask_head_size,
    const std::size_t mask_query_size,
    const std::size_t mask_key_size) {
    if (scores == nullptr || additive_attention_mask == nullptr ||
        output == nullptr) {
        throw std::invalid_argument("tensor pointers must not be null");
    }
    if (batch_size == 0 || heads == 0 || query_length == 0 ||
        key_length == 0) {
        throw std::invalid_argument("score dimensions must be positive");
    }
    if (batch_size > std::numeric_limits<std::size_t>::max() / heads ||
        batch_size * heads >
            std::numeric_limits<std::size_t>::max() / query_length) {
        throw std::invalid_argument("score row count overflows size_t");
    }

    const std::size_t mask_stride_query = mask_key_size;
    const std::size_t mask_stride_head = mask_query_size * mask_key_size;
    const std::size_t mask_stride_batch =
        mask_head_size * mask_query_size * mask_key_size;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t query = 0; query < query_length; ++query) {
                const std::size_t row =
                    (batch * heads + head) * query_length + query;
                const std::size_t row_offset = row * key_length;
                const std::size_t mask_offset =
                    (mask_batch_size == 1 ? 0 : batch) * mask_stride_batch +
                    (mask_head_size == 1 ? 0 : head) * mask_stride_head +
                    (mask_query_size == 1 ? 0 : query) * mask_stride_query;

                float row_max = -FLT_MAX;
                for (std::size_t key = 0; key < key_length; ++key) {
                    const float scaled = scores[row_offset + key] * scale;
                    const std::size_t mask_key = mask_key_size == 1 ? 0 : key;
                    const float value = scaled +
                        additive_attention_mask[mask_offset + mask_key];
                    output[row_offset + key] = value;
                    row_max = std::max(row_max, value);
                }

                float row_sum = 0.0F;
                for (std::size_t key = 0; key < key_length; ++key) {
                    const std::size_t index = row_offset + key;
                    const float exponential = std::exp(output[index] - row_max);
                    output[index] = exponential;
                    row_sum += exponential;
                }
                for (std::size_t key = 0; key < key_length; ++key) {
                    output[row_offset + key] /= row_sum;
                }
            }
        }
    }
}

}  // namespace flux
