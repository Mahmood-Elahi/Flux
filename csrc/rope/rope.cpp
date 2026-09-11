#include "rope.h"

#include <stdexcept>

namespace flux {
namespace {

void rotate_tensor(
    const float* input,
    const float* cos,
    const float* sin,
    float* output,
    const std::size_t batch_size,
    const std::size_t heads,
    const std::size_t sequence_length,
    const std::size_t head_dim,
    const std::size_t rope_batch_size,
    const RopeStrides input_strides,
    const RopeEmbeddingStrides cos_strides,
    const RopeEmbeddingStrides sin_strides) {
    const std::size_t half = head_dim / 2;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const std::size_t rope_batch = rope_batch_size == 1 ? 0 : batch;
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t sequence = 0; sequence < sequence_length; ++sequence) {
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    const std::size_t paired_dimension =
                        dimension < half ? dimension + half : dimension - half;
                    const float sign = dimension < half ? -1.0F : 1.0F;
                    const std::ptrdiff_t base =
                        static_cast<std::ptrdiff_t>(batch) * input_strides.batch +
                        static_cast<std::ptrdiff_t>(head) * input_strides.head +
                        static_cast<std::ptrdiff_t>(sequence) * input_strides.sequence;
                    const std::ptrdiff_t embedding_base =
                        static_cast<std::ptrdiff_t>(rope_batch) * cos_strides.batch +
                        static_cast<std::ptrdiff_t>(sequence) * cos_strides.sequence;
                    const std::ptrdiff_t sin_base =
                        static_cast<std::ptrdiff_t>(rope_batch) * sin_strides.batch +
                        static_cast<std::ptrdiff_t>(sequence) * sin_strides.sequence;
                    const std::size_t output_index =
                        ((batch * heads + head) * sequence_length + sequence) * head_dim + dimension;
                    const float value = input[base + static_cast<std::ptrdiff_t>(dimension) * input_strides.dimension];
                    const float paired = input[base + static_cast<std::ptrdiff_t>(paired_dimension) * input_strides.dimension];
                    output[output_index] =
                        value * cos[embedding_base + static_cast<std::ptrdiff_t>(dimension) * cos_strides.dimension] +
                        sign * paired * sin[sin_base + static_cast<std::ptrdiff_t>(dimension) * sin_strides.dimension];
                }
            }
        }
    }
}

}  // namespace

void rope_fp32(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_output,
    float* key_output,
    const std::size_t batch_size,
    const std::size_t query_heads,
    const std::size_t key_heads,
    const std::size_t sequence_length,
    const std::size_t head_dim,
    const std::size_t rope_batch_size,
    const RopeStrides query_strides,
    const RopeStrides key_strides,
    const RopeEmbeddingStrides cos_strides,
    const RopeEmbeddingStrides sin_strides) {
    if (query == nullptr || key == nullptr || cos == nullptr || sin == nullptr ||
        query_output == nullptr || key_output == nullptr) {
        throw std::invalid_argument("tensor pointers must not be null");
    }
    if (batch_size == 0 || query_heads == 0 || key_heads == 0 ||
        sequence_length == 0 || head_dim == 0 || head_dim % 2 != 0 ||
        (rope_batch_size != 1 && rope_batch_size != batch_size)) {
        throw std::invalid_argument("invalid RoPE dimensions");
    }
    rotate_tensor(query, cos, sin, query_output, batch_size, query_heads,
        sequence_length, head_dim, rope_batch_size, query_strides,
        cos_strides, sin_strides);
    rotate_tensor(key, cos, sin, key_output, batch_size, key_heads,
        sequence_length, head_dim, rope_batch_size, key_strides,
        cos_strides, sin_strides);
}

}  // namespace flux
