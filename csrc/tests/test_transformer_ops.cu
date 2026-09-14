#include "cuda_test_utils.cuh"

#include "attention_score_softmax_cuda.h"
#include "packed_swiglu_cuda.h"
#include "rope_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

using flux::test::DeviceBuffer;
using flux::test::Stream;
using flux::test::check_cuda;
using flux::test::copy_to_device;
using flux::test::copy_to_host;
using flux::test::deterministic_values;
using flux::test::expect;
using flux::test::expect_close;
using flux::test::expect_unchanged;

std::vector<float> softmax_reference(
    const std::vector<float>& scores,
    const std::vector<float>& mask,
    float scale,
    std::size_t batch_size,
    std::size_t heads,
    std::size_t query_length,
    std::size_t key_length) {
    std::vector<float> output(scores.size());
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t query = 0; query < query_length; ++query) {
                const std::size_t row =
                    (batch * heads + head) * query_length + query;
                const std::size_t offset = row * key_length;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t key = 0; key < key_length; ++key) {
                    output[offset + key] =
                        scores[offset + key] * scale +
                        mask[query * key_length + key];
                    maximum = std::max(maximum, output[offset + key]);
                }
                double sum = 0.0;
                for (std::size_t key = 0; key < key_length; ++key) {
                    output[offset + key] = static_cast<float>(
                        std::exp(static_cast<double>(output[offset + key] - maximum)));
                    sum += output[offset + key];
                }
                for (std::size_t key = 0; key < key_length; ++key) {
                    output[offset + key] = static_cast<float>(
                        static_cast<double>(output[offset + key]) / sum);
                }
            }
        }
    }
    return output;
}

void test_attention_score_softmax() {
    constexpr std::size_t batch_size = 2;
    constexpr std::size_t heads = 3;
    constexpr std::size_t query_length = 5;
    constexpr std::size_t key_length = 33;
    constexpr float scale = 0.125F;
    const std::size_t score_count =
        batch_size * heads * query_length * key_length;
    std::vector<float> scores = deterministic_values(score_count, 7.0F);
    std::vector<float> mask(query_length * key_length, 0.0F);
    for (std::size_t query = 0; query < query_length; ++query) {
        for (std::size_t key = query + 2; key < key_length; ++key) {
            mask[query * key_length + key] = -10000.0F;
        }
    }
    const std::vector<float> expected = softmax_reference(
        scores, mask, scale, batch_size, heads, query_length, key_length);

    DeviceBuffer<float> device_scores(score_count);
    DeviceBuffer<float> device_mask(mask.size());
    DeviceBuffer<float> device_output(score_count);
    Stream stream;
    copy_to_device(device_scores, scores, stream.get());
    copy_to_device(device_mask, mask, stream.get());
    check_cuda(flux::attention_score_softmax_cuda_fp32(
        device_scores.get(), device_mask.get(), device_output.get(), scale,
        batch_size, heads, query_length, key_length,
        1, 1, query_length, key_length,
        query_length * key_length, query_length * key_length, key_length,
        stream.get()), "attention_score_softmax_cuda_fp32");
    const std::vector<float> actual = copy_to_host(device_output, stream.get());
    expect_close(actual, expected, 2.0e-5F, 2.0e-6F,
                 "attention score softmax");
    expect_unchanged(copy_to_host(device_scores, stream.get()), scores,
                     "attention scores");

    float* pointer = reinterpret_cast<float*>(1);
    expect(flux::attention_score_softmax_cuda_fp32(
        nullptr, pointer, pointer, 1.0F, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, nullptr) == cudaErrorInvalidValue,
        "attention score softmax accepted null scores");
}

float rotate_reference(
    const std::vector<float>& input,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    std::size_t input_offset,
    std::size_t embedding_offset,
    std::size_t dimension,
    std::size_t head_dim) {
    const std::size_t half = head_dim / 2;
    const std::size_t paired =
        dimension < half ? dimension + half : dimension - half;
    const float sign = dimension < half ? -1.0F : 1.0F;
    return input[input_offset + dimension] *
               cosine[embedding_offset + dimension] +
           sign * input[input_offset + paired] *
               sine[embedding_offset + dimension];
}

void test_rope() {
    constexpr std::size_t batch_size = 2;
    constexpr std::size_t query_heads = 9;
    constexpr std::size_t key_heads = 3;
    constexpr std::size_t sequence_length = 7;
    constexpr std::size_t head_dim = 64;
    const std::size_t query_count =
        batch_size * query_heads * sequence_length * head_dim;
    const std::size_t key_count =
        batch_size * key_heads * sequence_length * head_dim;
    const std::size_t embedding_count = sequence_length * head_dim;
    std::vector<float> query = deterministic_values(query_count, 2.0F, 0.0F, 1);
    std::vector<float> key = deterministic_values(key_count, 2.0F, 0.0F, 2);
    std::vector<float> cosine(embedding_count);
    std::vector<float> sine(embedding_count);
    for (std::size_t sequence = 0; sequence < sequence_length; ++sequence) {
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            const float angle = static_cast<float>(sequence * head_dim + dimension) *
                0.0007F;
            cosine[sequence * head_dim + dimension] = std::cos(angle);
            sine[sequence * head_dim + dimension] = std::sin(angle);
        }
    }
    std::vector<float> expected_query(query_count);
    std::vector<float> expected_key(key_count);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < query_heads; ++head) {
            for (std::size_t sequence = 0; sequence < sequence_length; ++sequence) {
                const std::size_t base =
                    ((batch * query_heads + head) * sequence_length + sequence) *
                    head_dim;
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    expected_query[base + dimension] = rotate_reference(
                        query, cosine, sine, base, sequence * head_dim,
                        dimension, head_dim);
                }
            }
        }
        for (std::size_t head = 0; head < key_heads; ++head) {
            for (std::size_t sequence = 0; sequence < sequence_length; ++sequence) {
                const std::size_t base =
                    ((batch * key_heads + head) * sequence_length + sequence) *
                    head_dim;
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    expected_key[base + dimension] = rotate_reference(
                        key, cosine, sine, base, sequence * head_dim,
                        dimension, head_dim);
                }
            }
        }
    }

    DeviceBuffer<float> device_query(query_count);
    DeviceBuffer<float> device_key(key_count);
    DeviceBuffer<float> device_cosine(embedding_count);
    DeviceBuffer<float> device_sine(embedding_count);
    DeviceBuffer<float> device_query_output(query_count);
    DeviceBuffer<float> device_key_output(key_count);
    Stream stream;
    copy_to_device(device_query, query, stream.get());
    copy_to_device(device_key, key, stream.get());
    copy_to_device(device_cosine, cosine, stream.get());
    copy_to_device(device_sine, sine, stream.get());
    const flux::RopeStrides query_strides{
        static_cast<std::ptrdiff_t>(query_heads * sequence_length * head_dim),
        static_cast<std::ptrdiff_t>(sequence_length * head_dim),
        static_cast<std::ptrdiff_t>(head_dim), 1};
    const flux::RopeStrides key_strides{
        static_cast<std::ptrdiff_t>(key_heads * sequence_length * head_dim),
        static_cast<std::ptrdiff_t>(sequence_length * head_dim),
        static_cast<std::ptrdiff_t>(head_dim), 1};
    const flux::RopeEmbeddingStrides embedding_strides{
        0, static_cast<std::ptrdiff_t>(head_dim), 1};
    check_cuda(flux::rope_cuda_fp32(
        device_query.get(), device_key.get(), device_cosine.get(),
        device_sine.get(), device_query_output.get(), device_key_output.get(),
        batch_size, query_heads, key_heads, sequence_length, head_dim, 1,
        query_strides, key_strides, embedding_strides, embedding_strides,
        stream.get()), "rope_cuda_fp32");
    expect_close(copy_to_host(device_query_output, stream.get()), expected_query,
                 1.0e-6F, 2.0e-7F, "RoPE query");
    expect_close(copy_to_host(device_key_output, stream.get()), expected_key,
                 1.0e-6F, 2.0e-7F, "RoPE key");
    expect(flux::rope_cuda_fp32(
        device_query.get(), device_key.get(), device_cosine.get(),
        device_sine.get(), device_query_output.get(), device_key_output.get(),
        1, 1, 1, 1, 63, 1, query_strides, key_strides,
        embedding_strides, embedding_strides, nullptr) == cudaErrorInvalidValue,
        "RoPE accepted an odd head dimension");
}

void test_packed_swiglu() {
    constexpr std::size_t rows = 5;
    constexpr std::size_t width = 1536;
    std::vector<float> packed = deterministic_values(rows * width * 2, 4.0F);
    std::vector<float> expected(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            const float gate = packed[row * width * 2 + column];
            const float up = packed[row * width * 2 + width + column];
            expected[row * width + column] =
                (gate / (1.0F + std::exp(-gate))) * up;
        }
    }
    DeviceBuffer<float> device_packed(packed.size());
    DeviceBuffer<float> device_output(expected.size());
    Stream stream;
    copy_to_device(device_packed, packed, stream.get());
    check_cuda(flux::packed_swiglu_cuda_fp32(
        device_packed.get(), device_output.get(), rows, width, stream.get()),
        "packed_swiglu_cuda_fp32");
    expect_close(copy_to_host(device_output, stream.get()), expected,
                 2.0e-6F, 2.0e-6F, "packed SwiGLU");
    expect_unchanged(copy_to_host(device_packed, stream.get()), packed,
                     "packed SwiGLU input");
    expect(flux::packed_swiglu_cuda_fp32(
        nullptr, device_output.get(), rows, width, nullptr) ==
        cudaErrorInvalidValue, "packed SwiGLU accepted a null input");
}

}  // namespace

int main() {
    return flux::test::run("Flux transformer CUDA operators", [] {
        test_attention_score_softmax();
        test_rope();
        test_packed_swiglu();
    });
}
