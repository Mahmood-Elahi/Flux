#include "cuda_test_utils.cuh"

#include "gqa_decode_attention_cuda.h"
#include "packed_gate_up_gemv_cuda.h"
#include "packed_qkv_rope_cache_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr std::size_t kQueryHeads = 9;
constexpr std::size_t kKvHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::size_t kPackedWidth = kQueryWidth + 2 * kKvWidth;

float rotate(
    const std::vector<float>& packed,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    std::size_t packed_offset,
    std::size_t dimension) {
    const std::size_t half = kHeadDim / 2;
    const std::size_t paired =
        dimension < half ? dimension + half : dimension - half;
    const float sign = dimension < half ? -1.0F : 1.0F;
    return packed[packed_offset + dimension] * cosine[dimension] +
        sign * packed[packed_offset + paired] * sine[dimension];
}

void test_packed_qkv_rope_cache_and_graph() {
    constexpr std::size_t capacity = 7;
    constexpr std::int64_t position = 3;
    std::vector<float> packed = deterministic_values(kPackedWidth, 2.0F, 0.0F, 11);
    std::vector<float> cosine(kHeadDim);
    std::vector<float> sine(kHeadDim);
    for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
        const float angle = 0.01F * static_cast<float>(dimension + 1);
        cosine[dimension] = std::cos(angle);
        sine[dimension] = std::sin(angle);
    }
    const std::size_t cache_count = kKvHeads * capacity * kHeadDim;
    std::vector<float> initial_cache(cache_count, -7.0F);
    std::vector<float> expected_key = initial_cache;
    std::vector<float> expected_value = initial_cache;
    std::vector<float> expected_query(kQueryWidth);
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
            expected_query[head * kHeadDim + dimension] = rotate(
                packed, cosine, sine, head * kHeadDim, dimension);
        }
    }
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
            const std::size_t cache_index =
                (head * capacity + position) * kHeadDim + dimension;
            expected_key[cache_index] = rotate(
                packed, cosine, sine, kQueryWidth + head * kHeadDim,
                dimension);
            expected_value[cache_index] = packed[
                kQueryWidth + kKvWidth + head * kHeadDim + dimension];
        }
    }

    DeviceBuffer<float> device_packed(packed.size());
    DeviceBuffer<float> device_cosine(cosine.size());
    DeviceBuffer<float> device_sine(sine.size());
    DeviceBuffer<float> device_key(cache_count);
    DeviceBuffer<float> device_value(cache_count);
    DeviceBuffer<float> device_query(kQueryWidth);
    DeviceBuffer<std::int64_t> device_position(1);
    Stream stream;
    copy_to_device(device_packed, packed, stream.get());
    copy_to_device(device_cosine, cosine, stream.get());
    copy_to_device(device_sine, sine, stream.get());
    copy_to_device(device_key, initial_cache, stream.get());
    copy_to_device(device_value, initial_cache, stream.get());
    copy_to_device(device_position, std::vector<std::int64_t>{position}, stream.get());

    const flux::PackedQKVStrides packed_strides{
        static_cast<std::ptrdiff_t>(kPackedWidth),
        static_cast<std::ptrdiff_t>(kPackedWidth), 1};
    const flux::PackedQKVEmbeddingStrides embedding_strides{
        static_cast<std::ptrdiff_t>(kHeadDim),
        static_cast<std::ptrdiff_t>(kHeadDim), 1};
    const flux::PackedQKVCacheStrides cache_strides{
        static_cast<std::ptrdiff_t>(cache_count),
        static_cast<std::ptrdiff_t>(capacity * kHeadDim),
        static_cast<std::ptrdiff_t>(kHeadDim), 1};
    check_cuda(flux::packed_qkv_rope_cache_at_position_cuda_fp32(
        device_packed.get(), device_cosine.get(), device_sine.get(),
        device_key.get(), device_value.get(), device_position.get(),
        device_query.get(), capacity, packed_strides, embedding_strides,
        embedding_strides, cache_strides, cache_strides, stream.get()),
        "packed_qkv_rope_cache_at_position_cuda_fp32");
    expect_close(copy_to_host(device_query, stream.get()), expected_query,
                 1.0e-6F, 2.0e-7F, "packed QKV query");
    expect_close(copy_to_host(device_key, stream.get()), expected_key,
                 1.0e-6F, 2.0e-7F, "packed QKV key cache");
    expect_close(copy_to_host(device_value, stream.get()), expected_value,
                 0.0F, 0.0F, "packed QKV value cache");
    expect(copy_to_host(device_position, stream.get())[0] == position,
           "at-position launch advanced device state");

    copy_to_device(device_key, initial_cache, stream.get());
    copy_to_device(device_value, initial_cache, stream.get());
    copy_to_device(device_position, std::vector<std::int64_t>{position}, stream.get());
    stream.synchronize();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check_cuda(cudaStreamBeginCapture(
        stream.get(), cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture packed QKV");
    check_cuda(flux::packed_qkv_rope_cache_cuda_fp32(
        device_packed.get(), device_cosine.get(), device_sine.get(),
        device_key.get(), device_value.get(), device_position.get(),
        device_query.get(), capacity, packed_strides, embedding_strides,
        embedding_strides, cache_strides, cache_strides, stream.get()),
        "capture packed_qkv_rope_cache_cuda_fp32");
    check_cuda(cudaStreamEndCapture(stream.get(), &graph),
               "cudaStreamEndCapture packed QKV");
    check_cuda(cudaGraphInstantiate(&executable, graph, 0),
               "cudaGraphInstantiate packed QKV");
    check_cuda(cudaGraphLaunch(executable, stream.get()),
               "cudaGraphLaunch packed QKV");
    stream.synchronize();
    expect(copy_to_host(device_position, stream.get())[0] == position + 1,
           "captured packed QKV did not advance device state");
    check_cuda(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
    check_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");

    expect(flux::packed_qkv_rope_cache_cuda_fp32(
        nullptr, device_cosine.get(), device_sine.get(), device_key.get(),
        device_value.get(), device_position.get(), device_query.get(), capacity,
        packed_strides, embedding_strides, embedding_strides, cache_strides,
        cache_strides, nullptr) == cudaErrorInvalidValue,
        "packed QKV accepted null packed input");
}

std::vector<float> gqa_reference(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::size_t capacity,
    std::size_t valid_length,
    float scale) {
    std::vector<float> output(kQueryHeads * kHeadDim);
    std::vector<double> scores(valid_length);
    for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
        const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t position = 0; position < valid_length; ++position) {
            double dot = 0.0;
            for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[query_head * kHeadDim + dimension]) *
                    key[(kv_head * capacity + position) * kHeadDim + dimension];
            }
            scores[position] = dot * scale;
            maximum = std::max(maximum, scores[position]);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
            double result = 0.0;
            for (std::size_t position = 0; position < valid_length; ++position) {
                result += scores[position] * value[
                    (kv_head * capacity + position) * kHeadDim + dimension];
            }
            output[query_head * kHeadDim + dimension] =
                static_cast<float>(result / denominator);
        }
    }
    return output;
}

void run_gqa_case(std::size_t capacity, std::size_t valid_length) {
    constexpr float scale = 0.125F;
    std::vector<float> query = deterministic_values(kQueryWidth, 0.5F, 0.0F, 23);
    std::vector<float> key = deterministic_values(
        kKvHeads * capacity * kHeadDim, 0.5F, 0.0F, 29);
    std::vector<float> value = deterministic_values(
        kKvHeads * capacity * kHeadDim, 1.0F, 0.0F, 31);
    const std::vector<float> expected =
        gqa_reference(query, key, value, capacity, valid_length, scale);
    const std::size_t chunks = (capacity + 127) / 128;
    DeviceBuffer<float> device_query(query.size());
    DeviceBuffer<float> device_key(key.size());
    DeviceBuffer<float> device_value(value.size());
    DeviceBuffer<float> device_output(kQueryWidth);
    DeviceBuffer<float> device_workspace(
        kQueryHeads * chunks * (kHeadDim + 2));
    DeviceBuffer<std::int64_t> device_length(1);
    Stream stream;
    copy_to_device(device_query, query, stream.get());
    copy_to_device(device_key, key, stream.get());
    copy_to_device(device_value, value, stream.get());
    copy_to_device(device_length,
                   std::vector<std::int64_t>{static_cast<std::int64_t>(valid_length)},
                   stream.get());
    const std::array<std::int64_t, 4> query_strides{
        static_cast<std::int64_t>(kQueryWidth),
        static_cast<std::int64_t>(kHeadDim),
        static_cast<std::int64_t>(kHeadDim), 1};
    const std::array<std::int64_t, 4> cache_strides{
        static_cast<std::int64_t>(kKvHeads * capacity * kHeadDim),
        static_cast<std::int64_t>(capacity * kHeadDim),
        static_cast<std::int64_t>(kHeadDim), 1};
    check_cuda(flux::gqa_decode_attention_cuda_fp32(
        device_query.get(), device_key.get(), device_value.get(), nullptr,
        device_length.get(), device_output.get(), device_workspace.get(), scale,
        1, kQueryHeads, kKvHeads, capacity, kHeadDim, chunks,
        query_strides.data(), cache_strides.data(), cache_strides.data(),
        nullptr, nullptr, stream.get()), "gqa_decode_attention_cuda_fp32");
    const std::vector<float> actual = copy_to_host(device_output, stream.get());
    expect_close(actual, expected, 3.0e-4F, 3.0e-5F,
                 "GQA capacity " + std::to_string(capacity));
}

void test_gqa_short_and_long_context() {
    run_gqa_case(257, 193);
    run_gqa_case(8192, 8192);
    float* pointer = reinterpret_cast<float*>(1);
    const std::array<std::int64_t, 4> strides{576, 64, 64, 1};
    expect(flux::gqa_decode_attention_cuda_fp32(
        nullptr, pointer, pointer, nullptr, nullptr, pointer, pointer, 1.0F,
        1, 9, 3, 128, 64, 1, strides.data(), strides.data(), strides.data(),
        nullptr, nullptr, nullptr) == cudaErrorInvalidValue,
        "GQA accepted a null query");
}

void test_packed_gate_up_swiglu() {
    constexpr std::size_t input_width = 576;
    constexpr std::size_t intermediate_width = 1536;
    std::vector<float> input = deterministic_values(input_width, 0.25F, 0.0F, 41);
    std::vector<float> weights = deterministic_values(
        2 * intermediate_width * input_width, 0.05F, 0.0F, 43);
    std::vector<float> expected(intermediate_width);
    for (std::size_t row = 0; row < intermediate_width; ++row) {
        double gate = 0.0;
        double up = 0.0;
        for (std::size_t column = 0; column < input_width; ++column) {
            gate += static_cast<double>(input[column]) *
                weights[row * input_width + column];
            up += static_cast<double>(input[column]) *
                weights[(row + intermediate_width) * input_width + column];
        }
        const float gate_fp32 = static_cast<float>(gate);
        const float up_fp32 = static_cast<float>(up);
        expected[row] =
            (gate_fp32 / (1.0F + std::exp(-gate_fp32))) * up_fp32;
    }
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_weights(weights.size());
    DeviceBuffer<float> device_output(expected.size());
    Stream stream;
    copy_to_device(device_input, input, stream.get());
    copy_to_device(device_weights, weights, stream.get());
    check_cuda(flux::packed_gate_up_swiglu_cuda_fp32(
        device_input.get(), device_weights.get(), device_output.get(),
        stream.get()), "packed_gate_up_swiglu_cuda_fp32");
    expect_close(copy_to_host(device_output, stream.get()), expected,
                 5.0e-4F, 2.0e-5F, "packed gate/up SwiGLU");
    expect(flux::packed_gate_up_swiglu_cuda_fp32(
        nullptr, device_weights.get(), device_output.get(), nullptr) ==
        cudaErrorInvalidValue, "packed gate/up accepted null input");
}

}  // namespace

int main() {
    return flux::test::run("Flux decode CUDA operators", [] {
        test_packed_qkv_rope_cache_and_graph();
        test_gqa_short_and_long_context();
        test_packed_gate_up_swiglu();
    });
}
