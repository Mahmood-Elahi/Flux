#include "cuda_benchmark_utils.cuh"

#include "attention_score_softmax_cuda.h"
#include "gqa_decode_attention_cuda.h"
#include "packed_gate_up_gemv_cuda.h"
#include "packed_qkv_rope_cache_cuda.h"
#include "packed_swiglu_cuda.h"
#include "residual_rmsnorm_cuda.h"
#include "rmsnorm_cuda.h"
#include "rope_cuda.h"
#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using flux::benchmark::Options;
using flux::test::DeviceBuffer;
using flux::test::Stream;
using flux::test::check_cuda;
using flux::test::copy_to_device;
using flux::test::copy_to_host;
using flux::test::deterministic_values;
using flux::test::expect;
using flux::test::expect_close;

template <typename Operation>
void time_case(
    const Options& options,
    const std::string& name,
    cudaStream_t stream,
    Operation&& operation) {
    if (!flux::benchmark::selected(options, name)) {
        return;
    }
    flux::benchmark::report(
        name,
        flux::benchmark::median_cuda_ms(options, stream, operation));
}

void benchmark_rmsnorm(const Options& options, Stream& stream) {
    constexpr std::size_t rows = 14;
    constexpr std::size_t width = 576;
    constexpr float epsilon = 1.0e-5F;
    std::vector<float> input = deterministic_values(rows * width, 2.0F);
    std::vector<float> weight = deterministic_values(width, 0.4F, 1.0F, 2);
    std::vector<float> expected(input.size());
    for (std::size_t row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            const float value = input[row * width + column];
            squares += static_cast<double>(value) * value;
        }
        const double inverse = 1.0 / std::sqrt(squares / width + epsilon);
        for (std::size_t column = 0; column < width; ++column) {
            expected[row * width + column] = static_cast<float>(
                input[row * width + column] * inverse * weight[column]);
        }
    }
    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<float> d_weight(weight.size());
    DeviceBuffer<float> d_output(input.size());
    copy_to_device(d_input, input, stream.get());
    copy_to_device(d_weight, weight, stream.get());
    const auto operation = [&] {
        check_cuda(flux::rmsnorm_cuda_fp32(
            d_input.get(), d_weight.get(), d_output.get(), rows, width,
            epsilon, stream.get()), "benchmark RMSNorm launch");
    };
    operation();
    expect_close(copy_to_host(d_output, stream.get()), expected,
                 1.0e-5F, 2.0e-6F, "benchmark RMSNorm correctness");
    time_case(options, "rmsnorm_14x576", stream.get(), operation);
}

void benchmark_residual_rmsnorm(const Options& options, Stream& stream) {
    constexpr std::size_t rows = 14;
    constexpr std::size_t width = 576;
    constexpr float epsilon = 1.0e-5F;
    std::vector<float> hidden = deterministic_values(rows * width, 1.0F, 0.0F, 3);
    std::vector<float> residual = deterministic_values(rows * width, 1.0F, 0.0F, 4);
    std::vector<float> weight = deterministic_values(width, 0.3F, 1.0F, 5);
    std::vector<float> expected_norm(hidden.size());
    std::vector<float> expected_residual(hidden.size());
    for (std::size_t row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            const std::size_t index = row * width + column;
            expected_residual[index] = hidden[index] + residual[index];
            squares += static_cast<double>(expected_residual[index]) *
                expected_residual[index];
        }
        const double inverse = 1.0 / std::sqrt(squares / width + epsilon);
        for (std::size_t column = 0; column < width; ++column) {
            const std::size_t index = row * width + column;
            expected_norm[index] = static_cast<float>(
                expected_residual[index] * inverse * weight[column]);
        }
    }
    DeviceBuffer<float> d_hidden(hidden.size());
    DeviceBuffer<float> d_residual(residual.size());
    DeviceBuffer<float> d_weight(weight.size());
    DeviceBuffer<float> d_norm(hidden.size());
    DeviceBuffer<float> d_residual_out(hidden.size());
    copy_to_device(d_hidden, hidden, stream.get());
    copy_to_device(d_residual, residual, stream.get());
    copy_to_device(d_weight, weight, stream.get());
    const auto operation = [&] {
        check_cuda(flux::residual_rmsnorm_cuda_fp32(
            d_hidden.get(), d_residual.get(), d_weight.get(), d_norm.get(),
            d_residual_out.get(), rows, width, epsilon, stream.get()),
            "benchmark residual RMSNorm launch");
    };
    operation();
    expect_close(copy_to_host(d_norm, stream.get()), expected_norm,
                 1.0e-5F, 2.0e-6F, "benchmark residual RMSNorm correctness");
    expect_close(copy_to_host(d_residual_out, stream.get()), expected_residual,
                 0.0F, 0.0F, "benchmark residual output correctness");
    time_case(options, "residual_rmsnorm_14x576", stream.get(), operation);
}

std::vector<float> row_softmax(
    const std::vector<float>& input, std::size_t rows, std::size_t width) {
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * width;
        const float maximum = *std::max_element(
            input.begin() + static_cast<std::ptrdiff_t>(offset),
            input.begin() + static_cast<std::ptrdiff_t>(offset + width));
        double sum = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            output[offset + column] = std::exp(input[offset + column] - maximum);
            sum += output[offset + column];
        }
        for (std::size_t column = 0; column < width; ++column) {
            output[offset + column] = static_cast<float>(
                static_cast<double>(output[offset + column]) / sum);
        }
    }
    return output;
}

void benchmark_softmax(const Options& options, Stream& stream) {
    constexpr std::size_t rows = 27;
    constexpr std::size_t width = 8192;
    std::vector<float> input = deterministic_values(rows * width, 5.0F);
    const std::vector<float> expected = row_softmax(input, rows, width);
    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<float> d_output(input.size());
    copy_to_device(d_input, input, stream.get());
    const auto operation = [&] {
        check_cuda(flux::softmax_cuda_fp32(
            d_input.get(), d_output.get(), rows, width, stream.get()),
            "benchmark softmax launch");
    };
    operation();
    expect_close(copy_to_host(d_output, stream.get()), expected,
                 2.0e-5F, 2.0e-6F, "benchmark softmax correctness");
    time_case(options, "softmax_27x8192", stream.get(), operation);
}

void benchmark_attention_softmax(const Options& options, Stream& stream) {
    constexpr std::size_t heads = 9;
    constexpr std::size_t keys = 8192;
    constexpr float scale = 0.125F;
    std::vector<float> scores = deterministic_values(heads * keys, 4.0F);
    std::vector<float> mask(keys, 0.0F);
    std::vector<float> transformed(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index) {
        transformed[index] = scores[index] * scale + mask[index % keys];
    }
    const std::vector<float> expected = row_softmax(transformed, heads, keys);
    DeviceBuffer<float> d_scores(scores.size());
    DeviceBuffer<float> d_mask(mask.size());
    DeviceBuffer<float> d_output(scores.size());
    copy_to_device(d_scores, scores, stream.get());
    copy_to_device(d_mask, mask, stream.get());
    const auto operation = [&] {
        check_cuda(flux::attention_score_softmax_cuda_fp32(
            d_scores.get(), d_mask.get(), d_output.get(), scale,
            1, heads, 1, keys, 1, 1, 1, keys, keys, keys, keys,
            stream.get()), "benchmark attention softmax launch");
    };
    operation();
    expect_close(copy_to_host(d_output, stream.get()), expected,
                 2.0e-5F, 2.0e-6F,
                 "benchmark attention softmax correctness");
    time_case(options, "attention_softmax_1x9x1x8192", stream.get(), operation);
}

void benchmark_rope(const Options& options, Stream& stream) {
    constexpr std::size_t sequence = 512;
    constexpr std::size_t query_heads = 9;
    constexpr std::size_t key_heads = 3;
    constexpr std::size_t dimension = 64;
    std::vector<float> query = deterministic_values(
        query_heads * sequence * dimension, 2.0F, 0.0F, 7);
    std::vector<float> key = deterministic_values(
        key_heads * sequence * dimension, 2.0F, 0.0F, 8);
    std::vector<float> cosine(sequence * dimension);
    std::vector<float> sine(sequence * dimension);
    for (std::size_t index = 0; index < cosine.size(); ++index) {
        cosine[index] = std::cos(static_cast<float>(index) * 0.0001F);
        sine[index] = std::sin(static_cast<float>(index) * 0.0001F);
    }
    DeviceBuffer<float> d_query(query.size());
    DeviceBuffer<float> d_key(key.size());
    DeviceBuffer<float> d_cosine(cosine.size());
    DeviceBuffer<float> d_sine(sine.size());
    DeviceBuffer<float> d_query_output(query.size());
    DeviceBuffer<float> d_key_output(key.size());
    copy_to_device(d_query, query, stream.get());
    copy_to_device(d_key, key, stream.get());
    copy_to_device(d_cosine, cosine, stream.get());
    copy_to_device(d_sine, sine, stream.get());
    const flux::RopeStrides query_strides{
        static_cast<std::ptrdiff_t>(query.size()),
        static_cast<std::ptrdiff_t>(sequence * dimension),
        static_cast<std::ptrdiff_t>(dimension), 1};
    const flux::RopeStrides key_strides{
        static_cast<std::ptrdiff_t>(key.size()),
        static_cast<std::ptrdiff_t>(sequence * dimension),
        static_cast<std::ptrdiff_t>(dimension), 1};
    const flux::RopeEmbeddingStrides embedding_strides{
        static_cast<std::ptrdiff_t>(cosine.size()),
        static_cast<std::ptrdiff_t>(dimension), 1};
    const auto operation = [&] {
        check_cuda(flux::rope_cuda_fp32(
            d_query.get(), d_key.get(), d_cosine.get(), d_sine.get(),
            d_query_output.get(), d_key_output.get(), 1, query_heads,
            key_heads, sequence, dimension, 1, query_strides, key_strides,
            embedding_strides, embedding_strides, stream.get()),
            "benchmark RoPE launch");
    };
    operation();
    const std::vector<float> actual = copy_to_host(d_query_output, stream.get());
    const std::size_t paired = dimension / 2;
    const float expected_first = query[0] * cosine[0] - query[paired] * sine[0];
    expect(std::abs(actual[0] - expected_first) <= 2.0e-7F,
           "benchmark RoPE correctness check failed");
    time_case(options, "rope_q9_kv3_s512_d64", stream.get(), operation);
}

void benchmark_packed_swiglu(const Options& options, Stream& stream) {
    constexpr std::size_t rows = 512;
    constexpr std::size_t width = 1536;
    std::vector<float> packed = deterministic_values(rows * width * 2, 4.0F);
    DeviceBuffer<float> d_packed(packed.size());
    DeviceBuffer<float> d_output(rows * width);
    copy_to_device(d_packed, packed, stream.get());
    const auto operation = [&] {
        check_cuda(flux::packed_swiglu_cuda_fp32(
            d_packed.get(), d_output.get(), rows, width, stream.get()),
            "benchmark packed SwiGLU launch");
    };
    operation();
    const std::vector<float> actual = copy_to_host(d_output, stream.get());
    const float expected = (packed[0] / (1.0F + std::exp(-packed[0]))) *
        packed[width];
    expect(std::abs(actual[0] - expected) <= 2.0e-6F,
           "benchmark packed SwiGLU correctness check failed");
    time_case(options, "packed_swiglu_512x1536", stream.get(), operation);
}

void benchmark_packed_qkv(const Options& options, Stream& stream) {
    constexpr std::size_t query_heads = 9;
    constexpr std::size_t kv_heads = 3;
    constexpr std::size_t dimension = 64;
    constexpr std::size_t query_width = query_heads * dimension;
    constexpr std::size_t kv_width = kv_heads * dimension;
    constexpr std::size_t packed_width = query_width + 2 * kv_width;
    constexpr std::size_t capacity = 8192;
    std::vector<float> packed = deterministic_values(packed_width, 2.0F);
    std::vector<float> cosine(dimension, 1.0F);
    std::vector<float> sine(dimension, 0.0F);
    DeviceBuffer<float> d_packed(packed.size());
    DeviceBuffer<float> d_cosine(cosine.size());
    DeviceBuffer<float> d_sine(sine.size());
    DeviceBuffer<float> d_key(kv_heads * capacity * dimension);
    DeviceBuffer<float> d_value(kv_heads * capacity * dimension);
    DeviceBuffer<float> d_query(query_width);
    DeviceBuffer<std::int64_t> d_position(1);
    copy_to_device(d_packed, packed, stream.get());
    copy_to_device(d_cosine, cosine, stream.get());
    copy_to_device(d_sine, sine, stream.get());
    copy_to_device(d_position, std::vector<std::int64_t>{4096}, stream.get());
    const flux::PackedQKVStrides packed_strides{
        static_cast<std::ptrdiff_t>(packed_width),
        static_cast<std::ptrdiff_t>(packed_width), 1};
    const flux::PackedQKVEmbeddingStrides embedding_strides{
        static_cast<std::ptrdiff_t>(dimension),
        static_cast<std::ptrdiff_t>(dimension), 1};
    const flux::PackedQKVCacheStrides cache_strides{
        static_cast<std::ptrdiff_t>(kv_heads * capacity * dimension),
        static_cast<std::ptrdiff_t>(capacity * dimension),
        static_cast<std::ptrdiff_t>(dimension), 1};
    const auto operation = [&] {
        check_cuda(flux::packed_qkv_rope_cache_at_position_cuda_fp32(
            d_packed.get(), d_cosine.get(), d_sine.get(), d_key.get(),
            d_value.get(), d_position.get(), d_query.get(), capacity,
            packed_strides, embedding_strides, embedding_strides,
            cache_strides, cache_strides, stream.get()),
            "benchmark packed QKV launch");
    };
    operation();
    expect_close(copy_to_host(d_query, stream.get()),
                 std::vector<float>(packed.begin(), packed.begin() + query_width),
                 0.0F, 0.0F, "benchmark packed QKV correctness");
    time_case(options, "packed_qkv_rope_cache_token", stream.get(), operation);
}

void benchmark_gqa(const Options& options, Stream& stream) {
    constexpr std::size_t query_heads = 9;
    constexpr std::size_t kv_heads = 3;
    constexpr std::size_t dimension = 64;
    constexpr std::size_t capacity = 8192;
    constexpr std::size_t chunks = capacity / 128;
    std::vector<float> query(query_heads * dimension, 0.0F);
    std::vector<float> key = deterministic_values(
        kv_heads * capacity * dimension, 0.5F, 0.0F, 13);
    std::vector<float> value = deterministic_values(
        kv_heads * capacity * dimension, 1.0F, 0.0F, 14);
    DeviceBuffer<float> d_query(query.size());
    DeviceBuffer<float> d_key(key.size());
    DeviceBuffer<float> d_value(value.size());
    DeviceBuffer<float> d_output(query.size());
    DeviceBuffer<float> d_workspace(query_heads * chunks * (dimension + 2));
    DeviceBuffer<std::int64_t> d_length(1);
    copy_to_device(d_query, query, stream.get());
    copy_to_device(d_key, key, stream.get());
    copy_to_device(d_value, value, stream.get());
    copy_to_device(d_length, std::vector<std::int64_t>{capacity}, stream.get());
    const std::array<std::int64_t, 4> query_strides{576, 64, 64, 1};
    const std::array<std::int64_t, 4> cache_strides{
        static_cast<std::int64_t>(kv_heads * capacity * dimension),
        static_cast<std::int64_t>(capacity * dimension), 64, 1};
    const auto operation = [&] {
        check_cuda(flux::gqa_decode_attention_cuda_fp32(
            d_query.get(), d_key.get(), d_value.get(), nullptr,
            d_length.get(), d_output.get(), d_workspace.get(), 0.125F,
            1, query_heads, kv_heads, capacity, dimension, chunks,
            query_strides.data(), cache_strides.data(), cache_strides.data(),
            nullptr, nullptr, stream.get()), "benchmark GQA launch");
    };
    operation();
    const std::vector<float> actual = copy_to_host(d_output, stream.get());
    for (std::size_t head = 0; head < query_heads; ++head) {
        const std::size_t kv_head = head / 3;
        double sum = 0.0;
        for (std::size_t position = 0; position < capacity; ++position) {
            sum += value[(kv_head * capacity + position) * dimension];
        }
        const float expected = static_cast<float>(sum / capacity);
        expect(std::abs(actual[head * dimension] - expected) <= 3.0e-5F,
               "benchmark GQA correctness check failed");
    }
    time_case(options, "gqa_decode_b1_h9_kv3_s8192", stream.get(), operation);
}

void benchmark_gate_up(const Options& options, Stream& stream) {
    constexpr std::size_t input_width = 576;
    constexpr std::size_t output_width = 1536;
    std::vector<float> input = deterministic_values(input_width, 0.25F, 0.0F, 17);
    std::vector<float> weight = deterministic_values(
        2 * output_width * input_width, 0.05F, 0.0F, 18);
    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<float> d_weight(weight.size());
    DeviceBuffer<float> d_output(output_width);
    copy_to_device(d_input, input, stream.get());
    copy_to_device(d_weight, weight, stream.get());
    const auto operation = [&] {
        check_cuda(flux::packed_gate_up_swiglu_cuda_fp32(
            d_input.get(), d_weight.get(), d_output.get(), stream.get()),
            "benchmark gate/up launch");
    };
    operation();
    const std::vector<float> actual = copy_to_host(d_output, stream.get());
    double gate = 0.0;
    double up = 0.0;
    for (std::size_t column = 0; column < input_width; ++column) {
        gate += static_cast<double>(input[column]) * weight[column];
        up += static_cast<double>(input[column]) *
            weight[output_width * input_width + column];
    }
    const float gate_value = static_cast<float>(gate);
    const float expected =
        (gate_value / (1.0F + std::exp(-gate_value))) * static_cast<float>(up);
    expect(std::abs(actual[0] - expected) <=
               2.0e-5F + 5.0e-4F * std::abs(expected),
           "benchmark gate/up correctness check failed");
    time_case(options, "packed_gate_up_swiglu_token", stream.get(), operation);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = flux::benchmark::parse_options(argc, argv);
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        expect(device_count > 0, "no CUDA device is available");
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0),
                   "cudaGetDeviceProperties");
        std::cout << "Flux native CUDA microbenchmarks on " << properties.name
                  << " (sm_" << properties.major << properties.minor << ")\n"
                  << "warmup=" << options.warmup
                  << ", samples=" << options.samples << "\n";
        Stream stream;
        benchmark_rmsnorm(options, stream);
        benchmark_residual_rmsnorm(options, stream);
        benchmark_softmax(options, stream);
        benchmark_attention_softmax(options, stream);
        benchmark_rope(options, stream);
        benchmark_packed_swiglu(options, stream);
        benchmark_packed_qkv(options, stream);
        benchmark_gqa(options, stream);
        benchmark_gate_up(options, stream);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Flux CUDA microbenchmark failed: " << error.what() << '\n';
        return 1;
    }
}
