#include "cuda_test_utils.cuh"
#include "streaming_prefill_gqa_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
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
constexpr std::size_t kKeyValueHeads = 3;
constexpr std::size_t kHeadDimension = 64;
constexpr float kScale = 0.125F;

void check_cublas(const cublasStatus_t status, const char* operation) {
    expect(status == CUBLAS_STATUS_SUCCESS,
           std::string(operation) + " failed with cuBLAS status " +
               std::to_string(static_cast<int>(status)));
}

__device__ float reference_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ float reference_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ float reference_block_max(float value, float* reductions) {
    value = reference_warp_max(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        reductions[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32
        ? reductions[lane] : -FLT_MAX;
    if (warp == 0) {
        value = reference_warp_max(value);
        if (lane == 0) {
            reductions[0] = value;
        }
    }
    __syncthreads();
    return reductions[0];
}

__device__ float reference_block_sum(float value, float* reductions) {
    value = reference_warp_sum(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        reductions[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0F;
    if (warp == 0) {
        value = reference_warp_sum(value);
        if (lane == 0) {
            reductions[0] = value;
        }
    }
    __syncthreads();
    return reductions[0];
}

__global__ void retained_causal_softmax_kernel(
    float* scores, const float scale, const int sequence) {
    __shared__ float reductions[16];
    const int row = blockIdx.x;
    const int query_position = row % sequence;
    float* values = scores + static_cast<std::size_t>(row) * sequence;
    float local_maximum = -FLT_MAX;
    for (int key = threadIdx.x; key <= query_position; key += blockDim.x) {
        values[key] *= scale;
        local_maximum = fmaxf(local_maximum, values[key]);
    }
    for (int key = query_position + 1 + threadIdx.x;
         key < sequence; key += blockDim.x) {
        values[key] = 0.0F;
    }
    const float maximum = reference_block_max(local_maximum, reductions);
    float local_sum = 0.0F;
    for (int key = threadIdx.x; key <= query_position; key += blockDim.x) {
        values[key] = expf(values[key] - maximum);
        local_sum += values[key];
    }
    const float sum = reference_block_sum(local_sum, reductions);
    for (int key = threadIdx.x; key <= query_position; key += blockDim.x) {
        values[key] /= sum;
    }
}

void retained_attention(
    cublasHandle_t handle,
    const float* query,
    const float* key,
    const float* value,
    float* scores,
    float* output,
    const std::size_t sequence,
    cudaStream_t stream) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const long long query_stride = sequence * kHeadDimension;
    const long long score_stride = sequence * sequence;
    for (std::size_t kv_head = 0; kv_head < kKeyValueHeads; ++kv_head) {
        check_cublas(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(sequence), static_cast<int>(sequence),
            static_cast<int>(kHeadDimension), &alpha,
            key + kv_head * sequence * kHeadDimension,
            static_cast<int>(kHeadDimension), 0,
            query + kv_head * 3 * query_stride,
            static_cast<int>(kHeadDimension), query_stride, &beta,
            scores + kv_head * 3 * score_stride,
            static_cast<int>(sequence), score_stride, 3),
            "retained QK GEMM");
    }
    const int threads = sequence == 4096
        ? 512 : (sequence == 512 || sequence == 1024 ? 128 : 256);
    retained_causal_softmax_kernel<<<
        static_cast<unsigned int>(kQueryHeads * sequence), threads, 0, stream>>>(
        scores, kScale, static_cast<int>(sequence));
    check_cuda(cudaGetLastError(), "retained causal softmax launch");
    for (std::size_t kv_head = 0; kv_head < kKeyValueHeads; ++kv_head) {
        check_cublas(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(kHeadDimension), static_cast<int>(sequence),
            static_cast<int>(sequence), &alpha,
            value + kv_head * sequence * kHeadDimension,
            static_cast<int>(kHeadDimension), 0,
            scores + kv_head * 3 * score_stride,
            static_cast<int>(sequence), score_stride, &beta,
            output + kv_head * 3 * query_stride,
            static_cast<int>(kHeadDimension), query_stride, 3),
            "retained PV GEMM");
    }
}

template <typename Operation>
float median_cuda_ms(cudaStream_t stream, Operation&& operation) {
    for (int index = 0; index < 3; ++index) {
        operation();
    }
    check_cuda(cudaStreamSynchronize(stream), "timing warmup synchronization");
    std::array<float, 11> samples{};
    for (float& sample : samples) {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        check_cuda(cudaEventCreate(&start), "creating timing start event");
        check_cuda(cudaEventCreate(&end), "creating timing end event");
        check_cuda(cudaEventRecord(start, stream), "recording timing start");
        operation();
        check_cuda(cudaEventRecord(end, stream), "recording timing end");
        check_cuda(cudaEventSynchronize(end), "timing event synchronization");
        check_cuda(cudaEventElapsedTime(&sample, start, end),
                   "reading elapsed time");
        check_cuda(cudaEventDestroy(start), "destroying timing start event");
        check_cuda(cudaEventDestroy(end), "destroying timing end event");
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::vector<float> reference_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::size_t sequence,
    const std::size_t capacity) {
    std::vector<float> output(kQueryHeads * sequence * kHeadDimension);
    std::vector<double> scores(sequence);
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        const std::size_t kv_head = head / 3;
        for (std::size_t row = 0; row < sequence; ++row) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t column = 0; column <= row; ++column) {
                double dot = 0.0;
                for (std::size_t dimension = 0;
                     dimension < kHeadDimension; ++dimension) {
                    dot += static_cast<double>(
                        query[(head * sequence + row) * kHeadDimension + dimension]) *
                        key[(kv_head * capacity + column) * kHeadDimension + dimension];
                }
                scores[column] = dot * kScale;
                maximum = std::max(maximum, scores[column]);
            }
            double denominator = 0.0;
            for (std::size_t column = 0; column <= row; ++column) {
                scores[column] = std::exp(scores[column] - maximum);
                denominator += scores[column];
            }
            for (std::size_t dimension = 0;
                 dimension < kHeadDimension; ++dimension) {
                double sum = 0.0;
                for (std::size_t column = 0; column <= row; ++column) {
                    sum += scores[column] *
                        value[(kv_head * capacity + column) *
                            kHeadDimension + dimension];
                }
                output[(head * sequence + row) * kHeadDimension + dimension] =
                    static_cast<float>(sum / denominator);
            }
        }
    }
    return output;
}

std::vector<float> uniform_score_reference(
    const std::vector<float>& value,
    const std::size_t sequence,
    const std::size_t capacity) {
    std::vector<float> output(kQueryHeads * sequence * kHeadDimension);
    std::vector<double> prefix(kKeyValueHeads * kHeadDimension, 0.0);
    for (std::size_t row = 0; row < sequence; ++row) {
        for (std::size_t kv_head = 0; kv_head < kKeyValueHeads; ++kv_head) {
            for (std::size_t dimension = 0;
                 dimension < kHeadDimension; ++dimension) {
                prefix[kv_head * kHeadDimension + dimension] +=
                    value[(kv_head * capacity + row) *
                        kHeadDimension + dimension];
                const float average = static_cast<float>(
                    prefix[kv_head * kHeadDimension + dimension] / (row + 1));
                for (std::size_t group_head = 0; group_head < 3; ++group_head) {
                    const std::size_t head = kv_head * 3 + group_head;
                    output[(head * sequence + row) *
                        kHeadDimension + dimension] = average;
                }
            }
        }
    }
    return output;
}

void run_case(
    const std::size_t sequence,
    const std::size_t capacity,
    const flux::StreamingPrefillGQAVariant variant,
    const bool arbitrary_scores,
    Stream& stream) {
    std::vector<float> query = arbitrary_scores
        ? deterministic_values(kQueryHeads * sequence * kHeadDimension,
                               0.35F, 0.0F,
                               static_cast<std::uint32_t>(100 + sequence))
        : std::vector<float>(kQueryHeads * sequence * kHeadDimension, 0.0F);
    std::vector<float> key = deterministic_values(
        kKeyValueHeads * capacity * kHeadDimension, 0.4F, 0.0F,
        static_cast<std::uint32_t>(200 + sequence));
    std::vector<float> value = deterministic_values(
        kKeyValueHeads * capacity * kHeadDimension, 0.7F, 0.0F,
        static_cast<std::uint32_t>(300 + sequence));
    const std::vector<float> expected = arbitrary_scores
        ? reference_attention(query, key, value, sequence, capacity)
        : uniform_score_reference(value, sequence, capacity);

    DeviceBuffer<float> d_query(query.size());
    DeviceBuffer<float> d_key(key.size());
    DeviceBuffer<float> d_value(value.size());
    DeviceBuffer<float> d_output(expected.size());
    copy_to_device(d_query, query, stream.get());
    copy_to_device(d_key, key, stream.get());
    copy_to_device(d_value, value, stream.get());
    check_cuda(flux::streaming_prefill_gqa_cuda_fp32(
        d_query.get(), d_key.get(), d_value.get(), d_output.get(), kScale,
        sequence, capacity, variant, stream.get()),
        "streaming prefill GQA launch");
    const std::vector<float> actual = copy_to_host(d_output, stream.get());
    expect_close(actual, expected, 2.0e-4F, 2.0e-5F,
                 "streaming prefill GQA length " + std::to_string(sequence));
    std::cout << "  length " << sequence << " max abs error "
              << flux::test::maximum_absolute_error(actual, expected) << '\n';
}

void test_small_and_boundary_lengths(Stream& stream) {
    constexpr std::array<std::size_t, 10> lengths{
        1, 2, 7, 8, 9, 15, 16, 17, 32, 33};
    for (const std::size_t length : lengths) {
        run_case(length, length + 5,
                 flux::select_streaming_prefill_gqa_variant(length), true, stream);
    }
    for (const auto variant : {
             flux::StreamingPrefillGQAVariant::kQueryTile8,
             flux::StreamingPrefillGQAVariant::kQueryTile32,
             flux::StreamingPrefillGQAVariant::kQueryTile128}) {
        run_case(65, 79, variant, true, stream);
    }
}

void test_production_lengths(Stream& stream) {
    constexpr std::array<std::size_t, 6> lengths{
        128, 512, 1024, 2048, 4096, 8192};
    for (const std::size_t length : lengths) {
        run_case(length, length,
                 flux::select_streaming_prefill_gqa_variant(length), false,
                 stream);
    }
}

void test_repeated_deterministic_and_allocation_free(Stream& stream) {
    constexpr std::size_t sequence = 129;
    std::vector<float> query = deterministic_values(
        kQueryHeads * sequence * kHeadDimension, 0.25F, 0.0F, 901);
    std::vector<float> key = deterministic_values(
        kKeyValueHeads * sequence * kHeadDimension, 0.25F, 0.0F, 902);
    std::vector<float> value = deterministic_values(
        kKeyValueHeads * sequence * kHeadDimension, 0.5F, 0.0F, 903);
    DeviceBuffer<float> d_query(query.size());
    DeviceBuffer<float> d_key(key.size());
    DeviceBuffer<float> d_value(value.size());
    DeviceBuffer<float> d_output_a(query.size());
    DeviceBuffer<float> d_output_b(query.size());
    copy_to_device(d_query, query, stream.get());
    copy_to_device(d_key, key, stream.get());
    copy_to_device(d_value, value, stream.get());
    check_cuda(flux::streaming_prefill_gqa_cuda_fp32(
        d_query.get(), d_key.get(), d_value.get(), d_output_a.get(), kScale,
        sequence, sequence, flux::select_streaming_prefill_gqa_variant(sequence),
        stream.get()), "first repeated streaming launch");
    stream.synchronize();
    std::size_t free_before = 0;
    std::size_t total_before = 0;
    check_cuda(cudaMemGetInfo(&free_before, &total_before),
               "cudaMemGetInfo before repeated invocation");
    check_cuda(flux::streaming_prefill_gqa_cuda_fp32(
        d_query.get(), d_key.get(), d_value.get(), d_output_b.get(), kScale,
        sequence, sequence, flux::select_streaming_prefill_gqa_variant(sequence),
        stream.get()), "second repeated streaming launch");
    stream.synchronize();
    std::size_t free_after = 0;
    std::size_t total_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total_after),
               "cudaMemGetInfo after repeated invocation");
    expect(free_before == free_after && total_before == total_after,
           "streaming attention changed device allocation state");
    const std::vector<float> first = copy_to_host(d_output_a, stream.get());
    const std::vector<float> second = copy_to_host(d_output_b, stream.get());
    flux::test::expect_unchanged(second, first,
                                "streaming attention determinism");
}

void test_retained_path_comparison(Stream& stream) {
    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "creating retained-path handle");
    check_cublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH),
                 "setting retained-path math mode");
    check_cublas(cublasSetStream(handle, stream.get()),
                 "setting retained-path stream");
    constexpr std::array<std::size_t, 6> lengths{
        128, 512, 1024, 2048, 4096, 8192};
    for (const std::size_t sequence : lengths) {
        std::vector<float> query = deterministic_values(
            kQueryHeads * sequence * kHeadDimension, 0.35F, 0.0F,
            static_cast<std::uint32_t>(1200 + sequence));
        std::vector<float> key = deterministic_values(
            kKeyValueHeads * sequence * kHeadDimension, 0.35F, 0.0F,
            static_cast<std::uint32_t>(1300 + sequence));
        std::vector<float> value = deterministic_values(
            kKeyValueHeads * sequence * kHeadDimension, 0.7F, 0.0F,
            static_cast<std::uint32_t>(1400 + sequence));
        DeviceBuffer<float> d_query(query.size());
        DeviceBuffer<float> d_key(key.size());
        DeviceBuffer<float> d_value(value.size());
        DeviceBuffer<float> d_streaming(query.size());
        DeviceBuffer<float> d_retained(query.size());
        DeviceBuffer<float> d_scores(kQueryHeads * sequence * sequence);
        copy_to_device(d_query, query, stream.get());
        copy_to_device(d_key, key, stream.get());
        copy_to_device(d_value, value, stream.get());
        retained_attention(
            handle, d_query.get(), d_key.get(), d_value.get(), d_scores.get(),
            d_retained.get(), sequence, stream.get());
        check_cuda(flux::streaming_prefill_gqa_cuda_fp32(
            d_query.get(), d_key.get(), d_value.get(), d_streaming.get(),
            kScale, sequence, sequence,
            flux::select_streaming_prefill_gqa_variant(sequence), stream.get()),
            "retained-path comparison streaming launch");
        const std::vector<float> expected =
            copy_to_host(d_retained, stream.get());
        const std::vector<float> actual =
            copy_to_host(d_streaming, stream.get());
        expect_close(actual, expected, 2.0e-4F, 2.0e-5F,
                     "streaming versus retained length " +
                         std::to_string(sequence));
        const float retained_ms = median_cuda_ms(stream.get(), [&] {
            retained_attention(
                handle, d_query.get(), d_key.get(), d_value.get(),
                d_scores.get(), d_retained.get(), sequence, stream.get());
        });
        const float streaming_ms = median_cuda_ms(stream.get(), [&] {
            check_cuda(flux::streaming_prefill_gqa_cuda_fp32(
                d_query.get(), d_key.get(), d_value.get(), d_streaming.get(),
                kScale, sequence, sequence,
                flux::select_streaming_prefill_gqa_variant(sequence),
                stream.get()), "timed streaming launch");
        });
        std::cout << "  retained comparison length " << sequence
                  << " max abs error "
                  << flux::test::maximum_absolute_error(actual, expected)
                  << ", retained " << retained_ms << " ms, streaming "
                  << streaming_ms << " ms\n";
    }
    check_cublas(cublasDestroy(handle), "destroying retained-path handle");
}

void test_invalid_arguments(Stream& stream) {
    DeviceBuffer<float> buffer(kQueryHeads * kHeadDimension);
    expect(flux::streaming_prefill_gqa_cuda_fp32(
               nullptr, buffer.get(), buffer.get(), buffer.get(), kScale,
               1, 1, flux::StreamingPrefillGQAVariant::kQueryTile8,
               stream.get()) == cudaErrorInvalidValue,
           "null query was accepted");
    expect(flux::streaming_prefill_gqa_cuda_fp32(
               buffer.get(), buffer.get(), buffer.get(), buffer.get(), kScale,
               0, 1, flux::StreamingPrefillGQAVariant::kQueryTile8,
               stream.get()) == cudaErrorInvalidValue,
           "zero sequence length was accepted");
    expect(flux::streaming_prefill_gqa_cuda_fp32(
               buffer.get(), buffer.get(), buffer.get(), buffer.get(), kScale,
               2, 1, flux::StreamingPrefillGQAVariant::kQueryTile8,
               stream.get()) == cudaErrorInvalidValue,
           "sequence longer than capacity was accepted");
    expect(flux::streaming_prefill_gqa_cuda_fp32(
               buffer.get(), buffer.get(), buffer.get(), buffer.get(),
               std::numeric_limits<float>::infinity(), 1, 1,
               flux::StreamingPrefillGQAVariant::kQueryTile8,
               stream.get()) == cudaErrorInvalidValue,
           "non-finite scale was accepted");
    expect(flux::streaming_prefill_gqa_workspace_bytes(8192) == 0,
           "streaming attention unexpectedly requires workspace");
    expect(flux::streaming_prefill_gqa_cuda_fp32(
               buffer.get(), buffer.get(), buffer.get(), buffer.get(), kScale,
               1, 1, static_cast<flux::StreamingPrefillGQAVariant>(99),
               stream.get()) == cudaErrorInvalidValue,
           "unknown streaming attention variant was accepted");
    expect(flux::select_streaming_prefill_gqa_variant(128) ==
               flux::StreamingPrefillGQAVariant::kQueryTile8 &&
           flux::select_streaming_prefill_gqa_variant(129) ==
               flux::StreamingPrefillGQAVariant::kQueryTile32 &&
           flux::select_streaming_prefill_gqa_variant(384) ==
               flux::StreamingPrefillGQAVariant::kQueryTile32 &&
           flux::select_streaming_prefill_gqa_variant(385) ==
               flux::StreamingPrefillGQAVariant::kQueryTile128,
           "streaming attention dispatcher thresholds changed");
}

}  // namespace

int main() {
    return flux::test::run("streaming prefill GQA CUDA tests", [] {
        Stream stream;
        test_small_and_boundary_lengths(stream);
        test_production_lengths(stream);
        test_retained_path_comparison(stream);
        test_repeated_deterministic_and_allocation_free(stream);
        test_invalid_arguments(stream);
    });
}
