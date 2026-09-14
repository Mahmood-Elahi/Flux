#include "streaming_prefill_gqa_cuda.h"

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace flux {
namespace {

constexpr int kQueryHeads = 9;
constexpr int kQueryHeadsPerGroup = 3;
constexpr int kHeadDimension = 64;
constexpr int kKeyTile = 32;
constexpr std::size_t kMaximumSequenceLength = 8192;

template <int Width>
__device__ __forceinline__ float subgroup_sum(
    float value, const unsigned int mask) {
    if constexpr (Width == 1) {
        return value;
    } else {
#pragma unroll
        for (int offset = Width / 2; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(mask, value, offset, Width);
        }
        return __shfl_sync(mask, value, 0, Width);
    }
}

template <int Width>
__device__ __forceinline__ float subgroup_broadcast(
    const float value, const unsigned int mask) {
    if constexpr (Width == 1) {
        return value;
    }
    return __shfl_sync(mask, value, 0, Width);
}

// A subgroup owns one query row. K/V are staged once per CTA from compact
// three-head cache storage; each subgroup then updates stable (m, l, o) state
// without writing scores or probabilities to global memory.
template <int SubgroupWidth, int BlockThreads>
__global__ void streaming_prefill_gqa_fp32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    const float scale,
    const int sequence_length,
    const int capacity) {
    constexpr int block_warps = BlockThreads / 32;
    constexpr int rows_per_warp = 32 / SubgroupWidth;
    constexpr int query_tile = block_warps * rows_per_warp;
    constexpr int values_per_lane = kHeadDimension / SubgroupWidth;
    constexpr int vectors_per_tile = kKeyTile * kHeadDimension / 4;

    __shared__ __align__(16) float key_tile[kKeyTile][kHeadDimension];
    __shared__ __align__(16) float value_tile[kKeyTile][kHeadDimension];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane / SubgroupWidth;
    const int subgroup_lane = lane % SubgroupWidth;
    const unsigned int subgroup_mask = SubgroupWidth == 32
        ? 0xffffffffU
        : ((1U << SubgroupWidth) - 1U) << (subgroup * SubgroupWidth);
    const int query_head = blockIdx.y;
    const int kv_head = query_head / kQueryHeadsPerGroup;
    const int first_query = blockIdx.x * query_tile;
    const int query_position = first_query + warp + subgroup * block_warps;

    float query_values[values_per_lane];
    float output_values[values_per_lane] = {};
    float running_maximum = -FLT_MAX;
    float running_sum = 0.0F;
#pragma unroll
    for (int item = 0; item < values_per_lane; ++item) {
        const int dimension = subgroup_lane + item * SubgroupWidth;
        query_values[item] = query_position < sequence_length
            ? query[(static_cast<std::size_t>(query_head) * sequence_length +
                     query_position) * kHeadDimension + dimension]
            : 0.0F;
    }

    const int last_query = min(first_query + query_tile, sequence_length) - 1;
    const int key_limit = last_query + 1;
    for (int first_key = 0; first_key < key_limit; first_key += kKeyTile) {
        const int valid_keys = min(kKeyTile, key_limit - first_key);
        const float4* key_vectors = reinterpret_cast<const float4*>(
            key + (static_cast<std::size_t>(kv_head) * capacity + first_key) *
                kHeadDimension);
        const float4* value_vectors = reinterpret_cast<const float4*>(
            value + (static_cast<std::size_t>(kv_head) * capacity + first_key) *
                kHeadDimension);
        float4* shared_key_vectors = reinterpret_cast<float4*>(key_tile);
        float4* shared_value_vectors = reinterpret_cast<float4*>(value_tile);
        for (int vector_index = threadIdx.x;
             vector_index < vectors_per_tile;
             vector_index += blockDim.x) {
            const int key_index = vector_index / (kHeadDimension / 4);
            if (key_index < valid_keys) {
                shared_key_vectors[vector_index] = key_vectors[vector_index];
                shared_value_vectors[vector_index] = value_vectors[vector_index];
            }
        }
        __syncthreads();

        const int causal_keys =
            query_position < sequence_length && first_key <= query_position
            ? min(valid_keys, query_position - first_key + 1) : 0;
#pragma unroll 1
        for (int key_index = 0; key_index < causal_keys; ++key_index) {
            float partial = 0.0F;
#pragma unroll
            for (int item = 0; item < values_per_lane; ++item) {
                const int dimension = subgroup_lane + item * SubgroupWidth;
                partial = fmaf(query_values[item],
                               key_tile[key_index][dimension], partial);
            }
            const float score =
                subgroup_sum<SubgroupWidth>(partial, subgroup_mask) * scale;

            float previous_weight = 0.0F;
            float current_weight = 0.0F;
            if (subgroup_lane == 0) {
                const float new_maximum = fmaxf(running_maximum, score);
                previous_weight = expf(running_maximum - new_maximum);
                current_weight = expf(score - new_maximum);
                running_sum = running_sum * previous_weight + current_weight;
                running_maximum = new_maximum;
            }
            previous_weight = subgroup_broadcast<SubgroupWidth>(
                previous_weight, subgroup_mask);
            current_weight = subgroup_broadcast<SubgroupWidth>(
                current_weight, subgroup_mask);
#pragma unroll
            for (int item = 0; item < values_per_lane; ++item) {
                const int dimension = subgroup_lane + item * SubgroupWidth;
                output_values[item] =
                    output_values[item] * previous_weight +
                    current_weight * value_tile[key_index][dimension];
            }
        }
        __syncthreads();
    }

    if (query_position < sequence_length) {
        const float normalization = subgroup_broadcast<SubgroupWidth>(
            running_sum, subgroup_mask);
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int dimension = subgroup_lane + item * SubgroupWidth;
            output[(static_cast<std::size_t>(query_head) * sequence_length +
                    query_position) * kHeadDimension + dimension] =
                output_values[item] / normalization;
        }
    }
}

template <int SubgroupWidth, int BlockThreads>
cudaError_t launch(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    const float scale,
    const std::size_t sequence_length,
    const std::size_t capacity,
    cudaStream_t stream) {
    constexpr int query_tile =
        (BlockThreads / 32) * (32 / SubgroupWidth);
    const dim3 grid(
        static_cast<unsigned int>(
            (sequence_length + query_tile - 1) / query_tile),
        kQueryHeads);
    streaming_prefill_gqa_fp32_kernel<SubgroupWidth, BlockThreads><<<
        grid, BlockThreads, 0, stream>>>(
        query, key, value, output, scale, static_cast<int>(sequence_length),
        static_cast<int>(capacity));
    return cudaGetLastError();
}

}  // namespace

StreamingPrefillGQAVariant select_streaming_prefill_gqa_variant(
    const std::size_t sequence_length) {
    if (sequence_length <= 128) {
        return StreamingPrefillGQAVariant::kQueryTile8;
    }
    if (sequence_length <= 384) {
        return StreamingPrefillGQAVariant::kQueryTile32;
    }
    return StreamingPrefillGQAVariant::kQueryTile128;
}

cudaError_t streaming_prefill_gqa_cuda_fp32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    const float scale,
    const std::size_t sequence_length,
    const std::size_t capacity,
    const StreamingPrefillGQAVariant variant,
    cudaStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr ||
        output == nullptr || !std::isfinite(scale) || sequence_length == 0 ||
        sequence_length > capacity || capacity > kMaximumSequenceLength) {
        return cudaErrorInvalidValue;
    }
    switch (variant) {
        case StreamingPrefillGQAVariant::kQueryTile8:
            return launch<32, 256>(query, key, value, output, scale,
                                   sequence_length, capacity, stream);
        case StreamingPrefillGQAVariant::kQueryTile32:
            return launch<8, 256>(query, key, value, output, scale,
                                  sequence_length, capacity, stream);
        case StreamingPrefillGQAVariant::kQueryTile128:
            return launch<1, 128>(query, key, value, output, scale,
                                  sequence_length, capacity, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace flux
