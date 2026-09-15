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
constexpr int kPartialStateWidth = kHeadDimension + 2;
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

template <int Partitions, int BlockThreads>
__global__ void split_streaming_prefill_gqa_fp32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ partial_states,
    const float scale,
    const int sequence_length,
    const int capacity) {
    __shared__ __align__(16) float key_tile[kKeyTile][kHeadDimension];
    __shared__ __align__(16) float value_tile[kKeyTile][kHeadDimension];

    const int query_position = blockIdx.x * BlockThreads + threadIdx.x;
    const int query_head = blockIdx.y;
    const int kv_head = query_head / kQueryHeadsPerGroup;
    const int partition = blockIdx.z;
    const int partition_span =
        ((sequence_length + Partitions - 1) / Partitions + kKeyTile - 1) /
        kKeyTile * kKeyTile;
    const int partition_begin = partition * partition_span;
    const int partition_end = min(partition_begin + partition_span,
                                  sequence_length);
    const std::size_t state_index =
        ((static_cast<std::size_t>(query_head) * sequence_length +
          query_position) * Partitions + partition) * kPartialStateWidth;

    if (partition_begin >= partition_end ||
        partition_begin >= min((blockIdx.x + 1) * BlockThreads,
                               sequence_length)) {
        if (query_position < sequence_length) {
            partial_states[state_index] = -FLT_MAX;
            partial_states[state_index + 1] = 0.0F;
#pragma unroll
            for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
                partial_states[state_index + 2 + dimension] = 0.0F;
            }
        }
        return;
    }

    float query_values[kHeadDimension];
    float output_values[kHeadDimension] = {};
    float running_maximum = -FLT_MAX;
    float running_sum = 0.0F;
#pragma unroll
    for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
        query_values[dimension] = query_position < sequence_length
            ? query[(static_cast<std::size_t>(query_head) * sequence_length +
                     query_position) * kHeadDimension + dimension]
            : 0.0F;
    }

    const int last_query =
        min((blockIdx.x + 1) * BlockThreads, sequence_length) - 1;
    const int key_limit = min(partition_end, last_query + 1);
    for (int first_key = partition_begin; first_key < key_limit;
         first_key += kKeyTile) {
        const int valid_keys = min(kKeyTile, key_limit - first_key);
        const float4* key_vectors = reinterpret_cast<const float4*>(
            key + (static_cast<std::size_t>(kv_head) * capacity + first_key) *
                kHeadDimension);
        const float4* value_vectors = reinterpret_cast<const float4*>(
            value + (static_cast<std::size_t>(kv_head) * capacity + first_key) *
                kHeadDimension);
        float4* shared_key_vectors = reinterpret_cast<float4*>(key_tile);
        float4* shared_value_vectors = reinterpret_cast<float4*>(value_tile);
        constexpr int vectors_per_tile = kKeyTile * kHeadDimension / 4;
        for (int vector_index = threadIdx.x;
             vector_index < vectors_per_tile;
             vector_index += BlockThreads) {
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
            float score = 0.0F;
#pragma unroll
            for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
                score = fmaf(query_values[dimension],
                             key_tile[key_index][dimension], score);
            }
            score *= scale;
            const float new_maximum = fmaxf(running_maximum, score);
            const float previous_weight = expf(running_maximum - new_maximum);
            const float current_weight = expf(score - new_maximum);
            running_sum = running_sum * previous_weight + current_weight;
            running_maximum = new_maximum;
#pragma unroll
            for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
                output_values[dimension] =
                    output_values[dimension] * previous_weight +
                    current_weight * value_tile[key_index][dimension];
            }
        }
        __syncthreads();
    }

    if (query_position < sequence_length) {
        partial_states[state_index] = running_maximum;
        partial_states[state_index + 1] = running_sum;
#pragma unroll
        for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
            partial_states[state_index + 2 + dimension] =
                output_values[dimension];
        }
    }
}

template <int Partitions>
__global__ void merge_streaming_prefill_gqa_fp32_kernel(
    const float* __restrict__ partial_states,
    float* __restrict__ output,
    const int sequence_length) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int warps_per_block = 8;
    const int row = blockIdx.x * warps_per_block + warp;
    const int rows = kQueryHeads * sequence_length;
    if (row >= rows) {
        return;
    }
    const float* states = partial_states +
        static_cast<std::size_t>(row) * Partitions * kPartialStateWidth;
    float maximum = -FLT_MAX;
    if (lane == 0) {
#pragma unroll
        for (int partition = 0; partition < Partitions; ++partition) {
            maximum = fmaxf(maximum,
                            states[partition * kPartialStateWidth]);
        }
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    float normalization = 0.0F;
    float values[2] = {};
#pragma unroll
    for (int partition = 0; partition < Partitions; ++partition) {
        float weight = 0.0F;
        if (lane == 0) {
            const float local_maximum =
                states[partition * kPartialStateWidth];
            weight = local_maximum == -FLT_MAX
                ? 0.0F : expf(local_maximum - maximum);
            normalization +=
                states[partition * kPartialStateWidth + 1] * weight;
        }
        weight = __shfl_sync(0xffffffffU, weight, 0);
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            values[item] +=
                states[partition * kPartialStateWidth + 2 + lane + item * 32] *
                weight;
        }
    }
    normalization = __shfl_sync(0xffffffffU, normalization, 0);
#pragma unroll
    for (int item = 0; item < 2; ++item) {
        output[static_cast<std::size_t>(row) * kHeadDimension +
               lane + item * 32] = values[item] / normalization;
    }
}

template <int Partitions, int BlockThreads>
cudaError_t launch_split(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* workspace,
    const float scale,
    const std::size_t sequence_length,
    const std::size_t capacity,
    cudaStream_t stream) {
    const dim3 stage_grid(
        static_cast<unsigned int>(
            (sequence_length + BlockThreads - 1) / BlockThreads),
        kQueryHeads, Partitions);
    split_streaming_prefill_gqa_fp32_kernel<Partitions, BlockThreads><<<
        stage_grid, BlockThreads, 0, stream>>>(
        query, key, value, workspace, scale,
        static_cast<int>(sequence_length), static_cast<int>(capacity));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    constexpr int merge_threads = 256;
    constexpr int merge_rows = merge_threads / 32;
    const unsigned int merge_blocks = static_cast<unsigned int>(
        (kQueryHeads * sequence_length + merge_rows - 1) / merge_rows);
    merge_streaming_prefill_gqa_fp32_kernel<Partitions><<<
        merge_blocks, merge_threads, 0, stream>>>(
        workspace, output, static_cast<int>(sequence_length));
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
    if (sequence_length <= 1024) {
        return StreamingPrefillGQAVariant::kSplit4QueryTile128;
    }
    if (sequence_length <= 4096) {
        return StreamingPrefillGQAVariant::kSplit2QueryTile128;
    }
    return StreamingPrefillGQAVariant::kSplit4QueryTile128;
}

std::size_t streaming_prefill_gqa_workspace_bytes(
    const std::size_t sequence_length,
    const StreamingPrefillGQAVariant variant) {
    std::size_t partitions = 0;
    switch (variant) {
        case StreamingPrefillGQAVariant::kSplit2QueryTile128:
            partitions = 2;
            break;
        case StreamingPrefillGQAVariant::kSplit4QueryTile128:
            partitions = 4;
            break;
        default:
            return 0;
    }
    return kQueryHeads * sequence_length * partitions *
        kPartialStateWidth * sizeof(float);
}

cudaError_t streaming_prefill_gqa_cuda_fp32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* workspace,
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
        case StreamingPrefillGQAVariant::kSplit2QueryTile128:
            if (workspace == nullptr) return cudaErrorInvalidValue;
            return launch_split<2, 128>(query, key, value, output, workspace,
                                        scale, sequence_length, capacity, stream);
        case StreamingPrefillGQAVariant::kSplit4QueryTile128:
            if (workspace == nullptr) return cudaErrorInvalidValue;
            return launch_split<4, 128>(query, key, value, output, workspace,
                                        scale, sequence_length, capacity, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace flux
