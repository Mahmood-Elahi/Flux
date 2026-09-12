#include "gqa_decode_attention_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <limits>

namespace flux {
namespace {

constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kBlockSize = 256;
constexpr std::size_t kChunkSize = 256;
constexpr std::size_t kSingleBlockMaximum = 512;
constexpr std::size_t kMaximumCacheCapacity = 8192;

__device__ float warp_reduce_sum(float value) {
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

__device__ float warp_reduce_max(float value) {
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ float block_reduce_sum(float value, float* warp_values) {
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < kBlockSize / kWarpSize ? warp_values[lane] : 0.0F;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

__device__ float block_reduce_max(float value, float* warp_values) {
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < kBlockSize / kWarpSize ? warp_values[lane] : -FLT_MAX;
        value = warp_reduce_max(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

__global__ void gqa_decode_attention_chunk_cuda_fp32_kernel(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* mask,
    const std::int64_t* cache_length,
    float* workspace,
    const float scale,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t cache_capacity,
    const std::size_t head_dim,
    const std::size_t num_chunks,
    const std::int64_t q_s0,
    const std::int64_t q_s1,
    const std::int64_t q_s3,
    const std::int64_t k_s0,
    const std::int64_t k_s1,
    const std::int64_t k_s2,
    const std::int64_t k_s3,
    const std::int64_t v_s0,
    const std::int64_t v_s1,
    const std::int64_t v_s2,
    const std::int64_t v_s3,
    const std::int64_t mask_b,
    const std::int64_t mask_h,
    const std::int64_t mask_k,
    const std::int64_t mask_s0,
    const std::int64_t mask_s1,
    const std::int64_t mask_s3) {
    __shared__ float shared_query[64];
    __shared__ float scores[kChunkSize];
    __shared__ float reductions[kBlockSize / kWarpSize];

    const std::size_t chunk = blockIdx.x % num_chunks;
    const std::size_t head_block = blockIdx.x / num_chunks;
    const std::size_t batch = head_block / query_heads;
    const std::size_t query_head = head_block % query_heads;
    const std::size_t groups = query_heads / kv_heads;
    const std::size_t kv_head = query_head / groups;
    const std::int64_t requested_length = cache_length == nullptr
        ? static_cast<std::int64_t>(cache_capacity)
        : *cache_length;
    const std::size_t valid_length = static_cast<std::size_t>(
        max(static_cast<std::int64_t>(1),
            min(requested_length, static_cast<std::int64_t>(cache_capacity))));
    const std::size_t chunk_start = chunk * kChunkSize;
    const std::size_t chunk_length = chunk_start < valid_length
        ? min(kChunkSize, valid_length - chunk_start)
        : 0;
    const std::size_t workspace_offset =
        head_block * num_chunks * (head_dim + 2) + chunk * (head_dim + 2);

    if (threadIdx.x < head_dim) {
        shared_query[threadIdx.x] = query[
            batch * q_s0 + query_head * q_s1 + threadIdx.x * q_s3];
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    for (std::size_t local_position = warp; local_position < chunk_length;
         local_position += kBlockSize / kWarpSize) {
        const std::size_t position = chunk_start + local_position;
        float dot = 0.0F;
#pragma unroll
        for (unsigned int dimension = lane; dimension < 64; dimension += kWarpSize) {
            dot += shared_query[dimension] * key_cache[
                batch * k_s0 + kv_head * k_s1 + position * k_s2 +
                dimension * k_s3];
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float score = dot * scale;
            if (mask != nullptr) {
                const std::size_t mask_batch = mask_b == 1 ? 0 : batch;
                const std::size_t mask_head = mask_h == 1 ? 0 : query_head;
                const std::size_t mask_position = mask_k == 1 ? 0 : position;
                score += mask[
                    mask_batch * mask_s0 + mask_head * mask_s1 +
                    mask_position * mask_s3];
            }
            scores[local_position] = score;
        }
    }
    __syncthreads();

    float local_max = threadIdx.x < chunk_length ? scores[threadIdx.x] : -FLT_MAX;
    const float chunk_max = block_reduce_max(local_max, reductions);
    __syncthreads();
    float exponential = 0.0F;
    if (threadIdx.x < chunk_length) {
        exponential = expf(scores[threadIdx.x] - chunk_max);
        scores[threadIdx.x] = exponential;
    }
    const float chunk_sum = block_reduce_sum(exponential, reductions);

    constexpr unsigned int lanes_per_dimension = 4;
    const unsigned int dimension = threadIdx.x / lanes_per_dimension;
    const unsigned int reduction_lane = threadIdx.x % lanes_per_dimension;
    float weighted_value = 0.0F;
    for (std::size_t local_position = reduction_lane;
         local_position < chunk_length; local_position += lanes_per_dimension) {
        const std::size_t position = chunk_start + local_position;
        weighted_value += scores[local_position] * value_cache[
            batch * v_s0 + kv_head * v_s1 + position * v_s2 +
            dimension * v_s3];
    }
    weighted_value += __shfl_down_sync(
        0xFFFFFFFFU, weighted_value, 2, lanes_per_dimension);
    weighted_value += __shfl_down_sync(
        0xFFFFFFFFU, weighted_value, 1, lanes_per_dimension);
    if (threadIdx.x == 0) {
        workspace[workspace_offset] = chunk_length == 0 ? -FLT_MAX : chunk_max;
        workspace[workspace_offset + 1] = chunk_sum;
    }
    if (reduction_lane == 0) {
        workspace[workspace_offset + 2 + dimension] = weighted_value;
    }
}

__global__ void gqa_decode_attention_reduce_cuda_fp32_kernel(
    const float* workspace,
    float* output,
    const std::size_t head_dim,
    const std::size_t num_chunks) {
    __shared__ float global_max;
    __shared__ float global_sum;
    const std::size_t workspace_offset =
        static_cast<std::size_t>(blockIdx.x) * num_chunks * (head_dim + 2);
    if (threadIdx.x == 0) {
        float maximum = -FLT_MAX;
        for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
            maximum = fmaxf(maximum, workspace[
                workspace_offset + chunk * (head_dim + 2)]);
        }
        float sum = 0.0F;
        for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
            const std::size_t offset = workspace_offset + chunk * (head_dim + 2);
            sum += workspace[offset + 1] * expf(workspace[offset] - maximum);
        }
        global_max = maximum;
        global_sum = sum;
    }
    __syncthreads();
    for (std::size_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float value = 0.0F;
        for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
            const std::size_t offset = workspace_offset + chunk * (head_dim + 2);
            value += workspace[offset + 2 + dimension] *
                expf(workspace[offset] - global_max);
        }
        output[static_cast<std::size_t>(blockIdx.x) * head_dim + dimension] =
            value / global_sum;
    }
}

__global__ void gqa_decode_attention_cuda_fp32_kernel(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* mask,
    const std::int64_t* cache_length,
    float* output,
    const float scale,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t cache_capacity,
    const std::size_t head_dim,
    const std::int64_t q_s0,
    const std::int64_t q_s1,
    const std::int64_t q_s3,
    const std::int64_t k_s0,
    const std::int64_t k_s1,
    const std::int64_t k_s2,
    const std::int64_t k_s3,
    const std::int64_t v_s0,
    const std::int64_t v_s1,
    const std::int64_t v_s2,
    const std::int64_t v_s3,
    const std::int64_t mask_b,
    const std::int64_t mask_h,
    const std::int64_t mask_k,
    const std::int64_t mask_s0,
    const std::int64_t mask_s1,
    const std::int64_t mask_s3) {
    extern __shared__ float shared[];
    float* shared_query = shared;
    float* scores = shared + head_dim;
    __shared__ float reductions[kBlockSize / kWarpSize];

    const std::size_t batch = blockIdx.x / query_heads;
    const std::size_t query_head = blockIdx.x % query_heads;
    const std::size_t groups = query_heads / kv_heads;
    const std::size_t kv_head = query_head / groups;
    const std::int64_t requested_length = cache_length == nullptr
        ? static_cast<std::int64_t>(cache_capacity)
        : *cache_length;
    const std::size_t valid_length = static_cast<std::size_t>(
        max(static_cast<std::int64_t>(1),
            min(requested_length, static_cast<std::int64_t>(cache_capacity))));

    for (std::size_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        shared_query[dimension] = query[
            batch * q_s0 + query_head * q_s1 + dimension * q_s3];
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    for (std::size_t position = warp; position < valid_length;
         position += kBlockSize / kWarpSize) {
        float dot = 0.0F;
        for (std::size_t dimension = lane; dimension < head_dim;
             dimension += kWarpSize) {
            dot += shared_query[dimension] * key_cache[
                batch * k_s0 + kv_head * k_s1 + position * k_s2 +
                dimension * k_s3];
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float value = dot * scale;
            if (mask != nullptr) {
                const std::size_t mask_batch = mask_b == 1 ? 0 : batch;
                const std::size_t mask_head = mask_h == 1 ? 0 : query_head;
                const std::size_t mask_position = mask_k == 1 ? 0 : position;
                value += mask[
                    mask_batch * mask_s0 + mask_head * mask_s1 +
                    mask_position * mask_s3];
            }
            scores[position] = value;
        }
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    for (std::size_t position = threadIdx.x; position < valid_length;
         position += blockDim.x) {
        local_max = fmaxf(local_max, scores[position]);
    }
    const float row_max = block_reduce_max(local_max, reductions);
    __syncthreads();

    float local_sum = 0.0F;
    for (std::size_t position = threadIdx.x; position < valid_length;
         position += blockDim.x) {
        const float exponential = expf(scores[position] - row_max);
        scores[position] = exponential;
        local_sum += exponential;
    }
    const float row_sum = block_reduce_sum(local_sum, reductions);

    if (head_dim == 64) {
        constexpr unsigned int lanes_per_dimension = 4;
        const unsigned int dimension = threadIdx.x / lanes_per_dimension;
        const unsigned int reduction_lane = threadIdx.x % lanes_per_dimension;
        float value = 0.0F;
        for (std::size_t position = reduction_lane; position < valid_length;
             position += lanes_per_dimension) {
            value += scores[position] * value_cache[
                batch * v_s0 + kv_head * v_s1 + position * v_s2 +
                dimension * v_s3];
        }
        value += __shfl_down_sync(0xFFFFFFFFU, value, 2, lanes_per_dimension);
        value += __shfl_down_sync(0xFFFFFFFFU, value, 1, lanes_per_dimension);
        if (reduction_lane == 0) {
            output[(batch * query_heads + query_head) * head_dim + dimension] =
                value / row_sum;
        }
    } else {
        for (std::size_t dimension = threadIdx.x; dimension < head_dim;
             dimension += blockDim.x) {
            float value = 0.0F;
            for (std::size_t position = 0; position < valid_length; ++position) {
                value += scores[position] * value_cache[
                    batch * v_s0 + kv_head * v_s1 + position * v_s2 +
                    dimension * v_s3];
            }
            const std::size_t output_index =
                (batch * query_heads + query_head) * head_dim + dimension;
            output[output_index] = value / row_sum;
        }
    }
}

}  // namespace

cudaError_t gqa_decode_attention_cuda_fp32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* additive_attention_mask,
    const std::int64_t* cache_length,
    float* output,
    float* workspace,
    const float scale,
    const std::size_t batch_size,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t cache_capacity,
    const std::size_t head_dim,
    const std::size_t num_chunks,
    const std::int64_t* query_strides,
    const std::int64_t* key_strides,
    const std::int64_t* value_strides,
    const std::int64_t* mask_sizes,
    const std::int64_t* mask_strides,
    cudaStream_t stream) {
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        output == nullptr || batch_size == 0 || query_heads == 0 ||
        kv_heads == 0 || cache_capacity == 0 || head_dim == 0 ||
        query_heads % kv_heads != 0 || cache_capacity > kMaximumCacheCapacity ||
        num_chunks != (cache_capacity + kChunkSize - 1) / kChunkSize ||
        batch_size > std::numeric_limits<unsigned int>::max() / query_heads) {
        return cudaErrorInvalidValue;
    }
    const unsigned int blocks = static_cast<unsigned int>(batch_size * query_heads);
    const std::size_t shared_bytes = (head_dim + cache_capacity) * sizeof(float);
    const std::int64_t zero_sizes[4] = {1, 1, 1, 1};
    const std::int64_t zero_strides[4] = {0, 0, 0, 0};
    const std::int64_t* sizes = mask_sizes == nullptr ? zero_sizes : mask_sizes;
    const std::int64_t* strides = mask_strides == nullptr ? zero_strides : mask_strides;
    if (head_dim == 64 && cache_capacity > kSingleBlockMaximum) {
        if (workspace == nullptr) {
            return cudaErrorInvalidValue;
        }
        const unsigned int chunk_blocks = static_cast<unsigned int>(
            batch_size * query_heads * num_chunks);
        gqa_decode_attention_chunk_cuda_fp32_kernel<<<
            chunk_blocks, kBlockSize, 0, stream>>>(
            query, key_cache, value_cache, additive_attention_mask, cache_length,
            workspace, scale, query_heads, kv_heads, cache_capacity, head_dim,
            num_chunks, query_strides[0], query_strides[1], query_strides[3],
            key_strides[0], key_strides[1], key_strides[2], key_strides[3],
            value_strides[0], value_strides[1], value_strides[2], value_strides[3],
            sizes[0], sizes[1], sizes[3], strides[0], strides[1], strides[3]);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        gqa_decode_attention_reduce_cuda_fp32_kernel<<<
            blocks, 64, 0, stream>>>(workspace, output, head_dim, num_chunks);
        return cudaGetLastError();
    }
    gqa_decode_attention_cuda_fp32_kernel<<<blocks, kBlockSize, shared_bytes, stream>>>(
        query, key_cache, value_cache, additive_attention_mask, cache_length,
        output, scale, query_heads, kv_heads, cache_capacity, head_dim,
        query_strides[0], query_strides[1], query_strides[3],
        key_strides[0], key_strides[1], key_strides[2], key_strides[3],
        value_strides[0], value_strides[1], value_strides[2], value_strides[3],
        sizes[0], sizes[1], sizes[3], strides[0], strides[1], strides[3]);
    return cudaGetLastError();
}

}  // namespace flux
