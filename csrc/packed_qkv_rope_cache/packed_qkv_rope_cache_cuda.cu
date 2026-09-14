#include "packed_qkv_rope_cache_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace flux {
namespace {

constexpr unsigned int kBlockSize = 256;
constexpr std::size_t kQueryHeads = 9;
constexpr std::size_t kKeyValueHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr std::size_t kKeyWidth = kKeyValueHeads * kHeadDim;
constexpr std::size_t kValueOffset = kQueryWidth + kKeyWidth;
constexpr std::size_t kTotalWidth = kValueOffset + kKeyWidth;

__device__ __forceinline__ float rotate_value(
    const float* packed_qkv,
    const float* cos,
    const float* sin,
    const std::ptrdiff_t packed_base,
    const std::ptrdiff_t embedding_base,
    const std::size_t offset,
    const std::size_t dimension,
    const PackedQKVStrides packed_strides,
    const PackedQKVEmbeddingStrides cos_strides,
    const PackedQKVEmbeddingStrides sin_strides) {
    constexpr std::size_t half = kHeadDim / 2;
    const std::size_t paired_dimension =
        dimension < half ? dimension + half : dimension - half;
    const float sign = dimension < half ? -1.0F : 1.0F;
    const float value = packed_qkv[
        packed_base + static_cast<std::ptrdiff_t>(offset + dimension) *
            packed_strides.dimension];
    const float paired = packed_qkv[
        packed_base + static_cast<std::ptrdiff_t>(offset + paired_dimension) *
            packed_strides.dimension];
    const float first = __fmul_rn(
        value,
        cos[embedding_base + static_cast<std::ptrdiff_t>(dimension) *
                cos_strides.dimension]);
    const float second = __fmul_rn(
        sign * paired,
        sin[embedding_base + static_cast<std::ptrdiff_t>(dimension) *
                sin_strides.dimension]);
    return __fadd_rn(first, second);
}

__global__ void packed_qkv_rope_cache_cuda_fp32_kernel(
    const float* packed_qkv,
    const float* cos,
    const float* sin,
    float* key_cache,
    float* value_cache,
    const std::int64_t* cache_position,
    std::int64_t* updated_cache_length,
    float* query_output,
    const std::size_t cache_capacity,
    const PackedQKVStrides packed_strides,
    const PackedQKVEmbeddingStrides cos_strides,
    const PackedQKVEmbeddingStrides sin_strides,
    const PackedQKVCacheStrides key_cache_strides,
    const PackedQKVCacheStrides value_cache_strides) {
    __shared__ std::int64_t position;
    if (threadIdx.x == 0) {
        position = cache_position[0];
    }
    __syncthreads();

    if (position < 0 || static_cast<std::size_t>(position) >= cache_capacity) {
        return;
    }

    const std::ptrdiff_t packed_base =
        0 * packed_strides.batch + 0 * packed_strides.sequence;
    const std::ptrdiff_t embedding_base =
        0 * cos_strides.batch + 0 * cos_strides.sequence;

    for (std::size_t linear = threadIdx.x; linear < kTotalWidth;
         linear += blockDim.x) {
        if (linear < kQueryWidth) {
            const std::size_t head = linear / kHeadDim;
            const std::size_t dimension = linear % kHeadDim;
            query_output[head * kHeadDim + dimension] = rotate_value(
                packed_qkv, cos, sin, packed_base, embedding_base,
                head * kHeadDim, dimension, packed_strides, cos_strides,
                sin_strides);
        } else if (linear < kValueOffset) {
            const std::size_t key_linear = linear - kQueryWidth;
            const std::size_t head = key_linear / kHeadDim;
            const std::size_t dimension = key_linear % kHeadDim;
            const std::ptrdiff_t cache_index =
                static_cast<std::ptrdiff_t>(head) * key_cache_strides.head +
                position * key_cache_strides.sequence +
                static_cast<std::ptrdiff_t>(dimension) *
                    key_cache_strides.dimension;
            key_cache[cache_index] = rotate_value(
                packed_qkv, cos, sin, packed_base, embedding_base,
                kQueryWidth + head * kHeadDim, dimension, packed_strides,
                cos_strides, sin_strides);
        } else {
            const std::size_t value_linear = linear - kValueOffset;
            const std::size_t head = value_linear / kHeadDim;
            const std::size_t dimension = value_linear % kHeadDim;
            const std::ptrdiff_t cache_index =
                static_cast<std::ptrdiff_t>(head) * value_cache_strides.head +
                position * value_cache_strides.sequence +
                static_cast<std::ptrdiff_t>(dimension) *
                    value_cache_strides.dimension;
            value_cache[cache_index] = packed_qkv[
                packed_base + static_cast<std::ptrdiff_t>(linear) *
                    packed_strides.dimension];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && updated_cache_length != nullptr) {
        updated_cache_length[0] = position + 1;
    }
}

}  // namespace

cudaError_t packed_qkv_rope_cache_cuda_fp32(
    const float* packed_qkv,
    const float* cos,
    const float* sin,
    float* key_cache,
    float* value_cache,
    std::int64_t* cache_length,
    float* query_output,
    const std::size_t cache_capacity,
    const PackedQKVStrides packed_strides,
    const PackedQKVEmbeddingStrides cos_strides,
    const PackedQKVEmbeddingStrides sin_strides,
    const PackedQKVCacheStrides key_cache_strides,
    const PackedQKVCacheStrides value_cache_strides,
    cudaStream_t stream) {
    if (packed_qkv == nullptr || cos == nullptr || sin == nullptr ||
        key_cache == nullptr || value_cache == nullptr ||
        cache_length == nullptr || query_output == nullptr ||
        cache_capacity == 0) {
        return cudaErrorInvalidValue;
    }
    packed_qkv_rope_cache_cuda_fp32_kernel<<<1, kBlockSize, 0, stream>>>(
        packed_qkv, cos, sin, key_cache, value_cache, cache_length, cache_length,
        query_output, cache_capacity, packed_strides, cos_strides, sin_strides,
        key_cache_strides, value_cache_strides);
    return cudaGetLastError();
}

cudaError_t packed_qkv_rope_cache_at_position_cuda_fp32(
    const float* packed_qkv,
    const float* cos,
    const float* sin,
    float* key_cache,
    float* value_cache,
    const std::int64_t* cache_position,
    float* query_output,
    const std::size_t cache_capacity,
    const PackedQKVStrides packed_strides,
    const PackedQKVEmbeddingStrides cos_strides,
    const PackedQKVEmbeddingStrides sin_strides,
    const PackedQKVCacheStrides key_cache_strides,
    const PackedQKVCacheStrides value_cache_strides,
    cudaStream_t stream) {
    if (packed_qkv == nullptr || cos == nullptr || sin == nullptr ||
        key_cache == nullptr || value_cache == nullptr ||
        cache_position == nullptr || query_output == nullptr ||
        cache_capacity == 0) {
        return cudaErrorInvalidValue;
    }
    packed_qkv_rope_cache_cuda_fp32_kernel<<<1, kBlockSize, 0, stream>>>(
        packed_qkv, cos, sin, key_cache, value_cache, cache_position, nullptr,
        query_output, cache_capacity, packed_strides, cos_strides, sin_strides,
        key_cache_strides, value_cache_strides);
    return cudaGetLastError();
}

}  // namespace flux
