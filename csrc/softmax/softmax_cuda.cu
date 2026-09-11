#include "softmax_cuda.h"
#include "../attention_score_softmax/attention_score_softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <limits>

namespace flux {
namespace {

constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kSmallBlockSize = 128;
constexpr unsigned int kRegisterBlockSize = 256;
constexpr unsigned int kWideBlockSize = 512;
constexpr unsigned int kGenericBlockSize = 256;
constexpr unsigned int kWarpsPerBlock = 4;
constexpr std::size_t kWarpSoftmaxMaxWidth = 256;
constexpr std::size_t kLargeRegisterWidth = 8192;
constexpr std::size_t kHighRowCount = 512;
constexpr std::size_t kLowRowCount = 64;

struct IdentityTransform {
    const float* input;

    __device__ float load(
        const std::size_t index,
        const std::size_t,
        const std::size_t) const {
        return input[index];
    }
};

struct ScaledMaskedTransform {
    const float* scores;
    const float* mask;
    float scale;
    std::size_t heads;
    std::size_t query_length;
    std::size_t mask_batch_size;
    std::size_t mask_head_size;
    std::size_t mask_query_size;
    std::size_t mask_key_size;
    std::size_t mask_stride_batch;
    std::size_t mask_stride_head;
    std::size_t mask_stride_query;

    __device__ float load(
        const std::size_t index,
        const std::size_t row,
        const std::size_t column) const {
        const std::size_t query = row % query_length;
        const std::size_t head = (row / query_length) % heads;
        const std::size_t batch = row / (heads * query_length);
        const std::size_t mask_index =
            (mask_batch_size == 1 ? 0 : batch) * mask_stride_batch +
            (mask_head_size == 1 ? 0 : head) * mask_stride_head +
            (mask_query_size == 1 ? 0 : query) * mask_stride_query +
            (mask_key_size == 1 ? 0 : column);
        // PyTorch's unfused expression rounds the FP32 multiply before the
        // FP32 add.  Explicit round-to-nearest intrinsics prevent contraction
        // into an FMA and preserve that ordering.
        return __fadd_rn(__fmul_rn(scores[index], scale), mask[mask_index]);
    }
};

__device__ float warp_reduce_max(float value) {
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ float warp_reduce_sum(float value) {
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

template <unsigned int BlockSize>
__device__ float block_reduce_max(float value, float* warp_values) {
    constexpr unsigned int warp_count = BlockSize / kWarpSize;
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < warp_count ? warp_values[lane] : -FLT_MAX;
        value = warp_reduce_max(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

template <unsigned int BlockSize>
__device__ float block_reduce_sum(float value, float* warp_values) {
    constexpr unsigned int warp_count = BlockSize / kWarpSize;
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < warp_count ? warp_values[lane] : 0.0F;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

template <
    unsigned int ItemsPerThread,
    unsigned int WarpsPerBlock,
    typename Transform>
__global__ void warp_softmax_cuda_fp32_kernel(
    const Transform transform,
    float* output,
    const std::size_t num_rows,
    const std::size_t row_width) {
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * WarpsPerBlock + warp;
    if (row >= num_rows) {
        return;
    }

    const std::size_t row_offset = row * row_width;
    float values[ItemsPerThread];
    float local_max = -FLT_MAX;
#pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        const std::size_t column = lane + item * kWarpSize;
        const float value = column < row_width
            ? transform.load(row_offset + column, row, column)
            : -FLT_MAX;
        values[item] = value;
        local_max = fmaxf(local_max, value);
    }
    const float row_max =
        __shfl_sync(0xFFFFFFFFU, warp_reduce_max(local_max), 0);

    float local_sum = 0.0F;
#pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        const std::size_t column = lane + item * kWarpSize;
        if (column < row_width) {
            const float exponential = expf(values[item] - row_max);
            values[item] = exponential;
            local_sum += exponential;
        }
    }
    const float row_sum =
        __shfl_sync(0xFFFFFFFFU, warp_reduce_sum(local_sum), 0);

#pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        const std::size_t column = lane + item * kWarpSize;
        if (column < row_width) {
            output[row_offset + column] = values[item] / row_sum;
        }
    }
}

template <unsigned int BlockSize, typename Transform>
__global__ void block_softmax_cuda_fp32_kernel(
    const Transform transform,
    float* output,
    const std::size_t row_width) {
    constexpr unsigned int warp_count = BlockSize / kWarpSize;
    __shared__ float warp_values[warp_count];

    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * row_width;
    float local_max = -FLT_MAX;
    for (std::size_t column = threadIdx.x; column < row_width;
         column += blockDim.x) {
        local_max = fmaxf(
            local_max,
            transform.load(row_offset + column, blockIdx.x, column));
    }
    const float row_max = block_reduce_max<BlockSize>(local_max, warp_values);
    // Every thread must consume the shared reduction result before the same
    // storage is reused for the sum reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < row_width;
         column += blockDim.x) {
        const std::size_t index = row_offset + column;
        const float exponential = expf(
            transform.load(index, blockIdx.x, column) - row_max);
        output[index] = exponential;
        local_sum += exponential;
    }
    const float row_sum = block_reduce_sum<BlockSize>(local_sum, warp_values);

    for (std::size_t column = threadIdx.x; column < row_width;
         column += blockDim.x) {
        output[row_offset + column] /= row_sum;
    }
}

template <
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    typename Transform>
__global__ void register_softmax_cuda_fp32_kernel(
    const Transform transform,
    float* output,
    const std::size_t row_width) {
    constexpr unsigned int warp_count = BlockSize / kWarpSize;
    __shared__ float warp_values[warp_count];

    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * row_width;
    float values[ItemsPerThread];
    float local_max = -FLT_MAX;
#pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        const std::size_t column = threadIdx.x + item * BlockSize;
        const float value = column < row_width
            ? transform.load(row_offset + column, blockIdx.x, column)
            : -FLT_MAX;
        values[item] = value;
        local_max = fmaxf(local_max, value);
    }
    const float row_max = block_reduce_max<BlockSize>(local_max, warp_values);
    __syncthreads();

    float local_sum = 0.0F;
#pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        const std::size_t column = threadIdx.x + item * BlockSize;
        if (column < row_width) {
            const float exponential = expf(values[item] - row_max);
            values[item] = exponential;
            local_sum += exponential;
        }
    }
    const float row_sum = block_reduce_sum<BlockSize>(local_sum, warp_values);

#pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item) {
        const std::size_t column = threadIdx.x + item * BlockSize;
        if (column < row_width) {
            output[row_offset + column] = values[item] / row_sum;
        }
    }
}

template <typename Transform>
cudaError_t launch_softmax_cuda_fp32(
    const Transform transform,
    float* output,
    const std::size_t num_rows,
    const std::size_t row_width,
    cudaStream_t stream) {
    if (row_width == kWarpSoftmaxMaxWidth) {
        register_softmax_cuda_fp32_kernel<2, kSmallBlockSize><<<
            static_cast<unsigned int>(num_rows), kSmallBlockSize, 0, stream>>>(
            transform, output, row_width);
    } else if (row_width < kWarpSoftmaxMaxWidth) {
        const unsigned int block_count = static_cast<unsigned int>(
            (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
        if (row_width <= 32) {
            warp_softmax_cuda_fp32_kernel<1, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                transform, output, num_rows, row_width);
        } else if (row_width <= 64) {
            warp_softmax_cuda_fp32_kernel<2, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                transform, output, num_rows, row_width);
        } else if (row_width <= 128) {
            warp_softmax_cuda_fp32_kernel<4, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                transform, output, num_rows, row_width);
        } else {
            warp_softmax_cuda_fp32_kernel<8, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                transform, output, num_rows, row_width);
        }
    } else if (row_width == 512) {
        if (num_rows >= kHighRowCount) {
            register_softmax_cuda_fp32_kernel<4, kSmallBlockSize><<<
                static_cast<unsigned int>(num_rows), kSmallBlockSize, 0,
                stream>>>(
                transform, output, row_width);
        } else {
            register_softmax_cuda_fp32_kernel<2, kRegisterBlockSize><<<
                static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0,
                stream>>>(transform, output, row_width);
        }
    } else if (row_width < 512) {
        register_softmax_cuda_fp32_kernel<2, kRegisterBlockSize><<<
            static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0, stream>>>(
            transform, output, row_width);
    } else if (row_width < 1024) {
        register_softmax_cuda_fp32_kernel<4, kRegisterBlockSize><<<
            static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0, stream>>>(
            transform, output, row_width);
    } else if (row_width == 1024) {
        register_softmax_cuda_fp32_kernel<8, kSmallBlockSize><<<
            static_cast<unsigned int>(num_rows), kSmallBlockSize, 0, stream>>>(
            transform, output, row_width);
    } else if (row_width == 4096) {
        register_softmax_cuda_fp32_kernel<8, kWideBlockSize><<<
            static_cast<unsigned int>(num_rows), kWideBlockSize, 0, stream>>>(
            transform, output, row_width);
    } else if (row_width == kLargeRegisterWidth) {
        if (num_rows < kLowRowCount) {
            register_softmax_cuda_fp32_kernel<16, kWideBlockSize><<<
                static_cast<unsigned int>(num_rows), kWideBlockSize, 0,
                stream>>>(
                transform, output, row_width);
        } else {
            register_softmax_cuda_fp32_kernel<32, kRegisterBlockSize><<<
                static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0,
                stream>>>(transform, output, row_width);
        }
    } else {
        block_softmax_cuda_fp32_kernel<kGenericBlockSize><<<
            static_cast<unsigned int>(num_rows), kGenericBlockSize, 0, stream>>>(
            transform, output, row_width);
    }
    return cudaGetLastError();
}

bool valid_softmax_dimensions(
    const std::size_t num_rows,
    const std::size_t row_width) {
    return num_rows > 0 && row_width > 0 &&
        num_rows <= std::numeric_limits<unsigned int>::max() &&
        num_rows <= std::numeric_limits<std::size_t>::max() / row_width;
}

}  // namespace

cudaError_t softmax_cuda_fp32(
    const float* input,
    float* output,
    const std::size_t num_rows,
    const std::size_t row_width,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr ||
        !valid_softmax_dimensions(num_rows, row_width)) {
        return cudaErrorInvalidValue;
    }
    return launch_softmax_cuda_fp32(
        IdentityTransform{input}, output, num_rows, row_width, stream);
}

cudaError_t attention_score_softmax_cuda_fp32(
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
    const std::size_t mask_key_size,
    const std::size_t mask_stride_batch,
    const std::size_t mask_stride_head,
    const std::size_t mask_stride_query,
    cudaStream_t stream) {
    if (scores == nullptr || additive_attention_mask == nullptr ||
        output == nullptr || batch_size == 0 || heads == 0 ||
        query_length == 0 ||
        batch_size > std::numeric_limits<std::size_t>::max() / heads ||
        batch_size * heads >
            std::numeric_limits<std::size_t>::max() / query_length) {
        return cudaErrorInvalidValue;
    }
    const std::size_t num_rows = batch_size * heads * query_length;
    if (!valid_softmax_dimensions(num_rows, key_length)) {
        return cudaErrorInvalidValue;
    }
    return launch_softmax_cuda_fp32(
        ScaledMaskedTransform{
            scores,
            additive_attention_mask,
            scale,
            heads,
            query_length,
            mask_batch_size,
            mask_head_size,
            mask_query_size,
            mask_key_size,
            mask_stride_batch,
            mask_stride_head,
            mask_stride_query,
        },
        output,
        num_rows,
        key_length,
        stream);
}

}  // namespace flux
