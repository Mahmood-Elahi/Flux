#include "softmax_cuda.h"

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

template <unsigned int ItemsPerThread, unsigned int WarpsPerBlock>
__global__ void warp_softmax_cuda_fp32_kernel(
    const float* input,
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
        const float value =
            column < row_width ? input[row_offset + column] : -FLT_MAX;
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

template <unsigned int BlockSize>
__global__ void block_softmax_cuda_fp32_kernel(
    const float* input,
    float* output,
    const std::size_t row_width) {
    constexpr unsigned int warp_count = BlockSize / kWarpSize;
    __shared__ float warp_values[warp_count];

    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * row_width;
    float local_max = -FLT_MAX;
    for (std::size_t column = threadIdx.x; column < row_width;
         column += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + column]);
    }
    const float row_max = block_reduce_max<BlockSize>(local_max, warp_values);
    // Every thread must consume the shared reduction result before the same
    // storage is reused for the sum reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < row_width;
         column += blockDim.x) {
        const std::size_t index = row_offset + column;
        const float exponential = expf(input[index] - row_max);
        output[index] = exponential;
        local_sum += exponential;
    }
    const float row_sum = block_reduce_sum<BlockSize>(local_sum, warp_values);

    for (std::size_t column = threadIdx.x; column < row_width;
         column += blockDim.x) {
        output[row_offset + column] /= row_sum;
    }
}

template <unsigned int ItemsPerThread, unsigned int BlockSize>
__global__ void register_softmax_cuda_fp32_kernel(
    const float* input,
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
        const float value =
            column < row_width ? input[row_offset + column] : -FLT_MAX;
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

}  // namespace

cudaError_t softmax_cuda_fp32(
    const float* input,
    float* output,
    const std::size_t num_rows,
    const std::size_t row_width,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || num_rows == 0 ||
        row_width == 0) {
        return cudaErrorInvalidValue;
    }
    if (num_rows > std::numeric_limits<unsigned int>::max() ||
        num_rows > std::numeric_limits<std::size_t>::max() / row_width) {
        return cudaErrorInvalidValue;
    }

    if (row_width == kWarpSoftmaxMaxWidth) {
        register_softmax_cuda_fp32_kernel<2, kSmallBlockSize><<<
            static_cast<unsigned int>(num_rows), kSmallBlockSize, 0, stream>>>(
            input, output, row_width);
    } else if (row_width < kWarpSoftmaxMaxWidth) {
        const unsigned int block_count = static_cast<unsigned int>(
            (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
        if (row_width <= 32) {
            warp_softmax_cuda_fp32_kernel<1, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                input, output, num_rows, row_width);
        } else if (row_width <= 64) {
            warp_softmax_cuda_fp32_kernel<2, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                input, output, num_rows, row_width);
        } else if (row_width <= 128) {
            warp_softmax_cuda_fp32_kernel<4, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                input, output, num_rows, row_width);
        } else {
            warp_softmax_cuda_fp32_kernel<8, kWarpsPerBlock><<<
                block_count, kWarpsPerBlock * kWarpSize, 0, stream>>>(
                input, output, num_rows, row_width);
        }
    } else if (row_width == 512) {
        if (num_rows >= kHighRowCount) {
            register_softmax_cuda_fp32_kernel<4, kSmallBlockSize><<<
                static_cast<unsigned int>(num_rows), kSmallBlockSize, 0,
                stream>>>(
                input, output, row_width);
        } else {
            register_softmax_cuda_fp32_kernel<2, kRegisterBlockSize><<<
                static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0,
                stream>>>(input, output, row_width);
        }
    } else if (row_width < 512) {
        register_softmax_cuda_fp32_kernel<2, kRegisterBlockSize><<<
            static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0, stream>>>(
            input, output, row_width);
    } else if (row_width < 1024) {
        register_softmax_cuda_fp32_kernel<4, kRegisterBlockSize><<<
            static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0, stream>>>(
            input, output, row_width);
    } else if (row_width == 1024) {
        register_softmax_cuda_fp32_kernel<8, kSmallBlockSize><<<
            static_cast<unsigned int>(num_rows), kSmallBlockSize, 0, stream>>>(
            input, output, row_width);
    } else if (row_width == 4096) {
        register_softmax_cuda_fp32_kernel<8, kWideBlockSize><<<
            static_cast<unsigned int>(num_rows), kWideBlockSize, 0, stream>>>(
            input, output, row_width);
    } else if (row_width == kLargeRegisterWidth) {
        if (num_rows < kLowRowCount) {
            register_softmax_cuda_fp32_kernel<16, kWideBlockSize><<<
                static_cast<unsigned int>(num_rows), kWideBlockSize, 0,
                stream>>>(
                input, output, row_width);
        } else {
            register_softmax_cuda_fp32_kernel<32, kRegisterBlockSize><<<
                static_cast<unsigned int>(num_rows), kRegisterBlockSize, 0,
                stream>>>(input, output, row_width);
        }
    } else {
        block_softmax_cuda_fp32_kernel<kGenericBlockSize><<<
            static_cast<unsigned int>(num_rows), kGenericBlockSize, 0, stream>>>(
            input, output, row_width);
    }
    return cudaGetLastError();
}

}  // namespace flux
