#include "rmsnorm_cuda.h"

#include <cuda_runtime.h>

#include <limits>

namespace flux {
namespace {

constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kWarpCount = kBlockSize / kWarpSize;

static_assert(kBlockSize % kWarpSize == 0);

__device__ float warp_reduce_sum(float value) {
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

__global__ void rmsnorm_cuda_fp32_kernel(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t hidden_size,
    const float epsilon) {
    __shared__ float warp_sums[kWarpCount];

    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * hidden_size;
    float sum_squares = 0.0F;
    for (std::size_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const float value = input[row_offset + column];
        sum_squares += value * value;
    }
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    sum_squares = warp_reduce_sum(sum_squares);
    if (lane == 0) {
        warp_sums[warp] = sum_squares;
    }
    __syncthreads();

    if (warp == 0) {
        sum_squares = lane < kWarpCount ? warp_sums[lane] : 0.0F;
        sum_squares = warp_reduce_sum(sum_squares);
    }
    if (threadIdx.x == 0) {
        const float mean_square =
            sum_squares / static_cast<float>(hidden_size);
        warp_sums[0] = rsqrtf(mean_square + epsilon);
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (std::size_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const std::size_t index = row_offset + column;
        output[index] = input[index] * inv_rms * weight[column];
    }
}

}  // namespace

cudaError_t rmsnorm_cuda_fp32(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        num_rows == 0 || hidden_size == 0 || epsilon < 0.0F) {
        return cudaErrorInvalidValue;
    }
    if (num_rows > std::numeric_limits<unsigned int>::max() ||
        num_rows > std::numeric_limits<std::size_t>::max() / hidden_size) {
        return cudaErrorInvalidValue;
    }

    rmsnorm_cuda_fp32_kernel<<<
        static_cast<unsigned int>(num_rows), kBlockSize, 0, stream>>>(
        input, weight, output, hidden_size, epsilon);
    return cudaGetLastError();
}

}  // namespace flux
