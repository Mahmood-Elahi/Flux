#include "residual_rmsnorm_cuda.h"

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

__global__ void residual_rmsnorm_cuda_fp32_kernel(
    const float* hidden,
    const float* residual,
    const float* weight,
    float* residual_out,
    float* norm_out,
    const std::size_t hidden_size,
    const float epsilon) {
    __shared__ float warp_sums[kWarpCount];

    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * hidden_size;
    float sum_squares = 0.0F;
    for (std::size_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const std::size_t index = row_offset + column;
        const float value = hidden[index] + residual[index];
        residual_out[index] = value;
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
        norm_out[index] = residual_out[index] * inv_rms * weight[column];
    }
}

}  // namespace

cudaError_t residual_rmsnorm_cuda_fp32(
    const float* hidden,
    const float* residual,
    const float* weight,
    float* residual_out,
    float* norm_out,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    cudaStream_t stream) {
    if (hidden == nullptr || residual == nullptr || weight == nullptr ||
        residual_out == nullptr || norm_out == nullptr || num_rows == 0 ||
        hidden_size == 0 || epsilon < 0.0F) {
        return cudaErrorInvalidValue;
    }
    if (num_rows > std::numeric_limits<unsigned int>::max() ||
        num_rows > std::numeric_limits<std::size_t>::max() / hidden_size) {
        return cudaErrorInvalidValue;
    }

    residual_rmsnorm_cuda_fp32_kernel<<<
        static_cast<unsigned int>(num_rows), kBlockSize, 0, stream>>>(
        hidden,
        residual,
        weight,
        residual_out,
        norm_out,
        hidden_size,
        epsilon);
    return cudaGetLastError();
}

}  // namespace flux
