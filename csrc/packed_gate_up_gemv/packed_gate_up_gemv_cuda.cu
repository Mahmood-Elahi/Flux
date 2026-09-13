#include "packed_gate_up_gemv_cuda.h"

#include <cuda_runtime.h>

namespace flux {
namespace {

constexpr int kInputWidth = 576;
constexpr int kIntermediateWidth = 1536;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;
constexpr int kBlocks = kIntermediateWidth / kWarpsPerBlock;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

// One warp owns one matched gate/up feature. Adjacent lanes read adjacent
// row-major weights, and each input load feeds both FP32 accumulators.
__global__ void packed_gate_up_swiglu_cuda_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    const float* gate_weight = weight + row * kInputWidth;
    const float* up_weight =
        weight + (row + kIntermediateWidth) * kInputWidth;
    float gate = 0.0F;
    float up = 0.0F;
#pragma unroll
    for (int k = lane; k < kInputWidth; k += 32) {
        const float value = input[k];
        gate = fmaf(gate_weight[k], value, gate);
        up = fmaf(up_weight[k], value, up);
    }
    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0) {
        output[row] = (gate / (1.0F + expf(-gate))) * up;
    }
}

}  // namespace

cudaError_t packed_gate_up_swiglu_cuda_fp32(
    const float* input,
    const float* weight,
    float* output,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    packed_gate_up_swiglu_cuda_fp32_kernel
        <<<kBlocks, kThreadsPerBlock, 0, stream>>>(input, weight, output);
    return cudaGetLastError();
}

}  // namespace flux
