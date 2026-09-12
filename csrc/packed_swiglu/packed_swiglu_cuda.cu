#include "packed_swiglu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

namespace flux {
namespace {

constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kMaximumBlocks = 65535;

__global__ void packed_swiglu_cuda_fp32_kernel(
    const float* packed,
    float* output,
    const std::size_t output_elements,
    const std::size_t intermediate_size) {
    const std::size_t packed_size = 2 * intermediate_size;
    for (std::size_t output_index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_index < output_elements;
         output_index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t row = output_index / intermediate_size;
        const std::size_t column = output_index - row * intermediate_size;
        const std::size_t packed_offset = row * packed_size + column;
        const float gate = packed[packed_offset];
        const float up = packed[packed_offset + intermediate_size];
        output[output_index] = (gate / (1.0F + expf(-gate))) * up;
    }
}

}  // namespace

cudaError_t packed_swiglu_cuda_fp32(
    const float* packed,
    float* output,
    const std::size_t num_rows,
    const std::size_t intermediate_size,
    cudaStream_t stream) {
    if (packed == nullptr || output == nullptr || num_rows == 0 ||
        intermediate_size == 0) {
        return cudaErrorInvalidValue;
    }
    if (num_rows > std::numeric_limits<std::size_t>::max() / intermediate_size) {
        return cudaErrorInvalidValue;
    }
    const std::size_t output_elements = num_rows * intermediate_size;
    const std::size_t required_blocks =
        (output_elements + kBlockSize - 1) / kBlockSize;
    const unsigned int block_count = static_cast<unsigned int>(
        std::min(required_blocks, static_cast<std::size_t>(kMaximumBlocks)));
    packed_swiglu_cuda_fp32_kernel<<<block_count, kBlockSize, 0, stream>>>(
        packed, output, output_elements, intermediate_size);
    return cudaGetLastError();
}

}  // namespace flux
