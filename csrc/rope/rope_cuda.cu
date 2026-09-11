#include "rope_cuda.h"

#include <cuda_runtime.h>

#include <limits>

namespace flux {
namespace {

constexpr unsigned int kBlockSize = 256;

__global__ void rope_cuda_fp32_kernel(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_output,
    float* key_output,
    const std::size_t batch_size,
    const std::size_t query_heads,
    const std::size_t key_heads,
    const std::size_t sequence_length,
    const std::size_t head_dim,
    const std::size_t rope_batch_size,
    const RopeStrides query_strides,
    const RopeStrides key_strides,
    const RopeEmbeddingStrides cos_strides,
    const RopeEmbeddingStrides sin_strides,
    const std::size_t total_elements) {
    for (std::size_t linear =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total_elements;
         linear += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        std::size_t index = linear;
        const std::size_t dimension = index % head_dim;
        index /= head_dim;
        const std::size_t sequence = index % sequence_length;
        index /= sequence_length;
        const std::size_t combined_head = index % (query_heads + key_heads);
        const std::size_t batch = index / (query_heads + key_heads);
        const bool is_query = combined_head < query_heads;
        const std::size_t head =
            is_query ? combined_head : combined_head - query_heads;
        const float* input = is_query ? query : key;
        float* output = is_query ? query_output : key_output;
        const RopeStrides input_strides = is_query ? query_strides : key_strides;
        const std::size_t heads = is_query ? query_heads : key_heads;
        const std::size_t output_index =
            ((batch * heads + head) * sequence_length + sequence) * head_dim +
            dimension;
        const std::ptrdiff_t input_base =
            static_cast<std::ptrdiff_t>(batch) * input_strides.batch +
            static_cast<std::ptrdiff_t>(head) * input_strides.head +
            static_cast<std::ptrdiff_t>(sequence) * input_strides.sequence;
        const std::size_t half = head_dim / 2;
        const std::size_t paired_dimension =
            dimension < half ? dimension + half : dimension - half;
        const float sign = dimension < half ? -1.0F : 1.0F;
        const std::size_t rope_batch = rope_batch_size == 1 ? 0 : batch;
        const std::ptrdiff_t cos_base =
            static_cast<std::ptrdiff_t>(rope_batch) * cos_strides.batch +
            static_cast<std::ptrdiff_t>(sequence) * cos_strides.sequence;
        const std::ptrdiff_t sin_base =
            static_cast<std::ptrdiff_t>(rope_batch) * sin_strides.batch +
            static_cast<std::ptrdiff_t>(sequence) * sin_strides.sequence;
        const float value = input[
            input_base + static_cast<std::ptrdiff_t>(dimension) *
                input_strides.dimension];
        const float paired = input[
            input_base + static_cast<std::ptrdiff_t>(paired_dimension) *
                input_strides.dimension];
        // Hugging Face evaluates the two multiplies in separate pointwise
        // kernels before the add.  Explicit round-to-nearest operations avoid
        // contraction into an FMA and preserve those FP32 rounding semantics.
        const float first = __fmul_rn(
            value,
            cos[cos_base + static_cast<std::ptrdiff_t>(dimension) *
                    cos_strides.dimension]);
        const float second = __fmul_rn(
            sign * paired,
            sin[sin_base + static_cast<std::ptrdiff_t>(dimension) *
                    sin_strides.dimension]);
        output[output_index] = __fadd_rn(first, second);
    }
}

}  // namespace

cudaError_t rope_cuda_fp32(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_output,
    float* key_output,
    const std::size_t batch_size,
    const std::size_t query_heads,
    const std::size_t key_heads,
    const std::size_t sequence_length,
    const std::size_t head_dim,
    const std::size_t rope_batch_size,
    const RopeStrides query_strides,
    const RopeStrides key_strides,
    const RopeEmbeddingStrides cos_strides,
    const RopeEmbeddingStrides sin_strides,
    cudaStream_t stream) {
    if (query == nullptr || key == nullptr || cos == nullptr || sin == nullptr ||
        query_output == nullptr || key_output == nullptr || batch_size == 0 ||
        query_heads == 0 || key_heads == 0 || sequence_length == 0 ||
        head_dim == 0 || head_dim % 2 != 0 ||
        (rope_batch_size != 1 && rope_batch_size != batch_size)) {
        return cudaErrorInvalidValue;
    }
    const std::size_t combined_heads = query_heads + key_heads;
    if (batch_size > std::numeric_limits<std::size_t>::max() / combined_heads ||
        batch_size * combined_heads >
            std::numeric_limits<std::size_t>::max() / sequence_length ||
        batch_size * combined_heads * sequence_length >
            std::numeric_limits<std::size_t>::max() / head_dim) {
        return cudaErrorInvalidValue;
    }
    const std::size_t total_elements =
        batch_size * combined_heads * sequence_length * head_dim;
    const std::size_t required_blocks =
        (total_elements + kBlockSize - 1) / kBlockSize;
    const unsigned int blocks = static_cast<unsigned int>(
        required_blocks > 65535 ? 65535 : required_blocks);
    rope_cuda_fp32_kernel<<<blocks, kBlockSize, 0, stream>>>(
        query, key, cos, sin, query_output, key_output, batch_size,
        query_heads, key_heads, sequence_length, head_dim, rope_batch_size,
        query_strides, key_strides, cos_strides, sin_strides, total_elements);
    return cudaGetLastError();
}

}  // namespace flux
