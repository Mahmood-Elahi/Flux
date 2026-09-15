#include "greedy_generation_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace flux {
namespace {

constexpr int kGreedyThreads = 256;

struct ArgMaxPair {
    float value;
    std::int64_t index;
    bool valid;
};

__device__ __forceinline__ bool pair_precedes(
    const ArgMaxPair& candidate,
    const ArgMaxPair& incumbent) {
    if (!candidate.valid) {
        return false;
    }
    if (!incumbent.valid) {
        return true;
    }
    const bool candidate_nan = isnan(candidate.value);
    const bool incumbent_nan = isnan(incumbent.value);
    if (candidate_nan != incumbent_nan) {
        // PyTorch's CUDA argmax reduction propagates NaN as the maximum.
        return candidate_nan;
    }
    return candidate.value > incumbent.value ||
        (candidate.value == incumbent.value &&
         candidate.index < incumbent.index);
}

template <bool UpdateState>
__global__ void greedy_argmax_kernel(
    const float* logits,
    std::int64_t* input_token,
    std::int64_t* generated_tokens,
    std::int64_t* generation_step,
    const std::int64_t generated_capacity,
    std::int64_t* position,
    const std::int64_t* attention_length,
    const std::int64_t vocabulary_size) {
    __shared__ float values[kGreedyThreads];
    __shared__ std::int64_t indices[kGreedyThreads];
    __shared__ int valid[kGreedyThreads];

    ArgMaxPair best{0.0F, 0, false};
    for (std::int64_t index = threadIdx.x; index < vocabulary_size;
         index += blockDim.x) {
        const ArgMaxPair candidate{logits[index], index, true};
        if (pair_precedes(candidate, best)) {
            best = candidate;
        }
    }
    values[threadIdx.x] = best.value;
    indices[threadIdx.x] = best.index;
    valid[threadIdx.x] = best.valid ? 1 : 0;
    __syncthreads();

    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            const ArgMaxPair incumbent{
                values[threadIdx.x], indices[threadIdx.x],
                valid[threadIdx.x] != 0};
            const ArgMaxPair candidate{
                values[threadIdx.x + width], indices[threadIdx.x + width],
                valid[threadIdx.x + width] != 0};
            if (pair_precedes(candidate, incumbent)) {
                values[threadIdx.x] = candidate.value;
                indices[threadIdx.x] = candidate.index;
                valid[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const std::int64_t selected = indices[0];
        input_token[0] = selected;
        if constexpr (UpdateState) {
            const std::int64_t step = *generation_step;
            if (step >= 0 && step < generated_capacity) {
                generated_tokens[step] = selected;
            }
            *generation_step = step + 1;
            *position = *attention_length;
        }
    }
}

}  // namespace

cudaError_t greedy_argmax_cuda_fp32(
    const float* logits,
    std::int64_t* token,
    const std::int64_t vocabulary_size,
    cudaStream_t stream) {
    if (logits == nullptr || token == nullptr || vocabulary_size <= 0) {
        return cudaErrorInvalidValue;
    }
    greedy_argmax_kernel<false><<<1, kGreedyThreads, 0, stream>>>(
        logits, token, nullptr, nullptr, 0, nullptr, nullptr, vocabulary_size);
    return cudaGetLastError();
}

cudaError_t greedy_argmax_update_cuda_fp32(
    const float* logits,
    std::int64_t* input_token,
    std::int64_t* generated_tokens,
    std::int64_t* generation_step,
    const std::int64_t generated_capacity,
    std::int64_t* position,
    const std::int64_t* attention_length,
    const std::int64_t vocabulary_size,
    cudaStream_t stream) {
    if (logits == nullptr || input_token == nullptr ||
        generated_tokens == nullptr || generation_step == nullptr ||
        generated_capacity <= 0 || position == nullptr ||
        attention_length == nullptr || vocabulary_size <= 0) {
        return cudaErrorInvalidValue;
    }
    greedy_argmax_kernel<true><<<1, kGreedyThreads, 0, stream>>>(
        logits, input_token, generated_tokens, generation_step,
        generated_capacity, position, attention_length, vocabulary_size);
    return cudaGetLastError();
}

}  // namespace flux
