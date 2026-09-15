#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace flux {

// Select the first vocabulary index containing the maximum FP32 value.
cudaError_t greedy_argmax_cuda_fp32(
    const float* logits,
    std::int64_t* token,
    std::int64_t vocabulary_size,
    cudaStream_t stream);

// Decode-graph epilogue: select the next token, append it to the fixed
// generation buffer, make it the next graph input, and advance decode state.
cudaError_t greedy_argmax_update_cuda_fp32(
    const float* logits,
    std::int64_t* input_token,
    std::int64_t* generated_tokens,
    std::int64_t* generation_step,
    std::int64_t generated_capacity,
    std::int64_t* position,
    const std::int64_t* attention_length,
    std::int64_t vocabulary_size,
    cudaStream_t stream);

}  // namespace flux
