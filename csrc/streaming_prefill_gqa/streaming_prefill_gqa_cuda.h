#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace flux {

enum class StreamingPrefillGQAVariant : int {
    kQueryTile8 = 0,
    kQueryTile32 = 1,
    kQueryTile128 = 2,
    kSplit2QueryTile128 = 3,
    kSplit4QueryTile128 = 4,
};

StreamingPrefillGQAVariant select_streaming_prefill_gqa_variant(
    std::size_t sequence_length);

// SmolLM2-135M production specialization. Query is [9, sequence, 64], K/V
// are compact [3, capacity, 64], and output is [9, sequence, 64].
cudaError_t streaming_prefill_gqa_cuda_fp32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* workspace,
    float scale,
    std::size_t sequence_length,
    std::size_t capacity,
    StreamingPrefillGQAVariant variant,
    cudaStream_t stream);

std::size_t streaming_prefill_gqa_workspace_bytes(
    std::size_t sequence_length,
    StreamingPrefillGQAVariant variant);

}  // namespace flux
