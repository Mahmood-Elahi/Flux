#pragma once

#include <cstddef>

namespace flux {

// Applies FP32 RMSNorm independently to each contiguous input row. The
// hidden_size-element weight vector is shared by every row.
void rmsnorm_fp32(
    const float* input,
    const float* weight,
    float* output,
    std::size_t num_rows,
    std::size_t hidden_size,
    float epsilon);

}  // namespace flux
