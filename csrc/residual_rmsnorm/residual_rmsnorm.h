#pragma once

#include <cstddef>

namespace flux {

// Adds two FP32 inputs and applies RMSNorm independently to each contiguous
// row of the sum. The hidden_size-element weight vector is shared by every
// row. norm_out is written first in the public output order, followed by the
// preserved pre-normalization residual_out. All buffers must be distinct.
void residual_rmsnorm_fp32(
    const float* hidden,
    const float* residual,
    const float* weight,
    float* norm_out,
    float* residual_out,
    std::size_t num_rows,
    std::size_t hidden_size,
    float epsilon);

}  // namespace flux
