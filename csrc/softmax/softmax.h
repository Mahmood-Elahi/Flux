#pragma once

#include <cstddef>

namespace flux {

// Applies numerically stable FP32 softmax independently to each contiguous
// row. Input and output each contain num_rows * row_width elements.
void softmax_fp32(
    const float* input,
    float* output,
    std::size_t num_rows,
    std::size_t row_width);

}  // namespace flux
