#pragma once

#include <cstddef>

namespace flux {

// Applies SwiGLU to contiguous rows packed as [gate; up].
void packed_swiglu_fp32(
    const float* packed,
    float* output,
    std::size_t num_rows,
    std::size_t intermediate_size);

}  // namespace flux
