#include "packed_swiglu.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace flux {

void packed_swiglu_fp32(
    const float* packed,
    float* output,
    const std::size_t num_rows,
    const std::size_t intermediate_size) {
    if (packed == nullptr || output == nullptr) {
        throw std::invalid_argument("packed and output must not be null");
    }
    if (num_rows == 0 || intermediate_size == 0) {
        throw std::invalid_argument("num_rows and intermediate_size must be positive");
    }
    if (intermediate_size > std::numeric_limits<std::size_t>::max() / 2 ||
        num_rows > std::numeric_limits<std::size_t>::max() / (2 * intermediate_size)) {
        throw std::invalid_argument("packed tensor element count overflows size_t");
    }

    const std::size_t packed_size = 2 * intermediate_size;
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t packed_offset = row * packed_size;
        const std::size_t output_offset = row * intermediate_size;
        for (std::size_t column = 0; column < intermediate_size; ++column) {
            const float gate = packed[packed_offset + column];
            const float up = packed[packed_offset + intermediate_size + column];
            output[output_offset + column] =
                (gate / (1.0F + std::exp(-gate))) * up;
        }
    }
}

}  // namespace flux
