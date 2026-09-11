#include "softmax.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace flux {

void softmax_fp32(
    const float* input,
    float* output,
    const std::size_t num_rows,
    const std::size_t row_width) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("input and output must not be null");
    }
    if (num_rows == 0) {
        throw std::invalid_argument("num_rows must be positive");
    }
    if (row_width == 0) {
        throw std::invalid_argument("row_width must be positive");
    }
    if (num_rows > std::numeric_limits<std::size_t>::max() / row_width) {
        throw std::invalid_argument("num_rows * row_width overflows size_t");
    }

    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t row_offset = row * row_width;
        float row_max = input[row_offset];
        for (std::size_t column = 1; column < row_width; ++column) {
            const float value = input[row_offset + column];
            if (value > row_max) {
                row_max = value;
            }
        }

        float exponential_sum = 0.0F;
        for (std::size_t column = 0; column < row_width; ++column) {
            const std::size_t index = row_offset + column;
            const float exponential = std::exp(input[index] - row_max);
            output[index] = exponential;
            exponential_sum += exponential;
        }

        for (std::size_t column = 0; column < row_width; ++column) {
            const std::size_t index = row_offset + column;
            output[index] /= exponential_sum;
        }
    }
}

}  // namespace flux
