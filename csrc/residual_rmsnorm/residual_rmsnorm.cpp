#include "residual_rmsnorm.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace flux {

void residual_rmsnorm_fp32(
    const float* hidden,
    const float* residual,
    const float* weight,
    float* residual_out,
    float* norm_out,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon) {
    if (hidden == nullptr || residual == nullptr || weight == nullptr ||
        residual_out == nullptr || norm_out == nullptr) {
        throw std::invalid_argument(
            "hidden, residual, weight, residual_out, and norm_out must not be null");
    }
    if (num_rows == 0) {
        throw std::invalid_argument("num_rows must be positive");
    }
    if (hidden_size == 0) {
        throw std::invalid_argument("hidden_size must be positive");
    }
    if (num_rows > std::numeric_limits<std::size_t>::max() / hidden_size) {
        throw std::invalid_argument("num_rows * hidden_size overflows size_t");
    }
    if (epsilon < 0.0F) {
        throw std::invalid_argument("epsilon must be non-negative");
    }

    const float hidden_size_as_float = static_cast<float>(hidden_size);
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t row_offset = row * hidden_size;
        float sum_squares = 0.0F;
        for (std::size_t column = 0; column < hidden_size; ++column) {
            const std::size_t index = row_offset + column;
            const float value = hidden[index] + residual[index];
            residual_out[index] = value;
            sum_squares += value * value;
        }

        const float mean_square = sum_squares / hidden_size_as_float;
        const float inv_rms = 1.0F / std::sqrt(mean_square + epsilon);
        for (std::size_t column = 0; column < hidden_size; ++column) {
            const std::size_t index = row_offset + column;
            norm_out[index] = residual_out[index] * inv_rms * weight[column];
        }
    }
}

}  // namespace flux
