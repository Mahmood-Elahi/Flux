#include "softmax.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1.0e-6F;
constexpr float kRelativeTolerance = 1.0e-5F;

void expect_true(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(const float actual, const float expected, const std::string& message) {
    const float tolerance =
        kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual));
    }
}

std::vector<float> double_precision_reference(
    const std::vector<float>& input,
    const std::size_t num_rows,
    const std::size_t row_width) {
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t row_offset = row * row_width;
        double row_max = static_cast<double>(input[row_offset]);
        for (std::size_t column = 1; column < row_width; ++column) {
            const double value = static_cast<double>(input[row_offset + column]);
            if (value > row_max) {
                row_max = value;
            }
        }

        double exponential_sum = 0.0;
        for (std::size_t column = 0; column < row_width; ++column) {
            const std::size_t index = row_offset + column;
            const double exponential =
                std::exp(static_cast<double>(input[index]) - row_max);
            output[index] = static_cast<float>(exponential);
            exponential_sum += exponential;
        }
        for (std::size_t column = 0; column < row_width; ++column) {
            const std::size_t index = row_offset + column;
            output[index] =
                static_cast<float>(static_cast<double>(output[index]) /
                                   exponential_sum);
        }
    }
    return output;
}

std::vector<float> run_case(
    const std::size_t num_rows,
    const std::size_t row_width,
    const std::string& case_name) {
    std::vector<float> input(num_rows * row_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float position = static_cast<float>(index);
        input[index] = 2.0F * std::sin(position * 0.173F) +
                       0.5F * std::cos(position * 0.071F);
    }
    const std::vector<float> input_before = input;
    const std::vector<float> expected =
        double_precision_reference(input, num_rows, row_width);
    constexpr float guard = -12345.0F;
    std::vector<float> guarded_output(input.size() + 2, guard);

    flux::softmax_fp32(
        input.data(), guarded_output.data() + 1, num_rows, row_width);

    expect_true(guarded_output.front() == guard, case_name + ": wrote before output");
    expect_true(guarded_output.back() == guard, case_name + ": wrote after output");
    expect_true(input == input_before, case_name + ": input changed");
    for (std::size_t row = 0; row < num_rows; ++row) {
        float row_sum = 0.0F;
        for (std::size_t column = 0; column < row_width; ++column) {
            const std::size_t index = row * row_width + column;
            const float actual = guarded_output[index + 1];
            expect_true(std::isfinite(actual), case_name + ": output is not finite");
            expect_true(actual >= 0.0F, case_name + ": output is negative");
            expect_near(actual, expected[index], case_name + ": element mismatch");
            row_sum += actual;
        }
        expect_near(row_sum, 1.0F, case_name + ": row does not sum to one");
    }

    return std::vector<float>(guarded_output.begin() + 1, guarded_output.end() - 1);
}

void test_simple_known_values() {
    const std::vector<float> input{0.0F, std::log(2.0F), std::log(3.0F)};
    std::vector<float> output(input.size());

    flux::softmax_fp32(input.data(), output.data(), 1, input.size());

    expect_near(output[0], 1.0F / 6.0F, "known value 0");
    expect_near(output[1], 2.0F / 6.0F, "known value 1");
    expect_near(output[2], 3.0F / 6.0F, "known value 2");
}

void test_width_one() {
    const std::vector<float> input{-1000.0F, 0.0F, 1000.0F};
    std::vector<float> output(input.size());

    flux::softmax_fp32(input.data(), output.data(), 3, 1);

    for (const float value : output) {
        expect_true(value == 1.0F, "width-one row did not produce exactly one");
    }
}

void test_uniform_and_independent_rows() {
    const std::vector<float> input{
        5.0F, 5.0F, 5.0F, 5.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
        100.0F, -100.0F, -100.0F, -100.0F,
    };
    std::vector<float> output(input.size());

    flux::softmax_fp32(input.data(), output.data(), 3, 4);

    for (std::size_t index = 0; index < 8; ++index) {
        expect_near(output[index], 0.25F, "uniform row element");
    }
    expect_near(output[8], 1.0F, "independent peaked row maximum");
    for (std::size_t index = 9; index < 12; ++index) {
        expect_true(output[index] >= 0.0F, "peaked row output is negative");
        expect_near(output[index], 0.0F, "independent peaked row tail");
    }
}

void test_required_widths() {
    const std::vector<std::size_t> widths{
        1, 3, 7, 31, 32, 33, 63, 64, 65,
        127, 128, 129, 257, 511, 1024, 2048, 8192,
    };
    for (const std::size_t width : widths) {
        run_case(3, width, "width-" + std::to_string(width));
    }
}

void test_large_magnitude_logits() {
    const std::vector<float> input{
        1000.0F, 1001.0F, 999.0F,
        -1000.0F, -1001.0F, -999.0F,
        1000.0F, 0.0F, -1000.0F,
    };
    const std::vector<float> expected = double_precision_reference(input, 3, 3);
    std::vector<float> output(input.size());

    flux::softmax_fp32(input.data(), output.data(), 3, 3);

    for (std::size_t index = 0; index < output.size(); ++index) {
        expect_true(std::isfinite(output[index]), "large-logit output is not finite");
        expect_near(output[index], expected[index], "large-logit element");
    }
}

void test_shift_invariance() {
    constexpr std::size_t num_rows = 4;
    constexpr std::size_t row_width = 65;
    std::vector<float> input(num_rows * row_width);
    std::vector<float> shifted(input.size());
    std::vector<float> output(input.size());
    std::vector<float> shifted_output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.137F);
        shifted[index] = input[index] + 17.25F;
    }

    flux::softmax_fp32(input.data(), output.data(), num_rows, row_width);
    flux::softmax_fp32(
        shifted.data(), shifted_output.data(), num_rows, row_width);

    for (std::size_t index = 0; index < output.size(); ++index) {
        expect_near(shifted_output[index], output[index], "shift invariance");
    }
}

void test_deterministic() {
    constexpr std::size_t num_rows = 3;
    constexpr std::size_t row_width = 129;
    std::vector<float> input(num_rows * row_width);
    std::vector<float> first(input.size());
    std::vector<float> second(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::cos(static_cast<float>(index) * 0.091F);
    }

    flux::softmax_fp32(input.data(), first.data(), num_rows, row_width);
    flux::softmax_fp32(input.data(), second.data(), num_rows, row_width);

    expect_true(
        std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0,
        "repeated calls produced different output bytes");
}

void expect_invalid_argument(
    const float* input,
    float* output,
    const std::size_t num_rows,
    const std::size_t row_width,
    const std::string& message) {
    try {
        flux::softmax_fp32(input, output, num_rows, row_width);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_validation() {
    float input = 1.0F;
    float output = 0.0F;
    const std::size_t too_many_rows =
        std::numeric_limits<std::size_t>::max() / 2 + 1;

    expect_invalid_argument(nullptr, &output, 1, 1, "null input was accepted");
    expect_invalid_argument(&input, nullptr, 1, 1, "null output was accepted");
    expect_invalid_argument(&input, &output, 0, 1, "zero rows was accepted");
    expect_invalid_argument(&input, &output, 1, 0, "zero row width was accepted");
    expect_invalid_argument(
        &input, &output, too_many_rows, 2,
        "overflowing element count was accepted");
}

}  // namespace

int main() {
    try {
        test_simple_known_values();
        test_width_one();
        test_uniform_and_independent_rows();
        test_required_widths();
        test_large_magnitude_logits();
        test_shift_invariance();
        test_deterministic();
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << "Softmax test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All native softmax tests passed.\n";
    return 0;
}
