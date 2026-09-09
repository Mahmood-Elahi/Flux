#include "rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
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

void test_hand_verifiable_one_row() {
    const std::vector<float> input{3.0F, 4.0F};
    const std::vector<float> weight{2.0F, 0.5F};
    std::vector<float> output(input.size());

    flux::rmsnorm_fp32(input.data(), weight.data(), output.data(), 1, 2, 0.0F);

    // mean([3^2, 4^2]) = 12.5, calculated independently of the implementation.
    const float denominator = std::sqrt(12.5F);
    expect_near(output[0], 6.0F / denominator, "hand-verifiable element 0");
    expect_near(output[1], 2.0F / denominator, "hand-verifiable element 1");
}

void test_multiple_rows_and_output_count() {
    const std::vector<float> input{1.0F, 2.0F, 2.0F, 0.0F, 3.0F, 4.0F};
    const std::vector<float> weight{0.5F, 1.0F, 1.5F};
    constexpr float guard = -12345.0F;
    std::vector<float> guarded_output(input.size() + 2, guard);

    flux::rmsnorm_fp32(
        input.data(), weight.data(), guarded_output.data() + 1, 2, 3, 0.0F);

    expect_true(guarded_output.front() == guard, "wrote before output buffer");
    expect_true(guarded_output.back() == guard, "wrote after output buffer");

    const float first_inv_rms = 1.0F / std::sqrt(3.0F);
    const float second_inv_rms = 1.0F / std::sqrt(25.0F / 3.0F);
    const std::vector<float> expected{
        0.5F * first_inv_rms,
        2.0F * first_inv_rms,
        3.0F * first_inv_rms,
        0.0F,
        3.0F * second_inv_rms,
        6.0F * second_inv_rms,
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            guarded_output[index + 1],
            expected[index],
            "multiple-row element " + std::to_string(index));
    }
}

void test_smollm2_hidden_size_and_nontrivial_weights() {
    constexpr std::size_t hidden_size = 576;
    constexpr float epsilon = 1.0e-5F;
    std::vector<float> input(hidden_size, 1.0F);
    std::vector<float> weight(hidden_size);
    std::vector<float> output(hidden_size);

    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] = 0.25F + 1.5F * static_cast<float>(index) /
                                     static_cast<float>(hidden_size - 1);
    }

    flux::rmsnorm_fp32(
        input.data(), weight.data(), output.data(), 1, hidden_size, epsilon);

    const float expected_scale = 1.0F / std::sqrt(1.0F + epsilon);
    for (std::size_t index = 0; index < hidden_size; ++index) {
        expect_near(
            output[index],
            weight[index] * expected_scale,
            "hidden-size-576 element " + std::to_string(index));
    }
}

void test_zero_input() {
    constexpr std::size_t hidden_size = 576;
    std::vector<float> input(2 * hidden_size, 0.0F);
    std::vector<float> weight(hidden_size);
    std::vector<float> output(input.size(), 1.0F);
    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] = 0.5F + static_cast<float>(index) / 1000.0F;
    }

    flux::rmsnorm_fp32(
        input.data(), weight.data(), output.data(), 2, hidden_size, 1.0e-5F);

    expect_true(
        std::all_of(output.begin(), output.end(), [](const float value) {
            return value == 0.0F;
        }),
        "zero input did not produce exact zero output");
}

void test_deterministic() {
    constexpr std::size_t hidden_size = 17;
    constexpr std::size_t num_rows = 3;
    std::vector<float> input(num_rows * hidden_size);
    std::vector<float> weight(hidden_size);
    std::vector<float> first(input.size());
    std::vector<float> second(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.37F);
    }
    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] = 0.75F + static_cast<float>(index) * 0.02F;
    }

    flux::rmsnorm_fp32(
        input.data(), weight.data(), first.data(), num_rows, hidden_size, 1.0e-5F);
    flux::rmsnorm_fp32(
        input.data(), weight.data(), second.data(), num_rows, hidden_size, 1.0e-5F);

    expect_true(
        std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0,
        "repeated calls produced different output bytes");
}

void expect_invalid_argument(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    const std::string& message) {
    try {
        flux::rmsnorm_fp32(
            input, weight, output, num_rows, hidden_size, epsilon);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_validation() {
    float input = 1.0F;
    float weight = 1.0F;
    float output = 0.0F;

    expect_invalid_argument(
        nullptr, &weight, &output, 1, 1, 0.0F,
        "null input was accepted");
    expect_invalid_argument(
        &input, nullptr, &output, 1, 1, 0.0F,
        "null weight was accepted");
    expect_invalid_argument(
        &input, &weight, nullptr, 1, 1, 0.0F,
        "null output was accepted");
    expect_invalid_argument(
        &input, &weight, &output, 0, 1, 0.0F,
        "zero rows was accepted");
    expect_invalid_argument(
        &input, &weight, &output, 1, 0, 0.0F,
        "zero hidden size was accepted");
    expect_invalid_argument(
        &input, &weight, &output, 1, 1, -1.0e-5F,
        "negative epsilon was accepted");
}

}  // namespace

int main() {
    try {
        test_hand_verifiable_one_row();
        test_multiple_rows_and_output_count();
        test_smollm2_hidden_size_and_nontrivial_weights();
        test_zero_input();
        test_deterministic();
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << "RMSNorm test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All native RMSNorm tests passed.\n";
    return 0;
}
