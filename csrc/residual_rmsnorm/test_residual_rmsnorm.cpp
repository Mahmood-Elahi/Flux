#include "residual_rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

struct Outputs {
    std::vector<float> residual;
    std::vector<float> norm;
};

void expect_true(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(
    const float actual, const float expected, const std::string& message) {
    const float tolerance =
        kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual));
    }
}

Outputs reference(
    const std::vector<float>& hidden,
    const std::vector<float>& residual,
    const std::vector<float>& weight,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon) {
    Outputs outputs{std::vector<float>(hidden.size()),
                    std::vector<float>(hidden.size())};
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t row_offset = row * hidden_size;
        float sum_squares = 0.0F;
        for (std::size_t column = 0; column < hidden_size; ++column) {
            const std::size_t index = row_offset + column;
            const float value = hidden[index] + residual[index];
            outputs.residual[index] = value;
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0F /
            std::sqrt(sum_squares / static_cast<float>(hidden_size) + epsilon);
        for (std::size_t column = 0; column < hidden_size; ++column) {
            const std::size_t index = row_offset + column;
            outputs.norm[index] =
                outputs.residual[index] * inv_rms * weight[column];
        }
    }
    return outputs;
}

void expect_outputs(
    const Outputs& actual,
    const Outputs& expected,
    const std::string& case_name) {
    expect_true(
        actual.residual.size() == expected.residual.size(),
        case_name + ": residual output sizes differ");
    expect_true(
        actual.norm.size() == expected.norm.size(),
        case_name + ": normalized output sizes differ");
    for (std::size_t index = 0; index < actual.residual.size(); ++index) {
        expect_true(
            actual.residual[index] == expected.residual[index],
            case_name + ": residual element " + std::to_string(index) +
                " does not equal direct FP32 addition");
        expect_near(
            actual.norm[index],
            expected.norm[index],
            case_name + ": normalized element " + std::to_string(index));
    }
}

void fill_inputs(
    std::vector<float>& hidden,
    std::vector<float>& residual,
    std::vector<float>& weight) {
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        const float position = static_cast<float>(index);
        hidden[index] = std::sin(position * 0.013F) +
                        0.25F * std::cos(position * 0.031F);
        residual[index] = std::cos(position * 0.017F) -
                          0.125F * std::sin(position * 0.023F);
    }
    for (std::size_t index = 0; index < weight.size(); ++index) {
        weight[index] = 0.25F + 1.5F * static_cast<float>(index + 1) /
                                     static_cast<float>(weight.size());
    }
}

Outputs run_case(
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    const std::string& case_name) {
    std::vector<float> hidden(num_rows * hidden_size);
    std::vector<float> residual(hidden.size());
    std::vector<float> weight(hidden_size);
    fill_inputs(hidden, residual, weight);
    const std::vector<float> hidden_before = hidden;
    const std::vector<float> residual_before = residual;
    const std::vector<float> weight_before = weight;
    Outputs actual{std::vector<float>(hidden.size()),
                   std::vector<float>(hidden.size())};
    const Outputs expected =
        reference(hidden, residual, weight, num_rows, hidden_size, epsilon);

    flux::residual_rmsnorm_fp32(
        hidden.data(),
        residual.data(),
        weight.data(),
        actual.residual.data(),
        actual.norm.data(),
        num_rows,
        hidden_size,
        epsilon);

    expect_outputs(actual, expected, case_name);
    expect_true(hidden == hidden_before, case_name + ": hidden input changed");
    expect_true(
        residual == residual_before, case_name + ": residual input changed");
    expect_true(weight == weight_before, case_name + ": weight input changed");
    return actual;
}

void test_hand_verifiable_one_row() {
    const std::vector<float> hidden{1.0F, 3.0F};
    const std::vector<float> residual{2.0F, 1.0F};
    const std::vector<float> weight{2.0F, 0.5F};
    Outputs output{std::vector<float>(2), std::vector<float>(2)};

    flux::residual_rmsnorm_fp32(
        hidden.data(),
        residual.data(),
        weight.data(),
        output.residual.data(),
        output.norm.data(),
        1,
        2,
        0.0F);

    expect_true(output.residual[0] == 3.0F, "residual element 0 is wrong");
    expect_true(output.residual[1] == 4.0F, "residual element 1 is wrong");
    const float denominator = std::sqrt(12.5F);
    expect_near(output.norm[0], 6.0F / denominator, "normalized element 0");
    expect_near(output.norm[1], 2.0F / denominator, "normalized element 1");
}

void test_output_bounds() {
    const std::vector<float> hidden{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<float> residual{-0.5F, 0.5F, 1.0F, -1.0F, 2.0F, -2.0F};
    const std::vector<float> weight{0.5F, 1.0F, 1.5F};
    constexpr float guard = -12345.0F;
    std::vector<float> residual_out(hidden.size() + 2, guard);
    std::vector<float> norm_out(hidden.size() + 2, guard);

    flux::residual_rmsnorm_fp32(
        hidden.data(),
        residual.data(),
        weight.data(),
        residual_out.data() + 1,
        norm_out.data() + 1,
        2,
        3,
        1.0e-5F);

    expect_true(
        residual_out.front() == guard && residual_out.back() == guard,
        "wrote outside residual output buffer");
    expect_true(
        norm_out.front() == guard && norm_out.back() == guard,
        "wrote outside normalized output buffer");
}

void test_required_hidden_sizes() {
    const std::vector<std::size_t> hidden_sizes{1, 7, 63, 127, 575, 576, 577, 1024};
    for (const std::size_t hidden_size : hidden_sizes) {
        run_case(
            3,
            hidden_size,
            1.0e-5F,
            "hidden-size-" + std::to_string(hidden_size));
    }
}

void test_representative_shapes() {
    run_case(1, 576, 1.0e-5F, "shape-(576)");
    run_case(4, 576, 1.0e-5F, "shape-(4,576)");
    run_case(64, 576, 1.0e-5F, "shape-(2,32,576)");
}

void test_epsilon_values() {
    const std::vector<float> epsilons{0.0F, 1.0e-6F, 1.0e-5F, 1.0e-4F};
    for (const float epsilon : epsilons) {
        run_case(2, 127, epsilon, "epsilon-" + std::to_string(epsilon));
    }
}

void test_large_workload() {
    run_case(8192, 576, 1.0e-5F, "large-(8192,576)");
}

void test_deterministic() {
    constexpr std::size_t num_rows = 3;
    constexpr std::size_t hidden_size = 577;
    std::vector<float> hidden(num_rows * hidden_size);
    std::vector<float> residual(hidden.size());
    std::vector<float> weight(hidden_size);
    fill_inputs(hidden, residual, weight);
    Outputs first{std::vector<float>(hidden.size()), std::vector<float>(hidden.size())};
    Outputs second{std::vector<float>(hidden.size()), std::vector<float>(hidden.size())};

    flux::residual_rmsnorm_fp32(
        hidden.data(), residual.data(), weight.data(), first.residual.data(),
        first.norm.data(), num_rows, hidden_size, 1.0e-5F);
    flux::residual_rmsnorm_fp32(
        hidden.data(), residual.data(), weight.data(), second.residual.data(),
        second.norm.data(), num_rows, hidden_size, 1.0e-5F);

    expect_true(
        std::memcmp(
            first.residual.data(),
            second.residual.data(),
            first.residual.size() * sizeof(float)) == 0,
        "repeated calls produced different residual output bytes");
    expect_true(
        std::memcmp(
            first.norm.data(),
            second.norm.data(),
            first.norm.size() * sizeof(float)) == 0,
        "repeated calls produced different normalized output bytes");
}

void expect_invalid_argument(
    const float* hidden,
    const float* residual,
    const float* weight,
    float* residual_out,
    float* norm_out,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    const std::string& message) {
    try {
        flux::residual_rmsnorm_fp32(
            hidden,
            residual,
            weight,
            residual_out,
            norm_out,
            num_rows,
            hidden_size,
            epsilon);
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

    expect_invalid_argument(
        nullptr, &input, &input, &output, &output, 1, 1, 0.0F,
        "null hidden was accepted");
    expect_invalid_argument(
        &input, nullptr, &input, &output, &output, 1, 1, 0.0F,
        "null residual was accepted");
    expect_invalid_argument(
        &input, &input, nullptr, &output, &output, 1, 1, 0.0F,
        "null weight was accepted");
    expect_invalid_argument(
        &input, &input, &input, nullptr, &output, 1, 1, 0.0F,
        "null residual output was accepted");
    expect_invalid_argument(
        &input, &input, &input, &output, nullptr, 1, 1, 0.0F,
        "null normalized output was accepted");
    expect_invalid_argument(
        &input, &input, &input, &output, &output, 0, 1, 0.0F,
        "zero rows was accepted");
    expect_invalid_argument(
        &input, &input, &input, &output, &output, 1, 0, 0.0F,
        "zero hidden size was accepted");
    expect_invalid_argument(
        &input, &input, &input, &output, &output, 1, 1, -1.0e-5F,
        "negative epsilon was accepted");
    expect_invalid_argument(
        &input, &input, &input, &output, &output, too_many_rows, 2, 0.0F,
        "overflowing element count was accepted");
}

}  // namespace

int main() {
    try {
        test_hand_verifiable_one_row();
        test_output_bounds();
        test_required_hidden_sizes();
        test_representative_shapes();
        test_epsilon_values();
        test_large_workload();
        test_deterministic();
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << "Residual RMSNorm test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All native residual RMSNorm tests passed.\n";
    return 0;
}
