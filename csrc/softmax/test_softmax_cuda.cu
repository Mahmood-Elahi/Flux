#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1.0e-6F;
constexpr float kRelativeTolerance = 1.0e-5F;

struct CudaResult {
    std::vector<float> output;
    std::vector<float> input_after;
};

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void expect_true(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t element_count) {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), element_count * sizeof(float)),
            "cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    float* get() const { return data_; }

private:
    float* data_ = nullptr;
};

class CudaStream {
public:
    CudaStream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

std::vector<float> double_precision_reference(
    const std::vector<float>& input,
    const std::size_t num_rows,
    const std::size_t row_width) {
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t row_offset = row * row_width;
        double row_max = static_cast<double>(input[row_offset]);
        for (std::size_t column = 1; column < row_width; ++column) {
            row_max = std::max(
                row_max, static_cast<double>(input[row_offset + column]));
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

CudaResult run_cuda(
    const std::vector<float>& input,
    const std::size_t num_rows,
    const std::size_t row_width,
    cudaStream_t stream) {
    expect_true(
        num_rows > 0 && row_width > 0 && input.size() == num_rows * row_width,
        "invalid host test data");

    DeviceBuffer device_input(input.size());
    DeviceBuffer device_output(input.size());
    const std::size_t byte_count = input.size() * sizeof(float);

    check_cuda(
        cudaMemcpyAsync(
            device_input.get(),
            input.data(),
            byte_count,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync input host-to-device");
    check_cuda(
        flux::softmax_cuda_fp32(
            device_input.get(),
            device_output.get(),
            num_rows,
            row_width,
            stream),
        "softmax_cuda_fp32 kernel launch");

    CudaResult result{std::vector<float>(input.size()),
                      std::vector<float>(input.size())};
    check_cuda(
        cudaMemcpyAsync(
            result.output.data(),
            device_output.get(),
            byte_count,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync output device-to-host");
    check_cuda(
        cudaMemcpyAsync(
            result.input_after.data(),
            device_input.get(),
            byte_count,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync input preservation copy");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return result;
}

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& case_name) {
    expect_true(actual.size() == expected.size(), case_name + ": result sizes differ");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::abs(actual[index] - expected[index]);
        const float tolerance =
            kAbsoluteTolerance + kRelativeTolerance * std::abs(expected[index]);
        if (error > tolerance) {
            throw std::runtime_error(
                case_name + " element " + std::to_string(index) + ": expected " +
                std::to_string(expected[index]) + ", got " +
                std::to_string(actual[index]));
        }
    }
}

float maximum_absolute_error(
    const std::vector<float>& actual, const std::vector<float>& expected) {
    expect_true(actual.size() == expected.size(), "result sizes differ");
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum;
}

float run_case(
    const std::vector<float>& input,
    const std::size_t num_rows,
    const std::size_t row_width,
    cudaStream_t stream,
    const std::string& case_name) {
    const std::vector<float> expected =
        double_precision_reference(input, num_rows, row_width);
    const CudaResult actual = run_cuda(input, num_rows, row_width, stream);
    expect_close(actual.output, expected, case_name);
    expect_true(actual.input_after == input, case_name + ": input changed");

    for (std::size_t row = 0; row < num_rows; ++row) {
        float row_sum = 0.0F;
        for (std::size_t column = 0; column < row_width; ++column) {
            const float value = actual.output[row * row_width + column];
            expect_true(std::isfinite(value), case_name + ": output is not finite");
            expect_true(value >= 0.0F, case_name + ": output is negative");
            row_sum += value;
        }
        const float row_sum_error = std::abs(row_sum - 1.0F);
        expect_true(
            row_sum_error <= kAbsoluteTolerance + kRelativeTolerance,
            case_name + ": row does not sum to one");
    }
    return maximum_absolute_error(actual.output, expected);
}

float run_generated_case(
    const std::size_t num_rows,
    const std::size_t row_width,
    cudaStream_t stream,
    const std::string& case_name) {
    std::vector<float> input(num_rows * row_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float position = static_cast<float>(index);
        input[index] = 2.0F * std::sin(position * 0.173F) +
                       0.5F * std::cos(position * 0.071F);
    }
    return run_case(input, num_rows, row_width, stream, case_name);
}

void test_simple_known_values(cudaStream_t stream) {
    const std::vector<float> input{0.0F, std::log(2.0F), std::log(3.0F)};
    const CudaResult actual = run_cuda(input, 1, input.size(), stream);
    const std::vector<float> expected{1.0F / 6.0F, 2.0F / 6.0F, 3.0F / 6.0F};
    expect_close(actual.output, expected, "known values");
}

void test_width_one(cudaStream_t stream) {
    const std::vector<float> input{-1000.0F, 0.0F, 1000.0F};
    const CudaResult actual = run_cuda(input, 3, 1, stream);
    for (const float value : actual.output) {
        expect_true(value == 1.0F, "width-one row did not produce exactly one");
    }
}

void test_uniform_and_independent_rows(cudaStream_t stream) {
    const std::vector<float> input{
        5.0F, 5.0F, 5.0F, 5.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
        100.0F, -100.0F, -100.0F, -100.0F,
    };
    run_case(input, 3, 4, stream, "uniform and independent rows");
}

float test_required_widths(cudaStream_t stream) {
    const std::vector<std::size_t> widths{
        1,   3,   7,   31,  32,  33,   63,   64,   65,  127,
        128, 129, 255, 256, 257, 511,  512,  513,  1024, 2048,
        4096, 8192,
    };
    float maximum_error = 0.0F;
    for (const std::size_t width : widths) {
        maximum_error = std::max(
            maximum_error,
            run_generated_case(
                3, width, stream, "width-" + std::to_string(width)));
    }
    return maximum_error;
}

void test_large_magnitude_logits(cudaStream_t stream) {
    const std::vector<float> input{
        1000.0F, 1001.0F, 999.0F,
        -1000.0F, -1001.0F, -999.0F,
        1000.0F, 0.0F, -1000.0F,
    };
    run_case(input, 3, 3, stream, "large positive and negative logits");
}

void test_shift_invariance(cudaStream_t stream) {
    constexpr std::size_t num_rows = 4;
    constexpr std::size_t row_width = 65;
    std::vector<float> input(num_rows * row_width);
    std::vector<float> shifted(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.137F);
        shifted[index] = input[index] + 17.25F;
    }
    const CudaResult output = run_cuda(input, num_rows, row_width, stream);
    const CudaResult shifted_output =
        run_cuda(shifted, num_rows, row_width, stream);
    expect_close(shifted_output.output, output.output, "shift invariance");
}

void test_deterministic(cudaStream_t stream) {
    constexpr std::size_t num_rows = 3;
    constexpr std::size_t row_width = 513;
    std::vector<float> input(num_rows * row_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::cos(static_cast<float>(index) * 0.091F);
    }
    const CudaResult first = run_cuda(input, num_rows, row_width, stream);
    const CudaResult second = run_cuda(input, num_rows, row_width, stream);
    expect_true(
        std::memcmp(
            first.output.data(),
            second.output.data(),
            first.output.size() * sizeof(float)) == 0,
        "repeated CUDA execution produced different output bytes");
}

void test_launcher_validation() {
    float* pointer = reinterpret_cast<float*>(1);
    expect_true(
        flux::softmax_cuda_fp32(nullptr, pointer, 1, 1, nullptr) ==
            cudaErrorInvalidValue,
        "null input was accepted");
    expect_true(
        flux::softmax_cuda_fp32(pointer, nullptr, 1, 1, nullptr) ==
            cudaErrorInvalidValue,
        "null output was accepted");
    expect_true(
        flux::softmax_cuda_fp32(pointer, pointer, 0, 1, nullptr) ==
            cudaErrorInvalidValue,
        "zero rows was accepted");
    expect_true(
        flux::softmax_cuda_fp32(pointer, pointer, 1, 0, nullptr) ==
            cudaErrorInvalidValue,
        "zero row width was accepted");
    if (sizeof(std::size_t) > sizeof(unsigned int)) {
        const std::size_t too_many_rows =
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) + 1;
        expect_true(
            flux::softmax_cuda_fp32(pointer, pointer, too_many_rows, 1, nullptr) ==
                cudaErrorInvalidValue,
            "row count larger than the CUDA grid was accepted");
    }
    const std::size_t too_many_rows =
        std::numeric_limits<std::size_t>::max() / 2 + 1;
    expect_true(
        flux::softmax_cuda_fp32(pointer, pointer, too_many_rows, 2, nullptr) ==
            cudaErrorInvalidValue,
        "overflowing element count was accepted");
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        expect_true(device_count > 0, "no CUDA device is available");
        check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "CUDA device: " << properties.name << '\n';
        std::cout << "Compute capability: " << properties.major << '.'
                  << properties.minor << '\n';

        test_launcher_validation();
        test_simple_known_values(nullptr);
        test_width_one(nullptr);
        run_generated_case(3, 33, nullptr, "default-stream width-33");
        std::cout << "Default-stream softmax test passed.\n";

        CudaStream stream;
        test_uniform_and_independent_rows(stream.get());
        const float required_widths_max_error = test_required_widths(stream.get());
        test_large_magnitude_logits(stream.get());
        test_shift_invariance(stream.get());
        test_deterministic(stream.get());
        std::cout << "Non-default-stream softmax tests passed.\n";

        std::cout << std::setprecision(9)
                  << "Required-widths maximum absolute error: "
                  << required_widths_max_error << '\n';
    } catch (const std::exception& error) {
        std::cerr << "CUDA softmax test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All standalone CUDA softmax tests passed.\n";
    return 0;
}
