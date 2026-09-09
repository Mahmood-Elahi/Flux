#include "rmsnorm_cuda.h"

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

constexpr float kAbsoluteTolerance = 2.0e-6F;
constexpr float kRelativeTolerance = 1.0e-5F;

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

std::vector<float> cpu_reference(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon) {
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t row_offset = row * hidden_size;
        float sum_squares = 0.0F;
        for (std::size_t column = 0; column < hidden_size; ++column) {
            const float value = input[row_offset + column];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0F /
            std::sqrt(sum_squares / static_cast<float>(hidden_size) + epsilon);
        for (std::size_t column = 0; column < hidden_size; ++column) {
            const std::size_t index = row_offset + column;
            output[index] = input[index] * inv_rms * weight[column];
        }
    }
    return output;
}

std::vector<float> run_cuda(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    cudaStream_t stream) {
    expect_true(
        num_rows > 0 && hidden_size > 0 && input.size() == num_rows * hidden_size &&
            weight.size() == hidden_size,
        "invalid host test data");

    DeviceBuffer device_input(input.size());
    DeviceBuffer device_weight(weight.size());
    DeviceBuffer device_output(input.size());
    const std::size_t input_bytes = input.size() * sizeof(float);
    const std::size_t weight_bytes = weight.size() * sizeof(float);

    check_cuda(
        cudaMemcpyAsync(
            device_input.get(),
            input.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync input host-to-device");
    check_cuda(
        cudaMemcpyAsync(
            device_weight.get(),
            weight.data(),
            weight_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync weight host-to-device");
    check_cuda(
        flux::rmsnorm_cuda_fp32(
            device_input.get(),
            device_weight.get(),
            device_output.get(),
            num_rows,
            hidden_size,
            epsilon,
            stream),
        "rmsnorm_cuda_fp32 kernel launch");

    std::vector<float> output(input.size());
    check_cuda(
        cudaMemcpyAsync(
            output.data(),
            device_output.get(),
            input_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync output device-to-host");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return output;
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

void test_hand_verifiable_one_row(cudaStream_t stream) {
    const std::vector<float> input{3.0F, 4.0F};
    const std::vector<float> weight{2.0F, 0.5F};
    const std::vector<float> actual =
        run_cuda(input, weight, 1, 2, 0.0F, stream);
    const float denominator = std::sqrt(12.5F);
    const std::vector<float> expected{6.0F / denominator, 2.0F / denominator};
    expect_close(actual, expected, "hand-verifiable one-row case");
}

void test_multiple_rows(cudaStream_t stream) {
    constexpr std::size_t num_rows = 3;
    constexpr std::size_t hidden_size = 7;
    std::vector<float> input(num_rows * hidden_size);
    std::vector<float> weight(hidden_size);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.31F) - 0.2F;
    }
    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] = 0.4F + 0.17F * static_cast<float>(index);
    }
    const std::vector<float> expected =
        cpu_reference(input, weight, num_rows, hidden_size, 1.0e-5F);
    const std::vector<float> actual =
        run_cuda(input, weight, num_rows, hidden_size, 1.0e-5F, stream);
    expect_close(actual, expected, "multiple-row case");
}

float test_smollm2_shape(cudaStream_t stream) {
    constexpr std::size_t num_rows = 14;
    constexpr std::size_t hidden_size = 576;
    constexpr float epsilon = 1.0e-5F;
    std::vector<float> input(num_rows * hidden_size);
    std::vector<float> weight(hidden_size);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float position = static_cast<float>(index);
        input[index] = std::sin(position * 0.013F) + std::cos(position * 0.007F);
    }
    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] =
            0.25F + 1.5F * static_cast<float>(index) /
                        static_cast<float>(hidden_size - 1);
    }

    const std::vector<float> expected =
        cpu_reference(input, weight, num_rows, hidden_size, epsilon);
    const std::vector<float> actual =
        run_cuda(input, weight, num_rows, hidden_size, epsilon, stream);
    expect_close(actual, expected, "SmolLM2-sized case");
    return maximum_absolute_error(actual, expected);
}

void test_zero_input(cudaStream_t stream) {
    constexpr std::size_t num_rows = 2;
    constexpr std::size_t hidden_size = 576;
    std::vector<float> input(num_rows * hidden_size, 0.0F);
    std::vector<float> weight(hidden_size);
    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] = 0.5F + 0.001F * static_cast<float>(index);
    }
    const std::vector<float> actual =
        run_cuda(input, weight, num_rows, hidden_size, 1.0e-5F, stream);
    expect_true(
        std::all_of(actual.begin(), actual.end(), [](const float value) {
            return value == 0.0F;
        }),
        "zero input did not produce exact zero output");
}

void test_deterministic(cudaStream_t stream) {
    constexpr std::size_t num_rows = 4;
    constexpr std::size_t hidden_size = 513;
    std::vector<float> input(num_rows * hidden_size);
    std::vector<float> weight(hidden_size);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::cos(static_cast<float>(index) * 0.019F);
    }
    for (std::size_t index = 0; index < hidden_size; ++index) {
        weight[index] = 0.75F + 0.002F * static_cast<float>(index);
    }

    const std::vector<float> first =
        run_cuda(input, weight, num_rows, hidden_size, 1.0e-5F, stream);
    const std::vector<float> second =
        run_cuda(input, weight, num_rows, hidden_size, 1.0e-5F, stream);
    const std::vector<float> expected =
        cpu_reference(input, weight, num_rows, hidden_size, 1.0e-5F);
    expect_close(first, expected, "513-wide scalar-fallback case");
    expect_true(
        std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0,
        "repeated CUDA execution produced different output bytes");
}

void test_launcher_validation() {
    float* pointer = reinterpret_cast<float*>(1);
    expect_true(
        flux::rmsnorm_cuda_fp32(nullptr, pointer, pointer, 1, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "null input was accepted");
    expect_true(
        flux::rmsnorm_cuda_fp32(pointer, pointer, pointer, 0, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "zero rows was accepted");
    expect_true(
        flux::rmsnorm_cuda_fp32(pointer, pointer, pointer, 1, 0, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "zero hidden size was accepted");
    expect_true(
        flux::rmsnorm_cuda_fp32(pointer, pointer, pointer, 1, 1, -1.0F, nullptr) ==
            cudaErrorInvalidValue,
        "negative epsilon was accepted");
    if (sizeof(std::size_t) > sizeof(unsigned int)) {
        const std::size_t too_many_rows =
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) + 1;
        expect_true(
            flux::rmsnorm_cuda_fp32(
                pointer, pointer, pointer, too_many_rows, 1, 0.0F, nullptr) ==
                cudaErrorInvalidValue,
            "row count larger than the CUDA grid was accepted");
    }
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

        CudaStream stream;
        test_launcher_validation();
        test_hand_verifiable_one_row(stream.get());
        test_multiple_rows(stream.get());
        const float smollm2_max_error = test_smollm2_shape(stream.get());
        test_zero_input(stream.get());
        test_deterministic(stream.get());

        std::cout << std::setprecision(9)
                  << "SmolLM2-sized maximum absolute error: "
                  << smollm2_max_error << '\n';
    } catch (const std::exception& error) {
        std::cerr << "CUDA RMSNorm test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All standalone CUDA RMSNorm tests passed.\n";
    return 0;
}
