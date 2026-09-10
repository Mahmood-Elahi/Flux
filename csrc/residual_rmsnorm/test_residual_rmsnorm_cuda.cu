#include "residual_rmsnorm_cuda.h"

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

struct Outputs {
    std::vector<float> residual;
    std::vector<float> norm;
};

struct CudaResult {
    Outputs outputs;
    std::vector<float> hidden_after;
    std::vector<float> residual_after;
    std::vector<float> weight_after;
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

Outputs cpu_reference(
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

CudaResult run_cuda(
    const std::vector<float>& hidden,
    const std::vector<float>& residual,
    const std::vector<float>& weight,
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    cudaStream_t stream) {
    expect_true(
        num_rows > 0 && hidden_size > 0 &&
            hidden.size() == num_rows * hidden_size &&
            residual.size() == hidden.size() && weight.size() == hidden_size,
        "invalid host test data");
    expect_true(stream != nullptr, "test requires a non-default CUDA stream");

    DeviceBuffer device_hidden(hidden.size());
    DeviceBuffer device_residual(residual.size());
    DeviceBuffer device_weight(weight.size());
    DeviceBuffer device_residual_out(hidden.size());
    DeviceBuffer device_norm_out(hidden.size());
    const std::size_t input_bytes = hidden.size() * sizeof(float);
    const std::size_t weight_bytes = weight.size() * sizeof(float);

    check_cuda(
        cudaMemcpyAsync(
            device_hidden.get(),
            hidden.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync hidden host-to-device");
    check_cuda(
        cudaMemcpyAsync(
            device_residual.get(),
            residual.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync residual host-to-device");
    check_cuda(
        cudaMemcpyAsync(
            device_weight.get(),
            weight.data(),
            weight_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync weight host-to-device");
    check_cuda(
        flux::residual_rmsnorm_cuda_fp32(
            device_hidden.get(),
            device_residual.get(),
            device_weight.get(),
            device_norm_out.get(),
            device_residual_out.get(),
            num_rows,
            hidden_size,
            epsilon,
            stream),
        "residual_rmsnorm_cuda_fp32 kernel launch");

    CudaResult result{
        {std::vector<float>(hidden.size()), std::vector<float>(hidden.size())},
        std::vector<float>(hidden.size()),
        std::vector<float>(residual.size()),
        std::vector<float>(weight.size())};
    check_cuda(
        cudaMemcpyAsync(
            result.outputs.residual.data(),
            device_residual_out.get(),
            input_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync residual output device-to-host");
    check_cuda(
        cudaMemcpyAsync(
            result.outputs.norm.data(),
            device_norm_out.get(),
            input_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync norm output device-to-host");
    check_cuda(
        cudaMemcpyAsync(
            result.hidden_after.data(),
            device_hidden.get(),
            input_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync hidden preservation copy");
    check_cuda(
        cudaMemcpyAsync(
            result.residual_after.data(),
            device_residual.get(),
            input_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync residual preservation copy");
    check_cuda(
        cudaMemcpyAsync(
            result.weight_after.data(),
            device_weight.get(),
            weight_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync weight preservation copy");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return result;
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

float maximum_absolute_error(
    const std::vector<float>& actual, const std::vector<float>& expected) {
    expect_true(actual.size() == expected.size(), "result sizes differ");
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum;
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
            case_name + " residual element " + std::to_string(index) +
                " does not equal direct FP32 addition");
        const float error = std::abs(actual.norm[index] - expected.norm[index]);
        const float tolerance =
            kAbsoluteTolerance + kRelativeTolerance * std::abs(expected.norm[index]);
        if (error > tolerance) {
            throw std::runtime_error(
                case_name + " normalized element " + std::to_string(index) +
                ": expected " + std::to_string(expected.norm[index]) + ", got " +
                std::to_string(actual.norm[index]));
        }
    }
}

float run_case(
    const std::size_t num_rows,
    const std::size_t hidden_size,
    const float epsilon,
    cudaStream_t stream,
    const std::string& case_name) {
    std::vector<float> hidden(num_rows * hidden_size);
    std::vector<float> residual(hidden.size());
    std::vector<float> weight(hidden_size);
    fill_inputs(hidden, residual, weight);
    const Outputs expected =
        cpu_reference(hidden, residual, weight, num_rows, hidden_size, epsilon);
    const CudaResult actual = run_cuda(
        hidden, residual, weight, num_rows, hidden_size, epsilon, stream);

    expect_outputs(actual.outputs, expected, case_name);
    expect_true(actual.hidden_after == hidden, case_name + ": hidden input changed");
    expect_true(
        actual.residual_after == residual, case_name + ": residual input changed");
    expect_true(actual.weight_after == weight, case_name + ": weight input changed");
    return maximum_absolute_error(actual.outputs.norm, expected.norm);
}

void test_hand_verifiable_one_row(cudaStream_t stream) {
    const std::vector<float> hidden{1.0F, 3.0F};
    const std::vector<float> residual{2.0F, 1.0F};
    const std::vector<float> weight{2.0F, 0.5F};
    const CudaResult actual =
        run_cuda(hidden, residual, weight, 1, 2, 0.0F, stream);
    const float denominator = std::sqrt(12.5F);
    const Outputs expected{{3.0F, 4.0F},
                           {6.0F / denominator, 2.0F / denominator}};
    expect_outputs(actual.outputs, expected, "hand-verifiable one-row case");
}

void test_required_hidden_sizes(cudaStream_t stream) {
    const std::vector<std::size_t> hidden_sizes{1, 7, 63, 127, 575, 576, 577, 1024};
    for (const std::size_t hidden_size : hidden_sizes) {
        run_case(
            3,
            hidden_size,
            1.0e-5F,
            stream,
            "hidden-size-" + std::to_string(hidden_size));
    }
}

void test_representative_shapes(cudaStream_t stream) {
    run_case(1, 576, 1.0e-5F, stream, "shape-(576)");
    run_case(4, 576, 1.0e-5F, stream, "shape-(4,576)");
    run_case(64, 576, 1.0e-5F, stream, "shape-(2,32,576)");
}

void test_epsilon_values(cudaStream_t stream) {
    const std::vector<float> epsilons{0.0F, 1.0e-6F, 1.0e-5F, 1.0e-4F};
    for (const float epsilon : epsilons) {
        run_case(
            2,
            127,
            epsilon,
            stream,
            "epsilon-" + std::to_string(epsilon));
    }
}

float test_large_workload(cudaStream_t stream) {
    return run_case(8192, 576, 1.0e-5F, stream, "large-(8192,576)");
}

void test_deterministic(cudaStream_t stream) {
    constexpr std::size_t num_rows = 4;
    constexpr std::size_t hidden_size = 577;
    std::vector<float> hidden(num_rows * hidden_size);
    std::vector<float> residual(hidden.size());
    std::vector<float> weight(hidden_size);
    fill_inputs(hidden, residual, weight);

    const CudaResult first = run_cuda(
        hidden, residual, weight, num_rows, hidden_size, 1.0e-5F, stream);
    const CudaResult second = run_cuda(
        hidden, residual, weight, num_rows, hidden_size, 1.0e-5F, stream);
    expect_true(
        std::memcmp(
            first.outputs.residual.data(),
            second.outputs.residual.data(),
            first.outputs.residual.size() * sizeof(float)) == 0,
        "repeated CUDA execution produced different residual output bytes");
    expect_true(
        std::memcmp(
            first.outputs.norm.data(),
            second.outputs.norm.data(),
            first.outputs.norm.size() * sizeof(float)) == 0,
        "repeated CUDA execution produced different normalized output bytes");
}

void test_launcher_validation() {
    float* pointer = reinterpret_cast<float*>(1);
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            nullptr, pointer, pointer, pointer, pointer, 1, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "null hidden was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, nullptr, pointer, pointer, pointer, 1, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "null residual was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, pointer, nullptr, pointer, pointer, 1, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "null weight was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, pointer, pointer, nullptr, pointer, 1, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "null normalized output was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, pointer, pointer, pointer, nullptr, 1, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "null residual output was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, pointer, pointer, pointer, pointer, 0, 1, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "zero rows was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, pointer, pointer, pointer, pointer, 1, 0, 0.0F, nullptr) ==
            cudaErrorInvalidValue,
        "zero hidden size was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer, pointer, pointer, pointer, pointer, 1, 1, -1.0F, nullptr) ==
            cudaErrorInvalidValue,
        "negative epsilon was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            1,
            std::numeric_limits<float>::quiet_NaN(),
            nullptr) == cudaErrorInvalidValue,
        "NaN epsilon was accepted");
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            1,
            std::numeric_limits<float>::infinity(),
            nullptr) == cudaErrorInvalidValue,
        "infinite epsilon was accepted");
    if (sizeof(std::size_t) > sizeof(unsigned int)) {
        const std::size_t too_many_rows =
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) + 1;
        expect_true(
            flux::residual_rmsnorm_cuda_fp32(
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                too_many_rows,
                1,
                0.0F,
                nullptr) == cudaErrorInvalidValue,
            "row count larger than the CUDA grid was accepted");
    }
    const std::size_t too_many_rows =
        std::numeric_limits<std::size_t>::max() / 2 + 1;
    expect_true(
        flux::residual_rmsnorm_cuda_fp32(
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            too_many_rows,
            2,
            0.0F,
            nullptr) == cudaErrorInvalidValue,
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

        CudaStream stream;
        test_launcher_validation();
        test_hand_verifiable_one_row(stream.get());
        test_required_hidden_sizes(stream.get());
        test_representative_shapes(stream.get());
        test_epsilon_values(stream.get());
        const float large_max_error = test_large_workload(stream.get());
        test_deterministic(stream.get());

        std::cout << std::setprecision(9)
                  << "Large-workload maximum absolute norm error: "
                  << large_max_error << '\n';
    } catch (const std::exception& error) {
        std::cerr << "CUDA residual RMSNorm test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All standalone CUDA residual RMSNorm tests passed.\n";
    return 0;
}
