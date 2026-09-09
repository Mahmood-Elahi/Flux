#include "rmsnorm_cuda.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::size_t parse_size(const char* text, const char* name) {
    if (text[0] == '-') {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive size");
    }
    return static_cast<std::size_t>(value);
}

float parse_epsilon(const char* text) {
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || value < 0.0F) {
        throw std::invalid_argument("epsilon must be a non-negative float");
    }
    return value;
}

std::streamsize checked_byte_count(const std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("tensor byte count overflows size_t");
    }
    const std::size_t byte_count = count * sizeof(float);
    if (byte_count >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor is too large for a single file operation");
    }
    return static_cast<std::streamsize>(byte_count);
}

void read_exact(std::ifstream& stream, float* destination, const std::size_t count) {
    stream.read(reinterpret_cast<char*>(destination), checked_byte_count(count));
    if (!stream) {
        throw std::runtime_error("input file has fewer values than expected");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: rmsnorm_cuda_cli <input.bin> <output.bin> <rows> "
                     "<hidden-size> <epsilon>\n";
        return 2;
    }

    float* device_input = nullptr;
    float* device_weight = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    try {
        const std::size_t num_rows = parse_size(argv[3], "rows");
        const std::size_t hidden_size = parse_size(argv[4], "hidden-size");
        const float epsilon = parse_epsilon(argv[5]);
        if (num_rows > std::numeric_limits<std::size_t>::max() / hidden_size) {
            throw std::invalid_argument("rows * hidden-size overflows size_t");
        }
        const std::size_t input_count = num_rows * hidden_size;
        std::vector<float> input(input_count);
        std::vector<float> weight(hidden_size);
        std::vector<float> output(input_count);

        std::ifstream input_stream(argv[1], std::ios::binary);
        if (!input_stream) {
            throw std::runtime_error("could not open input file");
        }
        read_exact(input_stream, input.data(), input.size());
        read_exact(input_stream, weight.data(), weight.size());
        if (input_stream.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("input file has trailing data");
        }

        const std::size_t input_bytes =
            static_cast<std::size_t>(checked_byte_count(input.size()));
        const std::size_t weight_bytes =
            static_cast<std::size_t>(checked_byte_count(weight.size()));
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes),
                   "cudaMalloc input");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_weight), weight_bytes),
                   "cudaMalloc weight");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), input_bytes),
                   "cudaMalloc output");
        check_cuda(
            cudaMemcpyAsync(
                device_input,
                input.data(),
                input_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync input host-to-device");
        check_cuda(
            cudaMemcpyAsync(
                device_weight,
                weight.data(),
                weight_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync weight host-to-device");
        check_cuda(
            flux::rmsnorm_cuda_fp32(
                device_input,
                device_weight,
                device_output,
                num_rows,
                hidden_size,
                epsilon,
                stream),
            "rmsnorm_cuda_fp32 kernel launch");
        check_cuda(
            cudaMemcpyAsync(
                output.data(),
                device_output,
                input_bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync output device-to-host");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        std::ofstream output_stream(argv[2], std::ios::binary | std::ios::trunc);
        if (!output_stream) {
            throw std::runtime_error("could not open output file");
        }
        output_stream.write(
            reinterpret_cast<const char*>(output.data()),
            checked_byte_count(output.size()));
        if (!output_stream) {
            throw std::runtime_error("could not write complete output file");
        }
    } catch (const std::exception& error) {
        std::cerr << "rmsnorm_cuda_cli: " << error.what() << '\n';
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        if (device_weight != nullptr) {
            cudaFree(device_weight);
        }
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        return 1;
    }

    check_cuda(cudaFree(device_output), "cudaFree output");
    check_cuda(cudaFree(device_weight), "cudaFree weight");
    check_cuda(cudaFree(device_input), "cudaFree input");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
