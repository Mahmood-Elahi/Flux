#include "residual_rmsnorm_cuda.h"

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

void write_exact(
    std::ofstream& stream, const float* source, const std::size_t count) {
    stream.write(reinterpret_cast<const char*>(source), checked_byte_count(count));
    if (!stream) {
        throw std::runtime_error("could not write complete output file");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: residual_rmsnorm_cuda_cli <input.bin> <output.bin> "
                     "<rows> <hidden-size> <epsilon>\n";
        return 2;
    }

    float* device_hidden = nullptr;
    float* device_residual = nullptr;
    float* device_weight = nullptr;
    float* device_residual_out = nullptr;
    float* device_norm_out = nullptr;
    cudaStream_t stream = nullptr;
    try {
        const std::size_t num_rows = parse_size(argv[3], "rows");
        const std::size_t hidden_size = parse_size(argv[4], "hidden-size");
        const float epsilon = parse_epsilon(argv[5]);
        if (num_rows > std::numeric_limits<std::size_t>::max() / hidden_size) {
            throw std::invalid_argument("rows * hidden-size overflows size_t");
        }
        const std::size_t input_count = num_rows * hidden_size;
        std::vector<float> hidden(input_count);
        std::vector<float> residual(input_count);
        std::vector<float> weight(hidden_size);
        std::vector<float> residual_out(input_count);
        std::vector<float> norm_out(input_count);

        std::ifstream input_stream(argv[1], std::ios::binary);
        if (!input_stream) {
            throw std::runtime_error("could not open input file");
        }
        read_exact(input_stream, hidden.data(), hidden.size());
        read_exact(input_stream, residual.data(), residual.size());
        read_exact(input_stream, weight.data(), weight.size());
        if (input_stream.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("input file has trailing data");
        }

        const std::size_t input_bytes =
            static_cast<std::size_t>(checked_byte_count(input_count));
        const std::size_t weight_bytes =
            static_cast<std::size_t>(checked_byte_count(hidden_size));
        check_cuda(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_hidden), input_bytes),
            "cudaMalloc hidden");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_residual), input_bytes),
            "cudaMalloc residual");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_weight), weight_bytes),
            "cudaMalloc weight");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_residual_out), input_bytes),
            "cudaMalloc residual output");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_norm_out), input_bytes),
            "cudaMalloc norm output");
        check_cuda(
            cudaMemcpyAsync(
                device_hidden,
                hidden.data(),
                input_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync hidden host-to-device");
        check_cuda(
            cudaMemcpyAsync(
                device_residual,
                residual.data(),
                input_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync residual host-to-device");
        check_cuda(
            cudaMemcpyAsync(
                device_weight,
                weight.data(),
                weight_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync weight host-to-device");
        check_cuda(
            flux::residual_rmsnorm_cuda_fp32(
                device_hidden,
                device_residual,
                device_weight,
                device_residual_out,
                device_norm_out,
                num_rows,
                hidden_size,
                epsilon,
                stream),
            "residual_rmsnorm_cuda_fp32 kernel launch");
        check_cuda(
            cudaMemcpyAsync(
                residual_out.data(),
                device_residual_out,
                input_bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync residual output device-to-host");
        check_cuda(
            cudaMemcpyAsync(
                norm_out.data(),
                device_norm_out,
                input_bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync norm output device-to-host");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        std::ofstream output_stream(argv[2], std::ios::binary | std::ios::trunc);
        if (!output_stream) {
            throw std::runtime_error("could not open output file");
        }
        write_exact(output_stream, residual_out.data(), residual_out.size());
        write_exact(output_stream, norm_out.data(), norm_out.size());
    } catch (const std::exception& error) {
        std::cerr << "residual_rmsnorm_cuda_cli: " << error.what() << '\n';
        if (device_norm_out != nullptr) {
            cudaFree(device_norm_out);
        }
        if (device_residual_out != nullptr) {
            cudaFree(device_residual_out);
        }
        if (device_weight != nullptr) {
            cudaFree(device_weight);
        }
        if (device_residual != nullptr) {
            cudaFree(device_residual);
        }
        if (device_hidden != nullptr) {
            cudaFree(device_hidden);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        return 1;
    }

    check_cuda(cudaFree(device_norm_out), "cudaFree norm output");
    check_cuda(cudaFree(device_residual_out), "cudaFree residual output");
    check_cuda(cudaFree(device_weight), "cudaFree weight");
    check_cuda(cudaFree(device_residual), "cudaFree residual");
    check_cuda(cudaFree(device_hidden), "cudaFree hidden");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
