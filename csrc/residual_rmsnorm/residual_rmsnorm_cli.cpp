#include "residual_rmsnorm.h"

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
        std::cerr << "Usage: residual_rmsnorm_cli <input.bin> <output.bin> <rows> "
                     "<hidden-size> <epsilon>\n";
        return 2;
    }

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

        flux::residual_rmsnorm_fp32(
            hidden.data(),
            residual.data(),
            weight.data(),
            residual_out.data(),
            norm_out.data(),
            num_rows,
            hidden_size,
            epsilon);

        std::ofstream output_stream(argv[2], std::ios::binary | std::ios::trunc);
        if (!output_stream) {
            throw std::runtime_error("could not open output file");
        }
        write_exact(output_stream, residual_out.data(), residual_out.size());
        write_exact(output_stream, norm_out.data(), norm_out.size());
    } catch (const std::exception& error) {
        std::cerr << "residual_rmsnorm_cli: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
