#include "softmax.h"

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
    if (argc != 5) {
        std::cerr << "Usage: softmax_cli <input.bin> <output.bin> <rows> "
                     "<row-width>\n";
        return 2;
    }

    try {
        const std::size_t num_rows = parse_size(argv[3], "rows");
        const std::size_t row_width = parse_size(argv[4], "row-width");
        if (num_rows > std::numeric_limits<std::size_t>::max() / row_width) {
            throw std::invalid_argument("rows * row-width overflows size_t");
        }
        const std::size_t element_count = num_rows * row_width;
        std::vector<float> input(element_count);
        std::vector<float> output(element_count);

        std::ifstream input_stream(argv[1], std::ios::binary);
        if (!input_stream) {
            throw std::runtime_error("could not open input file");
        }
        read_exact(input_stream, input.data(), input.size());
        if (input_stream.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("input file has trailing data");
        }

        flux::softmax_fp32(
            input.data(), output.data(), num_rows, row_width);

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
        std::cerr << "softmax_cli: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
