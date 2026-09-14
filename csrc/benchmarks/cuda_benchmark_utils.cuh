#pragma once

#include "cuda_test_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flux::benchmark {

struct Options {
    std::string filter;
    int warmup = 10;
    int samples = 51;
};

inline Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--filter" && index + 1 < argc) {
            options.filter = argv[++index];
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup = std::stoi(argv[++index]);
        } else if (argument == "--samples" && index + 1 < argc) {
            options.samples = std::stoi(argv[++index]);
        } else if (argument == "--help") {
            std::cout << "Usage: flux_cuda_microbenchmarks "
                         "[--filter NAME] [--warmup N] [--samples N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
    }
    if (options.warmup < 0 || options.samples < 1) {
        throw std::invalid_argument(
            "warmup must be non-negative and samples must be positive");
    }
    return options;
}

inline bool selected(const Options& options, const std::string& name) {
    return options.filter.empty() || name.find(options.filter) != std::string::npos;
}

inline float median_cuda_ms(
    const Options& options,
    cudaStream_t stream,
    const std::function<void()>& operation) {
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        operation();
    }
    test::check_cuda(cudaStreamSynchronize(stream), "benchmark warmup synchronize");
    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(options.samples));
    test::Event start;
    test::Event end;
    for (int sample = 0; sample < options.samples; ++sample) {
        test::check_cuda(cudaEventRecord(start.get(), stream), "record start event");
        operation();
        test::check_cuda(cudaEventRecord(end.get(), stream), "record end event");
        test::check_cuda(cudaEventSynchronize(end.get()), "synchronize end event");
        float milliseconds = 0.0F;
        test::check_cuda(
            cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
            "cudaEventElapsedTime");
        samples.push_back(milliseconds);
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    return samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) * 0.5F
        : samples[middle];
}

inline void report(const std::string& name, float milliseconds) {
    std::cout << std::left << std::setw(34) << name << std::right
              << std::fixed << std::setprecision(6) << milliseconds
              << " ms\n";
}

}  // namespace flux::benchmark
