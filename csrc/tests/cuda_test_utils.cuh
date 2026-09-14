#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flux::test {

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        expect(count > 0, "device buffer size must be positive");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    T* get() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    cudaStream_t get() const { return stream_; }
    void synchronize() const {
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename T>
void copy_to_device(
    DeviceBuffer<T>& destination,
    const std::vector<T>& source,
    cudaStream_t stream) {
    expect(destination.size() == source.size(), "host/device sizes differ");
    check_cuda(
        cudaMemcpyAsync(destination.get(), source.data(), destination.bytes(),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync host-to-device");
}

template <typename T>
std::vector<T> copy_to_host(
    const DeviceBuffer<T>& source, cudaStream_t stream) {
    std::vector<T> result(source.size());
    check_cuda(
        cudaMemcpyAsync(result.data(), source.get(), source.bytes(),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync device-to-host");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize copy");
    return result;
}

inline std::vector<float> deterministic_values(
    std::size_t count, float scale = 1.0F, float bias = 0.0F,
    std::uint32_t seed = 1729U) {
    std::vector<float> values(count);
    std::uint32_t state = seed;
    for (std::size_t index = 0; index < count; ++index) {
        state = state * 1664525U + 1013904223U;
        const float unit = static_cast<float>(state >> 8U) /
            static_cast<float>(1U << 24U);
        values[index] = bias + scale * (2.0F * unit - 1.0F);
    }
    return values;
}

inline float maximum_absolute_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    expect(actual.size() == expected.size(), "comparison sizes differ");
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum;
}

inline void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float relative_tolerance,
    float absolute_tolerance,
    const std::string& name) {
    expect(actual.size() == expected.size(), name + ": comparison sizes differ");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float a = actual[index];
        const float e = expected[index];
        const float tolerance =
            absolute_tolerance + relative_tolerance * std::abs(e);
        if (!std::isfinite(a) || std::abs(a - e) > tolerance) {
            throw std::runtime_error(
                name + " at element " + std::to_string(index) +
                ": expected " + std::to_string(e) + ", got " +
                std::to_string(a) + ", tolerance " +
                std::to_string(tolerance));
        }
    }
}

inline void expect_unchanged(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& name) {
    expect(actual.size() == expected.size(), name + ": sizes differ");
    expect(
        std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(float)) == 0,
        name + ": storage changed");
}

inline int run(const char* suite_name, const std::function<void()>& body) {
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        expect(device_count > 0, "no CUDA device is available");
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0),
                   "cudaGetDeviceProperties");
        std::cout << suite_name << " on " << properties.name << " (sm_"
                  << properties.major << properties.minor << ")\n";
        body();
        std::cout << suite_name << ": all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << suite_name << " failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace flux::test
