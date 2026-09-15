#include "cuda_test_utils.cuh"

#include "greedy_generation_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kVocabulary = 49152;

using flux::test::DeviceBuffer;
using flux::test::Stream;
using flux::test::check_cuda;
using flux::test::copy_to_device;
using flux::test::copy_to_host;
using flux::test::deterministic_values;
using flux::test::expect;

void expect_argmax(
    const std::vector<float>& logits,
    const std::int64_t expected,
    Stream& stream,
    const std::string& name) {
    DeviceBuffer<float> device_logits(logits.size());
    DeviceBuffer<std::int64_t> device_token(1);
    copy_to_device(device_logits, logits, stream.get());
    for (int repetition = 0; repetition < 5; ++repetition) {
        check_cuda(flux::greedy_argmax_cuda_fp32(
            device_logits.get(), device_token.get(), logits.size(), stream.get()),
            "launch greedy argmax");
        const std::vector<std::int64_t> actual =
            copy_to_host(device_token, stream.get());
        expect(actual[0] == expected,
            name + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual[0]));
    }
}

void test_argmax_values_and_ties() {
    Stream stream;
    std::vector<float> logits = deterministic_values(kVocabulary, 2000.0F);
    const auto maximum = std::max_element(logits.begin(), logits.end());
    expect_argmax(logits, std::distance(logits.begin(), maximum), stream,
        "random FP32 logits");

    logits.assign(kVocabulary, -5000.0F);
    logits[177] = -3.0F;
    logits[321] = -3.0F;
    expect_argmax(logits, 177, stream, "negative tied logits");

    logits.assign(kVocabulary, -std::numeric_limits<float>::max());
    logits[0] = std::numeric_limits<float>::max();
    expect_argmax(logits, 0, stream, "first vocabulary winner");

    logits[0] = -std::numeric_limits<float>::max();
    logits.back() = std::numeric_limits<float>::max();
    expect_argmax(logits, kVocabulary - 1, stream, "last vocabulary winner");

    logits.assign(kVocabulary, -std::numeric_limits<float>::infinity());
    expect_argmax(logits, 0, stream, "all negative infinity");
}

void test_fused_token_and_state_update() {
    Stream stream;
    std::vector<float> logits(kVocabulary, -10.0F);
    logits[19] = 4.0F;
    logits[27] = 4.0F;
    DeviceBuffer<float> device_logits(kVocabulary);
    DeviceBuffer<std::int64_t> token(1);
    DeviceBuffer<std::int64_t> generated(4);
    DeviceBuffer<std::int64_t> step(1);
    DeviceBuffer<std::int64_t> position(1);
    DeviceBuffer<std::int64_t> attention_length(1);
    copy_to_device(device_logits, logits, stream.get());
    copy_to_device(step, std::vector<std::int64_t>{2}, stream.get());
    copy_to_device(position, std::vector<std::int64_t>{8}, stream.get());
    copy_to_device(
        attention_length, std::vector<std::int64_t>{9}, stream.get());
    copy_to_device(generated, std::vector<std::int64_t>{3, 7, -1, -1},
        stream.get());

    check_cuda(flux::greedy_argmax_update_cuda_fp32(
        device_logits.get(), token.get(), generated.get(), step.get(), 4,
        position.get(), attention_length.get(), kVocabulary, stream.get()),
        "launch fused greedy update");
    expect(copy_to_host(token, stream.get())[0] == 19,
        "fused update did not install the selected next token");
    const std::vector<std::int64_t> result =
        copy_to_host(generated, stream.get());
    expect(result == std::vector<std::int64_t>({3, 7, 19, -1}),
        "fused update wrote the wrong generated-token slot");
    expect(copy_to_host(step, stream.get())[0] == 3,
        "fused update did not increment generation step");
    expect(copy_to_host(position, stream.get())[0] == 9,
        "fused update did not advance absolute position");
}

}  // namespace

int main() {
    return flux::test::run("Flux greedy generation", [] {
        test_argmax_values_and_ties();
        test_fused_token_and_state_update();
    });
}
