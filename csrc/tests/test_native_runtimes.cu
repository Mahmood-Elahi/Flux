#include "cuda_test_utils.cuh"

#include "native_decode_runtime.h"
#include "native_prefill_runtime.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

constexpr std::int64_t kLayers = 30;
constexpr std::int64_t kHidden = 576;
constexpr std::int64_t kIntermediate = 1536;
constexpr std::int64_t kPackedQkv = 960;
constexpr std::int64_t kPackedGateUp = 3072;
constexpr std::int64_t kKvHeads = 3;
constexpr std::int64_t kHeadDim = 64;
constexpr std::int64_t kVocabulary = 32;

using flux::test::check_cuda;
using flux::test::expect;

struct ModelStorage {
    at::Tensor embedding;
    at::Tensor input_norm;
    at::Tensor packed_qkv;
    at::Tensor attention_output;
    at::Tensor post_attention_norm;
    at::Tensor packed_gate_up;
    at::Tensor down_projection;
    at::Tensor final_norm;
    at::Tensor lm_head;
    at::Tensor rope_cosine;
    at::Tensor rope_sine;
    std::vector<at::Tensor> input_norms;
    std::vector<at::Tensor> packed_qkvs;
    std::vector<at::Tensor> attention_outputs;
    std::vector<at::Tensor> post_attention_norms;
    std::vector<at::Tensor> packed_gate_ups;
    std::vector<at::Tensor> down_projections;
    std::vector<double> epsilons;
    std::vector<double> scales;

    explicit ModelStorage(std::int64_t capacity) {
        const at::TensorOptions options = at::TensorOptions()
            .device(at::kCUDA)
            .dtype(at::kFloat)
            .requires_grad(false);
        embedding = at::zeros({kVocabulary, kHidden}, options);
        input_norm = at::ones({kHidden}, options);
        packed_qkv = at::zeros({kPackedQkv, kHidden}, options);
        attention_output = at::zeros({kHidden, kHidden}, options);
        post_attention_norm = at::ones({kHidden}, options);
        packed_gate_up = at::zeros({kPackedGateUp, kHidden}, options);
        down_projection = at::zeros({kHidden, kIntermediate}, options);
        final_norm = at::ones({kHidden}, options);
        lm_head = at::zeros({kVocabulary, kHidden}, options);
        rope_cosine = at::ones({capacity, kHeadDim}, options);
        rope_sine = at::zeros({capacity, kHeadDim}, options);
        input_norms.assign(kLayers, input_norm);
        packed_qkvs.assign(kLayers, packed_qkv);
        attention_outputs.assign(kLayers, attention_output);
        post_attention_norms.assign(kLayers, post_attention_norm);
        packed_gate_ups.assign(kLayers, packed_gate_up);
        down_projections.assign(kLayers, down_projection);
        epsilons.assign(kLayers, 1.0e-5);
        scales.assign(kLayers, 0.125);
    }
};

std::vector<at::Tensor> cache_sources(std::int64_t length) {
    const at::Tensor cache = at::zeros(
        {1, kKvHeads, length, kHeadDim},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    return std::vector<at::Tensor>(kLayers, cache);
}

void assert_all_zero(const at::Tensor& tensor, const char* name) {
    const float maximum = tensor.abs().max().item<float>();
    expect(maximum == 0.0F, std::string(name) + " was not exactly zero");
}

void test_full_decode_runtime_on_non_default_stream() {
    constexpr std::int64_t capacity = 4;
    constexpr std::int64_t initial_position = 2;
    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    const c10::cuda::CUDAStreamGuard stream_guard(stream);
    ModelStorage model(capacity);
    at::Tensor token = at::full(
        {1, 1}, 3,
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    std::vector<at::Tensor> keys = cache_sources(initial_position);
    std::vector<at::Tensor> values = cache_sources(initial_position);

    flux::NativeSmolLM2Decode runtime(
        token, model.embedding, model.input_norms, model.packed_qkvs,
        model.attention_outputs, model.post_attention_norms,
        model.packed_gate_ups, model.down_projections, model.final_norm,
        model.lm_head, model.rope_cosine, model.rope_sine, keys, values,
        capacity, initial_position, model.epsilons, model.scales);
    const std::vector<std::int64_t> addresses = runtime.addresses();
    expect(!addresses.empty(), "decode runtime did not expose stable addresses");
    expect(runtime.position() == initial_position,
           "decode runtime initial position is incorrect");
    expect(runtime.cache_length() == initial_position,
           "decode runtime initial cache length is incorrect");

    std::size_t free_before = 0;
    std::size_t total_before = 0;
    check_cuda(cudaMemGetInfo(&free_before, &total_before),
               "cudaMemGetInfo before replay");
    const at::Tensor first = runtime.replay(token);
    check_cuda(cudaStreamSynchronize(stream.stream()),
               "synchronize first decode replay");
    std::size_t free_after = 0;
    std::size_t total_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total_after),
               "cudaMemGetInfo after replay");
    expect(total_before == total_after && free_before == free_after,
           "decode replay changed device allocation state");
    assert_all_zero(first, "decode logits");
    expect(runtime.position() == initial_position + 1,
           "decode replay did not advance position");
    expect(runtime.addresses() == addresses,
           "decode replay changed a stable address");

    runtime.replay(c10::nullopt);
    expect(runtime.position() == capacity,
           "repeated decode replay did not reach capacity");
    bool exhausted = false;
    try {
        runtime.replay(c10::nullopt);
    } catch (const c10::Error&) {
        exhausted = true;
    }
    expect(exhausted, "decode runtime did not reject capacity exhaustion");

    runtime.reset(token, keys, values, 1);
    expect(runtime.position() == 1 && runtime.cache_length() == 1,
           "decode reset did not restore device state");
    const at::Tensor reset_result = runtime.replay(token);
    check_cuda(cudaStreamSynchronize(stream.stream()),
               "synchronize reset decode replay");
    assert_all_zero(reset_result, "reset decode logits");
    expect(runtime.addresses() == addresses,
           "decode reset changed a stable address");
    expect(runtime.key_cache().sizes() ==
               at::IntArrayRef({kLayers, 1, kKvHeads, capacity, kHeadDim}),
           "decode cache layout invariant failed");
}

void test_prefill_handoff_and_reuse() {
    constexpr std::int64_t prompt_length = 2;
    constexpr std::int64_t capacity = 3;
    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    const c10::cuda::CUDAStreamGuard stream_guard(stream);
    ModelStorage model(capacity);
    at::Tensor input_ids = at::arange(
        1, 3, at::TensorOptions().device(at::kCUDA).dtype(at::kLong))
        .reshape({1, 2});
    flux::NativeSmolLM2Prefill runtime(
        input_ids, model.embedding, model.input_norms, model.packed_qkvs,
        model.attention_outputs, model.post_attention_norms,
        model.packed_gate_ups, model.down_projections, model.final_norm,
        model.lm_head, model.rope_cosine, model.rope_sine, capacity,
        model.epsilons, model.scales);
    const std::vector<std::int64_t> addresses = runtime.addresses();
    expect(runtime.position() == prompt_length &&
               runtime.cache_length() == prompt_length,
           "prefill did not install decode handoff state");
    expect(runtime.prompt_length() == prompt_length,
           "prefill prompt length is incorrect");
    assert_all_zero(runtime.logits(), "prefill logits");
    expect(runtime.key_cache().sizes() ==
               at::IntArrayRef({kLayers, 1, kKvHeads, capacity, kHeadDim}),
           "prefill cache layout invariant failed");

    std::size_t free_before = 0;
    std::size_t total_before = 0;
    check_cuda(cudaStreamSynchronize(stream.stream()),
               "synchronize before repeated prefill");
    check_cuda(cudaMemGetInfo(&free_before, &total_before),
               "cudaMemGetInfo before repeated prefill");
    runtime.prefill(input_ids);
    check_cuda(cudaStreamSynchronize(stream.stream()),
               "synchronize repeated prefill");
    std::size_t free_after = 0;
    std::size_t total_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total_after),
               "cudaMemGetInfo after repeated prefill");
    expect(free_before == free_after && total_before == total_after,
           "prefill reuse changed device allocation state");
    expect(runtime.addresses() == addresses,
           "prefill reuse changed a stable address");

    const at::Tensor decode_logits = runtime.replay(c10::nullopt);
    check_cuda(cudaStreamSynchronize(stream.stream()),
               "synchronize prefill/decode handoff");
    assert_all_zero(decode_logits, "prefill handoff decode logits");
    expect(runtime.position() == capacity && runtime.cache_length() == capacity,
           "prefill/decode handoff state did not advance");
    bool exhausted = false;
    try {
        runtime.replay(c10::nullopt);
    } catch (const c10::Error&) {
        exhausted = true;
    }
    expect(exhausted, "prefill-attached decode did not enforce capacity");
}

}  // namespace

int main() {
    return flux::test::run("Flux native CUDA runtimes", [] {
        at::InferenceMode inference_guard;
        test_full_decode_runtime_on_non_default_stream();
        test_prefill_handoff_and_reuse();
    });
}
