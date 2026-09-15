#pragma once

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace flux {

class NativeSmolLM2LayerDecode final : public torch::CustomClassHolder {
public:
    NativeSmolLM2LayerDecode(
        at::Tensor initial_hidden,
        at::Tensor input_norm_weight,
        at::Tensor packed_qkv_weight,
        at::Tensor attention_output_weight,
        at::Tensor post_attention_norm_weight,
        at::Tensor packed_gate_up_weight,
        at::Tensor down_projection_weight,
        at::Tensor rope_cos,
        at::Tensor rope_sin,
        at::Tensor initial_key_cache,
        at::Tensor initial_value_cache,
        std::int64_t cache_capacity,
        std::int64_t initial_position,
        double epsilon,
        double attention_scale);
    ~NativeSmolLM2LayerDecode() override;

    at::Tensor replay(const c10::optional<at::Tensor>& hidden);
    void reset(
        const at::Tensor& hidden,
        const at::Tensor& key_cache,
        const at::Tensor& value_cache,
        std::int64_t position);

    at::Tensor output() const;
    at::Tensor key_cache() const;
    at::Tensor value_cache() const;
    at::Tensor device_position() const;
    at::Tensor workspace() const;
    std::vector<std::int64_t> addresses() const;
    std::int64_t position();
    std::int64_t cache_length();
    std::int64_t capacity() const;
    std::int64_t replay_count() const;
    std::int64_t workspace_bytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class NativeSmolLM2Decode final : public torch::CustomClassHolder {
public:
    NativeSmolLM2Decode(
        at::Tensor initial_token,
        at::Tensor embedding_weight,
        std::vector<at::Tensor> input_norm_weights,
        std::vector<at::Tensor> packed_qkv_weights,
        std::vector<at::Tensor> attention_output_weights,
        std::vector<at::Tensor> post_attention_norm_weights,
        std::vector<at::Tensor> packed_gate_up_weights,
        std::vector<at::Tensor> down_projection_weights,
        at::Tensor final_norm_weight,
        at::Tensor lm_head_weight,
        at::Tensor rope_cos,
        at::Tensor rope_sin,
        std::vector<at::Tensor> initial_key_caches,
        std::vector<at::Tensor> initial_value_caches,
        std::int64_t cache_capacity,
        std::int64_t initial_position,
        std::vector<double> epsilons,
        std::vector<double> attention_scales);
    ~NativeSmolLM2Decode() override;

    at::Tensor replay(const c10::optional<at::Tensor>& token);
    at::Tensor generate_greedy(std::int64_t decode_steps);
    void reset(
        const at::Tensor& token,
        const std::vector<at::Tensor>& key_caches,
        const std::vector<at::Tensor>& value_caches,
        std::int64_t position);

    at::Tensor logits() const;
    at::Tensor current_token() const;
    at::Tensor generated_tokens() const;
    at::Tensor device_generation_step() const;
    at::Tensor key_cache() const;
    at::Tensor value_cache() const;
    at::Tensor device_position() const;
    at::Tensor device_cache_length() const;
    at::Tensor workspace() const;
    std::vector<std::int64_t> addresses() const;
    std::int64_t position();
    std::int64_t cache_length();
    std::int64_t capacity() const;
    std::int64_t replay_count() const;
    std::int64_t generation_step();
    std::int64_t workspace_bytes() const;
    std::int64_t stable_buffer_bytes() const;
    std::int64_t vocabulary_size() const;
    std::int64_t layer_count() const;

    // Native prefill writes directly into this runtime's owned compact cache.
    // These methods are intentionally not part of the Python registration.
    void wait_for_prefill(cudaStream_t stream);
    void install_prefilled_state(
        const at::Tensor& token,
        std::int64_t position,
        cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flux
