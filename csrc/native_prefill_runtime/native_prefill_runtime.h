#pragma once

#include "native_decode_runtime.h"

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace flux {

class NativeSmolLM2Prefill final : public torch::CustomClassHolder {
public:
    NativeSmolLM2Prefill(
        at::Tensor input_ids,
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
        std::int64_t cache_capacity,
        std::vector<double> epsilons,
        std::vector<double> attention_scales);
    ~NativeSmolLM2Prefill() override;

    at::Tensor prefill(const at::Tensor& input_ids);
    at::Tensor replay(const c10::optional<at::Tensor>& token);
    at::Tensor generate_greedy(std::int64_t max_new_tokens);
    at::Tensor logits() const;
    at::Tensor current_token() const;
    at::Tensor generated_tokens() const;
    at::Tensor device_generation_step() const;
    at::Tensor final_hidden() const;
    at::Tensor key_cache() const;
    at::Tensor value_cache() const;
    at::Tensor device_position() const;
    at::Tensor device_cache_length() const;
    std::vector<std::int64_t> addresses() const;
    std::int64_t position();
    std::int64_t cache_length();
    std::int64_t prompt_length() const;
    std::int64_t capacity() const;
    std::int64_t replay_count() const;
    std::int64_t generation_step();
    std::int64_t workspace_bytes() const;
    std::int64_t cache_bytes() const;
    std::int64_t stable_buffer_bytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flux
