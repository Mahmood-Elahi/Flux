#include "gqa_decode_attention_cuda.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace flux {
namespace {

void validate_gqa_decode_attention_arguments(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<at::Tensor>& additive_attention_mask,
    const double scale,
    const std::optional<at::Tensor>& cache_length) {
    TORCH_CHECK(query.defined() && key_cache.defined() && value_cache.defined(),
        "flux::gqa_decode_attention: Q/K/V tensors must be defined");
    TORCH_CHECK(query.dim() == 4 && key_cache.dim() == 4 && value_cache.dim() == 4,
        "flux::gqa_decode_attention: Q/K/V tensors must be rank four");
    TORCH_CHECK(query.size(2) == 1,
        "flux::gqa_decode_attention: query length must be one");
    TORCH_CHECK(query.size(0) == 1,
        "flux::gqa_decode_attention: batch size must be one");
    TORCH_CHECK(query.scalar_type() == at::kFloat &&
            key_cache.scalar_type() == at::kFloat &&
            value_cache.scalar_type() == at::kFloat,
        "flux::gqa_decode_attention: Q/K/V tensors must have dtype torch.float32");
    TORCH_CHECK(query.device() == key_cache.device() &&
            query.device() == value_cache.device(),
        "flux::gqa_decode_attention: Q/K/V tensor devices must match");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
        "flux::gqa_decode_attention: key/value cache shapes must match");
    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 &&
            query.size(3) > 0 && key_cache.size(1) > 0 && key_cache.size(2) > 0,
        "flux::gqa_decode_attention: tensor dimensions must be non-empty");
    TORCH_CHECK(query.size(0) == key_cache.size(0),
        "flux::gqa_decode_attention: cache batch must match query");
    TORCH_CHECK(query.size(3) == key_cache.size(3),
        "flux::gqa_decode_attention: cache head dimension must match query");
    TORCH_CHECK(query.size(1) % key_cache.size(1) == 0,
        "flux::gqa_decode_attention: query heads must be divisible by KV heads");
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= FLT_MAX,
        "flux::gqa_decode_attention: scale must be a finite FP32 value");
    for (const at::Tensor* tensor : {&query, &key_cache, &value_cache}) {
        for (const auto stride : tensor->strides()) {
            TORCH_CHECK(stride >= 0,
                "flux::gqa_decode_attention: negative tensor strides are unsupported");
        }
    }

    if (additive_attention_mask.has_value()) {
        const at::Tensor& mask = *additive_attention_mask;
        TORCH_CHECK(mask.defined() && mask.dim() == 4,
            "flux::gqa_decode_attention: additive_attention_mask must be rank four");
        TORCH_CHECK(mask.scalar_type() == at::kFloat,
            "flux::gqa_decode_attention: additive_attention_mask must be float32");
        TORCH_CHECK(mask.device() == query.device(),
            "flux::gqa_decode_attention: mask device must match query");
        const std::array<int64_t, 4> target = {
            query.size(0), query.size(1), 1, key_cache.size(2)};
        for (int64_t dimension = 0; dimension < 4; ++dimension) {
            TORCH_CHECK(mask.size(dimension) == 1 ||
                    mask.size(dimension) == target[dimension],
                "flux::gqa_decode_attention: mask is not broadcastable");
            TORCH_CHECK(mask.stride(dimension) >= 0,
                "flux::gqa_decode_attention: negative mask strides are unsupported");
        }
    }
    if (cache_length.has_value()) {
        const at::Tensor& length = *cache_length;
        TORCH_CHECK(length.defined() && length.numel() == 1 &&
                length.scalar_type() == at::kLong,
            "flux::gqa_decode_attention: cache_length must be a scalar int64 tensor");
        TORCH_CHECK(length.device() == query.device(),
            "flux::gqa_decode_attention: cache_length device must match query");
        TORCH_CHECK(!length.requires_grad(),
            "flux::gqa_decode_attention: cache_length must not require gradients");
    }
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            (!query.requires_grad() && !key_cache.requires_grad() &&
             !value_cache.requires_grad() &&
             (!additive_attention_mask.has_value() ||
              !additive_attention_mask->requires_grad())),
        "flux::gqa_decode_attention is inference-only; use torch.no_grad() or "
        "torch.inference_mode() for tensors that require gradients");
}

at::Tensor output_for(const at::Tensor& query) {
    return at::empty(query.sizes(), query.options());
}

void validate_cuda_outputs(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<at::Tensor>& additive_attention_mask,
    const std::optional<at::Tensor>& cache_length,
    const at::Tensor& output,
    const at::Tensor& workspace) {
    TORCH_CHECK(query.size(3) == 64 && key_cache.size(2) > 512,
        "flux::gqa_decode_attention_out: stable workspace path requires head_dim=64 and capacity > 512");
    TORCH_CHECK(output.defined() && output.sizes() == query.sizes(),
        "flux::gqa_decode_attention_out: output shape must match query");
    TORCH_CHECK(output.scalar_type() == at::kFloat &&
            output.device() == query.device() && output.is_contiguous(),
        "flux::gqa_decode_attention_out: output must be contiguous float32 on the query device");
    constexpr int64_t chunk_size = 128;
    const int64_t num_chunks =
        (key_cache.size(2) + chunk_size - 1) / chunk_size;
    const std::array<int64_t, 4> workspace_sizes = {
        query.size(0), query.size(1), num_chunks, query.size(3) + 2};
    TORCH_CHECK(workspace.defined() && workspace.sizes() == workspace_sizes,
        "flux::gqa_decode_attention_out: workspace shape is incorrect");
    TORCH_CHECK(workspace.scalar_type() == at::kFloat &&
            workspace.device() == query.device() && workspace.is_contiguous(),
        "flux::gqa_decode_attention_out: workspace must be contiguous float32 on the query device");
    at::assert_no_internal_overlap(output);
    at::assert_no_internal_overlap(workspace);
    TORCH_CHECK(!output.is_alias_of(workspace),
        "flux::gqa_decode_attention_out: output and workspace must not alias");
    for (const at::Tensor* input : {&query, &key_cache, &value_cache}) {
        TORCH_CHECK(!output.is_alias_of(*input) && !workspace.is_alias_of(*input),
            "flux::gqa_decode_attention_out: outputs must not alias Q/K/V");
    }
    if (additive_attention_mask.has_value()) {
        TORCH_CHECK(!output.is_alias_of(*additive_attention_mask) &&
                !workspace.is_alias_of(*additive_attention_mask),
            "flux::gqa_decode_attention_out: outputs must not alias the mask");
    }
    if (cache_length.has_value()) {
        TORCH_CHECK(!output.is_alias_of(*cache_length) &&
                !workspace.is_alias_of(*cache_length),
            "flux::gqa_decode_attention_out: outputs must not alias cache_length");
    }
}

at::Tensor launch_gqa_decode_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<at::Tensor>& additive_attention_mask,
    const double scale,
    const std::optional<at::Tensor>& cache_length,
    at::Tensor output,
    float* workspace_data,
    const int64_t num_chunks) {
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(query.get_device());
    const std::array<int64_t, 4> query_strides = {
        query.stride(0), query.stride(1), query.stride(2), query.stride(3)};
    const std::array<int64_t, 4> key_strides = {
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3)};
    const std::array<int64_t, 4> value_strides = {
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3)};
    std::array<int64_t, 4> mask_sizes = {1, 1, 1, 1};
    std::array<int64_t, 4> mask_strides = {0, 0, 0, 0};
    if (additive_attention_mask.has_value()) {
        for (int64_t dimension = 0; dimension < 4; ++dimension) {
            mask_sizes[dimension] = additive_attention_mask->size(dimension);
            mask_strides[dimension] = additive_attention_mask->stride(dimension);
        }
    }
    C10_CUDA_CHECK(gqa_decode_attention_cuda_fp32(
        query.const_data_ptr<float>(), key_cache.const_data_ptr<float>(),
        value_cache.const_data_ptr<float>(),
        additive_attention_mask.has_value()
            ? additive_attention_mask->const_data_ptr<float>() : nullptr,
        cache_length.has_value()
            ? cache_length->const_data_ptr<int64_t>() : nullptr,
        output.mutable_data_ptr<float>(), workspace_data,
        static_cast<float>(scale),
        static_cast<std::size_t>(query.size(0)),
        static_cast<std::size_t>(query.size(1)),
        static_cast<std::size_t>(key_cache.size(1)),
        static_cast<std::size_t>(key_cache.size(2)),
        static_cast<std::size_t>(query.size(3)),
        static_cast<std::size_t>(num_chunks),
        query_strides.data(), key_strides.data(), value_strides.data(),
        mask_sizes.data(), mask_strides.data(), stream.stream()));
    return output;
}

at::Tensor gqa_decode_attention_cpu(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<at::Tensor>& additive_attention_mask,
    const double scale,
    const std::optional<at::Tensor>& cache_length) {
    validate_gqa_decode_attention_arguments(
        query, key_cache, value_cache, additive_attention_mask, scale, cache_length);
    TORCH_CHECK(query.device().is_cpu(),
        "flux::gqa_decode_attention: CPU dispatch requires CPU tensors");
    const int64_t capacity = key_cache.size(2);
    const int64_t valid_length = cache_length.has_value()
        ? cache_length->item<int64_t>()
        : capacity;
    TORCH_CHECK(valid_length >= 1 && valid_length <= capacity,
        "flux::gqa_decode_attention: cache_length must be within cache capacity");

    at::Tensor output = output_for(query);
    const float* q = query.const_data_ptr<float>();
    const float* k = key_cache.const_data_ptr<float>();
    const float* v = value_cache.const_data_ptr<float>();
    const float* mask = additive_attention_mask.has_value()
        ? additive_attention_mask->const_data_ptr<float>()
        : nullptr;
    float* result = output.mutable_data_ptr<float>();
    const int64_t query_heads = query.size(1);
    const int64_t kv_heads = key_cache.size(1);
    const int64_t groups = query_heads / kv_heads;
    const int64_t head_dim = query.size(3);
    std::vector<float> scores(static_cast<std::size_t>(valid_length));

    for (int64_t batch = 0; batch < query.size(0); ++batch) {
        for (int64_t query_head = 0; query_head < query_heads; ++query_head) {
            const int64_t kv_head = query_head / groups;
            float row_max = -FLT_MAX;
            for (int64_t position = 0; position < valid_length; ++position) {
                float dot = 0.0F;
                for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                    dot += q[batch * query.stride(0) +
                             query_head * query.stride(1) +
                             dimension * query.stride(3)] *
                        k[batch * key_cache.stride(0) +
                          kv_head * key_cache.stride(1) +
                          position * key_cache.stride(2) +
                          dimension * key_cache.stride(3)];
                }
                float score = dot * static_cast<float>(scale);
                if (mask != nullptr) {
                    const at::Tensor& mask_tensor = *additive_attention_mask;
                    score += mask[
                        (mask_tensor.size(0) == 1 ? 0 : batch) * mask_tensor.stride(0) +
                        (mask_tensor.size(1) == 1 ? 0 : query_head) * mask_tensor.stride(1) +
                        (mask_tensor.size(3) == 1 ? 0 : position) * mask_tensor.stride(3)];
                }
                scores[static_cast<std::size_t>(position)] = score;
                row_max = std::max(row_max, score);
            }
            float row_sum = 0.0F;
            for (int64_t position = 0; position < valid_length; ++position) {
                const float value = std::exp(
                    scores[static_cast<std::size_t>(position)] - row_max);
                scores[static_cast<std::size_t>(position)] = value;
                row_sum += value;
            }
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                float value = 0.0F;
                for (int64_t position = 0; position < valid_length; ++position) {
                    value += scores[static_cast<std::size_t>(position)] *
                        v[batch * value_cache.stride(0) +
                          kv_head * value_cache.stride(1) +
                          position * value_cache.stride(2) +
                          dimension * value_cache.stride(3)];
                }
                result[(batch * query_heads + query_head) * head_dim + dimension] =
                    value / row_sum;
            }
        }
    }
    return output;
}

at::Tensor gqa_decode_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<at::Tensor>& additive_attention_mask,
    const double scale,
    const std::optional<at::Tensor>& cache_length) {
    validate_gqa_decode_attention_arguments(
        query, key_cache, value_cache, additive_attention_mask, scale, cache_length);
    TORCH_CHECK(query.is_cuda(),
        "flux::gqa_decode_attention: CUDA dispatch requires CUDA tensors");
    TORCH_CHECK(key_cache.size(2) <= 8192,
        "flux::gqa_decode_attention: CUDA cache capacity exceeds 8192 tokens");
    const c10::cuda::CUDAGuard device_guard(query.device());
    at::Tensor output = output_for(query);
    constexpr int64_t chunk_size = 128;
    const int64_t num_chunks =
        (key_cache.size(2) + chunk_size - 1) / chunk_size;
    at::Tensor workspace;
    float* workspace_data = nullptr;
    if (query.size(3) == 64 && key_cache.size(2) > 512) {
        workspace = at::empty(
            {query.size(0), query.size(1), num_chunks, query.size(3) + 2},
            query.options());
        workspace_data = workspace.mutable_data_ptr<float>();
    }
    return launch_gqa_decode_attention_cuda(
        query, key_cache, value_cache, additive_attention_mask, scale,
        cache_length, output, workspace_data, num_chunks);
}

at::Tensor gqa_decode_attention_cuda_out(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<at::Tensor>& additive_attention_mask,
    const double scale,
    const std::optional<at::Tensor>& cache_length,
    at::Tensor output,
    at::Tensor workspace) {
    validate_gqa_decode_attention_arguments(
        query, key_cache, value_cache, additive_attention_mask, scale, cache_length);
    TORCH_CHECK(query.is_cuda(),
        "flux::gqa_decode_attention_out: CUDA dispatch requires CUDA tensors");
    TORCH_CHECK(key_cache.size(2) <= 8192,
        "flux::gqa_decode_attention_out: CUDA cache capacity exceeds 8192 tokens");
    validate_cuda_outputs(
        query, key_cache, value_cache, additive_attention_mask, cache_length,
        output, workspace);
    const c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t num_chunks = (key_cache.size(2) + 127) / 128;
    return launch_gqa_decode_attention_cuda(
        query, key_cache, value_cache, additive_attention_mask, scale,
        cache_length, output, workspace.mutable_data_ptr<float>(), num_chunks);
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "gqa_decode_attention(Tensor query, Tensor key_cache, Tensor value_cache, "
        "Tensor? additive_attention_mask, float scale, Tensor? cache_length=None) -> Tensor");
    library.def(
        "gqa_decode_attention_out(Tensor query, Tensor key_cache, Tensor value_cache, "
        "Tensor? additive_attention_mask, float scale, Tensor? cache_length, "
        "Tensor(a!) output, Tensor(b!) workspace) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl("gqa_decode_attention", TORCH_FN(flux::gqa_decode_attention_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl("gqa_decode_attention", TORCH_FN(flux::gqa_decode_attention_cuda));
    library.impl(
        "gqa_decode_attention_out", TORCH_FN(flux::gqa_decode_attention_cuda_out));
}
