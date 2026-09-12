#include "packed_qkv_rope_cache_cuda.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstddef>
#include <cstdint>

namespace flux {
namespace {

constexpr int64_t kQueryHeads = 9;
constexpr int64_t kKeyValueHeads = 3;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kPackedWidth = 960;

void validate_arguments(
    const at::Tensor& packed_qkv,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& cache_length) {
    TORCH_CHECK(
        packed_qkv.defined() && cos.defined() && sin.defined() &&
            key_cache.defined() && value_cache.defined() &&
            cache_length.defined(),
        "flux::packed_qkv_rope_cache: all tensors must be defined");
    TORCH_CHECK(packed_qkv.dim() == 3 && packed_qkv.size(0) == 1 &&
            packed_qkv.size(1) == 1 && packed_qkv.size(2) == kPackedWidth,
        "flux::packed_qkv_rope_cache: packed_qkv must have shape [1, 1, 960]");
    TORCH_CHECK(cos.sizes() == at::IntArrayRef({1, 1, kHeadDim}) &&
            sin.sizes() == cos.sizes(),
        "flux::packed_qkv_rope_cache: cos and sin must have shape [1, 1, 64]");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4 &&
            key_cache.sizes() == value_cache.sizes() &&
            key_cache.size(0) == 1 && key_cache.size(1) == kKeyValueHeads &&
            key_cache.size(2) > 0 && key_cache.size(3) == kHeadDim,
        "flux::packed_qkv_rope_cache: K/V cache must have matching shape [1, 3, capacity, 64]");
    TORCH_CHECK(cache_length.dim() == 0 && cache_length.numel() == 1 &&
            cache_length.scalar_type() == at::kLong,
        "flux::packed_qkv_rope_cache: cache_length must be a scalar torch.int64 tensor");
    for (const at::Tensor* tensor :
         {&packed_qkv, &cos, &sin, &key_cache, &value_cache}) {
        TORCH_CHECK(tensor->scalar_type() == at::kFloat,
            "flux::packed_qkv_rope_cache: data tensors must have dtype torch.float32");
        TORCH_CHECK(tensor->device() == packed_qkv.device(),
            "flux::packed_qkv_rope_cache: all tensors must be on the same device");
    }
    TORCH_CHECK(cache_length.device() == packed_qkv.device(),
        "flux::packed_qkv_rope_cache: all tensors must be on the same device");
    for (int64_t dimension = 0; dimension < packed_qkv.dim(); ++dimension) {
        TORCH_CHECK(packed_qkv.stride(dimension) > 0,
            "flux::packed_qkv_rope_cache: packed_qkv strides must be positive");
    }
    TORCH_CHECK(key_cache.stride(0) > 0 && key_cache.stride(1) > 0 &&
            key_cache.stride(2) > 0 && key_cache.stride(3) > 0 &&
            value_cache.stride(0) > 0 && value_cache.stride(1) > 0 &&
            value_cache.stride(2) > 0 && value_cache.stride(3) > 0,
        "flux::packed_qkv_rope_cache: cache strides must be positive");
    TORCH_CHECK(
        !at::GradMode::is_enabled() ||
            (!packed_qkv.requires_grad() && !cos.requires_grad() &&
             !sin.requires_grad() && !key_cache.requires_grad() &&
             !value_cache.requires_grad() && !cache_length.requires_grad()),
        "flux::packed_qkv_rope_cache is inference-only; use torch.no_grad() or torch.inference_mode()");
}

PackedQKVStrides packed_strides(const at::Tensor& tensor) {
    return {tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}

PackedQKVEmbeddingStrides embedding_strides(const at::Tensor& tensor) {
    return {tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}

PackedQKVCacheStrides cache_strides(const at::Tensor& tensor) {
    return {
        tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)};
}

void validate_query_output(
    const at::Tensor& packed_qkv,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& cache_length,
    const at::Tensor& query_output) {
    TORCH_CHECK(query_output.defined() &&
            query_output.sizes() == at::IntArrayRef({1, kQueryHeads, 1, kHeadDim}),
        "flux::packed_qkv_rope_cache_out: query_output must have shape [1, 9, 1, 64]");
    TORCH_CHECK(query_output.scalar_type() == at::kFloat &&
            query_output.device() == packed_qkv.device() &&
            query_output.is_contiguous(),
        "flux::packed_qkv_rope_cache_out: query_output must be contiguous float32 on the input device");
    at::assert_no_internal_overlap(query_output);
    for (const at::Tensor* input :
         {&packed_qkv, &cos, &sin, &key_cache, &value_cache, &cache_length}) {
        TORCH_CHECK(!query_output.is_alias_of(*input),
            "flux::packed_qkv_rope_cache_out: query_output must not alias an input");
    }
}

at::Tensor launch_packed_qkv_rope_cache(
    const at::Tensor& packed_qkv,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& cache_length,
    at::Tensor query_output) {
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(packed_qkv.get_device());
    C10_CUDA_CHECK(packed_qkv_rope_cache_cuda_fp32(
        packed_qkv.const_data_ptr<float>(), cos.const_data_ptr<float>(),
        sin.const_data_ptr<float>(), key_cache.mutable_data_ptr<float>(),
        value_cache.mutable_data_ptr<float>(),
        cache_length.mutable_data_ptr<std::int64_t>(),
        query_output.mutable_data_ptr<float>(),
        static_cast<std::size_t>(key_cache.size(2)),
        packed_strides(packed_qkv), embedding_strides(cos),
        embedding_strides(sin), cache_strides(key_cache),
        cache_strides(value_cache), stream.stream()));
    return query_output;
}

at::Tensor packed_qkv_rope_cache_cuda(
    const at::Tensor& packed_qkv,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& cache_length) {
    validate_arguments(
        packed_qkv, cos, sin, key_cache, value_cache, cache_length);
    TORCH_CHECK(packed_qkv.is_cuda(),
        "flux::packed_qkv_rope_cache: CUDA dispatch requires CUDA tensors");
    const c10::cuda::CUDAGuard device_guard(packed_qkv.device());
    at::Tensor query_output = at::empty(
        {1, kQueryHeads, 1, kHeadDim}, packed_qkv.options());
    return launch_packed_qkv_rope_cache(
        packed_qkv, cos, sin, key_cache, value_cache, cache_length, query_output);
}

at::Tensor packed_qkv_rope_cache_cuda_out(
    const at::Tensor& packed_qkv,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& cache_length,
    at::Tensor query_output) {
    validate_arguments(
        packed_qkv, cos, sin, key_cache, value_cache, cache_length);
    TORCH_CHECK(packed_qkv.is_cuda(),
        "flux::packed_qkv_rope_cache_out: CUDA dispatch requires CUDA tensors");
    validate_query_output(
        packed_qkv, cos, sin, key_cache, value_cache, cache_length, query_output);
    const c10::cuda::CUDAGuard device_guard(packed_qkv.device());
    return launch_packed_qkv_rope_cache(
        packed_qkv, cos, sin, key_cache, value_cache, cache_length, query_output);
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "packed_qkv_rope_cache(Tensor packed_qkv, Tensor cos, Tensor sin, "
        "Tensor(a!) key_cache, Tensor(b!) value_cache, "
        "Tensor(c!) cache_length) -> Tensor");
    library.def(
        "packed_qkv_rope_cache_out(Tensor packed_qkv, Tensor cos, Tensor sin, "
        "Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor(c!) cache_length, "
        "Tensor(d!) query_output) -> Tensor(d!)");
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl(
        "packed_qkv_rope_cache", TORCH_FN(flux::packed_qkv_rope_cache_cuda));
    library.impl(
        "packed_qkv_rope_cache_out",
        TORCH_FN(flux::packed_qkv_rope_cache_cuda_out));
}
