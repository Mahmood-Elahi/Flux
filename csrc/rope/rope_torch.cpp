#include "rope.h"
#include "rope_cuda.h"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstddef>

namespace flux {
namespace {

void validate_rope_arguments(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
    TORCH_CHECK(query.defined() && key.defined() && cos.defined() && sin.defined(),
        "flux::rope: all tensors must be defined");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4,
        "flux::rope: query and key must be rank four");
    TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3,
        "flux::rope: cos and sin must be rank three");
    TORCH_CHECK(
        query.scalar_type() == at::kFloat && key.scalar_type() == at::kFloat,
        "flux::rope: query and key must have dtype torch.float32");
    TORCH_CHECK(
        cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
        "flux::rope: cos and sin must have dtype torch.float32");
    TORCH_CHECK(
        query.device() == key.device() && query.device() == cos.device() &&
            query.device() == sin.device(),
        "flux::rope: all tensors must be on the same device");
    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
        query.size(3) > 0 && key.size(1) > 0,
        "flux::rope: tensor dimensions must be non-empty");
    TORCH_CHECK(query.size(3) % 2 == 0,
        "flux::rope: head_dim must be even");
    TORCH_CHECK(key.size(0) == query.size(0) && key.size(2) == query.size(2) &&
        key.size(3) == query.size(3),
        "flux::rope: query and key batch, sequence, and head dimensions must match");
    TORCH_CHECK(cos.sizes() == sin.sizes(),
        "flux::rope: cos and sin shapes must match");
    TORCH_CHECK((cos.size(0) == 1 || cos.size(0) == query.size(0)) &&
        cos.size(1) == query.size(2) && cos.size(2) == query.size(3),
        "flux::rope: cos and sin shapes are not broadcastable to query and key");
    TORCH_CHECK(
        !at::GradMode::is_enabled() ||
            (!query.requires_grad() && !key.requires_grad() &&
             !cos.requires_grad() && !sin.requires_grad()),
        "flux::rope is inference-only; use torch.no_grad() or torch.inference_mode() "
        "for tensors that require gradients");
}

RopeStrides tensor_strides(const at::Tensor& tensor) {
    return {
        static_cast<std::ptrdiff_t>(tensor.stride(0)),
        static_cast<std::ptrdiff_t>(tensor.stride(1)),
        static_cast<std::ptrdiff_t>(tensor.stride(2)),
        static_cast<std::ptrdiff_t>(tensor.stride(3)),
    };
}

RopeEmbeddingStrides embedding_strides(const at::Tensor& tensor) {
    return {
        static_cast<std::ptrdiff_t>(tensor.stride(0)),
        static_cast<std::ptrdiff_t>(tensor.stride(1)),
        static_cast<std::ptrdiff_t>(tensor.stride(2)),
    };
}

std::tuple<at::Tensor, at::Tensor> rope_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
    validate_rope_arguments(query, key, cos, sin);
    TORCH_CHECK(query.device().is_cpu(),
        "flux::rope: CPU dispatch requires CPU tensors");
    at::Tensor query_output = at::empty(query.sizes(), query.options());
    at::Tensor key_output = at::empty(key.sizes(), key.options());
    rope_fp32(
        query.const_data_ptr<float>(), key.const_data_ptr<float>(),
        cos.const_data_ptr<float>(), sin.const_data_ptr<float>(),
        query_output.mutable_data_ptr<float>(),
        key_output.mutable_data_ptr<float>(),
        static_cast<std::size_t>(query.size(0)),
        static_cast<std::size_t>(query.size(1)),
        static_cast<std::size_t>(key.size(1)),
        static_cast<std::size_t>(query.size(2)),
        static_cast<std::size_t>(query.size(3)),
        static_cast<std::size_t>(cos.size(0)),
        tensor_strides(query), tensor_strides(key),
        embedding_strides(cos), embedding_strides(sin));
    return {query_output, key_output};
}

std::tuple<at::Tensor, at::Tensor> rope_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
    validate_rope_arguments(query, key, cos, sin);
    TORCH_CHECK(query.is_cuda(),
        "flux::rope: CUDA dispatch requires CUDA tensors");
    const c10::cuda::CUDAGuard device_guard(query.device());
    at::Tensor query_output = at::empty(query.sizes(), query.options());
    at::Tensor key_output = at::empty(key.sizes(), key.options());
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(query.get_device());
    C10_CUDA_CHECK(rope_cuda_fp32(
        query.const_data_ptr<float>(), key.const_data_ptr<float>(),
        cos.const_data_ptr<float>(), sin.const_data_ptr<float>(),
        query_output.mutable_data_ptr<float>(),
        key_output.mutable_data_ptr<float>(),
        static_cast<std::size_t>(query.size(0)),
        static_cast<std::size_t>(query.size(1)),
        static_cast<std::size_t>(key.size(1)),
        static_cast<std::size_t>(query.size(2)),
        static_cast<std::size_t>(query.size(3)),
        static_cast<std::size_t>(cos.size(0)),
        tensor_strides(query), tensor_strides(key),
        embedding_strides(cos), embedding_strides(sin), stream.stream()));
    return {query_output, key_output};
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "rope(Tensor query, Tensor key, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl("rope", TORCH_FN(flux::rope_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl("rope", TORCH_FN(flux::rope_cuda));
}
