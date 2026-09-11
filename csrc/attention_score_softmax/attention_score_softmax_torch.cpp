#include "attention_score_softmax.h"
#include "attention_score_softmax_cuda.h"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cmath>
#include <cfloat>
#include <cstddef>

namespace flux {
namespace {

void validate_attention_score_softmax_arguments(
    const at::Tensor& scores,
    const at::Tensor& additive_attention_mask,
    const double scale) {
    TORCH_CHECK(scores.defined(),
        "flux::attention_score_softmax: scores must be defined");
    TORCH_CHECK(additive_attention_mask.defined(),
        "flux::attention_score_softmax: additive_attention_mask must be defined");
    TORCH_CHECK(scores.dim() == 4,
        "flux::attention_score_softmax: scores must be rank four");
    TORCH_CHECK(additive_attention_mask.dim() == 4,
        "flux::attention_score_softmax: additive_attention_mask must be rank four");
    TORCH_CHECK(scores.numel() > 0,
        "flux::attention_score_softmax: scores dimensions must be non-empty");
    TORCH_CHECK(scores.scalar_type() == at::kFloat,
        "flux::attention_score_softmax: scores must have dtype torch.float32");
    TORCH_CHECK(additive_attention_mask.scalar_type() == at::kFloat,
        "flux::attention_score_softmax: additive_attention_mask must have dtype torch.float32");
    TORCH_CHECK(scores.device() == additive_attention_mask.device(),
        "flux::attention_score_softmax: tensor devices must match");
    for (int64_t dimension = 0; dimension < 4; ++dimension) {
        const int64_t mask_size = additive_attention_mask.size(dimension);
        TORCH_CHECK(mask_size == 1 || mask_size == scores.size(dimension),
            "flux::attention_score_softmax: additive_attention_mask is not broadcastable to scores");
    }
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= FLT_MAX,
        "flux::attention_score_softmax: scale must be a finite FP32 value");
    TORCH_CHECK(
        !at::GradMode::is_enabled() ||
            (!scores.requires_grad() && !additive_attention_mask.requires_grad()),
        "flux::attention_score_softmax is inference-only; use torch.no_grad() or "
        "torch.inference_mode() for tensors that require gradients");
}

at::Tensor attention_score_softmax_cpu(
    const at::Tensor& scores,
    const at::Tensor& additive_attention_mask,
    const double scale) {
    validate_attention_score_softmax_arguments(
        scores, additive_attention_mask, scale);
    TORCH_CHECK(scores.device().is_cpu(),
        "flux::attention_score_softmax: CPU dispatch requires CPU tensors");

    const at::Tensor contiguous_scores = scores.contiguous();
    const at::Tensor contiguous_mask = additive_attention_mask.contiguous();
    at::Tensor output = at::empty(scores.sizes(), scores.options());
    attention_score_softmax_fp32(
        contiguous_scores.const_data_ptr<float>(),
        contiguous_mask.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        static_cast<float>(scale),
        static_cast<std::size_t>(scores.size(0)),
        static_cast<std::size_t>(scores.size(1)),
        static_cast<std::size_t>(scores.size(2)),
        static_cast<std::size_t>(scores.size(3)),
        static_cast<std::size_t>(contiguous_mask.size(0)),
        static_cast<std::size_t>(contiguous_mask.size(1)),
        static_cast<std::size_t>(contiguous_mask.size(2)),
        static_cast<std::size_t>(contiguous_mask.size(3)));
    return output;
}

at::Tensor attention_score_softmax_cuda(
    const at::Tensor& scores,
    const at::Tensor& additive_attention_mask,
    const double scale) {
    validate_attention_score_softmax_arguments(
        scores, additive_attention_mask, scale);
    TORCH_CHECK(scores.is_cuda(),
        "flux::attention_score_softmax: CUDA dispatch requires CUDA tensors");

    const c10::cuda::CUDAGuard device_guard(scores.device());
    const at::Tensor contiguous_scores = scores.contiguous();
    const at::Tensor contiguous_mask = additive_attention_mask.contiguous();
    at::Tensor output = at::empty(scores.sizes(), scores.options());
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(scores.get_device());

    C10_CUDA_CHECK(attention_score_softmax_cuda_fp32(
        contiguous_scores.const_data_ptr<float>(),
        contiguous_mask.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        static_cast<float>(scale),
        static_cast<std::size_t>(scores.size(0)),
        static_cast<std::size_t>(scores.size(1)),
        static_cast<std::size_t>(scores.size(2)),
        static_cast<std::size_t>(scores.size(3)),
        static_cast<std::size_t>(contiguous_mask.size(0)),
        static_cast<std::size_t>(contiguous_mask.size(1)),
        static_cast<std::size_t>(contiguous_mask.size(2)),
        static_cast<std::size_t>(contiguous_mask.size(3)),
        static_cast<std::size_t>(contiguous_mask.stride(0)),
        static_cast<std::size_t>(contiguous_mask.stride(1)),
        static_cast<std::size_t>(contiguous_mask.stride(2)),
        stream.stream()));
    return output;
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "attention_score_softmax(Tensor scores, Tensor additive_attention_mask, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl(
        "attention_score_softmax",
        TORCH_FN(flux::attention_score_softmax_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl(
        "attention_score_softmax",
        TORCH_FN(flux::attention_score_softmax_cuda));
}
