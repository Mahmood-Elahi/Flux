#include "residual_rmsnorm.h"
#include "residual_rmsnorm_cuda.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstddef>
#include <limits>
#include <tuple>

namespace flux {
namespace {

void validate_residual_rmsnorm_arguments(
    const at::Tensor& hidden,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const double epsilon) {
    TORCH_CHECK(
        hidden.dim() >= 1,
        "flux::residual_rmsnorm: hidden must have at least one dimension");
    TORCH_CHECK(
        hidden.size(-1) > 0,
        "flux::residual_rmsnorm: hidden must have a non-empty final dimension");
    TORCH_CHECK(
        hidden.numel() > 0,
        "flux::residual_rmsnorm: hidden must contain at least one row");
    TORCH_CHECK(
        hidden.sizes() == residual.sizes(),
        "flux::residual_rmsnorm: hidden and residual must have the same shape");
    TORCH_CHECK(
        hidden.scalar_type() == at::kFloat,
        "flux::residual_rmsnorm: hidden must have dtype torch.float32");
    TORCH_CHECK(
        residual.scalar_type() == at::kFloat,
        "flux::residual_rmsnorm: residual must have dtype torch.float32");
    TORCH_CHECK(
        weight.scalar_type() == at::kFloat,
        "flux::residual_rmsnorm: weight must have dtype torch.float32");
    TORCH_CHECK(
        weight.dim() == 1,
        "flux::residual_rmsnorm: weight must be one-dimensional");
    TORCH_CHECK(
        weight.size(0) == hidden.size(-1),
        "flux::residual_rmsnorm: weight length must equal hidden.size(-1)");
    TORCH_CHECK(
        residual.device() == hidden.device() && weight.device() == hidden.device(),
        "flux::residual_rmsnorm: hidden, residual, and weight must be on the same device");
    TORCH_CHECK(
        epsilon >= 0.0 && epsilon <= std::numeric_limits<float>::max(),
        "flux::residual_rmsnorm: epsilon must be a non-negative finite FP32 value");
    TORCH_CHECK(
        !at::GradMode::is_enabled() ||
            (!hidden.requires_grad() && !residual.requires_grad() &&
             !weight.requires_grad()),
        "flux::residual_rmsnorm is inference-only; use torch.no_grad() or "
        "torch.inference_mode() for tensors that require gradients");
}

void validate_residual_rmsnorm_outputs(
    const at::Tensor& hidden,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& norm_out,
    const at::Tensor& residual_out) {
    TORCH_CHECK(
        hidden.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
        "flux::residual_rmsnorm_out: inputs must be contiguous");
    for (const at::Tensor* output : {&norm_out, &residual_out}) {
        TORCH_CHECK(output->defined() && output->sizes() == hidden.sizes(),
            "flux::residual_rmsnorm_out: output shapes must match hidden");
        TORCH_CHECK(output->scalar_type() == at::kFloat &&
                output->device() == hidden.device() && output->is_contiguous(),
            "flux::residual_rmsnorm_out: outputs must be contiguous float32 on the input device");
        at::assert_no_internal_overlap(*output);
        TORCH_CHECK(!output->is_alias_of(hidden) &&
                !output->is_alias_of(residual) && !output->is_alias_of(weight),
            "flux::residual_rmsnorm_out: outputs must not alias inputs");
    }
    TORCH_CHECK(!norm_out.is_alias_of(residual_out),
        "flux::residual_rmsnorm_out: outputs must not alias each other");
}

std::tuple<at::Tensor, at::Tensor> residual_rmsnorm_cpu(
    const at::Tensor& hidden,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const double epsilon) {
    validate_residual_rmsnorm_arguments(hidden, residual, weight, epsilon);
    TORCH_CHECK(
        hidden.device().is_cpu(),
        "flux::residual_rmsnorm: CPU dispatch requires CPU tensors");

    const at::Tensor contiguous_hidden = hidden.contiguous();
    const at::Tensor contiguous_residual = residual.contiguous();
    const at::Tensor contiguous_weight = weight.contiguous();
    at::Tensor residual_out = at::empty(hidden.sizes(), hidden.options());
    at::Tensor norm_out = at::empty(hidden.sizes(), hidden.options());
    const std::size_t hidden_size = static_cast<std::size_t>(hidden.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(hidden.numel() / hidden.size(-1));

    flux::residual_rmsnorm_fp32(
        contiguous_hidden.const_data_ptr<float>(),
        contiguous_residual.const_data_ptr<float>(),
        contiguous_weight.const_data_ptr<float>(),
        norm_out.mutable_data_ptr<float>(),
        residual_out.mutable_data_ptr<float>(),
        num_rows,
        hidden_size,
        static_cast<float>(epsilon));
    return {norm_out, residual_out};
}

std::tuple<at::Tensor, at::Tensor> residual_rmsnorm_cuda(
    const at::Tensor& hidden,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const double epsilon) {
    validate_residual_rmsnorm_arguments(hidden, residual, weight, epsilon);
    TORCH_CHECK(
        hidden.is_cuda(),
        "flux::residual_rmsnorm: CUDA dispatch requires CUDA tensors");

    const c10::cuda::CUDAGuard device_guard(hidden.device());
    const at::Tensor contiguous_hidden = hidden.contiguous();
    const at::Tensor contiguous_residual = residual.contiguous();
    const at::Tensor contiguous_weight = weight.contiguous();
    at::Tensor residual_out = at::empty(hidden.sizes(), hidden.options());
    at::Tensor norm_out = at::empty(hidden.sizes(), hidden.options());
    const std::size_t hidden_size = static_cast<std::size_t>(hidden.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(hidden.numel() / hidden.size(-1));
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(hidden.get_device());

    C10_CUDA_CHECK(flux::residual_rmsnorm_cuda_fp32(
        contiguous_hidden.const_data_ptr<float>(),
        contiguous_residual.const_data_ptr<float>(),
        contiguous_weight.const_data_ptr<float>(),
        norm_out.mutable_data_ptr<float>(),
        residual_out.mutable_data_ptr<float>(),
        num_rows,
        hidden_size,
        static_cast<float>(epsilon),
        stream.stream()));
    return {norm_out, residual_out};
}

std::tuple<at::Tensor, at::Tensor> residual_rmsnorm_cuda_out(
    const at::Tensor& hidden,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const double epsilon,
    at::Tensor norm_out,
    at::Tensor residual_out) {
    validate_residual_rmsnorm_arguments(hidden, residual, weight, epsilon);
    TORCH_CHECK(hidden.is_cuda(),
        "flux::residual_rmsnorm_out: CUDA dispatch requires CUDA tensors");
    validate_residual_rmsnorm_outputs(
        hidden, residual, weight, norm_out, residual_out);
    const c10::cuda::CUDAGuard device_guard(hidden.device());
    const std::size_t hidden_size = static_cast<std::size_t>(hidden.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(hidden.numel() / hidden.size(-1));
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(hidden.get_device());
    C10_CUDA_CHECK(flux::residual_rmsnorm_cuda_fp32(
        hidden.const_data_ptr<float>(), residual.const_data_ptr<float>(),
        weight.const_data_ptr<float>(), norm_out.mutable_data_ptr<float>(),
        residual_out.mutable_data_ptr<float>(), num_rows, hidden_size,
        static_cast<float>(epsilon), stream.stream()));
    return {norm_out, residual_out};
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "residual_rmsnorm(Tensor hidden, Tensor residual, Tensor weight, "
        "float epsilon) -> (Tensor norm_out, Tensor residual_out)");
    library.def(
        "residual_rmsnorm_out(Tensor hidden, Tensor residual, Tensor weight, "
        "float epsilon, Tensor(a!) norm_out, Tensor(b!) residual_out) -> "
        "(Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl("residual_rmsnorm", TORCH_FN(flux::residual_rmsnorm_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl("residual_rmsnorm", TORCH_FN(flux::residual_rmsnorm_cuda));
    library.impl(
        "residual_rmsnorm_out", TORCH_FN(flux::residual_rmsnorm_cuda_out));
}
