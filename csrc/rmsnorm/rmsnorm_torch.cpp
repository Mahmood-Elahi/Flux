#include "rmsnorm.h"
#include "rmsnorm_cuda.h"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstddef>
#include <limits>

namespace flux {
namespace {

void validate_rmsnorm_arguments(
    const at::Tensor& input,
    const at::Tensor& weight,
    const double epsilon) {
    TORCH_CHECK(input.dim() >= 1, "flux::rmsnorm: input must have at least one dimension");
    TORCH_CHECK(
        input.size(-1) > 0,
        "flux::rmsnorm: input must have a non-empty final dimension");
    TORCH_CHECK(
        input.numel() > 0,
        "flux::rmsnorm: input must contain at least one row");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat,
        "flux::rmsnorm: input must have dtype torch.float32");
    TORCH_CHECK(
        weight.scalar_type() == at::kFloat,
        "flux::rmsnorm: weight must have dtype torch.float32");
    TORCH_CHECK(
        weight.dim() == 1,
        "flux::rmsnorm: weight must be one-dimensional");
    TORCH_CHECK(
        weight.size(0) == input.size(-1),
        "flux::rmsnorm: weight length must equal input.size(-1)");
    TORCH_CHECK(
        weight.device() == input.device(),
        "flux::rmsnorm: input and weight must be on the same device");
    TORCH_CHECK(
        epsilon > 0.0 && epsilon <= std::numeric_limits<float>::max(),
        "flux::rmsnorm: epsilon must be a positive finite FP32 value");
    TORCH_CHECK(
        !at::GradMode::is_enabled() ||
            (!input.requires_grad() && !weight.requires_grad()),
        "flux::rmsnorm is inference-only; use torch.no_grad() or "
        "torch.inference_mode() for tensors that require gradients");
}

at::Tensor rmsnorm_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const double epsilon) {
    validate_rmsnorm_arguments(input, weight, epsilon);
    TORCH_CHECK(input.device().is_cpu(), "flux::rmsnorm: CPU dispatch requires CPU tensors");

    const at::Tensor contiguous_input = input.contiguous();
    const at::Tensor contiguous_weight = weight.contiguous();
    at::Tensor output = at::empty(input.sizes(), input.options());
    const std::size_t hidden_size = static_cast<std::size_t>(input.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(input.numel() / input.size(-1));

    flux::rmsnorm_fp32(
        contiguous_input.const_data_ptr<float>(),
        contiguous_weight.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        num_rows,
        hidden_size,
        static_cast<float>(epsilon));
    return output;
}

at::Tensor rmsnorm_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const double epsilon) {
    validate_rmsnorm_arguments(input, weight, epsilon);
    TORCH_CHECK(input.is_cuda(), "flux::rmsnorm: CUDA dispatch requires CUDA tensors");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const at::Tensor contiguous_input = input.contiguous();
    const at::Tensor contiguous_weight = weight.contiguous();
    at::Tensor output = at::empty(input.sizes(), input.options());
    const std::size_t hidden_size = static_cast<std::size_t>(input.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(input.numel() / input.size(-1));
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(input.get_device());

    C10_CUDA_CHECK(flux::rmsnorm_cuda_fp32(
        contiguous_input.const_data_ptr<float>(),
        contiguous_weight.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        num_rows,
        hidden_size,
        static_cast<float>(epsilon),
        stream.stream()));
    return output;
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY(flux, library) {
    library.def("rmsnorm(Tensor input, Tensor weight, float epsilon) -> Tensor");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl("rmsnorm", TORCH_FN(flux::rmsnorm_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl("rmsnorm", TORCH_FN(flux::rmsnorm_cuda));
}
