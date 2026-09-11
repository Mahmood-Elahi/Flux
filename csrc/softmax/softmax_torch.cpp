#include "softmax.h"
#include "softmax_cuda.h"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstddef>

namespace flux {
namespace {

void validate_softmax_argument(const at::Tensor& input) {
    TORCH_CHECK(input.defined(), "flux::softmax: input must be defined");
    TORCH_CHECK(
        input.dim() >= 1,
        "flux::softmax: input must have at least one dimension");
    TORCH_CHECK(
        input.size(-1) > 0,
        "flux::softmax: input must have a non-empty final dimension");
    TORCH_CHECK(
        input.numel() > 0,
        "flux::softmax: input must contain at least one row");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat,
        "flux::softmax: input must have dtype torch.float32");
    TORCH_CHECK(
        !at::GradMode::is_enabled() || !input.requires_grad(),
        "flux::softmax is inference-only; use torch.no_grad() or "
        "torch.inference_mode() for tensors that require gradients");
}

at::Tensor softmax_cpu(const at::Tensor& input) {
    validate_softmax_argument(input);
    TORCH_CHECK(
        input.device().is_cpu(),
        "flux::softmax: CPU dispatch requires a CPU tensor");

    const at::Tensor contiguous_input = input.contiguous();
    at::Tensor output = at::empty(input.sizes(), input.options());
    const std::size_t row_width = static_cast<std::size_t>(input.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(input.numel() / input.size(-1));

    flux::softmax_fp32(
        contiguous_input.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        num_rows,
        row_width);
    return output;
}

at::Tensor softmax_cuda(const at::Tensor& input) {
    validate_softmax_argument(input);
    TORCH_CHECK(
        input.is_cuda(),
        "flux::softmax: CUDA dispatch requires a CUDA tensor");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const at::Tensor contiguous_input = input.contiguous();
    at::Tensor output = at::empty(input.sizes(), input.options());
    const std::size_t row_width = static_cast<std::size_t>(input.size(-1));
    const std::size_t num_rows =
        static_cast<std::size_t>(input.numel() / input.size(-1));
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(input.get_device());

    C10_CUDA_CHECK(flux::softmax_cuda_fp32(
        contiguous_input.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        num_rows,
        row_width,
        stream.stream()));
    return output;
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def("softmax(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl("softmax", TORCH_FN(flux::softmax_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl("softmax", TORCH_FN(flux::softmax_cuda));
}
