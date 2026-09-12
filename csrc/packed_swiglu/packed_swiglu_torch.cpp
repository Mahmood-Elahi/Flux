#include "packed_swiglu.h"
#include "packed_swiglu_cuda.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstddef>
#include <vector>

namespace flux {
namespace {

void validate_packed_swiglu_arguments(const at::Tensor& packed) {
    TORCH_CHECK(
        packed.dim() >= 1,
        "flux::packed_swiglu: packed must have at least one dimension");
    TORCH_CHECK(
        packed.size(-1) > 0,
        "flux::packed_swiglu: packed final dimension must be non-empty");
    TORCH_CHECK(
        packed.size(-1) % 2 == 0,
        "flux::packed_swiglu: packed final dimension must be even");
    TORCH_CHECK(
        packed.numel() > 0,
        "flux::packed_swiglu: packed must contain at least one row");
    TORCH_CHECK(
        packed.scalar_type() == at::kFloat,
        "flux::packed_swiglu: packed must have dtype torch.float32");
    TORCH_CHECK(
        packed.is_contiguous(),
        "flux::packed_swiglu: packed must be contiguous; no implicit copy is performed");
    TORCH_CHECK(
        !at::GradMode::is_enabled() || !packed.requires_grad(),
        "flux::packed_swiglu is inference-only; use torch.no_grad() or "
        "torch.inference_mode() for tensors that require gradients");
}

at::Tensor output_for(const at::Tensor& packed) {
    std::vector<int64_t> output_sizes(packed.sizes().begin(), packed.sizes().end());
    output_sizes.back() /= 2;
    return at::empty(output_sizes, packed.options());
}

void validate_output(const at::Tensor& packed, const at::Tensor& output) {
    std::vector<int64_t> expected(packed.sizes().begin(), packed.sizes().end());
    expected.back() /= 2;
    TORCH_CHECK(output.defined() && output.sizes() == expected,
        "flux::packed_swiglu_out: output shape is incorrect");
    TORCH_CHECK(output.scalar_type() == at::kFloat &&
            output.device() == packed.device() && output.is_contiguous(),
        "flux::packed_swiglu_out: output must be contiguous float32 on the input device");
    at::assert_no_internal_overlap(output);
    TORCH_CHECK(!output.is_alias_of(packed),
        "flux::packed_swiglu_out: output must not alias packed");
}

at::Tensor packed_swiglu_cpu(const at::Tensor& packed) {
    validate_packed_swiglu_arguments(packed);
    TORCH_CHECK(
        packed.device().is_cpu(),
        "flux::packed_swiglu: CPU dispatch requires a CPU tensor");
    at::Tensor output = output_for(packed);
    const std::size_t intermediate_size =
        static_cast<std::size_t>(packed.size(-1) / 2);
    const std::size_t num_rows = static_cast<std::size_t>(
        packed.numel() / packed.size(-1));
    packed_swiglu_fp32(
        packed.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        num_rows,
        intermediate_size);
    return output;
}

at::Tensor packed_swiglu_cuda(const at::Tensor& packed) {
    validate_packed_swiglu_arguments(packed);
    TORCH_CHECK(
        packed.is_cuda(),
        "flux::packed_swiglu: CUDA dispatch requires a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(packed.device());
    at::Tensor output = output_for(packed);
    const std::size_t intermediate_size =
        static_cast<std::size_t>(packed.size(-1) / 2);
    const std::size_t num_rows = static_cast<std::size_t>(
        packed.numel() / packed.size(-1));
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(packed.get_device());
    C10_CUDA_CHECK(packed_swiglu_cuda_fp32(
        packed.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        num_rows,
        intermediate_size,
        stream.stream()));
    return output;
}

at::Tensor packed_swiglu_cuda_out(
    const at::Tensor& packed,
    at::Tensor output) {
    validate_packed_swiglu_arguments(packed);
    TORCH_CHECK(packed.is_cuda(),
        "flux::packed_swiglu_out: CUDA dispatch requires a CUDA tensor");
    validate_output(packed, output);
    const c10::cuda::CUDAGuard device_guard(packed.device());
    const std::size_t intermediate_size =
        static_cast<std::size_t>(packed.size(-1) / 2);
    const std::size_t num_rows = static_cast<std::size_t>(
        packed.numel() / packed.size(-1));
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(packed.get_device());
    C10_CUDA_CHECK(packed_swiglu_cuda_fp32(
        packed.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
        num_rows, intermediate_size, stream.stream()));
    return output;
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def("packed_swiglu(Tensor packed) -> Tensor");
    library.def(
        "packed_swiglu_out(Tensor packed, Tensor(a!) output) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flux, CPU, library) {
    library.impl("packed_swiglu", TORCH_FN(flux::packed_swiglu_cpu));
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl("packed_swiglu", TORCH_FN(flux::packed_swiglu_cuda));
    library.impl("packed_swiglu_out", TORCH_FN(flux::packed_swiglu_cuda_out));
}
