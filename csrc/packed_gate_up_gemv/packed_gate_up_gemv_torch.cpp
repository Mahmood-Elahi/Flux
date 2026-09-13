#include "packed_gate_up_gemv_cuda.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

namespace flux {
namespace {

constexpr int64_t kInputWidth = 576;
constexpr int64_t kOutputWidth = 3072;
constexpr int64_t kIntermediateWidth = 1536;

void validate_common(const at::Tensor& input, const at::Tensor& weight) {
    TORCH_CHECK(input.defined() && weight.defined(),
        "flux::packed_gate_up_gemv: input and weight must be defined");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && input.device() == weight.device(),
        "flux::packed_gate_up_gemv: input and weight must be CUDA tensors on the same device");
    TORCH_CHECK(input.sizes() == at::IntArrayRef({1, 1, kInputWidth}),
        "flux::packed_gate_up_gemv: input must have shape [1, 1, 576]");
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({kOutputWidth, kInputWidth}),
        "flux::packed_gate_up_gemv: weight must have shape [3072, 576]");
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
        "flux::packed_gate_up_gemv: input and weight must have dtype torch.float32");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
        "flux::packed_gate_up_gemv: input and weight must be contiguous");
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            (!input.requires_grad() && !weight.requires_grad()),
        "flux::packed_gate_up_gemv is inference-only; use torch.no_grad() or torch.inference_mode()");
}

void validate_output(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& output,
    int64_t width) {
    TORCH_CHECK(output.defined() && output.is_cuda() && output.device() == input.device(),
        "flux::packed_gate_up_gemv_out: output must be CUDA on the input device");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({1, 1, width}),
        "flux::packed_gate_up_gemv_out: output shape is incorrect");
    TORCH_CHECK(output.scalar_type() == at::kFloat && output.is_contiguous(),
        "flux::packed_gate_up_gemv_out: output must be contiguous float32");
    at::assert_no_internal_overlap(output);
    TORCH_CHECK(!output.is_alias_of(input) && !output.is_alias_of(weight),
        "flux::packed_gate_up_gemv_out: output must not alias an input");
}

at::Tensor packed_gate_up_swiglu_out_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor output) {
    validate_common(input, weight);
    validate_output(input, weight, output, kIntermediateWidth);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(input.get_device());
    C10_CUDA_CHECK(packed_gate_up_swiglu_cuda_fp32(
        input.const_data_ptr<float>(), weight.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(), stream.stream()));
    return output;
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "packed_gate_up_swiglu_out(Tensor input, Tensor weight, Tensor(a!) output) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl(
        "packed_gate_up_swiglu_out", TORCH_FN(flux::packed_gate_up_swiglu_out_cuda));
}
