#pragma once

#include <cuda_runtime.h>

namespace flux {

cudaError_t packed_gate_up_swiglu_cuda_fp32(
    const float* input,
    const float* weight,
    float* output,
    cudaStream_t stream);

}  // namespace flux
