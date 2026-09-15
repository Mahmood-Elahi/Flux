#include "native_prefill_runtime.h"

#include "greedy_generation_cuda.h"

#include "packed_swiglu_cuda.h"
#include "residual_rmsnorm_cuda.h"
#include "rmsnorm_cuda.h"
#include "streaming_prefill_gqa_cuda.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace flux {
namespace {

constexpr std::int64_t kLayerCount = 30;
constexpr std::int64_t kHiddenSize = 576;
constexpr std::int64_t kIntermediateSize = 1536;
constexpr std::int64_t kQueryHeads = 9;
constexpr std::int64_t kKeyValueHeads = 3;
constexpr std::int64_t kQueriesPerKeyValue = 3;
constexpr std::int64_t kHeadDim = 64;
constexpr std::int64_t kPackedQKVWidth = 960;
constexpr std::int64_t kPackedGateUpWidth = 3072;
constexpr std::int64_t kMaximumCapacity = 8192;
constexpr int kThreads = 256;

bool use_bounded_legacy_attention(const std::int64_t sequence_length) {
    return sequence_length > 128 && sequence_length <= 1024;
}

void check_cuda(const cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, "flux native prefill: ", operation,
        " failed: ", cudaGetErrorString(status));
}

void check_cublas(const cublasStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
        "flux native prefill: ", operation, " failed with cuBLAS status ",
        static_cast<int>(status));
}

void check_tensor(
    const at::Tensor& tensor,
    const at::Device& device,
    const at::ScalarType dtype,
    const at::IntArrayRef sizes,
    const char* name) {
    TORCH_CHECK(tensor.defined(), "flux native prefill: ", name, " is undefined");
    TORCH_CHECK(tensor.device() == device, "flux native prefill: ", name,
        " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == dtype, "flux native prefill: ", name,
        " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == sizes, "flux native prefill: ", name,
        " has shape ", tensor.sizes(), ", expected ", sizes);
    TORCH_CHECK(tensor.is_contiguous(), "flux native prefill: ", name,
        " must be contiguous");
}

std::int64_t tensor_bytes(const at::Tensor& tensor) {
    return tensor.numel() * tensor.element_size();
}

struct PrefillLayerDescriptor {
    at::Tensor input_norm_weight;
    at::Tensor packed_qkv_weight;
    at::Tensor attention_output_weight;
    at::Tensor post_attention_norm_weight;
    at::Tensor packed_gate_up_weight;
    at::Tensor down_projection_weight;
    float epsilon;
    float attention_scale;
};

__global__ void embedding_gather_fp32_kernel(
    const std::int64_t* input_ids,
    const float* embedding,
    float* hidden,
    const std::int64_t sequence_length,
    const std::int64_t vocabulary_size) {
    const std::int64_t total = sequence_length * kHiddenSize;
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row = linear / kHiddenSize;
        const std::int64_t column = linear % kHiddenSize;
        const std::int64_t token = input_ids[row];
        hidden[linear] = token >= 0 && token < vocabulary_size
            ? embedding[token * kHiddenSize + column]
            : 0.0F;
    }
}

__device__ __forceinline__ float rotate_packed(
    const float* packed,
    const float* cosine,
    const float* sine,
    const std::int64_t row,
    const std::int64_t offset,
    const std::int64_t dimension) {
    constexpr std::int64_t half = kHeadDim / 2;
    const std::int64_t paired = dimension < half
        ? dimension + half : dimension - half;
    const float sign = dimension < half ? -1.0F : 1.0F;
    const std::int64_t base = row * kPackedQKVWidth + offset;
    const float first = __fmul_rn(
        packed[base + dimension], cosine[row * kHeadDim + dimension]);
    const float second = __fmul_rn(
        sign * packed[base + paired], sine[row * kHeadDim + dimension]);
    return __fadd_rn(first, second);
}

__global__ void packed_qkv_rope_cache_prefill_fp32_kernel(
    const float* packed,
    const float* cosine,
    const float* sine,
    float* query,
    float* key_cache,
    float* value_cache,
    const std::int64_t sequence_length,
    const std::int64_t capacity) {
    const std::int64_t total = sequence_length * kPackedQKVWidth;
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row = linear / kPackedQKVWidth;
        const std::int64_t column = linear % kPackedQKVWidth;
        if (column < kHiddenSize) {
            const std::int64_t head = column / kHeadDim;
            const std::int64_t dimension = column % kHeadDim;
            query[(head * sequence_length + row) * kHeadDim + dimension] =
                rotate_packed(
                    packed, cosine, sine, row, head * kHeadDim, dimension);
        } else if (column < kHiddenSize + kKeyValueHeads * kHeadDim) {
            const std::int64_t compact = column - kHiddenSize;
            const std::int64_t head = compact / kHeadDim;
            const std::int64_t dimension = compact % kHeadDim;
            key_cache[(head * capacity + row) * kHeadDim + dimension] =
                rotate_packed(
                    packed, cosine, sine, row,
                    kHiddenSize + head * kHeadDim, dimension);
        } else {
            const std::int64_t compact =
                column - kHiddenSize - kKeyValueHeads * kHeadDim;
            const std::int64_t head = compact / kHeadDim;
            const std::int64_t dimension = compact % kHeadDim;
            value_cache[(head * capacity + row) * kHeadDim + dimension] =
                packed[row * kPackedQKVWidth + column];
        }
    }
}

__device__ __forceinline__ float legacy_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float legacy_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ float legacy_block_max(float value, float* reductions) {
    value = legacy_warp_max(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        reductions[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? reductions[lane] : -FLT_MAX;
    if (warp == 0) {
        value = legacy_warp_max(value);
        if (lane == 0) {
            reductions[0] = value;
        }
    }
    __syncthreads();
    return reductions[0];
}

__device__ float legacy_block_sum(float value, float* reductions) {
    value = legacy_warp_sum(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        reductions[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? reductions[lane] : 0.0F;
    if (warp == 0) {
        value = legacy_warp_sum(value);
        if (lane == 0) {
            reductions[0] = value;
        }
    }
    __syncthreads();
    return reductions[0];
}

__global__ void bounded_legacy_causal_softmax_fp32_kernel(
    float* scores,
    const float scale,
    const std::int64_t sequence_length) {
    __shared__ float reductions[16];
    const std::int64_t row = blockIdx.x;
    const std::int64_t query_position = row % sequence_length;
    float* values = scores + row * sequence_length;
    float local_maximum = -FLT_MAX;
    for (std::int64_t key = threadIdx.x; key <= query_position;
         key += blockDim.x) {
        values[key] *= scale;
        local_maximum = fmaxf(local_maximum, values[key]);
    }
    for (std::int64_t key = query_position + 1 + threadIdx.x;
         key < sequence_length; key += blockDim.x) {
        values[key] = 0.0F;
    }
    const float maximum = legacy_block_max(local_maximum, reductions);
    float local_sum = 0.0F;
    for (std::int64_t key = threadIdx.x; key <= query_position;
         key += blockDim.x) {
        values[key] = expf(values[key] - maximum);
        local_sum += values[key];
    }
    const float row_sum = legacy_block_sum(local_sum, reductions);
    for (std::int64_t key = threadIdx.x; key <= query_position;
         key += blockDim.x) {
        values[key] /= row_sum;
    }
}

__global__ void attention_heads_to_rows_fp32_kernel(
    const float* head_major,
    float* row_major,
    const std::int64_t sequence_length) {
    const std::int64_t total = sequence_length * kHiddenSize;
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row = linear / kHiddenSize;
        const std::int64_t column = linear % kHiddenSize;
        const std::int64_t head = column / kHeadDim;
        const std::int64_t dimension = column % kHeadDim;
        row_major[linear] =
            head_major[(head * sequence_length + row) * kHeadDim + dimension];
    }
}

__global__ void residual_add_fp32_kernel(
    const float* residual,
    const float* update,
    float* output,
    const std::int64_t elements) {
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        output[linear] = residual[linear] + update[linear];
    }
}

unsigned int blocks_for(const std::int64_t elements) {
    return static_cast<unsigned int>(
        std::min<std::int64_t>((elements + kThreads - 1) / kThreads, 65535));
}

void linear_fp32(
    cublasHandle_t handle,
    const float* input,
    const float* weight,
    float* output,
    const std::int64_t rows,
    const std::int64_t input_width,
    const std::int64_t output_width,
    const char* operation) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    check_cublas(cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(output_width), static_cast<int>(rows),
        static_cast<int>(input_width), &alpha, weight,
        static_cast<int>(input_width), input, static_cast<int>(input_width),
        &beta, output, static_cast<int>(output_width)), operation);
}

void bounded_legacy_grouped_attention(
    cublasHandle_t handle,
    const float* query,
    const float* key,
    const float* value,
    float* scores,
    float* output,
    const float scale,
    const std::int64_t sequence_length,
    const std::int64_t capacity,
    cudaStream_t stream) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const long long query_stride = sequence_length * kHeadDim;
    const long long score_stride = sequence_length * sequence_length;
    for (std::int64_t kv_head = 0; kv_head < kKeyValueHeads; ++kv_head) {
        check_cublas(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(sequence_length), static_cast<int>(sequence_length),
            static_cast<int>(kHeadDim), &alpha,
            key + kv_head * capacity * kHeadDim, static_cast<int>(kHeadDim), 0,
            query + kv_head * kQueriesPerKeyValue * query_stride,
            static_cast<int>(kHeadDim), query_stride, &beta,
            scores + kv_head * kQueriesPerKeyValue * score_stride,
            static_cast<int>(sequence_length), score_stride,
            static_cast<int>(kQueriesPerKeyValue)), "launching bounded legacy QK");
    }
    const int softmax_threads = sequence_length == 512 || sequence_length == 1024
        ? 128 : kThreads;
    bounded_legacy_causal_softmax_fp32_kernel<<<
        static_cast<unsigned int>(kQueryHeads * sequence_length),
        softmax_threads, 0, stream>>>(scores, scale, sequence_length);
    check_cuda(cudaGetLastError(), "launching bounded legacy causal softmax");
    for (std::int64_t kv_head = 0; kv_head < kKeyValueHeads; ++kv_head) {
        check_cublas(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(kHeadDim), static_cast<int>(sequence_length),
            static_cast<int>(sequence_length), &alpha,
            value + kv_head * capacity * kHeadDim, static_cast<int>(kHeadDim), 0,
            scores + kv_head * kQueriesPerKeyValue * score_stride,
            static_cast<int>(sequence_length), score_stride, &beta,
            output + kv_head * kQueriesPerKeyValue * query_stride,
            static_cast<int>(kHeadDim), query_stride,
            static_cast<int>(kQueriesPerKeyValue)), "launching bounded legacy PV");
    }
}

}  // namespace

class NativeSmolLM2Prefill::Impl {
public:
    Impl(
        at::Tensor input_ids,
        at::Tensor embedding_weight,
        std::vector<at::Tensor> input_norm_weights,
        std::vector<at::Tensor> packed_qkv_weights,
        std::vector<at::Tensor> attention_output_weights,
        std::vector<at::Tensor> post_attention_norm_weights,
        std::vector<at::Tensor> packed_gate_up_weights,
        std::vector<at::Tensor> down_projection_weights,
        at::Tensor final_norm_weight,
        at::Tensor lm_head_weight,
        at::Tensor rope_cos,
        at::Tensor rope_sin,
        const std::int64_t cache_capacity,
        std::vector<double> epsilons,
        std::vector<double> attention_scales)
        : device_(input_ids.device()),
          device_index_(input_ids.get_device()),
          sequence_length_(input_ids.size(1)),
          capacity_(cache_capacity),
          vocabulary_size_(embedding_weight.size(0)),
          embedding_weight_(std::move(embedding_weight)),
          final_norm_weight_(std::move(final_norm_weight)),
          lm_head_weight_(std::move(lm_head_weight)),
          rope_cos_(std::move(rope_cos)),
          rope_sin_(std::move(rope_sin)) {
        TORCH_CHECK(device_.is_cuda(),
            "flux native prefill: input_ids must be a CUDA tensor");
        TORCH_CHECK(sequence_length_ >= 1 && sequence_length_ <= capacity_,
            "flux native prefill: prompt length must be in [1, capacity]");
        TORCH_CHECK(capacity_ >= 1 && capacity_ <= kMaximumCapacity,
            "flux native prefill: cache capacity must be in [1, 8192]");
        TORCH_CHECK(input_norm_weights.size() == kLayerCount &&
                packed_qkv_weights.size() == kLayerCount &&
                attention_output_weights.size() == kLayerCount &&
                post_attention_norm_weights.size() == kLayerCount &&
                packed_gate_up_weights.size() == kLayerCount &&
                down_projection_weights.size() == kLayerCount &&
                epsilons.size() == kLayerCount &&
                attention_scales.size() == kLayerCount,
            "flux native prefill: exactly 30 complete layer descriptors are required");
        layers_.reserve(kLayerCount);
        for (std::int64_t index = 0; index < kLayerCount; ++index) {
            TORCH_CHECK(std::isfinite(epsilons[index]) && epsilons[index] > 0.0,
                "flux native prefill: every RMSNorm epsilon must be positive");
            TORCH_CHECK(std::isfinite(attention_scales[index]),
                "flux native prefill: every attention scale must be finite");
            layers_.push_back(PrefillLayerDescriptor{
                input_norm_weights[index], packed_qkv_weights[index],
                attention_output_weights[index], post_attention_norm_weights[index],
                packed_gate_up_weights[index], down_projection_weights[index],
                static_cast<float>(epsilons[index]),
                static_cast<float>(attention_scales[index])});
        }
        validate_weights(input_ids);

        const c10::cuda::CUDAGuard device_guard(device_);
        allocate_workspace(input_ids.options().dtype(at::kFloat).requires_grad(false));
        initial_token_ = at::empty({1, 1}, input_ids.options());
        prefill_logits_ = at::empty(
            {1, 1, vocabulary_size_}, input_ids.options().dtype(at::kFloat));
        at::Tensor empty_cache = at::empty(
            {1, kKeyValueHeads, 0, kHeadDim},
            input_ids.options().dtype(at::kFloat));
        std::vector<at::Tensor> empty_keys(kLayerCount, empty_cache);
        std::vector<at::Tensor> empty_values(kLayerCount, empty_cache);
        check_cuda(cudaMemsetAsync(
            initial_token_.mutable_data_ptr<std::int64_t>(), 0,
            sizeof(std::int64_t),
            c10::cuda::getCurrentCUDAStream(device_index_).stream()),
            "initializing decode token");
        decode_ = c10::make_intrusive<NativeSmolLM2Decode>(
            initial_token_, embedding_weight_, input_norm_weights,
            packed_qkv_weights, attention_output_weights,
            post_attention_norm_weights, packed_gate_up_weights,
            down_projection_weights, final_norm_weight_, lm_head_weight_,
            rope_cos_, rope_sin_, empty_keys, empty_values, capacity_, 0,
            epsilons, attention_scales);
        try {
            check_cublas(cublasCreate(&blas_handle_), "cublasCreate");
            check_cublas(cublasSetMathMode(blas_handle_, CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode");
            prefill(input_ids);
        } catch (...) {
            if (blas_handle_ != nullptr) {
                cublasDestroy(blas_handle_);
                blas_handle_ = nullptr;
            }
            throw;
        }
    }

    ~Impl() {
        if (blas_handle_ != nullptr) {
            // Reading the position waits for prefill or replay work that uses
            // this handle before its destruction.
            try {
                decode_->position();
            } catch (...) {
            }
            cublasDestroy(blas_handle_);
        }
    }

    void validate_weights(const at::Tensor& input_ids) const {
        check_tensor(input_ids, device_, at::kLong, {1, sequence_length_}, "input_ids");
        check_tensor(embedding_weight_, device_, at::kFloat,
            {vocabulary_size_, kHiddenSize}, "embedding weight");
        check_tensor(final_norm_weight_, device_, at::kFloat,
            {kHiddenSize}, "final norm weight");
        check_tensor(lm_head_weight_, device_, at::kFloat,
            {vocabulary_size_, kHiddenSize}, "LM-head weight");
        check_tensor(rope_cos_, device_, at::kFloat,
            {capacity_, kHeadDim}, "RoPE cosine table");
        check_tensor(rope_sin_, device_, at::kFloat,
            {capacity_, kHeadDim}, "RoPE sine table");
        for (const PrefillLayerDescriptor& layer : layers_) {
            check_tensor(layer.input_norm_weight, device_, at::kFloat,
                {kHiddenSize}, "input norm weight");
            check_tensor(layer.packed_qkv_weight, device_, at::kFloat,
                {kPackedQKVWidth, kHiddenSize}, "packed QKV weight");
            check_tensor(layer.attention_output_weight, device_, at::kFloat,
                {kHiddenSize, kHiddenSize}, "attention output weight");
            check_tensor(layer.post_attention_norm_weight, device_, at::kFloat,
                {kHiddenSize}, "post-attention norm weight");
            check_tensor(layer.packed_gate_up_weight, device_, at::kFloat,
                {kPackedGateUpWidth, kHiddenSize}, "packed gate/up weight");
            check_tensor(layer.down_projection_weight, device_, at::kFloat,
                {kHiddenSize, kIntermediateSize}, "down projection weight");
        }
    }

    void allocate_workspace(const at::TensorOptions& options) {
        const std::int64_t s = sequence_length_;
        const std::int64_t score_elements =
            use_bounded_legacy_attention(s) ? kQueryHeads * s * s : 0;
        const StreamingPrefillGQAVariant attention_variant =
            select_streaming_prefill_gqa_variant(s);
        const std::int64_t streaming_attention_elements =
            use_bounded_legacy_attention(s) ? 0 :
            static_cast<std::int64_t>(streaming_prefill_gqa_workspace_bytes(
                s, attention_variant) / sizeof(float));
        const std::array<std::int64_t, 13> sizes{
            s * kHiddenSize, s * kHiddenSize, s * kHiddenSize,
            s * kHiddenSize, s * kPackedQKVWidth,
            s * kQueryHeads * kHeadDim,
            s * kQueryHeads * kHeadDim, streaming_attention_elements,
            score_elements, s * kHiddenSize,
            s * kPackedGateUpWidth, s * kIntermediateSize,
            s * kHiddenSize};
        std::int64_t total = kHiddenSize;
        for (const std::int64_t size : sizes) {
            total += size;
        }
        workspace_ = at::empty({total}, options);
        std::int64_t offset = 0;
        auto take = [&](const std::int64_t size) {
            at::Tensor result = workspace_.slice(0, offset, offset + size);
            offset += size;
            return result;
        };
        hidden_a_ = take(s * kHiddenSize).view({1, s, kHiddenSize});
        hidden_b_ = take(s * kHiddenSize).view({1, s, kHiddenSize});
        norm_output_ = take(s * kHiddenSize).view({1, s, kHiddenSize});
        residual_output_ = take(s * kHiddenSize).view({1, s, kHiddenSize});
        qkv_output_ = take(s * kPackedQKVWidth).view({1, s, kPackedQKVWidth});
        query_output_ = take(s * kQueryHeads * kHeadDim)
            .view({kQueryHeads, s, kHeadDim});
        attention_heads_ = take(s * kQueryHeads * kHeadDim)
            .view({kQueryHeads, s, kHeadDim});
        streaming_attention_workspace_ = take(streaming_attention_elements);
        if (score_elements != 0) {
            attention_scores_ = take(score_elements)
                .view({kQueryHeads, s, s});
        }
        attention_projection_ = take(s * kHiddenSize).view({1, s, kHiddenSize});
        packed_gate_up_ = take(s * kPackedGateUpWidth)
            .view({1, s, kPackedGateUpWidth});
        swiglu_output_ = take(s * kIntermediateSize)
            .view({1, s, kIntermediateSize});
        down_projection_ = take(s * kHiddenSize).view({1, s, kHiddenSize});
        final_hidden_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        TORCH_INTERNAL_ASSERT(offset == total);
    }

    at::Tensor prefill(const at::Tensor& input_ids) {
        check_tensor(input_ids, device_, at::kLong,
            {1, sequence_length_}, "input_ids");
        const c10::cuda::CUDAGuard device_guard(device_);
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();
        decode_->wait_for_prefill(stream);
        check_cublas(cublasSetStream(blas_handle_, stream), "cublasSetStream");
        run(input_ids, stream);
        decode_->install_prefilled_state(initial_token_, sequence_length_, stream);
        return prefill_logits_;
    }

    void run(const at::Tensor& input_ids, cudaStream_t stream) {
        embedding_gather_fp32_kernel<<<
            blocks_for(sequence_length_ * kHiddenSize), kThreads, 0, stream>>>(
            input_ids.const_data_ptr<std::int64_t>(),
            embedding_weight_.const_data_ptr<float>(),
            hidden_a_.mutable_data_ptr<float>(), sequence_length_, vocabulary_size_);
        check_cuda(cudaGetLastError(), "launching embedding gather");

        at::Tensor keys = decode_->key_cache();
        at::Tensor values = decode_->value_cache();
        for (std::int64_t index = 0; index < kLayerCount; ++index) {
            const PrefillLayerDescriptor& layer = layers_[index];
            at::Tensor& input = index % 2 == 0 ? hidden_a_ : hidden_b_;
            at::Tensor& output = index % 2 == 0 ? hidden_b_ : hidden_a_;
            check_cuda(rmsnorm_cuda_fp32(
                input.const_data_ptr<float>(),
                layer.input_norm_weight.const_data_ptr<float>(),
                norm_output_.mutable_data_ptr<float>(), sequence_length_,
                kHiddenSize, layer.epsilon, stream), "launching input RMSNorm");
            linear_fp32(
                blas_handle_, norm_output_.const_data_ptr<float>(),
                layer.packed_qkv_weight.const_data_ptr<float>(),
                qkv_output_.mutable_data_ptr<float>(), sequence_length_,
                kHiddenSize, kPackedQKVWidth, "launching packed QKV projection");

            float* layer_keys = keys.mutable_data_ptr<float>() +
                index * kKeyValueHeads * capacity_ * kHeadDim;
            float* layer_values = values.mutable_data_ptr<float>() +
                index * kKeyValueHeads * capacity_ * kHeadDim;
            packed_qkv_rope_cache_prefill_fp32_kernel<<<
                blocks_for(sequence_length_ * kPackedQKVWidth), kThreads, 0,
                stream>>>(qkv_output_.const_data_ptr<float>(),
                rope_cos_.const_data_ptr<float>(), rope_sin_.const_data_ptr<float>(),
                query_output_.mutable_data_ptr<float>(), layer_keys, layer_values,
                sequence_length_, capacity_);
            check_cuda(cudaGetLastError(), "launching packed QKV/RoPE/cache fill");
            if (use_bounded_legacy_attention(sequence_length_)) {
                bounded_legacy_grouped_attention(
                    blas_handle_, query_output_.const_data_ptr<float>(),
                    layer_keys, layer_values,
                    attention_scores_.mutable_data_ptr<float>(),
                    attention_heads_.mutable_data_ptr<float>(),
                    layer.attention_scale, sequence_length_, capacity_, stream);
            } else {
                const StreamingPrefillGQAVariant attention_variant =
                    select_streaming_prefill_gqa_variant(sequence_length_);
                check_cuda(streaming_prefill_gqa_cuda_fp32(
                    query_output_.const_data_ptr<float>(), layer_keys,
                    layer_values, attention_heads_.mutable_data_ptr<float>(),
                    streaming_attention_workspace_.numel() == 0 ? nullptr :
                        streaming_attention_workspace_.mutable_data_ptr<float>(),
                    layer.attention_scale, sequence_length_, capacity_,
                    attention_variant, stream),
                    "launching streaming grouped-GQA attention");
            }
            attention_heads_to_rows_fp32_kernel<<<
                blocks_for(sequence_length_ * kHiddenSize), kThreads, 0, stream>>>(
                attention_heads_.const_data_ptr<float>(),
                qkv_output_.mutable_data_ptr<float>(), sequence_length_);
            check_cuda(cudaGetLastError(), "launching attention layout conversion");
            linear_fp32(
                blas_handle_, qkv_output_.const_data_ptr<float>(),
                layer.attention_output_weight.const_data_ptr<float>(),
                attention_projection_.mutable_data_ptr<float>(), sequence_length_,
                kHiddenSize, kHiddenSize, "launching attention output projection");
            check_cuda(residual_rmsnorm_cuda_fp32(
                attention_projection_.const_data_ptr<float>(),
                input.const_data_ptr<float>(),
                layer.post_attention_norm_weight.const_data_ptr<float>(),
                norm_output_.mutable_data_ptr<float>(),
                residual_output_.mutable_data_ptr<float>(), sequence_length_,
                kHiddenSize, layer.epsilon, stream),
                "launching residual RMSNorm");
            linear_fp32(
                blas_handle_, norm_output_.const_data_ptr<float>(),
                layer.packed_gate_up_weight.const_data_ptr<float>(),
                packed_gate_up_.mutable_data_ptr<float>(), sequence_length_,
                kHiddenSize, kPackedGateUpWidth,
                "launching packed gate/up projection");
            check_cuda(packed_swiglu_cuda_fp32(
                packed_gate_up_.const_data_ptr<float>(),
                swiglu_output_.mutable_data_ptr<float>(), sequence_length_,
                kIntermediateSize, stream), "launching packed SwiGLU");
            linear_fp32(
                blas_handle_, swiglu_output_.const_data_ptr<float>(),
                layer.down_projection_weight.const_data_ptr<float>(),
                down_projection_.mutable_data_ptr<float>(), sequence_length_,
                kIntermediateSize, kHiddenSize, "launching down projection");
            residual_add_fp32_kernel<<<
                blocks_for(sequence_length_ * kHiddenSize), kThreads, 0, stream>>>(
                residual_output_.const_data_ptr<float>(),
                down_projection_.const_data_ptr<float>(),
                output.mutable_data_ptr<float>(), sequence_length_ * kHiddenSize);
            check_cuda(cudaGetLastError(), "launching final residual add");
        }

        at::Tensor& final_input = kLayerCount % 2 == 0 ? hidden_a_ : hidden_b_;
        const float* last_row = final_input.const_data_ptr<float>() +
            (sequence_length_ - 1) * kHiddenSize;
        check_cuda(rmsnorm_cuda_fp32(
            last_row, final_norm_weight_.const_data_ptr<float>(),
            final_hidden_.mutable_data_ptr<float>(), 1, kHiddenSize,
            layers_.front().epsilon, stream), "launching final RMSNorm");
        linear_fp32(
            blas_handle_, final_hidden_.const_data_ptr<float>(),
            lm_head_weight_.const_data_ptr<float>(),
            prefill_logits_.mutable_data_ptr<float>(), 1, kHiddenSize,
            vocabulary_size_, "launching LM head");
        check_cuda(greedy_argmax_cuda_fp32(
            prefill_logits_.const_data_ptr<float>(),
            initial_token_.mutable_data_ptr<std::int64_t>(), vocabulary_size_,
            stream), "launching prefill argmax");
    }

    at::Device device_;
    int device_index_;
    std::int64_t sequence_length_;
    std::int64_t capacity_;
    std::int64_t vocabulary_size_;
    at::Tensor embedding_weight_;
    at::Tensor final_norm_weight_;
    at::Tensor lm_head_weight_;
    at::Tensor rope_cos_;
    at::Tensor rope_sin_;
    std::vector<PrefillLayerDescriptor> layers_;
    c10::intrusive_ptr<NativeSmolLM2Decode> decode_;
    cublasHandle_t blas_handle_ = nullptr;

    at::Tensor initial_token_;
    at::Tensor prefill_logits_;
    at::Tensor workspace_;
    at::Tensor hidden_a_;
    at::Tensor hidden_b_;
    at::Tensor norm_output_;
    at::Tensor residual_output_;
    at::Tensor qkv_output_;
    at::Tensor query_output_;
    at::Tensor attention_heads_;
    at::Tensor streaming_attention_workspace_;
    at::Tensor attention_scores_;
    at::Tensor attention_projection_;
    at::Tensor packed_gate_up_;
    at::Tensor swiglu_output_;
    at::Tensor down_projection_;
    at::Tensor final_hidden_;
};

NativeSmolLM2Prefill::NativeSmolLM2Prefill(
    at::Tensor input_ids,
    at::Tensor embedding_weight,
    std::vector<at::Tensor> input_norm_weights,
    std::vector<at::Tensor> packed_qkv_weights,
    std::vector<at::Tensor> attention_output_weights,
    std::vector<at::Tensor> post_attention_norm_weights,
    std::vector<at::Tensor> packed_gate_up_weights,
    std::vector<at::Tensor> down_projection_weights,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    const std::int64_t cache_capacity,
    std::vector<double> epsilons,
    std::vector<double> attention_scales)
    : impl_(std::make_unique<Impl>(
          std::move(input_ids), std::move(embedding_weight),
          std::move(input_norm_weights), std::move(packed_qkv_weights),
          std::move(attention_output_weights),
          std::move(post_attention_norm_weights),
          std::move(packed_gate_up_weights),
          std::move(down_projection_weights), std::move(final_norm_weight),
          std::move(lm_head_weight), std::move(rope_cos), std::move(rope_sin),
          cache_capacity, std::move(epsilons),
          std::move(attention_scales))) {}

NativeSmolLM2Prefill::~NativeSmolLM2Prefill() = default;

at::Tensor NativeSmolLM2Prefill::prefill(const at::Tensor& input_ids) {
    return impl_->prefill(input_ids);
}

at::Tensor NativeSmolLM2Prefill::replay(
    const c10::optional<at::Tensor>& token) {
    return impl_->decode_->replay(token);
}

at::Tensor NativeSmolLM2Prefill::generate_greedy(
    const std::int64_t max_new_tokens) {
    TORCH_CHECK(max_new_tokens >= 1,
        "flux native prefill: max_new_tokens must be positive");
    TORCH_CHECK(max_new_tokens <= impl_->capacity_ - impl_->sequence_length_ + 1,
        "flux native prefill: generation exceeds fixed cache capacity");
    at::Tensor storage = impl_->decode_->generate_greedy(max_new_tokens - 1);
    return storage.narrow(0, 0, max_new_tokens).view({1, max_new_tokens});
}

at::Tensor NativeSmolLM2Prefill::logits() const { return impl_->prefill_logits_; }
at::Tensor NativeSmolLM2Prefill::current_token() const {
    return impl_->decode_->current_token();
}
at::Tensor NativeSmolLM2Prefill::generated_tokens() const {
    return impl_->decode_->generated_tokens();
}
at::Tensor NativeSmolLM2Prefill::device_generation_step() const {
    return impl_->decode_->device_generation_step();
}
at::Tensor NativeSmolLM2Prefill::final_hidden() const { return impl_->final_hidden_; }
at::Tensor NativeSmolLM2Prefill::key_cache() const { return impl_->decode_->key_cache(); }
at::Tensor NativeSmolLM2Prefill::value_cache() const { return impl_->decode_->value_cache(); }
at::Tensor NativeSmolLM2Prefill::device_position() const {
    return impl_->decode_->device_position();
}
at::Tensor NativeSmolLM2Prefill::device_cache_length() const {
    return impl_->decode_->device_cache_length();
}

std::vector<std::int64_t> NativeSmolLM2Prefill::addresses() const {
    std::vector<std::int64_t> result{
        reinterpret_cast<std::int64_t>(impl_->prefill_logits_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->workspace_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->attention_heads_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->final_hidden_.data_ptr())};
    const std::vector<std::int64_t> decode = impl_->decode_->addresses();
    result.insert(result.end(), decode.begin(), decode.end());
    return result;
}

std::int64_t NativeSmolLM2Prefill::position() {
    return impl_->decode_->position();
}
std::int64_t NativeSmolLM2Prefill::cache_length() {
    return impl_->decode_->cache_length();
}
std::int64_t NativeSmolLM2Prefill::prompt_length() const {
    return impl_->sequence_length_;
}
std::int64_t NativeSmolLM2Prefill::capacity() const { return impl_->capacity_; }
std::int64_t NativeSmolLM2Prefill::replay_count() const {
    return impl_->decode_->replay_count();
}
std::int64_t NativeSmolLM2Prefill::generation_step() {
    return impl_->decode_->generation_step();
}
std::int64_t NativeSmolLM2Prefill::workspace_bytes() const {
    return tensor_bytes(impl_->workspace_);
}
std::int64_t NativeSmolLM2Prefill::cache_bytes() const {
    return tensor_bytes(impl_->decode_->key_cache()) +
        tensor_bytes(impl_->decode_->value_cache());
}
std::int64_t NativeSmolLM2Prefill::stable_buffer_bytes() const {
    return tensor_bytes(impl_->initial_token_) +
        tensor_bytes(impl_->prefill_logits_) +
        impl_->decode_->stable_buffer_bytes();
}

}  // namespace flux
