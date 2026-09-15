#include "native_decode_runtime.h"

#include "greedy_generation_cuda.h"

#include "cublaslt_linear_cuda.h"
#include "gqa_decode_attention_cuda.h"
#include "packed_gate_up_gemv_cuda.h"
#include "packed_qkv_rope_cache_cuda.h"
#include "residual_rmsnorm_cuda.h"
#include "rmsnorm_cuda.h"

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
#include <limits>
#include <utility>

namespace flux {
namespace {

constexpr std::int64_t kHiddenSize = 576;
constexpr std::int64_t kIntermediateSize = 1536;
constexpr std::int64_t kQueryHeads = 9;
constexpr std::int64_t kKeyValueHeads = 3;
constexpr std::int64_t kHeadDim = 64;
constexpr std::int64_t kPackedQKVWidth = 960;
constexpr std::int64_t kPackedGateUpWidth = 3072;
constexpr std::int64_t kMaximumCapacity = 8192;
constexpr std::int64_t kAttentionChunkSize = 128;

constexpr CublasLtAlgorithmConfig kRetainedProjectionAlgorithm{
    13, 0, 1, 0, 0, 91, 0};

void check_cuda(const cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, "flux native decode: ", operation,
        " failed: ", cudaGetErrorString(status));
}

void check_cublas(const cublasStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
        "flux native decode: ", operation, " failed with cuBLAS status ",
        static_cast<int>(status));
}

void check_tensor(
    const at::Tensor& tensor,
    const at::Device& device,
    const at::ScalarType dtype,
    const at::IntArrayRef sizes,
    const char* name) {
    TORCH_CHECK(tensor.defined(), "flux native decode: ", name, " is undefined");
    TORCH_CHECK(tensor.device() == device, "flux native decode: ", name,
        " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == dtype, "flux native decode: ", name,
        " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == sizes, "flux native decode: ", name,
        " has shape ", tensor.sizes(), ", expected ", sizes);
    TORCH_CHECK(tensor.is_contiguous(), "flux native decode: ", name,
        " must be contiguous");
}

__global__ void gather_rope_row_fp32_kernel(
    const float* cos_table,
    const float* sin_table,
    const std::int64_t* position,
    float* cos_row,
    float* sin_row,
    const std::int64_t capacity) {
    const std::int64_t row = *position;
    if (row < 0 || row >= capacity) {
        return;
    }
    for (std::int64_t column = threadIdx.x; column < kHeadDim;
         column += blockDim.x) {
        const std::int64_t index = row * kHeadDim + column;
        cos_row[column] = cos_table[index];
        sin_row[column] = sin_table[index];
    }
}

cudaError_t gather_rope_row_fp32(
    const float* cos_table,
    const float* sin_table,
    const std::int64_t* position,
    float* cos_row,
    float* sin_row,
    const std::int64_t capacity,
    cudaStream_t stream) {
    if (cos_table == nullptr || sin_table == nullptr || position == nullptr ||
        cos_row == nullptr || sin_row == nullptr || capacity <= 0 ||
        stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    gather_rope_row_fp32_kernel<<<1, 64, 0, stream>>>(
        cos_table, sin_table, position, cos_row, sin_row, capacity);
    return cudaGetLastError();
}

__global__ void prepare_full_decode_fp32_kernel(
    const std::int64_t* token,
    const float* embedding,
    const std::int64_t vocabulary_size,
    const float* cos_table,
    const float* sin_table,
    const std::int64_t* position,
    float* hidden,
    float* cos_row,
    float* sin_row,
    std::int64_t* attention_length,
    const std::int64_t capacity) {
    const std::int64_t token_value = *token;
    const std::int64_t row = *position;
    if (threadIdx.x == 0) {
        *attention_length = row + 1;
    }
    for (std::int64_t column = threadIdx.x; column < kHiddenSize;
         column += blockDim.x) {
        hidden[column] = token_value >= 0 && token_value < vocabulary_size
            ? embedding[token_value * kHiddenSize + column]
            : 0.0F;
    }
    for (std::int64_t column = threadIdx.x; column < kHeadDim;
         column += blockDim.x) {
        if (row >= 0 && row < capacity) {
            const std::int64_t index = row * kHeadDim + column;
            cos_row[column] = cos_table[index];
            sin_row[column] = sin_table[index];
        }
    }
}

cudaError_t prepare_full_decode_fp32(
    const std::int64_t* token,
    const float* embedding,
    const std::int64_t vocabulary_size,
    const float* cos_table,
    const float* sin_table,
    const std::int64_t* position,
    float* hidden,
    float* cos_row,
    float* sin_row,
    std::int64_t* attention_length,
    const std::int64_t capacity,
    cudaStream_t stream) {
    if (token == nullptr || embedding == nullptr || vocabulary_size <= 0 ||
        cos_table == nullptr || sin_table == nullptr || position == nullptr ||
        hidden == nullptr || cos_row == nullptr || sin_row == nullptr ||
        attention_length == nullptr || capacity <= 0 || stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    prepare_full_decode_fp32_kernel<<<1, 256, 0, stream>>>(
        token, embedding, vocabulary_size, cos_table, sin_table, position,
        hidden, cos_row, sin_row, attention_length, capacity);
    return cudaGetLastError();
}

std::int64_t tensor_bytes(const at::Tensor& tensor) {
    return tensor.numel() * tensor.element_size();
}

struct NativeLayerDescriptor {
    at::Tensor input_norm_weight;
    at::Tensor packed_qkv_weight;
    at::Tensor attention_output_weight;
    at::Tensor post_attention_norm_weight;
    at::Tensor packed_gate_up_weight;
    at::Tensor down_projection_weight;
    float epsilon;
    float attention_scale;
};

struct NativeLayerWorkspace {
    at::Tensor norm_output;
    at::Tensor residual_output;
    at::Tensor qkv_output;
    at::Tensor query_output;
    at::Tensor attention_output;
    at::Tensor attention_projection_output;
    at::Tensor swiglu_output;
    at::Tensor down_projection_output;
    at::Tensor attention_workspace;
};

void execute_decoder_layer(
    const NativeLayerDescriptor& layer,
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& cache_position,
    const at::Tensor& attention_length,
    const at::Tensor& rope_cos_row,
    const at::Tensor& rope_sin_row,
    NativeLayerWorkspace& workspace,
    cublasHandle_t blas_handle,
    const int device_index,
    const std::int64_t capacity,
    cudaStream_t stream,
    const bool advance_position) {
    check_cuda(rmsnorm_cuda_fp32(
        input.const_data_ptr<float>(),
        layer.input_norm_weight.const_data_ptr<float>(),
        workspace.norm_output.mutable_data_ptr<float>(), 1, kHiddenSize,
        layer.epsilon, stream), "launching input RMSNorm");
    cublaslt_linear_config_cuda_fp32(
        workspace.norm_output.const_data_ptr<float>(),
        layer.packed_qkv_weight.const_data_ptr<float>(),
        workspace.qkv_output.mutable_data_ptr<float>(), kHiddenSize,
        kPackedQKVWidth, nullptr, 0, device_index,
        kRetainedProjectionAlgorithm, stream);

    const PackedQKVStrides packed_strides{
        kPackedQKVWidth, kPackedQKVWidth, 1};
    const PackedQKVEmbeddingStrides embedding_strides{kHeadDim, kHeadDim, 1};
    const PackedQKVCacheStrides cache_strides{
        kKeyValueHeads * capacity * kHeadDim,
        capacity * kHeadDim, kHeadDim, 1};
    const cudaError_t qkv_status = advance_position
        ? packed_qkv_rope_cache_cuda_fp32(
              workspace.qkv_output.const_data_ptr<float>(),
              rope_cos_row.const_data_ptr<float>(),
              rope_sin_row.const_data_ptr<float>(),
              key_cache.mutable_data_ptr<float>(),
              value_cache.mutable_data_ptr<float>(),
              const_cast<std::int64_t*>(
                  cache_position.const_data_ptr<std::int64_t>()),
              workspace.query_output.mutable_data_ptr<float>(),
              static_cast<std::size_t>(capacity), packed_strides,
              embedding_strides, embedding_strides, cache_strides,
              cache_strides, stream)
        : packed_qkv_rope_cache_at_position_cuda_fp32(
              workspace.qkv_output.const_data_ptr<float>(),
              rope_cos_row.const_data_ptr<float>(),
              rope_sin_row.const_data_ptr<float>(),
              key_cache.mutable_data_ptr<float>(),
              value_cache.mutable_data_ptr<float>(),
              cache_position.const_data_ptr<std::int64_t>(),
              workspace.query_output.mutable_data_ptr<float>(),
              static_cast<std::size_t>(capacity), packed_strides,
              embedding_strides, embedding_strides, cache_strides,
              cache_strides, stream);
    check_cuda(qkv_status, "launching packed QKV/RoPE/cache update");

    const std::array<std::int64_t, 4> query_strides{
        kQueryHeads * kHeadDim, kHeadDim, kHeadDim, 1};
    const std::array<std::int64_t, 4> cache_stride_array{
        kKeyValueHeads * capacity * kHeadDim,
        capacity * kHeadDim, kHeadDim, 1};
    const std::array<std::int64_t, 4> no_mask_sizes{1, 1, 1, 1};
    const std::array<std::int64_t, 4> no_mask_strides{0, 0, 0, 0};
    const std::int64_t chunks =
        (capacity + kAttentionChunkSize - 1) / kAttentionChunkSize;
    check_cuda(gqa_decode_attention_cuda_fp32(
        workspace.query_output.const_data_ptr<float>(),
        key_cache.const_data_ptr<float>(), value_cache.const_data_ptr<float>(),
        nullptr, attention_length.const_data_ptr<std::int64_t>(),
        workspace.attention_output.mutable_data_ptr<float>(),
        workspace.attention_workspace.mutable_data_ptr<float>(),
        layer.attention_scale, 1, kQueryHeads, kKeyValueHeads, capacity,
        kHeadDim, chunks, query_strides.data(), cache_stride_array.data(),
        cache_stride_array.data(), no_mask_sizes.data(), no_mask_strides.data(),
        stream), "launching native GQA attention");
    cublaslt_linear_config_cuda_fp32(
        workspace.attention_output.const_data_ptr<float>(),
        layer.attention_output_weight.const_data_ptr<float>(),
        workspace.attention_projection_output.mutable_data_ptr<float>(),
        kHiddenSize, kHiddenSize, nullptr, 0, device_index,
        kRetainedProjectionAlgorithm, stream);
    check_cuda(residual_rmsnorm_cuda_fp32(
        workspace.attention_projection_output.const_data_ptr<float>(),
        input.const_data_ptr<float>(),
        layer.post_attention_norm_weight.const_data_ptr<float>(),
        workspace.norm_output.mutable_data_ptr<float>(),
        workspace.residual_output.mutable_data_ptr<float>(), 1, kHiddenSize,
        layer.epsilon, stream), "launching residual RMSNorm");
    check_cuda(packed_gate_up_swiglu_cuda_fp32(
        workspace.norm_output.const_data_ptr<float>(),
        layer.packed_gate_up_weight.const_data_ptr<float>(),
        workspace.swiglu_output.mutable_data_ptr<float>(), stream),
        "launching fused gate/up SwiGLU");

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    check_cublas(cublasSgemm(
        blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(kHiddenSize), 1,
        static_cast<int>(kIntermediateSize), &alpha,
        layer.down_projection_weight.const_data_ptr<float>(),
        static_cast<int>(kIntermediateSize),
        workspace.swiglu_output.const_data_ptr<float>(),
        static_cast<int>(kIntermediateSize), &beta,
        workspace.down_projection_output.mutable_data_ptr<float>(),
        static_cast<int>(kHiddenSize)), "launching down projection");
    check_cublas(cublasSgeam(
        blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(kHiddenSize), 1, &alpha,
        workspace.residual_output.const_data_ptr<float>(),
        static_cast<int>(kHiddenSize), &alpha,
        workspace.down_projection_output.const_data_ptr<float>(),
        static_cast<int>(kHiddenSize), output.mutable_data_ptr<float>(),
        static_cast<int>(kHiddenSize)), "launching final residual add");
}

}  // namespace

class NativeSmolLM2LayerDecode::Impl {
public:
    Impl(
        at::Tensor initial_hidden,
        at::Tensor input_norm_weight,
        at::Tensor packed_qkv_weight,
        at::Tensor attention_output_weight,
        at::Tensor post_attention_norm_weight,
        at::Tensor packed_gate_up_weight,
        at::Tensor down_projection_weight,
        at::Tensor rope_cos,
        at::Tensor rope_sin,
        at::Tensor initial_key_cache,
        at::Tensor initial_value_cache,
        const std::int64_t cache_capacity,
        const std::int64_t initial_position,
        const double epsilon,
        const double attention_scale)
        : device_(initial_hidden.device()),
          device_index_(initial_hidden.get_device()),
          capacity_(cache_capacity),
          initial_position_(initial_position),
          current_start_position_(initial_position),
          epsilon_(static_cast<float>(epsilon)),
          attention_scale_(static_cast<float>(attention_scale)),
          input_norm_weight_(std::move(input_norm_weight)),
          packed_qkv_weight_(std::move(packed_qkv_weight)),
          attention_output_weight_(std::move(attention_output_weight)),
          post_attention_norm_weight_(std::move(post_attention_norm_weight)),
          packed_gate_up_weight_(std::move(packed_gate_up_weight)),
          down_projection_weight_(std::move(down_projection_weight)),
          rope_cos_(std::move(rope_cos)),
          rope_sin_(std::move(rope_sin)) {
        TORCH_CHECK(device_.is_cuda(),
            "flux native decode: initial_hidden must be a CUDA tensor");
        TORCH_CHECK(capacity_ >= 1 && capacity_ <= kMaximumCapacity,
            "flux native decode: cache capacity must be in [1, 8192]");
        TORCH_CHECK(initial_position_ >= 0 && initial_position_ < capacity_,
            "flux native decode: initial position must be inside cache capacity");
        TORCH_CHECK(std::isfinite(epsilon_) && epsilon_ > 0.0F,
            "flux native decode: epsilon must be positive and finite");
        TORCH_CHECK(std::isfinite(attention_scale_) &&
                std::abs(attention_scale_) <= FLT_MAX,
            "flux native decode: attention scale must be finite FP32");

        validate_inputs(initial_hidden, initial_key_cache, initial_value_cache);
        const c10::cuda::CUDAGuard device_guard(device_);
        allocate_buffers(initial_hidden.options().requires_grad(false));

        try {
            check_cublas(cublasCreate(&blas_handle_), "cublasCreate");
            check_cublas(cublasSetMathMode(blas_handle_, CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode");
            check_cuda(cudaStreamCreateWithFlags(
                &capture_stream_, cudaStreamNonBlocking), "creating capture stream");
            check_cuda(cudaEventCreateWithFlags(
                &completion_event_, cudaEventDisableTiming),
                "creating completion event");

            const cudaStream_t current =
                c10::cuda::getCurrentCUDAStream(device_index_).stream();
            initialize_state(
                initial_hidden, initial_key_cache, initial_value_cache,
                initial_position_, current, false);
            check_cuda(cudaEventCreateWithFlags(
                &setup_event_, cudaEventDisableTiming), "creating setup event");
            check_cuda(cudaEventRecord(setup_event_, current),
                "recording setup event");
            check_cuda(cudaStreamWaitEvent(capture_stream_, setup_event_, 0),
                "waiting for setup inputs");

            run_body(capture_stream_);
            check_cuda(cudaStreamSynchronize(capture_stream_), "warming runtime");
            check_cuda(cudaEventDestroy(setup_event_), "destroying setup event");
            setup_event_ = nullptr;
            initialize_state(
                initial_hidden, initial_key_cache, initial_value_cache,
                initial_position_, capture_stream_, true);

            check_cuda(cudaStreamBeginCapture(
                capture_stream_, cudaStreamCaptureModeThreadLocal),
                "beginning CUDA Graph capture");
            capture_active_ = true;
            run_body(capture_stream_);
            check_cuda(cudaStreamEndCapture(capture_stream_, &graph_),
                "ending CUDA Graph capture");
            capture_active_ = false;
            TORCH_CHECK(graph_ != nullptr,
                "flux native decode: CUDA Graph capture returned a null graph");
            check_cuda(cudaGraphInstantiate(
                &graph_exec_, graph_, nullptr, nullptr, 0),
                "instantiating CUDA Graph");
            check_cuda(cudaStreamSynchronize(capture_stream_),
                "finalizing runtime capture");
        } catch (...) {
            release_resources_noexcept();
            throw;
        }
    }

    ~Impl() {
        release_resources_noexcept();
    }

    void release_resources_noexcept() noexcept {
        int previous_device = -1;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device_index_);
        if (capture_active_ && capture_stream_ != nullptr) {
            cudaGraph_t abandoned_graph = nullptr;
            cudaStreamEndCapture(capture_stream_, &abandoned_graph);
            if (abandoned_graph != nullptr) {
                cudaGraphDestroy(abandoned_graph);
            }
            capture_active_ = false;
        }
        if (pending_ && completion_event_ != nullptr) {
            cudaEventSynchronize(completion_event_);
        }
        if (graph_exec_ != nullptr) {
            cudaGraphExecDestroy(graph_exec_);
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
        }
        if (completion_event_ != nullptr) {
            cudaEventDestroy(completion_event_);
        }
        if (setup_event_ != nullptr) {
            cudaEventDestroy(setup_event_);
        }
        if (capture_stream_ != nullptr) {
            cudaStreamDestroy(capture_stream_);
        }
        if (blas_handle_ != nullptr) {
            cublasDestroy(blas_handle_);
        }
        if (previous_device >= 0 && previous_device != device_index_) {
            cudaSetDevice(previous_device);
        }
    }

    at::Tensor replay(const c10::optional<at::Tensor>& hidden) {
        TORCH_CHECK(replay_count_ < capacity_ - current_start_position_,
            "flux native decode: fixed-capacity state is exhausted");
        const c10::cuda::CUDAGuard device_guard(device_);
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();
        if (pending_) {
            check_cuda(cudaStreamWaitEvent(stream, completion_event_, 0),
                "ordering consecutive replays");
        }
        if (hidden.has_value()) {
            check_tensor(*hidden, device_, at::kFloat, {1, 1, kHiddenSize},
                "replay hidden state");
            check_cuda(cudaMemcpyAsync(
                input_hidden_.mutable_data_ptr<float>(),
                hidden->const_data_ptr<float>(),
                static_cast<std::size_t>(kHiddenSize) * sizeof(float),
                cudaMemcpyDeviceToDevice, stream), "copying replay hidden state");
        }
        check_cuda(cudaGraphLaunch(graph_exec_, stream), "launching CUDA Graph");
        check_cuda(cudaEventRecord(completion_event_, stream),
            "recording replay completion");
        pending_ = true;
        ++replay_count_;
        return output_;
    }

    void reset(
        const at::Tensor& hidden,
        const at::Tensor& key_cache,
        const at::Tensor& value_cache,
        const std::int64_t position) {
        validate_runtime_state(hidden, key_cache, value_cache, position);
        const c10::cuda::CUDAGuard device_guard(device_);
        if (pending_) {
            check_cuda(cudaEventSynchronize(completion_event_),
                "waiting to reset runtime");
            pending_ = false;
        }
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();
        initialize_state(hidden, key_cache, value_cache, position, stream, true);
        check_cuda(cudaStreamSynchronize(stream), "resetting runtime state");
        current_start_position_ = position;
        replay_count_ = 0;
    }

    std::int64_t position() {
        const c10::cuda::CUDAGuard device_guard(device_);
        if (pending_) {
            check_cuda(cudaEventSynchronize(completion_event_),
                "waiting to inspect device position");
        }
        std::int64_t result = -1;
        check_cuda(cudaMemcpy(
            &result, device_position_.const_data_ptr<std::int64_t>(),
            sizeof(result), cudaMemcpyDeviceToHost), "reading device position");
        return result;
    }

    void run_body(cudaStream_t stream) {
        check_cublas(cublasSetStream(blas_handle_, stream), "cublasSetStream");
        check_cuda(gather_rope_row_fp32(
            rope_cos_.const_data_ptr<float>(), rope_sin_.const_data_ptr<float>(),
            device_position_.const_data_ptr<std::int64_t>(),
            rope_cos_row_.mutable_data_ptr<float>(),
            rope_sin_row_.mutable_data_ptr<float>(), capacity_, stream),
            "gathering RoPE state");
        NativeLayerDescriptor layer{
            input_norm_weight_, packed_qkv_weight_, attention_output_weight_,
            post_attention_norm_weight_, packed_gate_up_weight_,
            down_projection_weight_, epsilon_, attention_scale_};
        NativeLayerWorkspace workspace{
            norm_output_, residual_output_, qkv_output_, query_output_,
            attention_output_, attention_projection_output_, swiglu_output_,
            down_projection_output_, attention_workspace_};
        execute_decoder_layer(
            layer, input_hidden_, output_, key_cache_, value_cache_,
            device_position_, device_position_, rope_cos_row_, rope_sin_row_,
            workspace, blas_handle_, device_index_, capacity_, stream, true);
    }

    void validate_inputs(
        const at::Tensor& initial_hidden,
        const at::Tensor& initial_key_cache,
        const at::Tensor& initial_value_cache) const {
        check_tensor(initial_hidden, device_, at::kFloat, {1, 1, kHiddenSize},
            "initial_hidden");
        check_tensor(input_norm_weight_, device_, at::kFloat, {kHiddenSize},
            "input_norm_weight");
        check_tensor(packed_qkv_weight_, device_, at::kFloat,
            {kPackedQKVWidth, kHiddenSize}, "packed_qkv_weight");
        check_tensor(attention_output_weight_, device_, at::kFloat,
            {kHiddenSize, kHiddenSize}, "attention_output_weight");
        check_tensor(post_attention_norm_weight_, device_, at::kFloat,
            {kHiddenSize}, "post_attention_norm_weight");
        check_tensor(packed_gate_up_weight_, device_, at::kFloat,
            {kPackedGateUpWidth, kHiddenSize}, "packed_gate_up_weight");
        check_tensor(down_projection_weight_, device_, at::kFloat,
            {kHiddenSize, kIntermediateSize}, "down_projection_weight");
        check_tensor(rope_cos_, device_, at::kFloat, {capacity_, kHeadDim},
            "rope_cos");
        check_tensor(rope_sin_, device_, at::kFloat, {capacity_, kHeadDim},
            "rope_sin");
        validate_runtime_state(
            initial_hidden, initial_key_cache, initial_value_cache,
            initial_position_);
    }

    void validate_runtime_state(
        const at::Tensor& hidden,
        const at::Tensor& key_cache,
        const at::Tensor& value_cache,
        const std::int64_t position) const {
        check_tensor(hidden, device_, at::kFloat, {1, 1, kHiddenSize},
            "hidden state");
        TORCH_CHECK(position >= 0 && position < capacity_,
            "flux native decode: reset position must be inside cache capacity");
        TORCH_CHECK(key_cache.defined() && value_cache.defined() &&
                key_cache.device() == device_ && value_cache.device() == device_ &&
                key_cache.scalar_type() == at::kFloat &&
                value_cache.scalar_type() == at::kFloat &&
                key_cache.is_contiguous() && value_cache.is_contiguous() &&
                key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes() &&
                key_cache.size(0) == 1 && key_cache.size(1) == kKeyValueHeads &&
                key_cache.size(2) >= position && key_cache.size(3) == kHeadDim,
            "flux native decode: initial/reset K/V must be matching contiguous "
            "float32 CUDA tensors shaped [1, 3, sequence>=position, 64]");
    }

    void allocate_buffers(const at::TensorOptions& options) {
        input_hidden_ = at::empty({1, 1, kHiddenSize}, options);
        output_ = at::empty({1, 1, kHiddenSize}, options);
        key_cache_ = at::empty(
            {1, kKeyValueHeads, capacity_, kHeadDim}, options);
        value_cache_ = at::empty(
            {1, kKeyValueHeads, capacity_, kHeadDim}, options);
        device_position_ = at::empty(
            {}, options.dtype(at::kLong));

        const std::int64_t chunks =
            (capacity_ + kAttentionChunkSize - 1) / kAttentionChunkSize;
        const std::int64_t attention_workspace_values =
            kQueryHeads * chunks * (kHeadDim + 2);
        const std::array<std::int64_t, 10> sizes{
            kHiddenSize,
            kHiddenSize,
            kPackedQKVWidth,
            kQueryHeads * kHeadDim,
            kQueryHeads * kHeadDim,
            kHiddenSize,
            kIntermediateSize,
            kHiddenSize,
            kHeadDim * 2,
            attention_workspace_values};
        std::int64_t total = 0;
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
        norm_output_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        residual_output_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        qkv_output_ = take(kPackedQKVWidth).view({1, 1, kPackedQKVWidth});
        query_output_ = take(kQueryHeads * kHeadDim)
            .view({1, kQueryHeads, 1, kHeadDim});
        attention_output_ = take(kQueryHeads * kHeadDim)
            .view({1, kQueryHeads, 1, kHeadDim});
        attention_projection_output_ =
            take(kHiddenSize).view({1, 1, kHiddenSize});
        swiglu_output_ =
            take(kIntermediateSize).view({1, 1, kIntermediateSize});
        down_projection_output_ =
            take(kHiddenSize).view({1, 1, kHiddenSize});
        at::Tensor rope_rows = take(kHeadDim * 2);
        rope_cos_row_ = rope_rows.slice(0, 0, kHeadDim).view({1, 1, kHeadDim});
        rope_sin_row_ = rope_rows.slice(0, kHeadDim, kHeadDim * 2)
            .view({1, 1, kHeadDim});
        attention_workspace_ = take(attention_workspace_values)
            .view({1, kQueryHeads, chunks, kHeadDim + 2});
        TORCH_INTERNAL_ASSERT(offset == total);
    }

    void initialize_state(
        const at::Tensor& hidden,
        const at::Tensor& source_key,
        const at::Tensor& source_value,
        const std::int64_t position,
        cudaStream_t stream,
        const bool clear_output) {
        const std::size_t hidden_bytes =
            static_cast<std::size_t>(kHiddenSize) * sizeof(float);
        const std::size_t cache_bytes = static_cast<std::size_t>(
            key_cache_.numel()) * sizeof(float);
        check_cuda(cudaMemcpyAsync(
            input_hidden_.mutable_data_ptr<float>(),
            hidden.const_data_ptr<float>(), hidden_bytes,
            cudaMemcpyDeviceToDevice, stream), "initializing hidden state");
        check_cuda(cudaMemsetAsync(
            key_cache_.mutable_data_ptr<float>(), 0, cache_bytes, stream),
            "clearing K cache");
        check_cuda(cudaMemsetAsync(
            value_cache_.mutable_data_ptr<float>(), 0, cache_bytes, stream),
            "clearing V cache");
        if (position > 0) {
            const std::size_t row_bytes = static_cast<std::size_t>(
                position * kHeadDim) * sizeof(float);
            const std::size_t source_pitch = static_cast<std::size_t>(
                source_key.size(2) * kHeadDim) * sizeof(float);
            const std::size_t destination_pitch = static_cast<std::size_t>(
                capacity_ * kHeadDim) * sizeof(float);
            check_cuda(cudaMemcpy2DAsync(
                key_cache_.mutable_data_ptr<float>(), destination_pitch,
                source_key.const_data_ptr<float>(), source_pitch, row_bytes,
                kKeyValueHeads, cudaMemcpyDeviceToDevice, stream),
                "importing K cache");
            check_cuda(cudaMemcpy2DAsync(
                value_cache_.mutable_data_ptr<float>(), destination_pitch,
                source_value.const_data_ptr<float>(), source_pitch, row_bytes,
                kKeyValueHeads, cudaMemcpyDeviceToDevice, stream),
                "importing V cache");
        }
        check_cuda(cudaMemcpyAsync(
            device_position_.mutable_data_ptr<std::int64_t>(), &position,
            sizeof(position), cudaMemcpyHostToDevice, stream),
            "initializing device position");
        if (clear_output) {
            check_cuda(cudaMemsetAsync(
                output_.mutable_data_ptr<float>(), 0, hidden_bytes, stream),
                "clearing layer output");
        }
    }

    at::Device device_;
    int device_index_;
    std::int64_t capacity_;
    std::int64_t initial_position_;
    std::int64_t current_start_position_;
    std::int64_t replay_count_ = 0;
    float epsilon_;
    float attention_scale_;
    bool pending_ = false;
    bool capture_active_ = false;

    at::Tensor input_norm_weight_;
    at::Tensor packed_qkv_weight_;
    at::Tensor attention_output_weight_;
    at::Tensor post_attention_norm_weight_;
    at::Tensor packed_gate_up_weight_;
    at::Tensor down_projection_weight_;
    at::Tensor rope_cos_;
    at::Tensor rope_sin_;

    at::Tensor input_hidden_;
    at::Tensor output_;
    at::Tensor key_cache_;
    at::Tensor value_cache_;
    at::Tensor device_position_;
    at::Tensor workspace_;
    at::Tensor norm_output_;
    at::Tensor residual_output_;
    at::Tensor qkv_output_;
    at::Tensor query_output_;
    at::Tensor attention_output_;
    at::Tensor attention_projection_output_;
    at::Tensor swiglu_output_;
    at::Tensor down_projection_output_;
    at::Tensor rope_cos_row_;
    at::Tensor rope_sin_row_;
    at::Tensor attention_workspace_;

    cublasHandle_t blas_handle_ = nullptr;
    cudaStream_t capture_stream_ = nullptr;
    cudaEvent_t completion_event_ = nullptr;
    cudaEvent_t setup_event_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

class NativeSmolLM2Decode::Impl {
public:
    Impl(
        at::Tensor initial_token,
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
        std::vector<at::Tensor> initial_key_caches,
        std::vector<at::Tensor> initial_value_caches,
        const std::int64_t cache_capacity,
        const std::int64_t initial_position,
        std::vector<double> epsilons,
        std::vector<double> attention_scales)
        : device_(initial_token.device()),
          device_index_(initial_token.get_device()),
          capacity_(cache_capacity),
          current_start_position_(initial_position),
          vocabulary_size_(embedding_weight.size(0)),
          embedding_weight_(std::move(embedding_weight)),
          final_norm_weight_(std::move(final_norm_weight)),
          lm_head_weight_(std::move(lm_head_weight)),
          rope_cos_(std::move(rope_cos)),
          rope_sin_(std::move(rope_sin)) {
        TORCH_CHECK(device_.is_cuda(),
            "flux native full decode: initial token must be a CUDA tensor");
        TORCH_CHECK(capacity_ >= 1 && capacity_ <= kMaximumCapacity,
            "flux native full decode: cache capacity must be in [1, 8192]");
        TORCH_CHECK(initial_position >= 0 && initial_position < capacity_,
            "flux native full decode: initial position must be inside capacity");
        constexpr std::size_t canonical_layers = 30;
        TORCH_CHECK(input_norm_weights.size() == canonical_layers &&
                packed_qkv_weights.size() == canonical_layers &&
                attention_output_weights.size() == canonical_layers &&
                post_attention_norm_weights.size() == canonical_layers &&
                packed_gate_up_weights.size() == canonical_layers &&
                down_projection_weights.size() == canonical_layers &&
                initial_key_caches.size() == canonical_layers &&
                initial_value_caches.size() == canonical_layers &&
                epsilons.size() == canonical_layers &&
                attention_scales.size() == canonical_layers,
            "flux native full decode: exactly 30 complete layer descriptors are required");

        layers_.reserve(canonical_layers);
        for (std::size_t index = 0; index < canonical_layers; ++index) {
            TORCH_CHECK(std::isfinite(epsilons[index]) && epsilons[index] > 0.0,
                "flux native full decode: every RMSNorm epsilon must be positive");
            TORCH_CHECK(std::isfinite(attention_scales[index]),
                "flux native full decode: every attention scale must be finite");
            layers_.push_back(NativeLayerDescriptor{
                std::move(input_norm_weights[index]),
                std::move(packed_qkv_weights[index]),
                std::move(attention_output_weights[index]),
                std::move(post_attention_norm_weights[index]),
                std::move(packed_gate_up_weights[index]),
                std::move(down_projection_weights[index]),
                static_cast<float>(epsilons[index]),
                static_cast<float>(attention_scales[index])});
        }
        validate_weights_and_state(
            initial_token, initial_key_caches, initial_value_caches,
            initial_position);

        const c10::cuda::CUDAGuard device_guard(device_);
        allocate_buffers(initial_token.options().dtype(at::kFloat).requires_grad(false));
        try {
            check_cublas(cublasCreate(&blas_handle_), "cublasCreate");
            check_cublas(cublasSetMathMode(blas_handle_, CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode");
            check_cuda(cudaStreamCreateWithFlags(
                &capture_stream_, cudaStreamNonBlocking),
                "creating full-runtime capture stream");
            check_cuda(cudaEventCreateWithFlags(
                &completion_event_, cudaEventDisableTiming),
                "creating full-runtime completion event");

            const cudaStream_t current =
                c10::cuda::getCurrentCUDAStream(device_index_).stream();
            initialize_state(
                initial_token, initial_key_caches, initial_value_caches,
                initial_position, current, false);
            check_cuda(cudaEventCreateWithFlags(
                &setup_event_, cudaEventDisableTiming),
                "creating full-runtime setup event");
            check_cuda(cudaEventRecord(setup_event_, current),
                "recording full-runtime setup event");
            check_cuda(cudaStreamWaitEvent(capture_stream_, setup_event_, 0),
                "ordering full-runtime setup");

            run_body(capture_stream_);
            check_cuda(cudaStreamSynchronize(capture_stream_),
                "warming full native runtime");
            check_cuda(cudaEventDestroy(setup_event_),
                "destroying full-runtime setup event");
            setup_event_ = nullptr;
            initialize_state(
                initial_token, initial_key_caches, initial_value_caches,
                initial_position, capture_stream_, true);

            check_cuda(cudaStreamBeginCapture(
                capture_stream_, cudaStreamCaptureModeThreadLocal),
                "beginning full CUDA Graph capture");
            capture_active_ = true;
            run_body(capture_stream_);
            check_cuda(cudaStreamEndCapture(capture_stream_, &graph_),
                "ending full CUDA Graph capture");
            capture_active_ = false;
            TORCH_CHECK(graph_ != nullptr,
                "flux native full decode: capture returned a null graph");
            check_cuda(cudaGraphInstantiate(
                &graph_exec_, graph_, nullptr, nullptr, 0),
                "instantiating full CUDA Graph");
            check_cuda(cudaStreamSynchronize(capture_stream_),
                "finalizing full runtime capture");
        } catch (...) {
            release_resources_noexcept();
            throw;
        }
    }

    ~Impl() { release_resources_noexcept(); }

    void release_resources_noexcept() noexcept {
        int previous_device = -1;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device_index_);
        if (capture_active_ && capture_stream_ != nullptr) {
            cudaGraph_t abandoned_graph = nullptr;
            cudaStreamEndCapture(capture_stream_, &abandoned_graph);
            if (abandoned_graph != nullptr) {
                cudaGraphDestroy(abandoned_graph);
            }
            capture_active_ = false;
        }
        if (pending_ && completion_event_ != nullptr) {
            cudaEventSynchronize(completion_event_);
        }
        if (graph_exec_ != nullptr) {
            cudaGraphExecDestroy(graph_exec_);
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
        }
        if (completion_event_ != nullptr) {
            cudaEventDestroy(completion_event_);
        }
        if (setup_event_ != nullptr) {
            cudaEventDestroy(setup_event_);
        }
        if (capture_stream_ != nullptr) {
            cudaStreamDestroy(capture_stream_);
        }
        if (blas_handle_ != nullptr) {
            cublasDestroy(blas_handle_);
        }
        if (previous_device >= 0 && previous_device != device_index_) {
            cudaSetDevice(previous_device);
        }
    }

    at::Tensor replay(const c10::optional<at::Tensor>& token) {
        TORCH_CHECK(replay_count_ < capacity_ - current_start_position_,
            "flux native full decode: fixed-capacity state is exhausted");
        const c10::cuda::CUDAGuard device_guard(device_);
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();
        if (pending_) {
            check_cuda(cudaStreamWaitEvent(stream, completion_event_, 0),
                "ordering consecutive full-runtime replays");
        }
        if (token.has_value()) {
            check_tensor(*token, device_, at::kLong, {1, 1}, "replay token");
            check_cuda(cudaMemcpyAsync(
                input_token_.mutable_data_ptr<std::int64_t>(),
                token->const_data_ptr<std::int64_t>(), sizeof(std::int64_t),
                cudaMemcpyDeviceToDevice, stream), "copying replay token");
        }
        check_cuda(cudaGraphLaunch(graph_exec_, stream),
            "launching full CUDA Graph");
        check_cuda(cudaEventRecord(completion_event_, stream),
            "recording full-runtime replay completion");
        pending_ = true;
        ++replay_count_;
        return logits_;
    }

    at::Tensor generate_greedy(const std::int64_t decode_steps) {
        TORCH_CHECK(decode_steps >= 0,
            "flux native full decode: generation step count must be non-negative");
        TORCH_CHECK(decode_steps <=
                capacity_ - current_start_position_ - replay_count_,
            "flux native full decode: fixed-capacity generation is exhausted");
        TORCH_CHECK(!generation_started_,
            "flux native full decode: prefill or reset is required before reuse");
        const c10::cuda::CUDAGuard device_guard(device_);
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();
        if (pending_) {
            check_cuda(cudaStreamWaitEvent(stream, completion_event_, 0),
                "ordering native greedy generation");
        }
        for (std::int64_t step = 0; step < decode_steps; ++step) {
            check_cuda(cudaGraphLaunch(graph_exec_, stream),
                "launching chained greedy decode graph");
        }
        if (decode_steps > 0) {
            check_cuda(cudaEventRecord(completion_event_, stream),
                "recording native greedy generation completion");
            pending_ = true;
            replay_count_ += decode_steps;
        }
        generation_started_ = true;
        return generated_tokens_;
    }

    void reset(
        const at::Tensor& token,
        const std::vector<at::Tensor>& key_caches,
        const std::vector<at::Tensor>& value_caches,
        const std::int64_t position) {
        validate_runtime_state(token, key_caches, value_caches, position);
        const c10::cuda::CUDAGuard device_guard(device_);
        if (pending_) {
            check_cuda(cudaEventSynchronize(completion_event_),
                "waiting to reset full runtime");
            pending_ = false;
        }
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();
        initialize_state(token, key_caches, value_caches, position, stream, true);
        check_cuda(cudaStreamSynchronize(stream), "resetting full runtime state");
        current_start_position_ = position;
        replay_count_ = 0;
        generation_started_ = false;
    }

    void wait_for_prefill(cudaStream_t stream) {
        if (pending_) {
            check_cuda(cudaStreamWaitEvent(stream, completion_event_, 0),
                "ordering native prefill after decode work");
            pending_ = false;
        }
    }

    void install_prefilled_state(
        const at::Tensor& token,
        const std::int64_t position,
        cudaStream_t stream) {
        check_tensor(token, device_, at::kLong, {1, 1}, "prefill token");
        TORCH_CHECK(position >= 1 && position <= capacity_,
            "flux native full decode: prefill position must be in [1, capacity]");
        check_cuda(cudaMemcpyAsync(
            input_token_.mutable_data_ptr<std::int64_t>(),
            token.const_data_ptr<std::int64_t>(), sizeof(std::int64_t),
            cudaMemcpyDeviceToDevice, stream), "installing native prefill token");
        check_cuda(cudaMemcpyAsync(
            device_position_.mutable_data_ptr<std::int64_t>(), &position,
            sizeof(position), cudaMemcpyHostToDevice, stream),
            "installing native prefill position");
        check_cuda(cudaMemcpyAsync(
            device_cache_length_.mutable_data_ptr<std::int64_t>(), &position,
            sizeof(position), cudaMemcpyHostToDevice, stream),
            "installing native prefill cache length");
        check_cuda(cudaMemcpyAsync(
            generated_tokens_.mutable_data_ptr<std::int64_t>(),
            input_token_.const_data_ptr<std::int64_t>(), sizeof(std::int64_t),
            cudaMemcpyDeviceToDevice, stream),
            "seeding native generated-token buffer");
        constexpr std::int64_t first_generation_step = 1;
        check_cuda(cudaMemcpyAsync(
            device_generation_step_.mutable_data_ptr<std::int64_t>(),
            &first_generation_step, sizeof(first_generation_step),
            cudaMemcpyHostToDevice, stream),
            "seeding native generation step");
        check_cuda(cudaEventRecord(completion_event_, stream),
            "recording native prefill completion");
        current_start_position_ = position;
        replay_count_ = 0;
        generation_started_ = false;
        pending_ = true;
    }

    std::int64_t read_scalar(const at::Tensor& state, const char* operation) {
        const c10::cuda::CUDAGuard device_guard(device_);
        if (pending_) {
            check_cuda(cudaEventSynchronize(completion_event_),
                "waiting to inspect full-runtime state");
        }
        std::int64_t result = -1;
        check_cuda(cudaMemcpy(
            &result, state.const_data_ptr<std::int64_t>(), sizeof(result),
            cudaMemcpyDeviceToHost), operation);
        return result;
    }

    void run_body(cudaStream_t stream) {
        check_cublas(cublasSetStream(blas_handle_, stream), "cublasSetStream");
        check_cuda(prepare_full_decode_fp32(
            input_token_.const_data_ptr<std::int64_t>(),
            embedding_weight_.const_data_ptr<float>(), vocabulary_size_,
            rope_cos_.const_data_ptr<float>(), rope_sin_.const_data_ptr<float>(),
            device_position_.const_data_ptr<std::int64_t>(),
            hidden_a_.mutable_data_ptr<float>(),
            rope_cos_row_.mutable_data_ptr<float>(),
            rope_sin_row_.mutable_data_ptr<float>(),
            device_cache_length_.mutable_data_ptr<std::int64_t>(), capacity_,
            stream), "preparing token embedding and device state");

        NativeLayerWorkspace workspace{
            norm_output_, residual_output_, qkv_output_, query_output_,
            attention_output_, attention_projection_output_, swiglu_output_,
            down_projection_output_, attention_workspace_};
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            at::Tensor& input = index % 2 == 0 ? hidden_a_ : hidden_b_;
            at::Tensor& output = index % 2 == 0 ? hidden_b_ : hidden_a_;
            execute_decoder_layer(
                layers_[index], input, output, key_layers_[index],
                value_layers_[index], device_position_, device_cache_length_,
                rope_cos_row_, rope_sin_row_, workspace, blas_handle_,
                device_index_, capacity_, stream, false);
        }
        at::Tensor& final_hidden =
            layers_.size() % 2 == 0 ? hidden_a_ : hidden_b_;
        check_cuda(rmsnorm_cuda_fp32(
            final_hidden.const_data_ptr<float>(),
            final_norm_weight_.const_data_ptr<float>(),
            final_norm_output_.mutable_data_ptr<float>(), 1, kHiddenSize,
            layers_.front().epsilon, stream), "launching final RMSNorm");

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        check_cublas(cublasSgemm(
            blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(vocabulary_size_), 1,
            static_cast<int>(kHiddenSize), &alpha,
            lm_head_weight_.const_data_ptr<float>(),
            static_cast<int>(kHiddenSize),
            final_norm_output_.const_data_ptr<float>(),
            static_cast<int>(kHiddenSize), &beta,
            logits_.mutable_data_ptr<float>(),
            static_cast<int>(vocabulary_size_)), "launching production LM head");
        check_cuda(greedy_argmax_update_cuda_fp32(
            logits_.const_data_ptr<float>(),
            input_token_.mutable_data_ptr<std::int64_t>(),
            generated_tokens_.mutable_data_ptr<std::int64_t>(),
            device_generation_step_.mutable_data_ptr<std::int64_t>(),
            generated_tokens_.numel(),
            device_position_.mutable_data_ptr<std::int64_t>(),
            device_cache_length_.const_data_ptr<std::int64_t>(),
            vocabulary_size_, stream),
            "selecting greedy token and advancing device state");
    }

    void validate_weights_and_state(
        const at::Tensor& token,
        const std::vector<at::Tensor>& key_caches,
        const std::vector<at::Tensor>& value_caches,
        const std::int64_t position) const {
        check_tensor(token, device_, at::kLong, {1, 1}, "initial token");
        TORCH_CHECK(vocabulary_size_ > 0,
            "flux native full decode: vocabulary must be non-empty");
        check_tensor(embedding_weight_, device_, at::kFloat,
            {vocabulary_size_, kHiddenSize}, "embedding weight");
        check_tensor(lm_head_weight_, device_, at::kFloat,
            {vocabulary_size_, kHiddenSize}, "LM-head weight");
        check_tensor(final_norm_weight_, device_, at::kFloat,
            {kHiddenSize}, "final norm weight");
        check_tensor(rope_cos_, device_, at::kFloat,
            {capacity_, kHeadDim}, "RoPE cosine table");
        check_tensor(rope_sin_, device_, at::kFloat,
            {capacity_, kHeadDim}, "RoPE sine table");
        for (const NativeLayerDescriptor& layer : layers_) {
            check_tensor(layer.input_norm_weight, device_, at::kFloat,
                {kHiddenSize}, "layer input norm weight");
            check_tensor(layer.packed_qkv_weight, device_, at::kFloat,
                {kPackedQKVWidth, kHiddenSize}, "layer packed QKV weight");
            check_tensor(layer.attention_output_weight, device_, at::kFloat,
                {kHiddenSize, kHiddenSize}, "layer attention output weight");
            check_tensor(layer.post_attention_norm_weight, device_, at::kFloat,
                {kHiddenSize}, "layer post-attention norm weight");
            check_tensor(layer.packed_gate_up_weight, device_, at::kFloat,
                {kPackedGateUpWidth, kHiddenSize}, "layer gate/up weight");
            check_tensor(layer.down_projection_weight, device_, at::kFloat,
                {kHiddenSize, kIntermediateSize}, "layer down weight");
        }
        validate_runtime_state(token, key_caches, value_caches, position);
    }

    void validate_runtime_state(
        const at::Tensor& token,
        const std::vector<at::Tensor>& key_caches,
        const std::vector<at::Tensor>& value_caches,
        const std::int64_t position) const {
        check_tensor(token, device_, at::kLong, {1, 1}, "token");
        TORCH_CHECK(position >= 0 && position < capacity_,
            "flux native full decode: reset position must be inside capacity");
        TORCH_CHECK(key_caches.size() == layers_.size() &&
                value_caches.size() == layers_.size(),
            "flux native full decode: reset requires one K/V tensor per layer");
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            const at::Tensor& key = key_caches[index];
            const at::Tensor& value = value_caches[index];
            TORCH_CHECK(key.defined() && value.defined() &&
                    key.device() == device_ && value.device() == device_ &&
                    key.scalar_type() == at::kFloat &&
                    value.scalar_type() == at::kFloat &&
                    key.is_contiguous() && value.is_contiguous() &&
                    key.dim() == 4 && value.sizes() == key.sizes() &&
                    key.size(0) == 1 && key.size(1) == kKeyValueHeads &&
                    key.size(2) >= position && key.size(3) == kHeadDim,
                "flux native full decode: every K/V source must be contiguous "
                "float32 [1, 3, sequence>=position, 64]");
        }
    }

    void allocate_buffers(const at::TensorOptions& float_options) {
        const at::TensorOptions long_options = float_options.dtype(at::kLong);
        input_token_ = at::empty({1, 1}, long_options);
        device_position_ = at::empty({}, long_options);
        device_cache_length_ = at::empty({}, long_options);
        device_generation_step_ = at::empty({}, long_options);
        generated_tokens_ = at::empty({capacity_}, long_options);
        logits_ = at::empty({1, 1, vocabulary_size_}, float_options);
        key_cache_ = at::empty(
            {static_cast<std::int64_t>(layers_.size()), 1, kKeyValueHeads,
             capacity_, kHeadDim}, float_options);
        value_cache_ = at::empty_like(key_cache_);
        key_layers_.reserve(layers_.size());
        value_layers_.reserve(layers_.size());
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            key_layers_.push_back(key_cache_.select(0, index));
            value_layers_.push_back(value_cache_.select(0, index));
        }

        const std::int64_t chunks =
            (capacity_ + kAttentionChunkSize - 1) / kAttentionChunkSize;
        const std::int64_t attention_workspace_values =
            kQueryHeads * chunks * (kHeadDim + 2);
        const std::array<std::int64_t, 11> sizes{
            kHiddenSize, kHiddenSize, kHiddenSize, kHiddenSize,
            kPackedQKVWidth, kQueryHeads * kHeadDim,
            kQueryHeads * kHeadDim, kHiddenSize, kIntermediateSize,
            kHiddenSize, kHeadDim * 2 + attention_workspace_values};
        std::int64_t total = 0;
        for (const std::int64_t size : sizes) {
            total += size;
        }
        workspace_ = at::empty({total}, float_options);
        std::int64_t offset = 0;
        auto take = [&](const std::int64_t size) {
            at::Tensor result = workspace_.slice(0, offset, offset + size);
            offset += size;
            return result;
        };
        hidden_a_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        hidden_b_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        norm_output_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        residual_output_ = take(kHiddenSize).view({1, 1, kHiddenSize});
        qkv_output_ = take(kPackedQKVWidth).view({1, 1, kPackedQKVWidth});
        query_output_ = take(kQueryHeads * kHeadDim)
            .view({1, kQueryHeads, 1, kHeadDim});
        attention_output_ = take(kQueryHeads * kHeadDim)
            .view({1, kQueryHeads, 1, kHeadDim});
        attention_projection_output_ =
            take(kHiddenSize).view({1, 1, kHiddenSize});
        swiglu_output_ =
            take(kIntermediateSize).view({1, 1, kIntermediateSize});
        down_projection_output_ =
            take(kHiddenSize).view({1, 1, kHiddenSize});
        at::Tensor tail = take(kHeadDim * 2 + attention_workspace_values);
        rope_cos_row_ = tail.slice(0, 0, kHeadDim).view({1, 1, kHeadDim});
        rope_sin_row_ = tail.slice(0, kHeadDim, kHeadDim * 2)
            .view({1, 1, kHeadDim});
        attention_workspace_ = tail.slice(0, kHeadDim * 2)
            .view({1, kQueryHeads, chunks, kHeadDim + 2});
        final_norm_output_ = norm_output_;
        TORCH_INTERNAL_ASSERT(offset == total);
    }

    void initialize_state(
        const at::Tensor& token,
        const std::vector<at::Tensor>& source_keys,
        const std::vector<at::Tensor>& source_values,
        const std::int64_t position,
        cudaStream_t stream,
        const bool clear_outputs) {
        check_cuda(cudaMemcpyAsync(
            input_token_.mutable_data_ptr<std::int64_t>(),
            token.const_data_ptr<std::int64_t>(), sizeof(std::int64_t),
            cudaMemcpyDeviceToDevice, stream), "initializing full-runtime token");
        const std::size_t cache_bytes =
            static_cast<std::size_t>(key_cache_.numel()) * sizeof(float);
        check_cuda(cudaMemsetAsync(
            key_cache_.mutable_data_ptr<float>(), 0, cache_bytes, stream),
            "clearing full-runtime K cache");
        check_cuda(cudaMemsetAsync(
            value_cache_.mutable_data_ptr<float>(), 0, cache_bytes, stream),
            "clearing full-runtime V cache");
        if (position > 0) {
            const std::size_t row_bytes = static_cast<std::size_t>(
                position * kHeadDim) * sizeof(float);
            const std::size_t destination_pitch = static_cast<std::size_t>(
                capacity_ * kHeadDim) * sizeof(float);
            for (std::size_t index = 0; index < layers_.size(); ++index) {
                const std::size_t source_pitch = static_cast<std::size_t>(
                    source_keys[index].size(2) * kHeadDim) * sizeof(float);
                check_cuda(cudaMemcpy2DAsync(
                    key_layers_[index].mutable_data_ptr<float>(),
                    destination_pitch, source_keys[index].const_data_ptr<float>(),
                    source_pitch, row_bytes, kKeyValueHeads,
                    cudaMemcpyDeviceToDevice, stream), "importing full-runtime K cache");
                check_cuda(cudaMemcpy2DAsync(
                    value_layers_[index].mutable_data_ptr<float>(),
                    destination_pitch, source_values[index].const_data_ptr<float>(),
                    source_pitch, row_bytes, kKeyValueHeads,
                    cudaMemcpyDeviceToDevice, stream), "importing full-runtime V cache");
            }
        }
        check_cuda(cudaMemcpyAsync(
            device_position_.mutable_data_ptr<std::int64_t>(), &position,
            sizeof(position), cudaMemcpyHostToDevice, stream),
            "initializing full-runtime position");
        check_cuda(cudaMemcpyAsync(
            device_cache_length_.mutable_data_ptr<std::int64_t>(), &position,
            sizeof(position), cudaMemcpyHostToDevice, stream),
            "initializing full-runtime cache length");
        check_cuda(cudaMemsetAsync(
            device_generation_step_.mutable_data_ptr<std::int64_t>(), 0,
            sizeof(std::int64_t), stream),
            "initializing full-runtime generation state");
        if (clear_outputs) {
            check_cuda(cudaMemsetAsync(
                logits_.mutable_data_ptr<float>(), 0,
                static_cast<std::size_t>(logits_.numel()) * sizeof(float), stream),
                "clearing full-runtime logits");
            check_cuda(cudaMemsetAsync(
                workspace_.mutable_data_ptr<float>(), 0,
                static_cast<std::size_t>(workspace_.numel()) * sizeof(float), stream),
                "clearing full-runtime workspace");
        }
    }

    at::Device device_;
    int device_index_;
    std::int64_t capacity_;
    std::int64_t current_start_position_;
    std::int64_t vocabulary_size_;
    std::int64_t replay_count_ = 0;
    bool pending_ = false;
    bool capture_active_ = false;
    bool generation_started_ = false;

    at::Tensor embedding_weight_;
    at::Tensor final_norm_weight_;
    at::Tensor lm_head_weight_;
    at::Tensor rope_cos_;
    at::Tensor rope_sin_;
    std::vector<NativeLayerDescriptor> layers_;

    at::Tensor input_token_;
    at::Tensor generated_tokens_;
    at::Tensor device_generation_step_;
    at::Tensor logits_;
    at::Tensor key_cache_;
    at::Tensor value_cache_;
    std::vector<at::Tensor> key_layers_;
    std::vector<at::Tensor> value_layers_;
    at::Tensor device_position_;
    at::Tensor device_cache_length_;
    at::Tensor workspace_;
    at::Tensor hidden_a_;
    at::Tensor hidden_b_;
    at::Tensor norm_output_;
    at::Tensor residual_output_;
    at::Tensor qkv_output_;
    at::Tensor query_output_;
    at::Tensor attention_output_;
    at::Tensor attention_projection_output_;
    at::Tensor swiglu_output_;
    at::Tensor down_projection_output_;
    at::Tensor rope_cos_row_;
    at::Tensor rope_sin_row_;
    at::Tensor attention_workspace_;
    at::Tensor final_norm_output_;

    cublasHandle_t blas_handle_ = nullptr;
    cudaStream_t capture_stream_ = nullptr;
    cudaEvent_t completion_event_ = nullptr;
    cudaEvent_t setup_event_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

NativeSmolLM2LayerDecode::NativeSmolLM2LayerDecode(
    at::Tensor initial_hidden,
    at::Tensor input_norm_weight,
    at::Tensor packed_qkv_weight,
    at::Tensor attention_output_weight,
    at::Tensor post_attention_norm_weight,
    at::Tensor packed_gate_up_weight,
    at::Tensor down_projection_weight,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    at::Tensor initial_key_cache,
    at::Tensor initial_value_cache,
    const std::int64_t cache_capacity,
    const std::int64_t initial_position,
    const double epsilon,
    const double attention_scale)
    : impl_(std::make_unique<Impl>(
          std::move(initial_hidden), std::move(input_norm_weight),
          std::move(packed_qkv_weight), std::move(attention_output_weight),
          std::move(post_attention_norm_weight),
          std::move(packed_gate_up_weight), std::move(down_projection_weight),
          std::move(rope_cos), std::move(rope_sin),
          std::move(initial_key_cache), std::move(initial_value_cache),
          cache_capacity, initial_position, epsilon, attention_scale)) {}

NativeSmolLM2LayerDecode::~NativeSmolLM2LayerDecode() = default;

at::Tensor NativeSmolLM2LayerDecode::replay(
    const c10::optional<at::Tensor>& hidden) {
    return impl_->replay(hidden);
}

void NativeSmolLM2LayerDecode::reset(
    const at::Tensor& hidden,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::int64_t position) {
    impl_->reset(hidden, key_cache, value_cache, position);
}

at::Tensor NativeSmolLM2LayerDecode::output() const { return impl_->output_; }
at::Tensor NativeSmolLM2LayerDecode::key_cache() const { return impl_->key_cache_; }
at::Tensor NativeSmolLM2LayerDecode::value_cache() const { return impl_->value_cache_; }
at::Tensor NativeSmolLM2LayerDecode::device_position() const {
    return impl_->device_position_;
}
at::Tensor NativeSmolLM2LayerDecode::workspace() const { return impl_->workspace_; }

std::vector<std::int64_t> NativeSmolLM2LayerDecode::addresses() const {
    return {
        reinterpret_cast<std::int64_t>(impl_->input_hidden_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->key_cache_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->value_cache_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->device_position_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->workspace_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->norm_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->residual_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->qkv_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->query_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->attention_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(
            impl_->attention_projection_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->swiglu_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(
            impl_->down_projection_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->attention_workspace_.data_ptr())};
}

std::int64_t NativeSmolLM2LayerDecode::position() { return impl_->position(); }
std::int64_t NativeSmolLM2LayerDecode::cache_length() { return impl_->position(); }
std::int64_t NativeSmolLM2LayerDecode::capacity() const { return impl_->capacity_; }
std::int64_t NativeSmolLM2LayerDecode::replay_count() const {
    return impl_->replay_count_;
}
std::int64_t NativeSmolLM2LayerDecode::workspace_bytes() const {
    return tensor_bytes(impl_->workspace_);
}

NativeSmolLM2Decode::NativeSmolLM2Decode(
    at::Tensor initial_token,
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
    std::vector<at::Tensor> initial_key_caches,
    std::vector<at::Tensor> initial_value_caches,
    const std::int64_t cache_capacity,
    const std::int64_t initial_position,
    std::vector<double> epsilons,
    std::vector<double> attention_scales)
    : impl_(std::make_unique<Impl>(
          std::move(initial_token), std::move(embedding_weight),
          std::move(input_norm_weights), std::move(packed_qkv_weights),
          std::move(attention_output_weights),
          std::move(post_attention_norm_weights),
          std::move(packed_gate_up_weights),
          std::move(down_projection_weights), std::move(final_norm_weight),
          std::move(lm_head_weight), std::move(rope_cos), std::move(rope_sin),
          std::move(initial_key_caches), std::move(initial_value_caches),
          cache_capacity, initial_position, std::move(epsilons),
          std::move(attention_scales))) {}

NativeSmolLM2Decode::~NativeSmolLM2Decode() = default;

at::Tensor NativeSmolLM2Decode::replay(
    const c10::optional<at::Tensor>& token) {
    return impl_->replay(token);
}

at::Tensor NativeSmolLM2Decode::generate_greedy(
    const std::int64_t decode_steps) {
    return impl_->generate_greedy(decode_steps);
}

void NativeSmolLM2Decode::reset(
    const at::Tensor& token,
    const std::vector<at::Tensor>& key_caches,
    const std::vector<at::Tensor>& value_caches,
    const std::int64_t position) {
    impl_->reset(token, key_caches, value_caches, position);
}

at::Tensor NativeSmolLM2Decode::logits() const { return impl_->logits_; }
at::Tensor NativeSmolLM2Decode::current_token() const {
    return impl_->input_token_;
}
at::Tensor NativeSmolLM2Decode::generated_tokens() const {
    return impl_->generated_tokens_;
}
at::Tensor NativeSmolLM2Decode::device_generation_step() const {
    return impl_->device_generation_step_;
}
at::Tensor NativeSmolLM2Decode::key_cache() const { return impl_->key_cache_; }
at::Tensor NativeSmolLM2Decode::value_cache() const { return impl_->value_cache_; }
at::Tensor NativeSmolLM2Decode::device_position() const {
    return impl_->device_position_;
}
at::Tensor NativeSmolLM2Decode::device_cache_length() const {
    return impl_->device_cache_length_;
}
at::Tensor NativeSmolLM2Decode::workspace() const { return impl_->workspace_; }

std::vector<std::int64_t> NativeSmolLM2Decode::addresses() const {
    return {
        reinterpret_cast<std::int64_t>(impl_->input_token_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->logits_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->generated_tokens_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->device_generation_step_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->key_cache_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->value_cache_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->device_position_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->device_cache_length_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->workspace_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->hidden_a_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->hidden_b_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->final_norm_output_.data_ptr()),
        reinterpret_cast<std::int64_t>(impl_->attention_workspace_.data_ptr())};
}

std::int64_t NativeSmolLM2Decode::position() {
    return impl_->read_scalar(impl_->device_position_, "reading device position");
}
std::int64_t NativeSmolLM2Decode::cache_length() {
    return impl_->read_scalar(
        impl_->device_cache_length_, "reading device cache length");
}
std::int64_t NativeSmolLM2Decode::capacity() const { return impl_->capacity_; }
std::int64_t NativeSmolLM2Decode::replay_count() const {
    return impl_->replay_count_;
}
std::int64_t NativeSmolLM2Decode::generation_step() {
    return impl_->read_scalar(
        impl_->device_generation_step_, "reading device generation step");
}
std::int64_t NativeSmolLM2Decode::workspace_bytes() const {
    return tensor_bytes(impl_->workspace_);
}
std::int64_t NativeSmolLM2Decode::stable_buffer_bytes() const {
    return tensor_bytes(impl_->input_token_) + tensor_bytes(impl_->logits_) +
        tensor_bytes(impl_->generated_tokens_) +
        tensor_bytes(impl_->device_generation_step_) +
        tensor_bytes(impl_->device_position_) +
        tensor_bytes(impl_->device_cache_length_);
}
std::int64_t NativeSmolLM2Decode::vocabulary_size() const {
    return impl_->vocabulary_size_;
}
std::int64_t NativeSmolLM2Decode::layer_count() const {
    return static_cast<std::int64_t>(impl_->layers_.size());
}

void NativeSmolLM2Decode::wait_for_prefill(cudaStream_t stream) {
    impl_->wait_for_prefill(stream);
}

void NativeSmolLM2Decode::install_prefilled_state(
    const at::Tensor& token,
    const std::int64_t position,
    cudaStream_t stream) {
    impl_->install_prefilled_state(token, position, stream);
}

}  // namespace flux
