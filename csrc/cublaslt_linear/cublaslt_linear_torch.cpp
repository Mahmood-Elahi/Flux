#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace flux {
namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "flux::cublaslt_linear: ", operation, " failed with ",
        cublasGetStatusString(status));
}

class MatmulPlan {
public:
    MatmulPlan(int64_t output_width, int64_t input_width, int64_t max_workspace_bytes)
        : output_width_(output_width), input_width_(input_width) {
        check_cublas(
            cublasLtMatmulDescCreate(
                &operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "cublasLtMatmulDescCreate");
        cublasOperation_t trans_a = CUBLAS_OP_N;
        cublasOperation_t trans_b = CUBLAS_OP_T;
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a, sizeof(trans_a)),
            "setting TRANSA");
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b, sizeof(trans_b)),
            "setting TRANSB");

        check_cublas(
            cublasLtMatrixLayoutCreate(
                &input_, CUDA_R_32F, 1, input_width_, input_width_),
            "creating input layout");
        check_cublas(
            cublasLtMatrixLayoutCreate(
                &weight_, CUDA_R_32F, output_width_, input_width_, input_width_),
            "creating weight layout");
        check_cublas(
            cublasLtMatrixLayoutCreate(
                &output_, CUDA_R_32F, 1, output_width_, output_width_),
            "creating output layout");
        cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
        for (cublasLtMatrixLayout_t layout : {input_, weight_, output_}) {
            check_cublas(
                cublasLtMatrixLayoutSetAttribute(
                    layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                    &row_order, sizeof(row_order)),
                "setting row-major layout");
        }

        check_cublas(
            cublasLtMatmulPreferenceCreate(&preference_),
            "cublasLtMatmulPreferenceCreate");
        const std::size_t workspace_limit =
            static_cast<std::size_t>(max_workspace_bytes);
        check_cublas(
            cublasLtMatmulPreferenceSetAttribute(
                preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_limit, sizeof(workspace_limit)),
            "setting workspace preference");
    }

    ~MatmulPlan() {
        if (preference_ != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference_);
        }
        if (output_ != nullptr) {
            cublasLtMatrixLayoutDestroy(output_);
        }
        if (weight_ != nullptr) {
            cublasLtMatrixLayoutDestroy(weight_);
        }
        if (input_ != nullptr) {
            cublasLtMatrixLayoutDestroy(input_);
        }
        if (operation_ != nullptr) {
            cublasLtMatmulDescDestroy(operation_);
        }
    }

    MatmulPlan(const MatmulPlan&) = delete;
    MatmulPlan& operator=(const MatmulPlan&) = delete;

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(int requested) const;
    cublasLtMatmulAlgo_t configured_algorithm(
        int32_t algorithm_id,
        uint32_t tile_id,
        int32_t split_k,
        uint32_t reduction_scheme,
        uint32_t cta_swizzle,
        uint32_t custom_option,
        uint32_t stages_id) const;

    cublasLtMatmulDesc_t operation() const { return operation_; }
    cublasLtMatrixLayout_t input() const { return input_; }
    cublasLtMatrixLayout_t weight() const { return weight_; }
    cublasLtMatrixLayout_t output() const { return output_; }

private:
    int64_t output_width_;
    int64_t input_width_;
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t input_ = nullptr;
    cublasLtMatrixLayout_t weight_ = nullptr;
    cublasLtMatrixLayout_t output_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    mutable std::once_flag heuristic_once_;
    mutable std::vector<cublasLtMatmulHeuristicResult_t> cached_heuristics_;
    mutable std::mutex configured_mutex_;
    using AlgorithmKey = std::tuple<
        int32_t, uint32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t>;

    struct AlgorithmKeyHash {
        std::size_t operator()(const AlgorithmKey& key) const noexcept {
            return std::apply([](const auto... values) {
                std::size_t result = 0;
                ((result ^= std::hash<uint64_t>{}(
                    static_cast<uint64_t>(values)) + 0x9e3779b9U +
                    (result << 6U) + (result >> 2U)), ...);
                return result;
            }, key);
        }
    };

    mutable std::unordered_map<
        AlgorithmKey, cublasLtMatmulAlgo_t, AlgorithmKeyHash>
        configured_algorithms_;
};

cublasLtHandle_t cublaslt_handle() {
    static cublasLtHandle_t handle = [] {
        cublasLtHandle_t result = nullptr;
        check_cublas(cublasLtCreate(&result), "cublasLtCreate");
        return result;
    }();
    return handle;
}

std::vector<cublasLtMatmulHeuristicResult_t> MatmulPlan::heuristics(
    int requested) const {
    std::call_once(heuristic_once_, [&] {
        constexpr int maximum = 64;
        cached_heuristics_.resize(maximum);
        int returned = 0;
        check_cublas(
            cublasLtMatmulAlgoGetHeuristic(
                cublaslt_handle(), operation_, input_, weight_, output_, output_,
                preference_, maximum, cached_heuristics_.data(), &returned),
            "cublasLtMatmulAlgoGetHeuristic");
        cached_heuristics_.resize(static_cast<std::size_t>(returned));
    });
    const std::size_t count = std::min(
        cached_heuristics_.size(), static_cast<std::size_t>(requested));
    return {cached_heuristics_.begin(), cached_heuristics_.begin() + count};
}

cublasLtMatmulAlgo_t MatmulPlan::configured_algorithm(
    int32_t algorithm_id,
    uint32_t tile_id,
    int32_t split_k,
    uint32_t reduction_scheme,
    uint32_t cta_swizzle,
    uint32_t custom_option,
    uint32_t stages_id) const {
    const AlgorithmKey key{
        algorithm_id, tile_id, split_k, reduction_scheme, cta_swizzle,
        custom_option, stages_id};
    std::lock_guard<std::mutex> lock(configured_mutex_);
    const auto existing = configured_algorithms_.find(key);
    if (existing != configured_algorithms_.end()) {
        return existing->second;
    }
    cublasLtMatmulAlgo_t algorithm{};
    check_cublas(
        cublasLtMatmulAlgoInit(
            cublaslt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F,
            CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
            algorithm_id, &algorithm),
        "cublasLtMatmulAlgoInit");
    const auto set_attribute = [&](auto attribute, const auto& value) {
        check_cublas(
            cublasLtMatmulAlgoConfigSetAttribute(
                &algorithm, attribute, &value, sizeof(value)),
            "setting explicit algorithm configuration");
    };
    set_attribute(CUBLASLT_ALGO_CONFIG_TILE_ID, tile_id);
    set_attribute(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split_k);
    set_attribute(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction_scheme);
    set_attribute(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, cta_swizzle);
    set_attribute(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom_option);
    set_attribute(CUBLASLT_ALGO_CONFIG_STAGES_ID, stages_id);
    cublasLtMatmulHeuristicResult_t checked{};
    check_cublas(
        cublasLtMatmulAlgoCheck(
            cublaslt_handle(), operation_, input_, weight_, output_, output_,
            &algorithm, &checked),
        "checking explicit algorithm configuration");
    TORCH_CHECK(checked.state == CUBLAS_STATUS_SUCCESS,
        "flux::cublaslt_linear: explicit algorithm configuration is not runnable");
    configured_algorithms_.emplace(key, algorithm);
    return algorithm;
}

using PlanKey = std::tuple<int, int64_t, int64_t, int64_t>;

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const noexcept {
        const auto [device, output_width, input_width, workspace] = key;
        std::size_t result = std::hash<int>{}(device);
        for (int64_t value : {output_width, input_width, workspace}) {
            result ^= std::hash<int64_t>{}(value) + 0x9e3779b9U +
                (result << 6U) + (result >> 2U);
        }
        return result;
    }
};

std::shared_ptr<MatmulPlan> plan_for(
    int device, int64_t output_width, int64_t input_width,
    int64_t max_workspace_bytes) {
    static std::mutex mutex;
    static std::unordered_map<PlanKey, std::shared_ptr<MatmulPlan>, PlanKeyHash> plans;
    const PlanKey key{device, output_width, input_width, max_workspace_bytes};
    std::lock_guard<std::mutex> lock(mutex);
    auto [iterator, inserted] = plans.try_emplace(key);
    if (inserted) {
        iterator->second = std::make_shared<MatmulPlan>(
            output_width, input_width, max_workspace_bytes);
    }
    return iterator->second;
}

void validate_common(
    const at::Tensor& input,
    const at::Tensor& weight,
    int64_t max_workspace_bytes) {
    TORCH_CHECK(input.defined() && weight.defined(),
        "flux::cublaslt_linear: input and weight must be defined");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
        "flux::cublaslt_linear: input and weight must be CUDA tensors");
    TORCH_CHECK(input.device() == weight.device(),
        "flux::cublaslt_linear: tensor devices must match");
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
        "flux::cublaslt_linear: input and weight must be float32");
    TORCH_CHECK(input.dim() >= 1 && weight.dim() == 2,
        "flux::cublaslt_linear: input must have at least one dimension and weight must be rank two");
    TORCH_CHECK(input.size(-1) > 0 && weight.size(0) > 0 &&
            input.size(-1) == weight.size(1),
        "flux::cublaslt_linear: weight shape must be [N, input.size(-1)]");
    TORCH_CHECK(input.numel() == input.size(-1),
        "flux::cublaslt_linear: only M=1 inputs are supported");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
        "flux::cublaslt_linear: input and weight must be contiguous; no implicit copy is performed");
    TORCH_CHECK(max_workspace_bytes >= 0,
        "flux::cublaslt_linear: max_workspace_bytes must be non-negative");
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            (!input.requires_grad() && !weight.requires_grad()),
        "flux::cublaslt_linear is inference-only; use torch.no_grad() or torch.inference_mode()");
}

template <typename T>
int64_t algo_attribute(
    const cublasLtMatmulAlgo_t& algorithm,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
    T value{};
    std::size_t written = 0;
    check_cublas(
        cublasLtMatmulAlgoConfigGetAttribute(
            &algorithm, attribute, &value, sizeof(value), &written),
        "reading algorithm configuration");
    TORCH_CHECK(written == sizeof(value),
        "flux::cublaslt_linear: unexpected algorithm attribute size");
    return static_cast<int64_t>(value);
}

at::Tensor cublaslt_algorithm_info_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    int64_t max_workspace_bytes,
    int64_t max_algorithms) {
    validate_common(input, weight, max_workspace_bytes);
    TORCH_CHECK(max_algorithms >= 1 && max_algorithms <= 64,
        "flux::cublaslt_algorithm_info: max_algorithms must be in [1, 64]");
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto plan = plan_for(
        input.get_device(), weight.size(0), weight.size(1), max_workspace_bytes);
    auto results = plan->heuristics(static_cast<int>(max_algorithms));
    constexpr int64_t columns = 10;
    at::Tensor information = at::empty(
        {static_cast<int64_t>(results.size()), columns},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    auto accessor = information.accessor<int64_t, 2>();
    for (int64_t index = 0; index < static_cast<int64_t>(results.size()); ++index) {
        const auto& result = results[static_cast<std::size_t>(index)];
        accessor[index][0] = index;
        accessor[index][1] = algo_attribute<int32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_ID);
        accessor[index][2] = algo_attribute<uint32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
        accessor[index][3] = algo_attribute<int32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        accessor[index][4] = algo_attribute<uint32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
        accessor[index][5] = algo_attribute<uint32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
        accessor[index][6] = algo_attribute<uint32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
        accessor[index][7] = algo_attribute<uint32_t>(
            result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
        accessor[index][8] = static_cast<int64_t>(result.workspaceSize);
        accessor[index][9] = static_cast<int64_t>(result.wavesCount * 1000000.0F);
    }
    return information;
}

at::Tensor cublaslt_linear_out_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor output,
    const at::Tensor& workspace,
    int64_t algorithm_index,
    int64_t max_workspace_bytes) {
    validate_common(input, weight, max_workspace_bytes);
    TORCH_CHECK(output.defined() && output.is_cuda() && output.device() == input.device() &&
            output.scalar_type() == at::kFloat && output.is_contiguous(),
        "flux::cublaslt_linear_out: output must be contiguous float32 CUDA on the input device");
    TORCH_CHECK(output.dim() == input.dim() && output.numel() == weight.size(0),
        "flux::cublaslt_linear_out: output must preserve the input rank with final width weight.size(0)");
    for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
        TORCH_CHECK(output.size(dimension) == input.size(dimension),
            "flux::cublaslt_linear_out: output leading dimensions must match input");
    }
    TORCH_CHECK(output.size(-1) == weight.size(0),
        "flux::cublaslt_linear_out: output final dimension is incorrect");
    TORCH_CHECK(workspace.defined() && workspace.is_cuda() &&
            workspace.device() == input.device() && workspace.scalar_type() == at::kByte &&
            workspace.is_contiguous(),
        "flux::cublaslt_linear_out: workspace must be contiguous uint8 CUDA on the input device");
    TORCH_CHECK(workspace.numel() <= max_workspace_bytes,
        "flux::cublaslt_linear_out: workspace exceeds max_workspace_bytes");
    at::assert_no_internal_overlap(output);
    at::assert_no_internal_overlap(workspace);
    TORCH_CHECK(!output.is_alias_of(input) && !output.is_alias_of(weight) &&
            !output.is_alias_of(workspace) && !workspace.is_alias_of(input) &&
            !workspace.is_alias_of(weight),
        "flux::cublaslt_linear_out: output, inputs, and workspace must not alias");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto plan = plan_for(
        input.get_device(), weight.size(0), weight.size(1), max_workspace_bytes);
    auto results = plan->heuristics(64);
    TORCH_CHECK(algorithm_index >= 0 &&
            algorithm_index < static_cast<int64_t>(results.size()),
        "flux::cublaslt_linear_out: algorithm_index is outside the heuristic result set");
    const auto& selected = results[static_cast<std::size_t>(algorithm_index)];
    TORCH_CHECK(selected.state == CUBLAS_STATUS_SUCCESS,
        "flux::cublaslt_linear_out: selected heuristic is not runnable");
    TORCH_CHECK(selected.workspaceSize <= static_cast<std::size_t>(workspace.numel()),
        "flux::cublaslt_linear_out: workspace is smaller than the selected algorithm requires");

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    void* workspace_pointer = workspace.numel() == 0
        ? nullptr
        : workspace.mutable_data_ptr<uint8_t>();
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(input.get_device());
    check_cublas(
        cublasLtMatmul(
            cublaslt_handle(), plan->operation(), &alpha,
            input.const_data_ptr<float>(), plan->input(),
            weight.const_data_ptr<float>(), plan->weight(), &beta,
            output.const_data_ptr<float>(), plan->output(),
            output.mutable_data_ptr<float>(), plan->output(),
            &selected.algo, workspace_pointer, selected.workspaceSize,
            stream.stream()),
        "cublasLtMatmul");
    return output;
}

at::Tensor cublaslt_linear_config_out_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor output,
    const at::Tensor& workspace,
    int64_t algorithm_id,
    int64_t tile_id,
    int64_t split_k,
    int64_t reduction_scheme,
    int64_t cta_swizzle,
    int64_t custom_option,
    int64_t stages_id) {
    validate_common(input, weight, workspace.numel());
    TORCH_CHECK(output.defined() && output.is_cuda() && output.device() == input.device() &&
            output.scalar_type() == at::kFloat && output.is_contiguous(),
        "flux::cublaslt_linear_config_out: output must be contiguous float32 CUDA on the input device");
    TORCH_CHECK(output.dim() == input.dim() && output.numel() == weight.size(0) &&
            output.size(-1) == weight.size(0),
        "flux::cublaslt_linear_config_out: output shape is incorrect");
    for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
        TORCH_CHECK(output.size(dimension) == input.size(dimension),
            "flux::cublaslt_linear_config_out: output leading dimensions must match input");
    }
    TORCH_CHECK(workspace.defined() && workspace.is_cuda() &&
            workspace.device() == input.device() && workspace.scalar_type() == at::kByte &&
            workspace.is_contiguous(),
        "flux::cublaslt_linear_config_out: workspace must be contiguous uint8 CUDA on the input device");
    TORCH_CHECK(
        algorithm_id >= std::numeric_limits<int32_t>::min() &&
            algorithm_id <= std::numeric_limits<int32_t>::max() &&
            split_k >= std::numeric_limits<int32_t>::min() &&
            split_k <= std::numeric_limits<int32_t>::max() &&
            tile_id >= 0 && tile_id <= std::numeric_limits<uint32_t>::max() &&
            reduction_scheme >= 0 && reduction_scheme <= std::numeric_limits<uint32_t>::max() &&
            cta_swizzle >= 0 && cta_swizzle <= std::numeric_limits<uint32_t>::max() &&
            custom_option >= 0 && custom_option <= std::numeric_limits<uint32_t>::max() &&
            stages_id >= 0 && stages_id <= std::numeric_limits<uint32_t>::max(),
        "flux::cublaslt_linear_config_out: algorithm configuration value is out of range");
    at::assert_no_internal_overlap(output);
    at::assert_no_internal_overlap(workspace);
    TORCH_CHECK(!output.is_alias_of(input) && !output.is_alias_of(weight) &&
            !output.is_alias_of(workspace) && !workspace.is_alias_of(input) &&
            !workspace.is_alias_of(weight),
        "flux::cublaslt_linear_config_out: output, inputs, and workspace must not alias");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto plan = plan_for(
        input.get_device(), weight.size(0), weight.size(1), workspace.numel());
    const cublasLtMatmulAlgo_t algorithm = plan->configured_algorithm(
        static_cast<int32_t>(algorithm_id), static_cast<uint32_t>(tile_id),
        static_cast<int32_t>(split_k), static_cast<uint32_t>(reduction_scheme),
        static_cast<uint32_t>(cta_swizzle), static_cast<uint32_t>(custom_option),
        static_cast<uint32_t>(stages_id));
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    void* workspace_pointer = workspace.numel() == 0
        ? nullptr
        : workspace.mutable_data_ptr<uint8_t>();
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(input.get_device());
    check_cublas(
        cublasLtMatmul(
            cublaslt_handle(), plan->operation(), &alpha,
            input.const_data_ptr<float>(), plan->input(),
            weight.const_data_ptr<float>(), plan->weight(), &beta,
            output.const_data_ptr<float>(), plan->output(),
            output.mutable_data_ptr<float>(), plan->output(),
            &algorithm, workspace_pointer, static_cast<std::size_t>(workspace.numel()),
            stream.stream()),
        "cublasLtMatmul with explicit configuration");
    return output;
}

}  // namespace
}  // namespace flux

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.def(
        "cublaslt_algorithm_info(Tensor input, Tensor weight, int max_workspace_bytes, int max_algorithms=16) -> Tensor");
    library.def(
        "cublaslt_linear_out(Tensor input, Tensor weight, Tensor(a!) output, Tensor(b!) workspace, int algorithm_index, int max_workspace_bytes) -> Tensor(a!)");
    library.def(
        "cublaslt_linear_config_out(Tensor input, Tensor weight, Tensor(a!) output, Tensor(b!) workspace, int algorithm_id, int tile_id, int split_k, int reduction_scheme, int cta_swizzle, int custom_option, int stages_id) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flux, CUDA, library) {
    library.impl(
        "cublaslt_algorithm_info", TORCH_FN(flux::cublaslt_algorithm_info_cuda));
    library.impl(
        "cublaslt_linear_out", TORCH_FN(flux::cublaslt_linear_out_cuda));
    library.impl(
        "cublaslt_linear_config_out", TORCH_FN(flux::cublaslt_linear_config_out_cuda));
}
