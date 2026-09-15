from __future__ import annotations

import pytest
import torch

from flux.ops import (
    cublaslt_algorithms,
    cublaslt_linear_config_out,
    cublaslt_linear_out,
    native_cublaslt_linear_is_available,
)


_AVAILABLE = torch.cuda.is_available() and native_cublaslt_linear_is_available()
_SHAPES = ((960, 576), (49152, 576))
_WORKSPACE_BYTES = 4 * 1024 * 1024


@pytest.mark.skipif(not _AVAILABLE, reason="CUDA cuBLASLt extension is required")
@pytest.mark.parametrize(("output_width", "input_width"), _SHAPES)
def test_cublaslt_linear_matches_pytorch_and_repeats(
    output_width: int, input_width: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(output_width + input_width)
    input = torch.randn((1, 1, input_width), generator=generator, device="cuda")
    weight = torch.randn(
        (output_width, input_width), generator=generator, device="cuda"
    )
    output = torch.empty((1, 1, output_width), device="cuda")
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    algorithms = cublaslt_algorithms(
        input, weight, max_workspace_bytes=_WORKSPACE_BYTES, max_algorithms=16
    )
    assert algorithms
    assert all(item.workspace_bytes <= _WORKSPACE_BYTES for item in algorithms)

    with torch.inference_mode():
        expected = torch.nn.functional.linear(input, weight)
        returned = cublaslt_linear_out(
            input,
            weight,
            output,
            workspace,
            algorithm_index=algorithms[0].index,
            max_workspace_bytes=_WORKSPACE_BYTES,
        )
        first = output.clone()
        cublaslt_linear_out(
            input,
            weight,
            output,
            workspace,
            algorithm_index=algorithms[0].index,
            max_workspace_bytes=_WORKSPACE_BYTES,
        )

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output, first, rtol=0, atol=0)


@pytest.mark.skipif(not _AVAILABLE, reason="CUDA cuBLASLt extension is required")
@pytest.mark.parametrize(("output_width", "input_width"), ((960, 576), (576, 576)))
def test_retained_explicit_configuration_matches_pytorch(
    output_width: int, input_width: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1200 + output_width)
    input = torch.randn((1, 1, input_width), generator=generator, device="cuda")
    weight = torch.randn(
        (output_width, input_width), generator=generator, device="cuda"
    )
    output = torch.empty((1, 1, output_width), device="cuda")
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    algorithm = cublaslt_algorithms(
        input, weight, max_workspace_bytes=_WORKSPACE_BYTES, max_algorithms=16
    )[5]
    assert (
        algorithm.algorithm_id,
        algorithm.tile_id,
        algorithm.split_k,
        algorithm.reduction_scheme,
        algorithm.cta_swizzle,
        algorithm.custom_option,
        algorithm.stages_id,
        algorithm.workspace_bytes,
    ) == (13, 0, 1, 0, 0, 91, 0, 0)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(input, weight)
        cublaslt_linear_config_out(input, weight, output, workspace, algorithm)
        first = output.clone()
        cublaslt_linear_config_out(input, weight, output, workspace, algorithm)
    # Different valid FP32 reduction orders need not be bit-identical. For
    # unit-normal inputs the observed worst-case accumulation delta stays
    # below 2e-5; real model weights are checked end to end separately.
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(output, first, rtol=0, atol=0)


@pytest.mark.skipif(not _AVAILABLE, reason="CUDA cuBLASLt extension is required")
def test_cublaslt_linear_fake_tensor_contract() -> None:
    input = torch.randn((1, 1, 576), device="cuda")
    weight = torch.randn((960, 576), device="cuda")
    output = torch.empty((1, 1, 960), device="cuda")
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    torch.library.opcheck(
        torch.ops.flux.cublaslt_linear_out.default,
        (input, weight, output, workspace, 0, _WORKSPACE_BYTES),
        test_utils=("test_schema", "test_faketensor"),
    )
    algorithm = cublaslt_algorithms(
        input, weight, max_workspace_bytes=_WORKSPACE_BYTES, max_algorithms=16
    )[5]
    exact_workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    torch.library.opcheck(
        torch.ops.flux.cublaslt_linear_config_out.default,
        (
            input,
            weight,
            output,
            exact_workspace,
            algorithm.algorithm_id,
            algorithm.tile_id,
            algorithm.split_k,
            algorithm.reduction_scheme,
            algorithm.cta_swizzle,
            algorithm.custom_option,
            algorithm.stages_id,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )
