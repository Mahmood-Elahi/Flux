"""Correctness, metadata, stream, and graph tests for fused post-QKV decode."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flux.ops import (
    native_packed_qkv_rope_cache_is_available,
    packed_qkv_rope_cache_native,
    rope_native,
)


pytestmark = pytest.mark.skipif(
    not native_packed_qkv_rope_cache_is_available(),
    reason="Flux native packed-QKV RoPE/cache operator has not been built",
)


def _inputs(
    position: int,
    capacity: int,
    *,
    strided_packed: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cuda").manual_seed(1234 + position)
    if strided_packed:
        storage = torch.randn(
            (1, 1, 1920), generator=generator, device="cuda"
        )
        packed = storage[..., ::2]
        assert packed.shape == (1, 1, 960) and packed.stride(-1) == 2
    else:
        packed = torch.randn(
            (1, 1, 960), generator=generator, device="cuda"
        )
    frequencies = torch.randn(
        (1, 1, 64), generator=generator, device="cuda"
    )
    cos = frequencies.cos()
    sin = frequencies.sin()
    key_cache = torch.randn(
        (1, 3, capacity, 64), generator=generator, device="cuda"
    )
    value_cache = torch.randn(
        (1, 3, capacity, 64), generator=generator, device="cuda"
    )
    length = torch.tensor(position, dtype=torch.int64, device="cuda")
    return packed, cos, sin, key_cache, value_cache, length


def _reference(
    packed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = packed[..., :576].view(1, 1, 9, 64).transpose(1, 2)
    key = packed[..., 576:768].view(1, 1, 3, 64).transpose(1, 2)
    value = packed[..., 768:].view(1, 1, 3, 64).transpose(1, 2)
    query_output, key_output = rope_native(query, key, cos, sin)
    expected_keys = key_cache.clone()
    expected_values = value_cache.clone()
    expected_keys[..., position : position + 1, :].copy_(key_output)
    expected_values[..., position : position + 1, :].copy_(value)
    return query_output, expected_keys, expected_values


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("position", "capacity"),
    [
        (128, 130),
        (512, 514),
        (1024, 1026),
        (2048, 2050),
        (4096, 4098),
        (8190, 8192),
    ],
)
def test_matches_retained_rope_and_static_cache_update(
    position: int, capacity: int
) -> None:
    packed, cos, sin, keys, values, length = _inputs(position, capacity)
    expected_q, expected_keys, expected_values = _reference(
        packed, cos, sin, keys, values, position
    )

    with torch.inference_mode():
        actual_q = packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length
        )

    assert actual_q.shape == (1, 9, 1, 64)
    assert actual_q.is_contiguous()
    assert int(length.item()) == position + 1
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_reads_positive_stride_packed_projection_layout() -> None:
    inputs = _inputs(127, 129, strided_packed=True)
    packed, cos, sin, keys, values, length = inputs
    expected_q, expected_keys, expected_values = _reference(
        packed, cos, sin, keys, values, 127
    )

    with torch.inference_mode():
        actual_q = packed_qkv_rope_cache_native(*inputs)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    assert int(length.item()) == 128


def test_fake_tensor_metadata() -> None:
    mode = FakeTensorMode()
    with mode:
        packed = torch.empty((1, 1, 960), device="cuda")
        cos = torch.empty((1, 1, 64), device="cuda")
        sin = torch.empty_like(cos)
        keys = torch.empty((1, 3, 4097, 64), device="cuda")
        values = torch.empty_like(keys)
        length = torch.empty((), dtype=torch.int64, device="cuda")
        output = packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length
        )
    assert isinstance(output, FakeTensor)
    assert output.shape == (1, 9, 1, 64)
    assert output.dtype == torch.float32
    assert output.device == packed.device
    assert output.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_library_opcheck() -> None:
    inputs = _inputs(11, 17)
    with torch.inference_mode():
        result = torch.library.opcheck(
            torch.ops.flux.packed_qkv_rope_cache.default,
            inputs,
            rtol=0,
            atol=0,
        )
    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uses_current_non_default_stream_and_is_deterministic() -> None:
    packed, cos, sin, keys, values, length = _inputs(256, 258)
    expected_q, expected_keys, expected_values = _reference(
        packed, cos, sin, keys, values, 256
    )
    original_keys = keys.clone()
    original_values = values.clone()
    stream = torch.cuda.Stream()
    assert stream != torch.cuda.default_stream()

    with torch.cuda.stream(stream), torch.inference_mode():
        torch.cuda._sleep(10_000_000)
        first = packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length
        )
        consumed = first + 0.0
    stream.synchronize()
    torch.testing.assert_close(consumed, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)

    keys.copy_(original_keys)
    values.copy_(original_values)
    length.fill_(256)
    with torch.inference_mode():
        second = packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length
        )
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_capture_and_replay_advance_device_length() -> None:
    packed, cos, sin, keys, values, length = _inputs(128, 4098)
    before_keys = keys.clone()
    before_values = values.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        output = packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length
        )
    address = output.data_ptr()

    for position in (128, 512, 1024, 2048, 4096):
        keys.copy_(before_keys)
        values.copy_(before_values)
        length.fill_(position)
        expected_q, expected_keys, expected_values = _reference(
            packed, cos, sin, before_keys, before_values, position
        )
        graph.replay()
        torch.cuda.synchronize()
        assert output.data_ptr() == address
        assert int(length.item()) == position + 1
        torch.testing.assert_close(output, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rejects_invalid_shapes_dtypes_and_autograd() -> None:
    packed, cos, sin, keys, values, length = _inputs(11, 17)
    with pytest.raises(RuntimeError, match=r"\[1, 1, 960\]"):
        packed_qkv_rope_cache_native(
            packed[..., :-1], cos, sin, keys, values, length
        )
    with pytest.raises(RuntimeError, match="float32"):
        packed_qkv_rope_cache_native(
            packed.double(), cos, sin, keys, values, length
        )
    with pytest.raises(RuntimeError, match="int64"):
        packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length.int()
        )
    packed.requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        packed_qkv_rope_cache_native(
            packed, cos, sin, keys, values, length
        )
