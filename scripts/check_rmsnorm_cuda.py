"""Compare the standalone CUDA RMSNorm with the Flux PyTorch oracle."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from flux.ops.rmsnorm import rms_norm


SHAPE = (2, 7, 576)
EPSILON = 1e-5
RTOL = 1e-5
ATOL = 2e-6


def _write_fp32(path: Path, *tensors: torch.Tensor) -> None:
    with path.open("wb") as stream:
        for tensor in tensors:
            values = array("f", tensor.contiguous().view(-1).tolist())
            values.tofile(stream)


def _read_fp32(path: Path, count: int) -> torch.Tensor:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, count)
        if stream.read(1):
            raise RuntimeError("CUDA output contains trailing data")
    if len(values) != count:
        raise RuntimeError(f"CUDA output has {len(values)} values; expected {count}")
    return torch.tensor(values, dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="path to rmsnorm_cuda_cli.exe")
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")

    element_count = 1
    for dimension in SHAPE:
        element_count *= dimension
    positions = torch.arange(element_count, dtype=torch.float32).reshape(SHAPE)
    input_tensor = torch.sin(positions * 0.013) + torch.cos(positions * 0.007)
    weight = torch.linspace(0.25, 1.75, SHAPE[-1], dtype=torch.float32)
    expected = rms_norm(input_tensor, weight, EPSILON)

    with tempfile.TemporaryDirectory(prefix="flux-rmsnorm-cuda-") as temp_directory:
        input_path = Path(temp_directory) / "input.bin"
        output_path = Path(temp_directory) / "output.bin"
        _write_fp32(input_path, input_tensor, weight)
        subprocess.run(
            [
                str(executable),
                str(input_path),
                str(output_path),
                str(SHAPE[0] * SHAPE[1]),
                str(SHAPE[2]),
                str(EPSILON),
            ],
            check=True,
        )
        actual = _read_fp32(output_path, element_count).reshape(SHAPE)

    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / expected.abs().clamp_min(
        torch.finfo(torch.float32).eps
    )
    max_absolute_error = absolute_error.max().item()
    max_relative_error = relative_error.max().item()

    print(f"shape: {tuple(actual.shape)}")
    print(f"epsilon: {EPSILON}")
    print(f"maximum absolute error: {max_absolute_error:.9g}")
    print(f"maximum relative error: {max_relative_error:.9g}")
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    print("CUDA RMSNorm matches the PyTorch oracle within FP32 tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
