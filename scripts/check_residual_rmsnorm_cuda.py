"""Compare standalone CUDA residual RMSNorm with the Flux PyTorch oracle."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from flux.ops.residual_rmsnorm import residual_rmsnorm


CASES = [
    ((1,), 0.0),
    ((3, 7), 1e-6),
    ((3, 63), 1e-5),
    ((3, 127), 1e-4),
    ((3, 575), 1e-5),
    ((576,), 1e-5),
    ((4, 576), 1e-5),
    ((2, 32, 576), 1e-5),
    ((3, 577), 1e-5),
    ((3, 1024), 1e-5),
    ((8192, 576), 1e-5),
]
RTOL = 1e-5
ATOL = 2e-6


def _write_fp32(path: Path, *tensors: torch.Tensor) -> None:
    with path.open("wb") as stream:
        for tensor in tensors:
            values = array("f", tensor.contiguous().view(-1).tolist())
            values.tofile(stream)


def _read_outputs(
    path: Path, count: int, shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, 2 * count)
        if stream.read(1):
            raise RuntimeError("CUDA output contains trailing data")
    if len(values) != 2 * count:
        raise RuntimeError(
            f"CUDA output has {len(values)} values; expected {2 * count}"
        )
    outputs = torch.tensor(values, dtype=torch.float32)
    return outputs[:count].reshape(shape), outputs[count:].reshape(shape)


def _maximum_errors(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(torch.finfo(torch.float32).eps)
    return absolute.max().item(), relative.max().item()


def _make_inputs(
    shape: tuple[int, ...], seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(shape, dtype=torch.float32, generator=generator)
    residual = torch.randn(shape, dtype=torch.float32, generator=generator)
    weight = torch.linspace(0.25, 1.75, shape[-1], dtype=torch.float32)
    return hidden, residual, weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "executable", type=Path, help="path to residual_rmsnorm_cuda_cli.exe"
    )
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")

    overall_residual_absolute = 0.0
    overall_residual_relative = 0.0
    overall_norm_absolute = 0.0
    overall_norm_relative = 0.0
    with tempfile.TemporaryDirectory(
        prefix="flux-residual-rmsnorm-cuda-"
    ) as temp_directory:
        temp_path = Path(temp_directory)
        for case_index, (shape, epsilon) in enumerate(CASES):
            hidden, residual, weight = _make_inputs(shape, case_index)
            expected_norm, expected_residual = residual_rmsnorm(
                hidden, residual, weight, epsilon
            )
            input_path = temp_path / f"input-{case_index}.bin"
            output_path = temp_path / f"output-{case_index}.bin"
            _write_fp32(input_path, hidden, residual, weight)
            subprocess.run(
                [
                    str(executable),
                    str(input_path),
                    str(output_path),
                    str(hidden.numel() // shape[-1]),
                    str(shape[-1]),
                    str(epsilon),
                ],
                check=True,
            )
            actual_norm, actual_residual = _read_outputs(
                output_path, hidden.numel(), shape
            )

            torch.testing.assert_close(
                actual_residual, expected_residual, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                actual_norm, expected_norm, rtol=RTOL, atol=ATOL
            )
            residual_errors = _maximum_errors(actual_residual, expected_residual)
            norm_errors = _maximum_errors(actual_norm, expected_norm)
            overall_residual_absolute = max(
                overall_residual_absolute, residual_errors[0]
            )
            overall_residual_relative = max(
                overall_residual_relative, residual_errors[1]
            )
            overall_norm_absolute = max(overall_norm_absolute, norm_errors[0])
            overall_norm_relative = max(overall_norm_relative, norm_errors[1])
            print(
                f"shape={shape}, epsilon={epsilon:g}: "
                f"residual max abs/rel={residual_errors[0]:.9g}/"
                f"{residual_errors[1]:.9g}, norm max abs/rel="
                f"{norm_errors[0]:.9g}/{norm_errors[1]:.9g}"
            )

    print(
        "overall residual maximum absolute/relative error: "
        f"{overall_residual_absolute:.9g}/{overall_residual_relative:.9g}"
    )
    print(
        "overall norm maximum absolute/relative error: "
        f"{overall_norm_absolute:.9g}/{overall_norm_relative:.9g}"
    )
    print("CUDA residual RMSNorm matches the PyTorch oracle within FP32 tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
