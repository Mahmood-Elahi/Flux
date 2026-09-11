"""Compare standalone C++ softmax with the Flux PyTorch oracle."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from flux.ops.softmax import softmax


RANDOM_SHAPES = [
    (1,),
    (2, 3),
    (2, 3, 7),
    (3, 31),
    (3, 32),
    (3, 33),
    (3, 63),
    (3, 64),
    (3, 65),
    (3, 127),
    (3, 128),
    (3, 129),
    (2, 9, 257),
    (4, 1024),
    (2, 2048),
    (2, 8192),
]
RTOL = 1e-5
ATOL = 1e-6


def _write_fp32(path: Path, tensor: torch.Tensor) -> None:
    with path.open("wb") as stream:
        values = array("f", tensor.contiguous().view(-1).tolist())
        values.tofile(stream)


def _read_fp32(path: Path, count: int, shape: tuple[int, ...]) -> torch.Tensor:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, count)
        if stream.read(1):
            raise RuntimeError("C++ output contains trailing data")
    if len(values) != count:
        raise RuntimeError(f"C++ output has {len(values)} values; expected {count}")
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _maximum_errors(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(torch.finfo(torch.float32).eps)
    return absolute.max().item(), relative.max().item()


def _make_cases() -> list[tuple[str, torch.Tensor]]:
    cases: list[tuple[str, torch.Tensor]] = []
    for case_index, shape in enumerate(RANDOM_SHAPES):
        generator = torch.Generator().manual_seed(1000 + case_index)
        values = 3.0 * torch.randn(shape, dtype=torch.float32, generator=generator)
        cases.append((f"random-{shape}", values))

    cases.extend(
        [
            (
                "large-magnitude",
                torch.tensor(
                    [
                        [1000.0, 1001.0, 999.0],
                        [-1000.0, -1001.0, -999.0],
                        [1000.0, 0.0, -1000.0],
                    ],
                    dtype=torch.float32,
                ),
            ),
            ("constant-width-65", torch.full((3, 65), 42.0, dtype=torch.float32)),
        ]
    )

    generator = torch.Generator().manual_seed(2000)
    shifted = torch.randn((4, 129), dtype=torch.float32, generator=generator)
    cases.append(("shifted-random", shifted + 250.0))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="path to softmax_cli.exe")
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")

    overall_absolute = 0.0
    overall_relative = 0.0
    overall_row_sum_error = 0.0
    with tempfile.TemporaryDirectory(prefix="flux-softmax-") as temp_directory:
        temp_path = Path(temp_directory)
        for case_index, (case_name, input_tensor) in enumerate(_make_cases()):
            expected = softmax(input_tensor)
            torch.testing.assert_close(
                expected,
                torch.softmax(input_tensor, dim=-1),
                rtol=RTOL,
                atol=ATOL,
            )
            input_path = temp_path / f"input-{case_index}.bin"
            output_path = temp_path / f"output-{case_index}.bin"
            _write_fp32(input_path, input_tensor)
            subprocess.run(
                [
                    str(executable),
                    str(input_path),
                    str(output_path),
                    str(input_tensor.numel() // input_tensor.shape[-1]),
                    str(input_tensor.shape[-1]),
                ],
                check=True,
            )
            actual = _read_fp32(output_path, input_tensor.numel(), input_tensor.shape)

            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            if not torch.isfinite(actual).all():
                raise AssertionError(f"{case_name}: C++ output is not finite")
            if not torch.all(actual >= 0):
                raise AssertionError(f"{case_name}: C++ output is negative")
            row_sum_error = (actual.sum(dim=-1) - 1.0).abs().max().item()
            if row_sum_error > ATOL + RTOL:
                raise AssertionError(
                    f"{case_name}: maximum row-sum error {row_sum_error:.9g} "
                    f"exceeds {ATOL + RTOL:.9g}"
                )

            absolute_error, relative_error = _maximum_errors(actual, expected)
            overall_absolute = max(overall_absolute, absolute_error)
            overall_relative = max(overall_relative, relative_error)
            overall_row_sum_error = max(overall_row_sum_error, row_sum_error)
            print(
                f"{case_name}: shape={tuple(input_tensor.shape)}, "
                f"max abs/rel={absolute_error:.9g}/{relative_error:.9g}, "
                f"max row-sum error={row_sum_error:.9g}"
            )

    print(
        "overall maximum absolute/relative error: "
        f"{overall_absolute:.9g}/{overall_relative:.9g}"
    )
    print(f"overall maximum row-sum error: {overall_row_sum_error:.9g}")
    print("C++ softmax matches the Flux PyTorch oracle within FP32 tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
