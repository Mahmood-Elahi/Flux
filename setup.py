"""Optional development build for Flux native PyTorch operators.

Normal package builds remain Python-only. Set ``FLUX_BUILD_NATIVE=1`` and use
``build_ext --inplace`` to compile against the PyTorch already installed in the
active development environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent.resolve()


def _native_extension_config() -> tuple[list[object], dict[str, object]]:
    if os.environ.get("FLUX_BUILD_NATIVE") != "1":
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as error:
        raise RuntimeError(
            "Building Flux native operators requires PyTorch in the active "
            "environment. Install Flux dependencies first and do not use PEP "
            "517 build isolation for this explicit development build."
        ) from error

    rmsnorm_dir = ROOT / "csrc" / "rmsnorm"
    residual_rmsnorm_dir = ROOT / "csrc" / "residual_rmsnorm"
    softmax_dir = ROOT / "csrc" / "softmax"
    attention_score_softmax_dir = ROOT / "csrc" / "attention_score_softmax"
    rope_dir = ROOT / "csrc" / "rope"
    packed_swiglu_dir = ROOT / "csrc" / "packed_swiglu"
    gqa_decode_attention_dir = ROOT / "csrc" / "gqa_decode_attention"
    extension = CUDAExtension(
        name="flux._C",
        sources=[
            str(rmsnorm_dir / "rmsnorm_torch.cpp"),
            str(rmsnorm_dir / "rmsnorm.cpp"),
            str(rmsnorm_dir / "rmsnorm_cuda.cu"),
            str(residual_rmsnorm_dir / "residual_rmsnorm_torch.cpp"),
            str(residual_rmsnorm_dir / "residual_rmsnorm.cpp"),
            str(residual_rmsnorm_dir / "residual_rmsnorm_cuda.cu"),
            str(softmax_dir / "softmax_torch.cpp"),
            str(softmax_dir / "softmax.cpp"),
            str(softmax_dir / "softmax_cuda.cu"),
            str(attention_score_softmax_dir / "attention_score_softmax_torch.cpp"),
            str(attention_score_softmax_dir / "attention_score_softmax.cpp"),
            str(rope_dir / "rope_torch.cpp"),
            str(rope_dir / "rope.cpp"),
            str(rope_dir / "rope_cuda.cu"),
            str(packed_swiglu_dir / "packed_swiglu_torch.cpp"),
            str(packed_swiglu_dir / "packed_swiglu.cpp"),
            str(packed_swiglu_dir / "packed_swiglu_cuda.cu"),
            str(gqa_decode_attention_dir / "gqa_decode_attention_torch.cpp"),
            str(gqa_decode_attention_dir / "gqa_decode_attention_cuda.cu"),
        ],
        include_dirs=[
            str(rmsnorm_dir),
            str(residual_rmsnorm_dir),
            str(softmax_dir),
            str(attention_score_softmax_dir),
            str(rope_dir),
            str(packed_swiglu_dir),
            str(gqa_decode_attention_dir),
        ],
        extra_compile_args={
            "cxx": ["/W4", "/permissive-"],
            "nvcc": [
                "-arch=sm_120",
                "-Xcompiler=/W4",
                "-Xcompiler=/EHsc",
            ],
        },
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = _native_extension_config()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
