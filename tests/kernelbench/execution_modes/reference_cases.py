"""Small-shape adapters for representative real KernelBench references."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceCase:
    relative_path: str
    overrides: dict[str, Any]

    @property
    def test_id(self) -> str:
        return self.relative_path.replace("/", "-").removesuffix(".py")


REFERENCE_CASES = (
    ReferenceCase("level1/19_ReLU.py", {"batch_size": 4, "dim": 16}),
    ReferenceCase(
        "level1/33_BatchNorm.py",
        {"batch_size": 2, "features": 4, "dim1": 8, "dim2": 8},
    ),
    ReferenceCase(
        "level1/40_LayerNorm.py",
        {"batch_size": 2, "features": 4, "dim1": 8, "dim2": 8},
    ),
    ReferenceCase(
        "level1/89_cumsum.py",
        {"batch_size": 4, "input_shape": (8,), "dim": 1},
    ),
    ReferenceCase(
        "level1/97_ScaledDotProductAttention.py",
        {"batch_size": 2, "num_heads": 2, "sequence_length": 4, "embedding_dimension": 8},
    ),
    ReferenceCase(
        "level2/11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm.py",
        {
            "batch_size": 2,
            "in_channels": 4,
            "out_channels": 8,
            "height": 8,
            "width": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "groups": 1,
            "num_groups": 4,
        },
    ),
    ReferenceCase(
        "level2/66_Matmul_Dropout_Softmax.py",
        {"batch_size": 2, "in_features": 8, "out_features": 8, "dropout_p": 0.2},
    ),
    ReferenceCase(
        "level3/1_MLP.py",
        {"batch_size": 2, "input_size": 8, "layer_sizes": [8, 8], "output_size": 4},
    ),
    ReferenceCase(
        "level3/33_VanillaRNN.py",
        {
            "batch_size": 2,
            "input_size": 8,
            "hidden_size": 8,
            "output_size": 4,
            "sequence_length": 2,
        },
    ),
    ReferenceCase(
        "level3/35_LSTM.py",
        {
            "batch_size": 2,
            "sequence_length": 4,
            "input_size": 8,
            "hidden_size": 8,
            "num_layers": 2,
            "output_size": 4,
            "dropout": 0.0,
        },
    ),
)


def find_kernelbench_data_root() -> Path | None:
    configured = os.environ.get("KERNELBENCH_DATA_ROOT")
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(configured).expanduser() if configured else None,
        repo_root.parent / "KernelBench-oldsize" / "KernelBench",
        repo_root.parent / "kernel_bench_verified" / "KernelBench",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "level1").is_dir():
            return candidate.resolve()
    return None


def load_reference_namespace(data_root: Path, case: ReferenceCase) -> dict[str, Any]:
    source_path = data_root / case.relative_path
    namespace: dict[str, Any] = {}
    source = source_path.read_text(encoding="utf-8")
    exec(compile(source, str(source_path), "exec"), namespace)
    namespace.update(case.overrides)
    return namespace
