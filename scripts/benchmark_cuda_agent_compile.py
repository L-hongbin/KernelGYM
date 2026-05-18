"""Compare CUDA-Agent compile latency across two KernelGYM worktrees."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from textwrap import dedent
from typing import Any


CHILD = r"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

repo = os.environ["KG_BENCH_REPO"]
sys.path.insert(0, repo)

from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend


bench_id = os.environ["KG_BENCH_ID"]
variant = os.environ["KG_BENCH_VARIANT"]
artifact_cache = os.environ.get("KG_BENCH_ARTIFACT_CACHE") == "1"

common_sources = {}
for index in range(3):
    common_sources[f"kernels/common_{index}.cu"] = f'''
#include <torch/extension.h>

template <typename T, int N>
struct BenchFunctor{index} {{
    __device__ __forceinline__ T operator()(T value) const {{
        #pragma unroll
        for (int i = 0; i < N; ++i) {{
            value = value * static_cast<T>(1.000001) + static_cast<T>(0.000001);
        }}
        return value;
    }}
}};

__global__ void bench_kernel_{index}(float* x, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        BenchFunctor{index}<float, 64> fn;
        x[i] = fn(x[i]);
    }}
}}
'''

common_sources["kernels/common_0.cu"] += '''
torch::Tensor identity_impl(torch::Tensor x) {
    return x;
}
'''

binding = f'''
#include <torch/extension.h>

torch::Tensor identity_impl(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    // benchmark id: {bench_id}
    // variant marker: {variant}
    m.def("identity", &identity_impl);
}}
'''

model_code = f'''
import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def forward(self, x):
        # benchmark id: {bench_id}
        return cuda_extension.identity(x)
'''

cuda_sources = dict(common_sources)
cuda_sources["kernels/generated_binding.cpp"] = binding

backend = KernelBenchCudaAgentBackend()
start = time.perf_counter()
artifact = backend.compile(
    model_code,
    cuda_sources=cuda_sources,
    device="cuda:0",
    entry_point="ModelNew",
    enable_compile_artifact_cache=artifact_cache,
)
wall = time.perf_counter() - start
payload = {
    "repo": repo,
    "variant": variant,
    "compiled": bool(artifact.get("compiled")),
    "error": artifact.get("error"),
    "wall_sec": round(wall, 6),
    "build_backend": artifact.get("build_backend"),
    "compile_artifact_cache_hit": artifact.get("compile_artifact_cache_hit"),
    "compile_timing": artifact.get("compile_timing"),
}
print(json.dumps(payload, sort_keys=True))

work_dir = artifact.get("work_dir")
if work_dir and not artifact.get("persistent_work_dir"):
    shutil.rmtree(work_dir, ignore_errors=True)
"""


def run_compile(
    *,
    python: str,
    repo: Path,
    bench_id: str,
    variant: str,
    artifact_cache: bool,
    extra_env: dict[str, str],
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(extra_env)
    env.update(
        {
            "KG_BENCH_REPO": str(repo),
            "KG_BENCH_ID": bench_id,
            "KG_BENCH_VARIANT": variant,
            "KG_BENCH_ARTIFACT_CACHE": "1" if artifact_cache else "0",
            "KERNELGYM_NVCC_THREADS": env.get("KERNELGYM_NVCC_THREADS", "1"),
        }
    )
    env["PYTHONPATH"] = str(repo)
    completed = subprocess.run(
        [python, "-c", CHILD],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "repo": str(repo),
            "variant": variant,
            "compiled": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr[-4000:],
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    payload["stderr_tail"] = completed.stderr[-1000:]
    return payload


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["wall_sec"] for row in rows if row.get("compiled") and isinstance(row.get("wall_sec"), float)]
    return {
        "count": len(values),
        "mean_wall_sec": round(statistics.mean(values), 6) if values else None,
        "median_wall_sec": round(statistics.median(values), 6) if values else None,
        "min_wall_sec": round(min(values), 6) if values else None,
        "max_wall_sec": round(max(values), 6) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--candidate-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bench_id = f"compile-accel-{int(time.time())}"
    cache_root = Path("/dev/shm/kernelgym/compile_benchmark") / bench_id
    candidate_cache = cache_root / "candidate"
    candidate_cache.mkdir(parents=True, exist_ok=True)

    main_env = {
        "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DISABLE": "1",
    }
    candidate_env = {
        "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE": "true",
        "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_INDEX": "fs",
        "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_DIR": str(candidate_cache / "objects"),
        "KERNELGYM_COMPILE_ARTIFACT_CACHE_DIR": str(candidate_cache / "artifacts"),
    }

    rows: list[dict[str, Any]] = []
    matrix = [
        ("main_no_exact_cache", args.main_repo, main_env, False, ["cold-a", "cold-b", "cold-c"]),
        ("candidate_object_cache", args.candidate_repo, candidate_env, False, ["cold-a", "cold-b", "cold-c"]),
        ("main_exact_repeat", args.main_repo, {}, False, ["exact", "exact"]),
        ("candidate_exact_repeat", args.candidate_repo, candidate_env, True, ["exact", "exact"]),
    ]

    for scenario, repo, env, artifact_cache, variants in matrix:
        for index, variant in enumerate(variants):
            row = run_compile(
                python=args.python,
                repo=repo,
                bench_id=bench_id,
                variant=variant,
                artifact_cache=artifact_cache,
                extra_env=env,
            )
            row["scenario"] = scenario
            row["iteration"] = index
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if not row.get("compiled"):
                break

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["scenario"]), []).append(row)

    result = {
        "bench_id": bench_id,
        "python": args.python,
        "main_repo": str(args.main_repo),
        "candidate_repo": str(args.candidate_repo),
        "rows": rows,
        "summary": {name: summarize(group) for name, group in groups.items()},
        "notes": [
            "main_no_exact_cache disables the old whole-extension cache to measure cpp_extension.load compile cost.",
            "candidate_object_cache disables exact artifact cache and enables object cache to measure cross-variant reuse.",
            "exact_repeat scenarios measure exact repeated payload behavior with each branch's exact cache path.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        dedent(f"""
    wrote {args.output}
    """).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
