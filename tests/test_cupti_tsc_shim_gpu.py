"""GPU end-to-end test for single-forward profiling under the CUPTI TSC shim.

Runs a subprocess with the shim preloaded and the Kineto TSC fix declared, the
way the deployed service runs workers, and asserts that auto resolution drops
to one profiler forward, every context still captures the CUDA kernel with a
sane duration, and the shim reports the engaged (or passthrough-fixed) state.
"""

import json
import shutil
import subprocess
import sys

import pytest

from kernelgym.utils import cupti_tsc_shim

_SUBPROCESS_SCRIPT = """
import json
import torch

from kernelgym.toolkit.kernelbench import timing
from kernelgym.utils import cupti_tsc_shim

device = torch.device("cuda:0")

target_ms = 150.0
cycles = 2_000_000
for _ in range(3):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device=device)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize(device=device)
    measured_ms = max(start.elapsed_time(end), 1e-3)
    if abs(measured_ms - target_ms) / target_ms < 0.2:
        break
    cycles = int(cycles * target_ms / measured_ms)


def slow_forward(x):
    torch.cuda._sleep(cycles)
    return x


inputs = torch.randn(64, device=device)
contexts = []
for _ in range(3):
    elapsed, metrics, timing_info = timing.time_execution_with_cuda_event(
        slow_forward,
        inputs,
        num_warmup=1,
        num_trials=2,
        verbose=False,
        device=device,
        enable_profiling=True,
    )
    kernels = metrics.get("kernels", [])
    contexts.append(
        {
            "num_profiling_trials": timing_info["num_profiling_trials"],
            "kernel_count": len(kernels),
            "max_kernel_cuda_ms": max((k["cuda_time_us"] for k in kernels), default=0.0) / 1000.0,
            "cuda_event_ms": elapsed[0],
            "shim_state_metric": metrics.get("cupti_tsc_shim_state"),
        }
    )

print(
    "RESULT:"
    + json.dumps(
        {
            "resolved_trials": timing.resolve_num_profiling_trials(100),
            "fix_verified": timing.kineto_tsc_fix_verified(),
            "shim_state": cupti_tsc_shim.shim_state(),
            "cupti_version": cupti_tsc_shim.shim_cupti_version(),
            "contexts": contexts,
        }
    )
)
"""


@pytest.mark.gpu
def test_single_forward_profiling_under_shim() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no C++ compiler available")

    shim_path = cupti_tsc_shim.ensure_shim_built()
    assert shim_path is not None, "shim build failed"

    import os

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(shim_path)
    env["KINETO_TSC_FIXED"] = "true"
    env[cupti_tsc_shim.SHIM_EXPECTED_ENV] = str(shim_path)

    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT:")]
    assert result_lines, completed.stdout[-2000:]
    report = json.loads(result_lines[-1][len("RESULT:") :])

    # Auto resolution trusts the declared fix and uses a single forward.
    assert report["resolved_trials"] == 1
    assert report["fix_verified"] is True
    assert cupti_tsc_shim.shim_state_healthy(report["shim_state"])

    for idx, context in enumerate(report["contexts"]):
        assert context["num_profiling_trials"] == 1
        assert context["kernel_count"] > 0, f"context {idx}: no CUDA kernels captured"
        assert cupti_tsc_shim.shim_state_healthy(context["shim_state_metric"])
        # Native timestamps must produce a duration consistent with CUDA events.
        ratio = context["max_kernel_cuda_ms"] / max(context["cuda_event_ms"], 1e-6)
        assert 0.7 < ratio < 1.3, f"context {idx}: profiler/cuda-event mismatch {context}"
