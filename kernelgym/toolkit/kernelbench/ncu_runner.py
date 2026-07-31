"""Dedicated target process launched by Nsight Compute."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _move_to_device(value: Any, device: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, torch) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device, torch) for key, item in value.items()}
    return value


def _invoke(model: Any, inputs: Any) -> Any:
    return model(**inputs) if isinstance(inputs, dict) else model(*inputs)


def run(payload_path: Path) -> None:
    import torch

    from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
    from kernelgym.toolkit.kernelbench.exec_types import set_seed
    from kernelgym.toolkit.kernelbench.loading import load_original_model_and_inputs

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    device = torch.device(payload.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    set_seed(42)

    context = {}
    _, get_init_inputs, get_inputs = load_original_model_and_inputs(
        payload["reference_code"],
        context,
        payload.get("entry_point", "Model"),
    )
    init_inputs = _move_to_device(get_init_inputs(), device, torch)

    backend_name = payload.get("backend", "triton")
    backend = KernelBenchBackend()
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict) or not artifact.get("compiled"):
        artifact = backend.compile(
            payload["kernel_code"],
            device=device,
            backend=backend_name,
            entry_point=f"{payload.get('entry_point', 'Model')}New",
            enable_compile_artifact_cache=True,
        )
    if not artifact.get("compiled"):
        raise RuntimeError(f"NCU runner could not compile candidate: {artifact.get('error', 'unknown error')}")

    artifact["device"] = str(device)
    artifact.setdefault("backend", backend_name)
    artifact.setdefault("code", payload["kernel_code"])
    artifact.setdefault("entry_point", f"{payload.get('entry_point', 'Model')}New")
    handle = backend.load(artifact, device=device, context=context, build_dir=artifact.get("build_dir"))
    session = backend.open_session(handle, device=device)
    try:
        model = session.create_model(init_inputs, no_grad=True, synchronize=False)
        inputs = _move_to_device(get_inputs(), device, torch)

        with torch.no_grad():
            for _ in range(max(0, int(payload.get("warmup", 2)))):
                _invoke(model, inputs)
            torch.cuda.synchronize(device=device)
            with torch.cuda.profiler.profile():
                _invoke(model, inputs)
                torch.cuda.synchronize(device=device)
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m kernelgym.toolkit.kernelbench.ncu_runner PAYLOAD.json", file=sys.stderr)
        return 2
    run(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
