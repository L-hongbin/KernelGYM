"""Dedicated candidate process launched under NVIDIA Compute Sanitizer."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
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


@contextmanager
def _input_generation_device_context(device: Any, torch: Any, *, enabled: bool):
    if not enabled or not hasattr(torch, "set_default_device"):
        yield
        return
    previous_device = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous_device or "cpu")


def run(payload_path: Path) -> None:
    import torch

    from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
    from kernelgym.toolkit.kernelbench.exec_types import set_seed
    from kernelgym.toolkit.kernelbench.input_perturbation import (
        PERTURBATION_ORIGINAL,
        apply_input_perturbation,
        capture_random_input_origins,
    )
    from kernelgym.toolkit.kernelbench.loading import load_original_model_and_inputs

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    device = torch.device(payload.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    model_seed = int(payload.get("model_seed", 42))
    set_seed(model_seed)

    context = {}
    _, get_init_inputs, get_inputs = load_original_model_and_inputs(
        payload["reference_code"], context, payload.get("entry_point", "Model")
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
        raise RuntimeError(f"Sanitizer runner could not compile candidate: {artifact.get('error', 'unknown error')}")

    artifact["device"] = str(device)
    artifact.setdefault("backend", backend_name)
    artifact.setdefault("code", payload["kernel_code"])
    artifact.setdefault("entry_point", f"{payload.get('entry_point', 'Model')}New")
    handle = backend.load(artifact, device=device, context=context, build_dir=artifact.get("build_dir"))
    session = backend.open_session(handle, device=device)
    try:
        set_seed(model_seed)
        model = session.create_model(init_inputs, no_grad=True, synchronize=False)
        replay_seed = payload.get("input_seed")
        input_seed = model_seed if replay_seed is None else int(replay_seed)
        set_seed(input_seed)
        with _input_generation_device_context(
            device,
            torch,
            enabled=(
                bool(payload.get("generate_inputs_on_gpu", True))
                and os.environ.get("KERNELGYM_COMPUTE_SANITIZER_TOOL") != "initcheck"
            ),
        ):
            input_perturbation = payload.get("input_perturbation") or PERTURBATION_ORIGINAL
            if input_perturbation != PERTURBATION_ORIGINAL:
                with capture_random_input_origins() as origins:
                    inputs = get_inputs()
                inputs, _ = apply_input_perturbation(inputs, origins, input_perturbation)
            else:
                inputs = get_inputs()
        inputs = _move_to_device(inputs, device, torch)
        with torch.no_grad():
            set_seed(input_seed)
            _invoke(model, inputs)
            torch.cuda.synchronize(device=device)
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m kernelgym.toolkit.kernelbench.compute_sanitizer_runner PAYLOAD.json",
            file=sys.stderr,
        )
        return 2
    run(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
