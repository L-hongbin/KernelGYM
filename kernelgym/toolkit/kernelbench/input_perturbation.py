"""Distribution-aware input perturbations for hidden correctness checks."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

INPUT_KIND_RAND = "torch.rand"
INPUT_KIND_RANDN = "torch.randn"

PERTURBATION_ORIGINAL = "original"
PERTURBATION_SCALE_UP = "scale_up"
PERTURBATION_SCALE_DOWN = "scale_down"
PERTURBATION_SIGN_CHALLENGE = "sign_challenge"

CORRECTNESS_INPUT_PERTURBATIONS = (
    PERTURBATION_ORIGINAL,
    PERTURBATION_SCALE_UP,
    PERTURBATION_SCALE_DOWN,
    PERTURBATION_SIGN_CHALLENGE,
)

_CAPTURE_LOCK = threading.RLock()
_RANDOM_FACTORIES = {
    "rand": INPUT_KIND_RAND,
    "rand_like": INPUT_KIND_RAND,
    "randn": INPUT_KIND_RANDN,
    "randn_like": INPUT_KIND_RANDN,
}


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int] | None:
    try:
        storage = tensor.untyped_storage()
        return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())
    except Exception:
        return None


@dataclass
class RandomInputOrigins:
    """Tracks tensors returned directly by torch.rand/torch.randn factories."""

    by_object_id: dict[int, str] = field(default_factory=dict)
    by_storage: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def record(self, value: Any, kind: str) -> None:
        for tensor in _iter_tensors(value):
            self.by_object_id[id(tensor)] = kind
            storage_key = _storage_key(tensor)
            if storage_key is not None:
                self.by_storage[storage_key] = kind

    def kind_for(self, tensor: torch.Tensor) -> str | None:
        kind = self.by_object_id.get(id(tensor))
        if kind is not None:
            return kind
        storage_key = _storage_key(tensor)
        return self.by_storage.get(storage_key) if storage_key is not None else None


@contextmanager
def capture_random_input_origins() -> Iterator[RandomInputOrigins]:
    """Capture direct torch.rand/torch.randn outputs while get_inputs() runs.

    KernelGYM GPU workers execute one task at a time. The lock additionally
    prevents overlapping capture contexts in unit tests or future threaded use.
    """

    origins = RandomInputOrigins()
    originals: dict[str, Any] = {}
    with _CAPTURE_LOCK:
        for factory_name, kind in _RANDOM_FACTORIES.items():
            original = getattr(torch, factory_name, None)
            if original is None:
                continue
            originals[factory_name] = original

            def wrapped(*args: Any, _original=original, _kind=kind, **kwargs: Any) -> Any:
                value = _original(*args, **kwargs)
                origins.record(value, _kind)
                return value

            setattr(torch, factory_name, wrapped)
        try:
            yield origins
        finally:
            for factory_name, original in originals.items():
                setattr(torch, factory_name, original)


def _clone_preserve_format(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.preserve_format)


def _transform_tensor(tensor: torch.Tensor, kind: str, perturbation: str) -> tuple[torch.Tensor, str | None]:
    if perturbation == PERTURBATION_ORIGINAL:
        return tensor, None
    transformed = _clone_preserve_format(tensor)
    if perturbation == PERTURBATION_SCALE_UP:
        transformed.mul_(3.0)
        return transformed, "multiply_3"
    if perturbation == PERTURBATION_SCALE_DOWN:
        transformed.mul_(0.01)
        return transformed, "multiply_0.01"
    if perturbation == PERTURBATION_SIGN_CHALLENGE:
        if kind == INPUT_KIND_RAND:
            transformed.neg_()
            return transformed, "negate"
        if kind == INPUT_KIND_RANDN:
            transformed.abs_()
            return transformed, "absolute"
    raise ValueError(f"Unsupported input perturbation {perturbation!r} for {kind!r}")


def apply_input_perturbation(
    inputs: Any,
    origins: RandomInputOrigins,
    perturbation: str,
) -> tuple[Any, dict[str, Any]]:
    """Apply a perturbation recursively to recognized floating-point inputs."""

    if perturbation not in CORRECTNESS_INPUT_PERTURBATIONS:
        raise ValueError(f"Unsupported correctness input perturbation: {perturbation!r}")

    detected_counts = {INPUT_KIND_RAND: 0, INPUT_KIND_RANDN: 0}
    transform_counts: dict[str, int] = {}

    def transform(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            kind = origins.kind_for(value)
            if kind not in detected_counts or not value.is_floating_point():
                return value
            detected_counts[kind] += 1
            transformed, transform_name = _transform_tensor(value, kind, perturbation)
            if transform_name is not None:
                transform_counts[transform_name] = transform_counts.get(transform_name, 0) + 1
            return transformed
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, tuple):
            return tuple(transform(item) for item in value)
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        return value

    transformed_inputs = transform(inputs)
    summary = {
        "name": perturbation,
        "detected_input_kinds": {kind: count for kind, count in detected_counts.items() if count},
        "transforms": transform_counts,
        "transformed_tensor_count": sum(transform_counts.values()),
    }
    return transformed_inputs, summary
