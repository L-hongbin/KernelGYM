"""Differential tests against the pre-fusion implementation on real CUDA."""

import math

import pytest
import torch

from kernelgym.toolkit.kernelbench import correctness as c
from kernelgym.toolkit.kernelbench import correctness_fused as fused

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def assert_equivalent(reference, candidate, atol=1e-3, rtol=1e-3):
    actual = fused.compare(reference, candidate, atol=atol, rtol=rtol, output_path="output")
    assert actual is not None, "fusion must run, not silently fall back"
    expected = c._compare_tensors_inplace_with_diagnostics_torch(
        reference.clone(),
        candidate.clone(),
        atol=atol,
        rtol=rtol,
    )
    assert actual[0] == expected[0]
    for i in (1, 2):
        if math.isnan(expected[i]):
            assert math.isnan(actual[i])
        else:
            assert actual[i] == pytest.approx(expected[i], rel=2e-13, abs=1e-14)
    assert actual[3:7] == expected[3:7]
    for field in ("first", "last", "localizations"):
        assert actual[7][field] == expected[7][field]
    # torch.topk has unspecified tie order. Check scores and the score at every
    # returned coordinate; different choices among ties do not lose accuracy.
    assert [x[0] for x in actual[7]["top"]] == [x[0] for x in expected[7]["top"]]
    diff = (candidate - reference).abs()
    tol = reference.abs().mul(rtol).add(atol)
    scores = c._normalize_difference_inplace(diff, tol, tolerance_can_be_zero=atol == 0)
    coords = [tuple(record["coordinate"]) for _, record in actual[7]["top"]]
    assert len(set(coords)) == len(coords)
    for score, record in actual[7]["top"]:
        assert scores[tuple(record["coordinate"])].item() == score
    return actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(), (1,), (1057,), (33, 65), (2, 33, 65), (2, 2, 3, 35, 67)])
@pytest.mark.parametrize("pattern", ["correct", "random", "tail", "nonfinite"])
def test_differential(dtype, shape, pattern):
    generator = torch.Generator(device="cuda").manual_seed(173)
    reference = torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
    candidate = reference.clone()
    if pattern == "random":
        candidate += torch.randn(shape, generator=generator, device="cuda", dtype=dtype) * 0.02
    elif pattern == "tail":
        candidate.reshape(-1)[-min(17, reference.numel()) :] += 1
    elif pattern == "nonfinite":
        candidate.reshape(-1)[-1] = float("nan")
        if reference.numel() > 1:
            candidate.reshape(-1)[0] = float("inf")
    assert_equivalent(reference, candidate)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_tolerance_boundaries(dtype):
    thresholds = torch.tensor([0, 1, 2, 4, 8, 16], device="cuda", dtype=dtype)
    candidate = torch.cat(
        (
            thresholds,
            torch.nextafter(thresholds, torch.full_like(thresholds, float("inf"))),
            torch.nextafter(thresholds, torch.full_like(thresholds, -float("inf"))),
        )
    )
    assert_equivalent(torch.zeros_like(candidate), candidate, atol=1, rtol=0)
    assert_equivalent(torch.zeros_like(candidate), candidate, atol=0, rtol=0)
    # Half/BF16 rounding of separate scalar multiply/add must match PyTorch.
    x = torch.linspace(-50, 50, 8192, dtype=dtype, device="cuda")
    y = x + x.abs() * 0.00123456789 + 0.00087654321
    assert_equivalent(x, y, atol=0.00087654321, rtol=0.00123456789)


def test_reference_nonfinite_and_curve_truncation():
    x = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.0, 1.0], device="cuda")
    y = torch.tensor([float("inf"), 1.0, 0.0, 1.0, 2.0], device="cuda")
    assert_equivalent(x, y)
    result = assert_equivalent(torch.zeros(3, device="cuda"), torch.tensor([0.0, 1.5, 3.0], device="cuda"), 1, 0)
    assert list(c._format_element_correctness_curve(result[3], result[4])) == ["1x", "2x", "4x"]


def test_tied_top3_is_deterministic_and_lists_bounded():
    x = torch.zeros((4, 512, 512), device="cuda")
    result = assert_equivalent(x, torch.ones_like(x))
    assert [r["coordinate"] for _, r in result[7]["top"]] == [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
    for field in ("batch", "row", "tile"):
        assert len(result[7]["localizations"][0][field]["mismatches"]) <= 8


def test_nondefault_stream():
    with torch.cuda.stream(torch.cuda.Stream()):
        x = torch.zeros((35, 63), device="cuda")
        y = torch.ones_like(x)
        assert_equivalent(x, y)


def test_fallback_and_default_gate(monkeypatch):
    x = torch.zeros((33, 65), device="cuda")
    assert fused.compare(x.t(), x.t(), atol=0.001, rtol=0.001, output_path="output") is None
    for dtype in (torch.bool, torch.int64, torch.complex64):
        value = x.to(dtype)
        assert fused.compare(value, value, atol=0.001, rtol=0.001, output_path="output") is None

    def forbidden(*args, **kwargs):
        raise AssertionError("default mode must not enter fused diagnostics")

    monkeypatch.setattr(fused, "compare", forbidden)
    assert c._compare_tensors_inplace(x.clone(), x.clone())[0]
    monkeypatch.setenv("KERNELGYM_FUSED_CORRECTNESS", "false")
    assert c._compare_tensors_inplace_with_diagnostics(x.clone(), x.clone())[0]


def test_unavailable_build_uses_torch(monkeypatch):
    monkeypatch.setattr(fused, "_library", lambda capability: None)
    x = torch.zeros((33, 65), device="cuda")
    assert c._compare_tensors_inplace_with_diagnostics(x, torch.ones_like(x))[0] is False
