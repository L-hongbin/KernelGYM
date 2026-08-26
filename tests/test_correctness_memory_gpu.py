import gc

import pytest

MIB = 1024**2


def _require_cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")
    return torch


def _begin_memory_measurement(torch, device) -> int:
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _finish_memory_measurement(torch, device, baseline: int) -> dict[str, int]:
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "baseline_allocated_bytes": baseline,
        "peak_increment_bytes": max(0, peak - baseline),
        "persistent_increment_bytes": max(0, after - baseline),
        "temporary_peak_bytes": max(0, peak - max(baseline, after)),
    }


def _measured_forward(torch, model, inputs, device):
    baseline = _begin_memory_measurement(torch, device)
    output = model(*inputs)
    stats = _finish_memory_measurement(torch, device, baseline)
    return output, stats


def _cleanup_cuda_values(torch) -> None:
    gc.collect()
    torch.cuda.synchronize()


def _mib_stats(stats: dict[str, int]) -> dict[str, float]:
    return {
        key.removesuffix("_bytes") + "_mib": round(value / MIB, 6)
        for key, value in stats.items()
    }


@pytest.mark.gpu
def test_correctness_memory_matches_independent_trial_for_capture_batch_norm() -> None:
    """Compare in-correctness and post-warmup memory measurement semantics.

    The model and tensor shape come from slime's
    ``examples/kernel_agent/test/capture_kernelgym_feedback_cases.py`` pass
    case. The candidate intentionally uses the same eager BatchNorm so this
    test isolates measurement placement from custom-kernel implementation
    differences.
    """
    torch = _require_cuda_runtime()
    from kernelgym.toolkit.kernelbench import correctness

    device = torch.device("cuda:0")
    num_trials = 3
    num_warmup = 3

    class CaptureCaseBatchNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_norm = torch.nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)

        def forward(self, x):
            return self.batch_norm(x)

    def make_inputs():
        # capture_kernelgym_feedback_cases.py uses [128, 3, 1024, 32].
        return [torch.randn(128, 3, 1024, 32, device=device)]

    torch.cuda.empty_cache()
    torch.manual_seed(42)
    reference = CaptureCaseBatchNorm().to(device)
    torch.manual_seed(42)
    candidate = CaptureCaseBatchNorm().to(device)

    correctness_reference_trials = []
    correctness_candidate_trials = []

    with torch.no_grad():
        # Reproduce the relevant ordering in run_and_check_correctness:
        # reference -> cache poison -> candidate -> compare.
        for trial in range(num_trials):
            torch.manual_seed(1234 + trial)
            inputs = make_inputs()

            reference_output, reference_stats = _measured_forward(
                torch, reference, inputs, device
            )

            poison_scratch = correctness._zero_poison_like(reference_output)
            torch.cuda.synchronize(device)
            del poison_scratch

            candidate_output, candidate_stats = _measured_forward(
                torch, candidate, inputs, device
            )

            torch.testing.assert_close(reference_output, candidate_output)
            correctness_reference_trials.append(reference_stats)
            correctness_candidate_trials.append(candidate_stats)

            del inputs, reference_output, candidate_output
            _cleanup_cuda_values(torch)

        # Independent strategy: warm both models, discard all warmup outputs,
        # then run one additional measured forward for each model.
        independent_inputs = make_inputs()
        for _ in range(num_warmup):
            reference_output = reference(*independent_inputs)
            candidate_output = candidate(*independent_inputs)
            torch.cuda.synchronize(device)
            del reference_output, candidate_output
        _cleanup_cuda_values(torch)

        independent_reference_output, independent_reference = _measured_forward(
            torch, reference, independent_inputs, device
        )
        del independent_reference_output
        _cleanup_cuda_values(torch)

        independent_candidate_output, independent_candidate = _measured_forward(
            torch, candidate, independent_inputs, device
        )
        del independent_candidate_output, independent_inputs
        _cleanup_cuda_values(torch)

    report = {
        "correctness_reference_trials": [
            _mib_stats(stats) for stats in correctness_reference_trials
        ],
        "correctness_candidate_trials": [
            _mib_stats(stats) for stats in correctness_candidate_trials
        ],
        "independent_reference": _mib_stats(independent_reference),
        "independent_candidate": _mib_stats(independent_candidate),
    }
    print(f"memory measurement comparison: {report}")

    output_bytes = (
        128 * 3 * 1024 * 32 * torch.tensor([], dtype=torch.float32).element_size()
    )
    allocator_tolerance = MIB

    # Both strategies capture the output plus the same small BatchNorm
    # workspace. They should agree to within one allocator block.
    for stats in correctness_reference_trials + correctness_candidate_trials:
        assert stats["peak_increment_bytes"] >= output_bytes
    assert (
        abs(
            correctness_reference_trials[-1]["peak_increment_bytes"]
            - independent_reference["peak_increment_bytes"]
        )
        <= allocator_tolerance
    )
    assert (
        abs(
            correctness_candidate_trials[-1]["peak_increment_bytes"]
            - independent_candidate["peak_increment_bytes"]
        )
        <= allocator_tolerance
    )

    # During correctness, reference_output remains live while candidate runs,
    # so the candidate's absolute baseline is about one output tensor larger.
    # The independent trial deletes the reference output first. This proves
    # absolute max_allocated is not comparable, while peak-baseline is.
    correctness_baseline_gap = (
        correctness_candidate_trials[-1]["baseline_allocated_bytes"]
        - correctness_reference_trials[-1]["baseline_allocated_bytes"]
    )
    assert abs(correctness_baseline_gap - output_bytes) <= allocator_tolerance
    assert (
        correctness_candidate_trials[-1]["baseline_allocated_bytes"]
        > independent_candidate["baseline_allocated_bytes"]
    )
