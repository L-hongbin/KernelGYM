"""Exercise actual HTTP compiler diagnostics after deployment; retain responses."""

import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid

from benchmarks.kernels.tvm_ffi_vector_add import KERNEL_CODE, REFERENCE_CODE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:20111/evaluate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assignment = "out[idx] = a[idx] + b[idx];"
    cases = {
        "correct": KERNEL_CODE,
        "many_errors": KERNEL_CODE.replace(assignment, "\n".join(
            f"out[idx] += missing_value_{i};" for i in range(12)
        )),
        "ptxas_mma": KERNEL_CODE.replace(assignment, '''
            float value = 0.0f;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%0,%0,%0}, {%0,%0,%0,%0}, {%0,%0}, {%0,%0,%0,%0};"
                         : "+f"(value));
            out[idx] = value;
        '''),
        "load_error": KERNEL_CODE.replace("import tvm_ffi_extension", "import tvm_ffi_extension\nraise RuntimeError('intentional model load failure')"),
        "mismatch": KERNEL_CODE.replace(assignment, "out[idx] = a[idx] + b[idx] + 1.0f;"),
    }
    report = {"endpoint": args.endpoint, "runs": [], "all_checks_passed": False}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    for name, code in cases.items():
        payload = {
            "task_id": f"compile-feedback-{name}-{uuid.uuid4().hex}",
            "reference_code": REFERENCE_CODE, "kernel_code": code, "backend": "tvm_ffi",
            "num_correct_trials": 1, "num_perf_trials": 1, "timeout": 180,
            "force_refresh": True, "run_performance": False, "enable_profiling": False,
            "enable_ncu": False, "enable_compute_sanitizer": False,
            "enable_correctness_input_perturbations": False, "return_detail_correctness": False,
            "use_reference_cache": False, "enable_compile_artifact_cache": True,
            "split_compile_and_execute": True, "detect_decoy_kernel": False,
            "enable_triton_detection": False, "run_triton_detection": False,
        }
        request = urllib.request.Request(args.endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        report["runs"].append({"case": name, "payload": payload, "response": result, "wall_s": time.perf_counter() - start})
        target.write_text(json.dumps(report, indent=2) + "\n")
        metadata = result.get("metadata", {})
        artifact = metadata.get("compile_artifact", {})
        assert "compiled" not in artifact and "error" not in artifact, (name, artifact)
        assert "correctness_failed_input_perturbation" not in metadata
        if name in {"many_errors", "ptxas_mma"}:
            assert result["compiled"] is False and result["error_code"] == "COMPILATION_ERROR"
            details = metadata["compilation_error_detail"]
            category = "undefined_identifier" if name == "many_errors" else "instruction_argument_mismatch"
            group = details[category]
            assert set(group) == {"count", "truncated", "errors"}
            assert 0 < len(group["errors"]) <= 8
            assert "First diagnostic:" in result["error_message"]
            assert len(result["error_message"]) < len(metadata["compilation_error"])
            if name == "many_errors":
                assert group["count"] == 12 and group["truncated"] and len(group["errors"]) == 8
        elif name == "load_error":
            assert result["compiled"] is False and result["error_code"] == "COMPILATION_ERROR"
            assert "compilation_error_detail" not in metadata
            assert "intentional model load failure" in metadata["compilation_error"]
            assert result["error_message"] == "Kernel compilation failed: " + metadata["compilation_error"]
        else:
            assert result["compiled"] is True
            assert result["correctness"] is (name == "correct")
            assert "compilation_error_detail" not in metadata
        print(f"{name}: passed", flush=True)
    report["all_checks_passed"] = True
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
