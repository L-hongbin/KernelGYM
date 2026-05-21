#!/usr/bin/env bash
# Run the 30-request subset at three concurrency levels.
# Before each run, clear the manual-ninja object cache on .40 so each
# setting starts from an equally cold compile-cache state.
set -u

PYTHON=python3
PROJECT=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
RESULTS=$PROJECT/benchmarks/results

clear_obj_cache() {
  ssh -o StrictHostKeyChecking=no chenshuailin@192.168.16.40 \
    "docker exec kernelgym-reward-only-40 bash -c 'rm -rf /dev/shm/kernelgym/compile_cache/manual_ninja_objects/* 2>/dev/null; rm -rf /dev/shm/kernelgym/compile_cache/cuda_agent_artifacts/* 2>/dev/null; rm -rf /dev/shm/kernelgym/compile_cache/tvm_ffi_artifacts/* 2>/dev/null; echo cleared'" \
    >/dev/null 2>&1
}

for conc in 3 8 16; do
  tag="calib_c${conc}"
  echo "================ concurrency=$conc tag=$tag ================"
  echo "  $(date +%H:%M:%S) clearing object/artifact caches on .40..."
  clear_obj_cache

  # Remove any prior calibration results for this tag (so resume logic
  # doesn't skip everything when we re-run).
  rm -f "$RESULTS/${tag}_27b_breakdown_"*.jsonl

  echo "  $(date +%H:%M:%S) starting run..."
  start=$(date +%s)
  cd "$PROJECT" && $PYTHON benchmarks/run_27b_breakdown.py \
    --tag "$tag" \
    --limit 10 \
    --concurrency "$conc" \
    --timeout 240 \
    --no-health \
    2>&1 | tail -5
  rc=$?
  end=$(date +%s)
  wall=$((end - start))
  echo "  $(date +%H:%M:%S) finished rc=$rc wall=${wall}s"
  echo
done
echo "================ all 3 calibration runs done ================"
