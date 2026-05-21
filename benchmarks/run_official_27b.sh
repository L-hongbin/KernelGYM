#!/usr/bin/env bash
# Official 27B 3-binding paired benchmark: full 74 problems × 3 bindings = 222
# requests, run at BOTH conc=3 and conc=8 sequentially so the two
# concurrency settings can be compared side-by-side.
#
# Per codex audit: timeout dropped from 240s -> 180s, same seed for
# both passes, object + artifact caches cleared on .40 before each
# pass, results land under benchmarks/results/official_c{3,8}_*.

set -u
PYTHON=python3
PROJECT=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
RESULTS=$PROJECT/benchmarks/results
SEED=2026  # fixed seed: both passes shuffle problems in the same order

clear_obj_cache() {
  ssh -o StrictHostKeyChecking=no chenshuailin@192.168.16.40 \
    "docker exec kernelgym-reward-only-40 bash -c 'rm -rf /dev/shm/kernelgym/compile_cache/manual_ninja_objects/* 2>/dev/null; rm -rf /dev/shm/kernelgym/compile_cache/cuda_agent_artifacts/* 2>/dev/null; rm -rf /dev/shm/kernelgym/compile_cache/tvm_ffi_artifacts/* 2>/dev/null; echo cleared'" \
    >/dev/null 2>&1
}

for conc in 3 8; do
  tag="official_c${conc}"
  echo "================ concurrency=$conc tag=$tag seed=$SEED ================"
  echo "  $(date +%H:%M:%S) clearing caches on .40..."
  clear_obj_cache

  # Wipe any prior official run with this tag so we start fresh and
  # resume-by-uid logic doesn't no-op when re-launching.
  rm -f "$RESULTS/${tag}_27b_breakdown_"*.jsonl

  echo "  $(date +%H:%M:%S) starting full 222-sample run..."
  start=$(date +%s)
  cd "$PROJECT" && $PYTHON benchmarks/run_27b_breakdown.py \
    --tag "$tag" \
    --concurrency "$conc" \
    --timeout 180 \
    --seed "$SEED" \
    --no-health \
    2>&1 | tail -10
  rc=$?
  end=$(date +%s)
  wall=$((end - start))
  echo "  $(date +%H:%M:%S) finished rc=$rc wall=${wall}s"
  echo
done
echo "================ both passes done ================"
