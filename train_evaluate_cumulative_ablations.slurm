#!/bin/bash
#SBATCH --job-name=elasticdino-cumulative-ablations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-/home/ubuntu/ElasticDino}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/ubuntu/datasets/hypersim/hypersim}"
TEST="${TEST:-false}"
LOG_DIR="${REPO_DIR}/slurm_logs/${SLURM_JOB_ID}"

case "${TEST,,}" in
  1|true|yes|on) TEST_ENABLED=true ;;
  0|false|no|off|"") TEST_ENABLED=false ;;
  *)
    echo "TEST must be a boolean (true/false, 1/0, yes/no, or on/off), got: ${TEST}" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPU_IDS[@]} < 4 )); then
  echo "Expected four allocated GPUs, found: ${CUDA_VISIBLE_DEVICES:-none}" >&2
  exit 1
fi

run_ablation() {
  local gpu_id="$1"
  local head_count="$2"
  local label="heads-cumulative-${head_count}"
  local command=(
    "${PYTHON_BIN}" -u train_evaluate_ablations.py
    --ablation heads
    --head-ablation-count "${head_count}"
    --dataset-root "${DATASET_ROOT}"
    --upscaler-epochs 20
    --depth-epochs 20
    --num-workers 8
  )

  if [[ "${TEST_ENABLED}" == true ]]; then
    command+=(--test)
  fi

  echo "[$(date --iso-8601=seconds)] GPU ${gpu_id}: starting ${label}"
  if CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}" \
      >"${LOG_DIR}/${label}.log" 2>&1; then
    echo "[$(date --iso-8601=seconds)] GPU ${gpu_id}: finished ${label}"
  else
    local status=$?
    echo "[$(date --iso-8601=seconds)] GPU ${gpu_id}: ${label} failed with status ${status}" >&2
    return "${status}"
  fi
}

trap 'kill $(jobs -pr) 2>/dev/null || true' INT TERM

# HYPERSIM_TASKS has five heads. Counts 1-4 remove increasingly long
# prefixes while retaining at least one supervision head for training.
run_ablation "${GPU_IDS[0]}" 1 &
pid_0=$!
run_ablation "${GPU_IDS[1]}" 2 &
pid_1=$!
run_ablation "${GPU_IDS[2]}" 3 &
pid_2=$!
run_ablation "${GPU_IDS[3]}" 4 &
pid_3=$!

status=0
for pid in "${pid_0}" "${pid_1}" "${pid_2}" "${pid_3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
