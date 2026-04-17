#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Config (override via env vars)
# -----------------------------
ROOT_DIR="${ROOT_DIR:-/mnt/nj-aigc/usr/wangtong2/eagle_sql}"
EAGLE_DIR="${EAGLE_DIR:-${ROOT_DIR}/EAGLE-main}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT_DIR}/model/Qwen2.5-Coder-14B-Instruct}"
BIRD_DEV_JSON="${BIRD_DEV_JSON:-${ROOT_DIR}/bird/dev_20240627/dev.json}"
BIRD_DEV_DB_ROOT="${BIRD_DEV_DB_ROOT:-${ROOT_DIR}/bird/dev_20240627/dev_databases}"

WORKDIR="${WORKDIR:-${ROOT_DIR}/artifacts/prompt_eval_audit_qwen25}"
QUESTION_JSONL="${QUESTION_JSONL:-${WORKDIR}/bird_dev_infer.jsonl}"
PROMPT_VALIDATE_DIR="${PROMPT_VALIDATE_DIR:-${WORKDIR}/prompt_validate}"
EVAL_LOGIC_VALIDATE_DIR="${EVAL_LOGIC_VALIDATE_DIR:-${WORKDIR}/eval_logic_validate}"
LOG_FILE="${LOG_FILE:-${WORKDIR}/run_bird_prompt_eval_audit.log}"

NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-auto}"                # auto/bf16/fp16/fp32
DEVICE="${DEVICE:-cuda}"              # auto/cuda/cpu
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-1}" # 1/0
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-15}"

# If QUESTION_JSONL is missing, build from BIRD dev json.
AUTO_PREP_IF_MISSING="${AUTO_PREP_IF_MISSING:-1}"  # 1/0
FORCE_REBUILD_QUESTION_JSONL="${FORCE_REBUILD_QUESTION_JSONL:-0}" # 1/0

is_true() {
  local v
  v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "y" || "$v" == "on" ]]
}

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

die() {
  echo "[$(date '+%F %T')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
  exit 1
}

mkdir -p "${WORKDIR}" "${PROMPT_VALIDATE_DIR}" "${EVAL_LOGIC_VALIDATE_DIR}"
: > "${LOG_FILE}"

for cmd in python bash tee; do
  command -v "${cmd}" >/dev/null 2>&1 || die "missing command: ${cmd}"
done

# Auto fallback to repository-local BIRD dev layout when ROOT_DIR/bird is absent.
if [[ ! -f "${BIRD_DEV_JSON}" ]]; then
  ALT_DEV_JSON="${EAGLE_DIR}/eagle/bird/dev_20240627/dev.json"
  if [[ -f "${ALT_DEV_JSON}" ]]; then
    BIRD_DEV_JSON="${ALT_DEV_JSON}"
  fi
fi
if [[ ! -d "${BIRD_DEV_DB_ROOT}" ]]; then
  ALT_DEV_DB_ROOT="${EAGLE_DIR}/eagle/bird/dev_20240627/dev_databases"
  if [[ -d "${ALT_DEV_DB_ROOT}" ]]; then
    BIRD_DEV_DB_ROOT="${ALT_DEV_DB_ROOT}"
  fi
fi

[[ -d "${EAGLE_DIR}" ]] || die "EAGLE_DIR not found: ${EAGLE_DIR}"
[[ -d "${BASE_MODEL_PATH}" ]] || die "BASE_MODEL_PATH not found: ${BASE_MODEL_PATH}"
[[ -f "${BIRD_DEV_JSON}" ]] || die "BIRD_DEV_JSON not found: ${BIRD_DEV_JSON}"
[[ -d "${BIRD_DEV_DB_ROOT}" ]] || die "BIRD_DEV_DB_ROOT not found: ${BIRD_DEV_DB_ROOT}"

if [[ "${DEVICE}" == "cuda" ]]; then
  python - <<'PY' 2>&1 | tee -a "${LOG_FILE}" || die "DEVICE=cuda but CUDA is unavailable"
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() == False")

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
print(f"[gpu-check] cuda_available=1 device_count={count} devices={names}")
PY
fi

log "paths root=${ROOT_DIR} eagle=${EAGLE_DIR}"
log "model base_model=${BASE_MODEL_PATH}"
log "data dev_json=${BIRD_DEV_JSON} db_root=${BIRD_DEV_DB_ROOT}"
log "workdir=${WORKDIR}"
log "question_jsonl=${QUESTION_JSONL}"
log "prompt_validate_dir=${PROMPT_VALIDATE_DIR}"
log "eval_logic_validate_dir=${EVAL_LOGIC_VALIDATE_DIR}"
log "params num_samples=${NUM_SAMPLES} seed=${SEED} temperature=${TEMPERATURE} top_p=${TOP_P} max_new_tokens=${MAX_NEW_TOKENS}"
log "runtime dtype=${DTYPE} device=${DEVICE} device_map_auto=${DEVICE_MAP_AUTO} eval_timeout_sec=${EVAL_TIMEOUT_SEC}"
log "prep auto_if_missing=${AUTO_PREP_IF_MISSING} force_rebuild=${FORCE_REBUILD_QUESTION_JSONL}"

need_prep=0
if is_true "${FORCE_REBUILD_QUESTION_JSONL}"; then
  need_prep=1
elif [[ ! -f "${QUESTION_JSONL}" ]]; then
  if is_true "${AUTO_PREP_IF_MISSING}"; then
    need_prep=1
  else
    die "QUESTION_JSONL missing and AUTO_PREP_IF_MISSING=0: ${QUESTION_JSONL}"
  fi
fi

if [[ "${need_prep}" == "1" ]]; then
  log "[1/4] build question_jsonl via prep_bird"
  (
    cd "${EAGLE_DIR}"
    python -m eagle.text2sql.bird.prep_bird \
      --input-json "${BIRD_DEV_JSON}" \
      --db-root "${BIRD_DEV_DB_ROOT}" \
      --output-jsonl "${QUESTION_JSONL}"
  )
else
  log "[1/4] skip prep_bird, reuse existing QUESTION_JSONL"
fi

log "[2/4] prompt small-batch direct generation + optional EX eval"
device_map_flag="--no-device-map-auto"
if is_true "${DEVICE_MAP_AUTO}"; then
  device_map_flag="--device-map-auto"
fi

(
  cd "${EAGLE_DIR}"
  python -m eagle.text2sql.bird.validate_prompt_smallbatch \
    --base-model-path "${BASE_MODEL_PATH}" \
    --question-jsonl "${QUESTION_JSONL}" \
    --output-dir "${PROMPT_VALIDATE_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    ${device_map_flag} \
    --run-eval \
    --db-root "${BIRD_DEV_DB_ROOT}" \
    --eval-timeout-sec "${EVAL_TIMEOUT_SEC}"
)

log "[3/4] strict eval_exec logic validation on minimal positive/negative cases"
(
  cd "${EAGLE_DIR}"
  python -m eagle.text2sql.bird.validate_eval_exec_minimal \
    --output-dir "${EVAL_LOGIC_VALIDATE_DIR}" \
    | tee "${EVAL_LOGIC_VALIDATE_DIR}/validate_eval_exec_minimal.stdout.log"
)

log "[4/4] aggregate key audit metrics"
python - \
  "${PROMPT_VALIDATE_DIR}/prompt_smallbatch_summary.json" \
  "${EVAL_LOGIC_VALIDATE_DIR}/eval_ignore_summary.json" \
  "${EVAL_LOGIC_VALIDATE_DIR}/eval_strict_summary.json" <<'PY'
import json
import sys
from pathlib import Path

prompt_summary_path = Path(sys.argv[1])
ignore_summary_path = Path(sys.argv[2])
strict_summary_path = Path(sys.argv[3])

def read_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

ps = read_json(prompt_summary_path)
ig = read_json(ignore_summary_path)
st = read_json(strict_summary_path)

out = {
    "prompt_validate": {
        "total_samples": ps.get("total_samples", 0),
        "valid_sql_rate": ps.get("valid_sql_rate", 0.0),
        "avg_wall_time_sec": ps.get("avg_wall_time_sec", 0.0),
        "invalid_breakdown": ps.get("invalid_breakdown", {}),
        "eval_exec_accuracy": (ps.get("eval_summary") or {}).get("exec_accuracy", 0.0),
        "eval_pred_executable_rate": (ps.get("eval_summary") or {}).get("pred_executable_rate", 0.0),
    },
    "eval_logic_validate": {
        "ignore_row_order_exec_accuracy": ig.get("exec_accuracy", 0.0),
        "strict_row_order_exec_accuracy": st.get("exec_accuracy", 0.0),
        "ignore_failure_breakdown": ig.get("failure_breakdown", {}),
        "strict_failure_breakdown": st.get("failure_breakdown", {}),
    },
    "artifacts": {
        "prompt_summary_json": str(prompt_summary_path),
        "eval_ignore_summary_json": str(ignore_summary_path),
        "eval_strict_summary_json": str(strict_summary_path),
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

log "DONE"
log "prompt_summary=${PROMPT_VALIDATE_DIR}/prompt_smallbatch_summary.json"
log "eval_ignore_summary=${EVAL_LOGIC_VALIDATE_DIR}/eval_ignore_summary.json"
log "eval_strict_summary=${EVAL_LOGIC_VALIDATE_DIR}/eval_strict_summary.json"
log "eval_logic_stdout=${EVAL_LOGIC_VALIDATE_DIR}/validate_eval_exec_minimal.stdout.log"
