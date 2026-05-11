#!/usr/bin/env bash
set -Eeuo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# EAGLE Full Eval: prep_bird → inference → eval_exec
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="${ROOT_DIR:-/mnt/nj-aigc/usr/wangtong2/eagle_sql}"
EAGLE_DIR="${EAGLE_DIR:-${ROOT_DIR}/EAGLE-main}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT_DIR}/model/Qwen2.5-Coder-14B-Instruct}"
# Inference-format dir: config.json + pytorch_model.bin (flat).
# NOT the accelerate save_state output (artifacts/eagle2/checkpoints/state_X/model/).
EA_MODEL_PATH="${EA_MODEL_PATH:-${ROOT_DIR}/artifacts/eagle2/infer}"
BIRD_DEV_JSON="${BIRD_DEV_JSON:-${ROOT_DIR}/bird/dev_20240627/dev.json}"
BIRD_DEV_DB_ROOT="${BIRD_DEV_DB_ROOT:-${ROOT_DIR}/bird/dev_20240627/dev_databases}"

WORKDIR="${WORKDIR:-${ROOT_DIR}/artifacts/bird_dev_full_eval}"
TIMEOUT_SEC="${TIMEOUT_SEC:-15}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-128}"
# Tree config follows EAGLE-2 paper Appendix A for 13B class models
# (also a sane default for 14B). Set TOTAL_TOKEN=-1 to enable auto-tune.
TOTAL_TOKEN="${TOTAL_TOKEN:-50}"
DEPTH="${DEPTH:-6}"
TOP_K="${TOP_K:-10}"
DECODE_MODE="${DECODE_MODE:-eagle}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
USE_EAGLE3="${USE_EAGLE3:-0}"
# Stage A BIRD baseline: keep retry OFF for clean α/τ. Set RETRY_INVALID_SQL=1 to enable.
RETRY_INVALID_SQL="${RETRY_INVALID_SQL:-0}"

# Positional overrides
if [[ $# -ge 5 ]]; then
  BASE_MODEL_PATH="$1"; EA_MODEL_PATH="$2"; BIRD_DEV_JSON="$3"
  BIRD_DEV_DB_ROOT="$4"; WORKDIR="$5"; TIMEOUT_SEC="${6:-${TIMEOUT_SEC}}"
fi

QUESTION_JSONL="${WORKDIR}/bird_dev_infer.jsonl"
PRED_JSONL="${WORKDIR}/pred_dev.jsonl"
SUMMARY_JSON="${WORKDIR}/eval_summary.json"
FAILURES_JSONL="${WORKDIR}/eval_failures.jsonl"
REPORT_MD="${WORKDIR}/eval_report.md"
LOG_FILE="${WORKDIR}/full_eval.log"

mkdir -p "${WORKDIR}"
: > "${LOG_FILE}"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

export PYTHONPATH="${EAGLE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

log "=== Full Eval Pipeline ==="
log "base_model=${BASE_MODEL_PATH}"
log "ea_model=${EA_MODEL_PATH}"
log "eagle_version=$([ "${USE_EAGLE3}" = "1" ] && echo "3" || echo "2")"
log "decode=${DECODE_MODE} max_new=${MAX_NEW_TOKEN} depth=${DEPTH} top_k=${TOP_K}"

# ─── Step 1: Prepare BIRD dev JSONL ──────────────────────────────────────────
log "[1/3] prep_bird → ${QUESTION_JSONL}"
(cd "${EAGLE_DIR}" && python -m eagle.text2sql.bird.prep_bird \
  --input-json "${BIRD_DEV_JSON}" \
  --db-root "${BIRD_DEV_DB_ROOT}" \
  --output-jsonl "${QUESTION_JSONL}")

# ─── Step 2: Run inference ────────────────────────────────────────────────────
log "[2/3] inference → ${PRED_JSONL}"
USE_EAGLE3="${USE_EAGLE3}" \
MAX_NEW_TOKEN="${MAX_NEW_TOKEN}" \
TOTAL_TOKEN="${TOTAL_TOKEN}" \
DEPTH="${DEPTH}" \
TOP_K="${TOP_K}" \
DECODE_MODE="${DECODE_MODE}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
RETRY_INVALID_SQL="${RETRY_INVALID_SQL}" \
bash "${SCRIPT_DIR}/run_bird_eagle3_infer.sh" \
  "${BASE_MODEL_PATH}" \
  "${EA_MODEL_PATH}" \
  "${QUESTION_JSONL}" \
  "${PRED_JSONL}"

# ─── Step 3: Execution eval ──────────────────────────────────────────────────
log "[3/3] eval_exec → ${SUMMARY_JSON}"
bash "${SCRIPT_DIR}/run_bird_eval_exec.sh" \
  "${QUESTION_JSONL}" \
  "${PRED_JSONL}" \
  "${BIRD_DEV_DB_ROOT}" \
  "${SUMMARY_JSON}" \
  "${FAILURES_JSONL}" \
  "${REPORT_MD}" \
  "${TIMEOUT_SEC}"

# ─── Summary ──────────────────────────────────────────────────────────────────
python3 - "${SUMMARY_JSON}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
print(json.dumps({
    "exec_accuracy": s.get("exec_accuracy", 0.0),
    "acceptance_rate_mean": s.get("acceptance_rate_mean", 0.0),
    "pred_executable_rate": s.get("pred_executable_rate", 0.0),
}, indent=2))
PY

log "DONE. summary=${SUMMARY_JSON} report=${REPORT_MD}"
