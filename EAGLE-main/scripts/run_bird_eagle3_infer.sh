#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGLE_DIR="${EAGLE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ ! -d "${EAGLE_DIR}/eagle" ]]; then
  echo "ERROR: invalid EAGLE_DIR (missing eagle package): ${EAGLE_DIR}" >&2
  exit 1
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EAGLE_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${EAGLE_DIR}"
fi

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <base_model_path> <ea_model_path> <question_file.jsonl> <answer_file.jsonl>"
  exit 1
fi

BASE_MODEL_PATH="$1"
EA_MODEL_PATH="$2"
QUESTION_FILE="$3"
ANSWER_FILE="$4"
KV_CACHE_MIN_LENGTH="${KV_CACHE_MIN_LENGTH:-3072}"
KV_CACHE_MARGIN="${KV_CACHE_MARGIN:-256}"
DEVICE="${DEVICE:-cuda}"                    # auto/cuda/cpu
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-0}"    # 1/0
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-128}"
TOTAL_TOKEN="${TOTAL_TOKEN:-16}"
DEPTH="${DEPTH:-3}"
TOP_K="${TOP_K:-3}"
DECODE_MODE="${DECODE_MODE:-eagle}"           # eagle/naive/hf
LAYERED_PROBE_SAMPLES="${LAYERED_PROBE_SAMPLES:-0}"
LAYERED_PROBE_DEVICE="${LAYERED_PROBE_DEVICE:-cpu}"    # same/auto/cuda/cpu
LAYERED_PROBE_DTYPE="${LAYERED_PROBE_DTYPE:-auto}"     # auto/bf16/fp16/fp32
LAYERED_PROBE_OUTPUT="${LAYERED_PROBE_OUTPUT:-}"
LAYERED_PROBE_LOG_EVERY="${LAYERED_PROBE_LOG_EVERY:-1}"
LAYERED_PROBE_ONLY="${LAYERED_PROBE_ONLY:-0}"          # 1/0
WARMUP="${WARMUP:-1}"                         # 1/0
WARMUP_DECODE_MODE="${WARMUP_DECODE_MODE:-auto}" # auto/eagle/naive/hf
WARMUP_PROBE="${WARMUP_PROBE:-0}"             # 1/0
HARD_RESET_AFTER_WARMUP="${HARD_RESET_AFTER_WARMUP:-0}"   # 1/0
HARD_RESET_BEFORE_PROBE="${HARD_RESET_BEFORE_PROBE:-0}"   # 1/0
HARD_RESET_BEFORE_GENERATE="${HARD_RESET_BEFORE_GENERATE:-0}" # 1/0
LOG_STATE_SNAPSHOT="${LOG_STATE_SNAPSHOT:-0}" # 1/0
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}" # eager/sdpa/flash_attention_2
DEBUG_TOKEN_DUMP="${DEBUG_TOKEN_DUMP:-16}"
LOGITS_PROBE_SAMPLES="${LOGITS_PROBE_SAMPLES:-0}"
DIAG_LOG_FIRST_SAMPLES="${DIAG_LOG_FIRST_SAMPLES:-3}"
DIAG_LOG_INVALID_MAX="${DIAG_LOG_INVALID_MAX:-20}"
DIAG_RAW_PREVIEW_CHARS="${DIAG_RAW_PREVIEW_CHARS:-120}"
ABORT_ON_DEGENERATE="${ABORT_ON_DEGENERATE:-0}"   # 1/0
DEGENERATE_WINDOW="${DEGENERATE_WINDOW:-20}"
RESET_KV_PER_SAMPLE="${RESET_KV_PER_SAMPLE:-1}"   # 1/0
MAX_SAMPLES="${MAX_SAMPLES:-0}"

device_map_flag="--no-device-map-auto"
if [[ "${DEVICE_MAP_AUTO}" == "1" || "${DEVICE_MAP_AUTO}" == "true" ]]; then
  device_map_flag="--device-map-auto"
fi

abort_flag=""
if [[ "${ABORT_ON_DEGENERATE}" == "1" || "${ABORT_ON_DEGENERATE}" == "true" ]]; then
  abort_flag="--abort-on-degenerate"
fi

reset_kv_flag="--reset-kv-per-sample"
if [[ "${RESET_KV_PER_SAMPLE}" == "0" || "${RESET_KV_PER_SAMPLE}" == "false" ]]; then
  reset_kv_flag="--no-reset-kv-per-sample"
fi

warmup_flag="--warmup"
if [[ "${WARMUP}" == "0" || "${WARMUP}" == "false" ]]; then
  warmup_flag="--no-warmup"
fi

warmup_probe_flag=""
if [[ "${WARMUP_PROBE}" == "1" || "${WARMUP_PROBE}" == "true" ]]; then
  warmup_probe_flag="--warmup-probe"
fi

hard_reset_after_warmup_flag=""
if [[ "${HARD_RESET_AFTER_WARMUP}" == "1" || "${HARD_RESET_AFTER_WARMUP}" == "true" ]]; then
  hard_reset_after_warmup_flag="--hard-reset-after-warmup"
fi

hard_reset_before_probe_flag=""
if [[ "${HARD_RESET_BEFORE_PROBE}" == "1" || "${HARD_RESET_BEFORE_PROBE}" == "true" ]]; then
  hard_reset_before_probe_flag="--hard-reset-before-probe"
fi

hard_reset_before_generate_flag=""
if [[ "${HARD_RESET_BEFORE_GENERATE}" == "1" || "${HARD_RESET_BEFORE_GENERATE}" == "true" ]]; then
  hard_reset_before_generate_flag="--hard-reset-before-generate"
fi

log_state_snapshot_flag=""
if [[ "${LOG_STATE_SNAPSHOT}" == "1" || "${LOG_STATE_SNAPSHOT}" == "true" ]]; then
  log_state_snapshot_flag="--log-state-snapshot"
fi

layered_probe_only_flag=""
if [[ "${LAYERED_PROBE_ONLY}" == "1" || "${LAYERED_PROBE_ONLY}" == "true" ]]; then
  layered_probe_only_flag="--layered-probe-only"
fi

(
  cd "${EAGLE_DIR}"
  python -m eagle.evaluation.gen_ea_answer_qwen3_bird \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$EA_MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --answer-file "$ANSWER_FILE" \
    --max-new-token "$MAX_NEW_TOKEN" \
    --total-token "$TOTAL_TOKEN" \
    --depth "$DEPTH" \
    --top-k "$TOP_K" \
    --decode-mode "$DECODE_MODE" \
    --layered-probe-samples "$LAYERED_PROBE_SAMPLES" \
    --layered-probe-device "$LAYERED_PROBE_DEVICE" \
    --layered-probe-dtype "$LAYERED_PROBE_DTYPE" \
    --layered-probe-output "$LAYERED_PROBE_OUTPUT" \
    --layered-probe-log-every "$LAYERED_PROBE_LOG_EVERY" \
    ${layered_probe_only_flag} \
    ${warmup_flag} \
    --warmup-decode-mode "$WARMUP_DECODE_MODE" \
    ${warmup_probe_flag} \
    ${hard_reset_after_warmup_flag} \
    ${hard_reset_before_probe_flag} \
    ${hard_reset_before_generate_flag} \
    ${log_state_snapshot_flag} \
    --attn-implementation "$ATTN_IMPLEMENTATION" \
    --debug-token-dump "$DEBUG_TOKEN_DUMP" \
    --logits-probe-samples "$LOGITS_PROBE_SAMPLES" \
    --diag-log-first-samples "$DIAG_LOG_FIRST_SAMPLES" \
    --diag-log-invalid-max "$DIAG_LOG_INVALID_MAX" \
    --diag-raw-preview-chars "$DIAG_RAW_PREVIEW_CHARS" \
    --degenerate-window "$DEGENERATE_WINDOW" \
    ${abort_flag} \
    ${reset_kv_flag} \
    --temperature 0 \
    --device "$DEVICE" \
    ${device_map_flag} \
    --kv-cache-min-length "$KV_CACHE_MIN_LENGTH" \
    --kv-cache-margin "$KV_CACHE_MARGIN" \
    --max-samples "$MAX_SAMPLES" \
    --bf16
)
