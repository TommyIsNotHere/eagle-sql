#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Config (override via env vars)
# -----------------------------
ROOT_DIR="${ROOT_DIR:-/mnt/nj-aigc/usr/wangtong2/eagle_sql}"
EAGLE_DIR="${EAGLE_DIR:-${ROOT_DIR}/EAGLE-main}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT_DIR}/model/Qwen2.5-Coder-14B-Instruct}"
# Keep default in sync with run_bird_eagle3_train.sh SAVE_DIR.
EA_MODEL_PATH="${EA_MODEL_PATH:-${ROOT_DIR}/model/Qwen2.5-Coder-14B-Instruct_eagle3_head}"
BIRD_DEV_JSON="${BIRD_DEV_JSON:-${ROOT_DIR}/bird/dev_20240627/dev.json}"
BIRD_DEV_DB_ROOT="${BIRD_DEV_DB_ROOT:-${ROOT_DIR}/bird/dev_20240627/dev_databases}"

WORKDIR="${WORKDIR:-${ROOT_DIR}/artifacts/bird_dev_full_eval_qwen25}"
TIMEOUT_SEC="${TIMEOUT_SEC:-15}"
DEVICE="${DEVICE:-cuda}"                  # auto/cuda/cpu
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-0}"   # 1/0
KV_CACHE_MIN_LENGTH="${KV_CACHE_MIN_LENGTH:-3072}"
KV_CACHE_MARGIN="${KV_CACHE_MARGIN:-256}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-128}"
TOTAL_TOKEN="${TOTAL_TOKEN:-16}"
DEPTH="${DEPTH:-3}"
TOP_K="${TOP_K:-3}"
DECODE_MODE="${DECODE_MODE:-eagle}"           # eagle/naive/hf
LAYERED_PROBE_SAMPLES="${LAYERED_PROBE_SAMPLES:-0}"
LAYERED_PROBE_DEVICE="${LAYERED_PROBE_DEVICE:-cpu}"    # same/auto/cuda/cpu
LAYERED_PROBE_DTYPE="${LAYERED_PROBE_DTYPE:-auto}"     # auto/bf16/fp16/fp32
LAYERED_PROBE_OUTPUT="${LAYERED_PROBE_OUTPUT:-}"        # empty => pred_jsonl.layered_probe.json
LAYERED_PROBE_ONLY="${LAYERED_PROBE_ONLY:-0}"           # 1/0
WARMUP="${WARMUP:-1}"                         # 1/0
WARMUP_DECODE_MODE="${WARMUP_DECODE_MODE:-auto}" # auto/eagle/naive/hf
WARMUP_PROBE="${WARMUP_PROBE:-0}"             # 1/0
HARD_RESET_AFTER_WARMUP="${HARD_RESET_AFTER_WARMUP:-0}"   # 1/0
HARD_RESET_BEFORE_PROBE="${HARD_RESET_BEFORE_PROBE:-0}"   # 1/0
HARD_RESET_BEFORE_GENERATE="${HARD_RESET_BEFORE_GENERATE:-0}" # 1/0
LOG_STATE_SNAPSHOT="${LOG_STATE_SNAPSHOT:-0}" # 1/0
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}" # eager/sdpa/flash_attention_2
EAGLE_STRICT_HEAD_LOAD="${EAGLE_STRICT_HEAD_LOAD:-1}"
EAGLE_STRICT_DRAFT_MAP="${EAGLE_STRICT_DRAFT_MAP:-1}"
DEBUG_TOKEN_DUMP="${DEBUG_TOKEN_DUMP:-16}"
LOGITS_PROBE_SAMPLES="${LOGITS_PROBE_SAMPLES:-0}"
DIAG_LOG_FIRST_SAMPLES="${DIAG_LOG_FIRST_SAMPLES:-3}"
DIAG_LOG_INVALID_MAX="${DIAG_LOG_INVALID_MAX:-20}"
DIAG_RAW_PREVIEW_CHARS="${DIAG_RAW_PREVIEW_CHARS:-120}"
ABORT_ON_DEGENERATE="${ABORT_ON_DEGENERATE:-1}"   # 1/0
DEGENERATE_WINDOW="${DEGENERATE_WINDOW:-20}"
RESET_KV_PER_SAMPLE="${RESET_KV_PER_SAMPLE:-1}"   # 1/0
MAX_SAMPLES="${MAX_SAMPLES:-0}"

QUESTION_JSONL="${QUESTION_JSONL:-${WORKDIR}/bird_dev_infer.jsonl}"
PRED_JSONL="${PRED_JSONL:-${WORKDIR}/pred_dev.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORKDIR}/eval_summary.json}"
FAILURES_JSONL="${FAILURES_JSONL:-${WORKDIR}/eval_failures.jsonl}"
REPORT_MD="${REPORT_MD:-${WORKDIR}/eval_report.md}"
LOG_FILE="${LOG_FILE:-${WORKDIR}/run_bird_eagle3_full_eval.log}"

# Optional positional overrides for compatibility:
# bash run_bird_eagle3_full_eval.sh <base_model_path> <ea_model_path> <bird_dev_json> <bird_dev_db_root> <workdir> [timeout_sec]
if [[ $# -gt 0 ]]; then
  if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <base_model_path> <ea_model_path> <bird_dev_json> <bird_dev_db_root> <workdir> [timeout_sec]"
    exit 1
  fi
  BASE_MODEL_PATH="$1"
  EA_MODEL_PATH="$2"
  BIRD_DEV_JSON="$3"
  BIRD_DEV_DB_ROOT="$4"
  WORKDIR="$5"
  TIMEOUT_SEC="${6:-${TIMEOUT_SEC}}"

  QUESTION_JSONL="${WORKDIR}/bird_dev_infer.jsonl"
  PRED_JSONL="${WORKDIR}/pred_dev.jsonl"
  SUMMARY_JSON="${WORKDIR}/eval_summary.json"
  FAILURES_JSONL="${WORKDIR}/eval_failures.jsonl"
  REPORT_MD="${WORKDIR}/eval_report.md"
  LOG_FILE="${WORKDIR}/run_bird_eagle3_full_eval.log"
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

die() {
  echo "[$(date '+%F %T')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
  exit 1
}

is_valid_ea_ckpt_dir() {
  local d="$1"
  [[ -f "${d}/config.json" && ( -f "${d}/model.safetensors" || -f "${d}/pytorch_model.bin" ) ]]
}

has_ea_model_file() {
  local d="$1"
  [[ -f "${d}/model.safetensors" || -f "${d}/pytorch_model.bin" ]]
}

build_resolved_ea_config() {
  local template_path="$1"
  local out_path="$2"
  EA_CFG_TEMPLATE="${template_path}" \
  EA_CFG_OUT="${out_path}" \
  BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
  python - <<'PY'
import json
import os
import sys
from pathlib import Path

from transformers import AutoConfig

template = Path(os.environ["EA_CFG_TEMPLATE"])
out_path = Path(os.environ["EA_CFG_OUT"])
base_model_path = os.environ["BASE_MODEL_PATH"]

if not template.exists():
    raise SystemExit(f"template config not found: {template}")

with template.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

base_cfg = AutoConfig.from_pretrained(base_model_path)
sync_keys = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "hidden_act",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "pretraining_tp",
]
for key in sync_keys:
    if hasattr(base_cfg, key):
        cfg[key] = getattr(base_cfg, key)


def normalize_rope_scaling(value):
    if not isinstance(value, dict):
        return None
    scaling_type = value.get("type") or value.get("rope_type") or value.get("name")
    try:
        scaling_factor = float(value.get("factor"))
    except Exception:
        return None
    if scaling_type not in {"linear", "dynamic"}:
        return None
    if scaling_factor <= 1.0:
        return None
    return {"type": scaling_type, "factor": scaling_factor}


# EConfig in this repo only accepts {"type": "linear|dynamic", "factor": float>1}.
# Newer HF model configs may expose rope_scaling in incompatible formats
# (e.g. {"rope_type": "...", ...}); normalize or drop to avoid load-time failure.
rope_from_base = normalize_rope_scaling(getattr(base_cfg, "rope_scaling", None))
rope_from_template = normalize_rope_scaling(cfg.get("rope_scaling"))
cfg["rope_scaling"] = rope_from_base if rope_from_base is not None else rope_from_template

draft_vocab = int(cfg.get("draft_vocab_size", 32000))
vocab_size = int(cfg.get("vocab_size", getattr(base_cfg, "vocab_size", draft_vocab)))
cfg["draft_vocab_size"] = min(draft_vocab, vocab_size)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(
    "[resolved-ea-config] "
    f"template={template} out={out_path} "
    f"hidden={cfg.get('hidden_size')} inter={cfg.get('intermediate_size')} "
    f"heads={cfg.get('num_attention_heads')}/{cfg.get('num_key_value_heads')} "
    f"vocab={cfg.get('vocab_size')} draft_vocab={cfg.get('draft_vocab_size')} "
    f"rope_scaling={cfg.get('rope_scaling')}",
    file=sys.stderr,
)
PY
}

resolve_latest_state_with_model() {
  local p="$1"
  local best=""
  local best_idx=-1
  local sub=""
  if [[ -d "${p}" ]]; then
    for sub in "${p}"/state_*; do
      [[ -d "${sub}" ]] || continue
      local bn idx
      bn="$(basename "${sub}")"
      if [[ "${bn}" =~ ^state_([0-9]+)$ ]]; then
        idx="${BASH_REMATCH[1]}"
        if has_ea_model_file "${sub}" && (( idx > best_idx )); then
          best_idx="${idx}"
          best="${sub}"
        fi
      fi
    done
  fi

  echo "${best}"
}

resolve_ea_model_path() {
  local p="$1"
  local out_dir="${WORKDIR}/resolved_ea_model"

  if [[ -d "${p}" ]] && is_valid_ea_ckpt_dir "${p}"; then
    echo "${p}"
    return 0
  fi

  local model_dir=""
  local latest_state=""
  local parent_dir=""
  local config_file=""

  if [[ -d "${p}" ]] && has_ea_model_file "${p}"; then
    model_dir="${p}"
  fi

  if [[ -z "${model_dir}" ]]; then
    latest_state="$(resolve_latest_state_with_model "${p}")"
    if [[ -n "${latest_state}" ]]; then
      model_dir="${latest_state}"
    fi
  fi

  if [[ -z "${model_dir}" && "${p}" =~ /state_[0-9]+$ ]]; then
    parent_dir="$(dirname "${p}")"
    if [[ -d "${p}" ]] && has_ea_model_file "${p}"; then
      model_dir="${p}"
    else
      latest_state="$(resolve_latest_state_with_model "${parent_dir}")"
      if [[ -n "${latest_state}" ]]; then
        model_dir="${latest_state}"
      fi
    fi
  fi

  # config 搜索顺序：input -> model_dir -> input父目录 -> trianeagle3/config.json
  if [[ -f "${p}/config.json" ]]; then
    config_file="${p}/config.json"
  elif [[ -n "${model_dir}" && -f "${model_dir}/config.json" ]]; then
    config_file="${model_dir}/config.json"
  else
    parent_dir="$(dirname "${p}")"
    if [[ -f "${parent_dir}/config.json" ]]; then
      config_file="${parent_dir}/config.json"
    elif [[ -f "${EAGLE_DIR}/eagle/traineagle3/config.json" ]]; then
      config_file="${EAGLE_DIR}/eagle/traineagle3/config.json"
    fi
  fi

  if [[ -n "${model_dir}" && -n "${config_file}" ]]; then
    mkdir -p "${out_dir}"
    rm -f "${out_dir}/config.json" "${out_dir}/model.safetensors" "${out_dir}/pytorch_model.bin"
    build_resolved_ea_config "${config_file}" "${out_dir}/config.json"
    if [[ -f "${model_dir}/model.safetensors" ]]; then
      ln -sfn "${model_dir}/model.safetensors" "${out_dir}/model.safetensors"
    else
      ln -sfn "${model_dir}/pytorch_model.bin" "${out_dir}/pytorch_model.bin"
    fi
    echo "${out_dir}"
    return 0
  fi

  echo "${p}"
}

mkdir -p "${WORKDIR}"
: > "${LOG_FILE}"

for cmd in python bash tee; do
  command -v "${cmd}" >/dev/null 2>&1 || die "missing command: ${cmd}"
done

# Auto fallback for repository-local BIRD dev layout.
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

EA_MODEL_PATH_INPUT="${EA_MODEL_PATH}"
EA_MODEL_PATH="$(resolve_ea_model_path "${EA_MODEL_PATH_INPUT}")"

[[ -d "${EAGLE_DIR}" ]] || die "EAGLE_DIR not found: ${EAGLE_DIR}"
[[ -d "${BASE_MODEL_PATH}" ]] || die "BASE_MODEL_PATH not found: ${BASE_MODEL_PATH}"
[[ -d "${EA_MODEL_PATH}" ]] || die "EA_MODEL_PATH not found: ${EA_MODEL_PATH}"
[[ -f "${BIRD_DEV_JSON}" ]] || die "BIRD_DEV_JSON not found: ${BIRD_DEV_JSON}"
[[ -d "${BIRD_DEV_DB_ROOT}" ]] || die "BIRD_DEV_DB_ROOT not found: ${BIRD_DEV_DB_ROOT}"
is_valid_ea_ckpt_dir "${EA_MODEL_PATH}" || die "EA_MODEL_PATH invalid checkpoint dir (need config.json + model.safetensors/pytorch_model.bin): ${EA_MODEL_PATH}"

log "paths root=${ROOT_DIR} eagle=${EAGLE_DIR}"
if [[ "${EA_MODEL_PATH}" != "${EA_MODEL_PATH_INPUT}" ]]; then
  log "models base_model=${BASE_MODEL_PATH} ea_model_input=${EA_MODEL_PATH_INPUT}"
  log "models auto-resolved ea_model=${EA_MODEL_PATH}"
else
  log "models base_model=${BASE_MODEL_PATH} ea_model=${EA_MODEL_PATH}"
fi
log "data dev_json=${BIRD_DEV_JSON} dev_db_root=${BIRD_DEV_DB_ROOT}"
log "artifacts workdir=${WORKDIR}"
log "outputs question_jsonl=${QUESTION_JSONL} pred_jsonl=${PRED_JSONL}"
log "outputs summary_json=${SUMMARY_JSON} failures_jsonl=${FAILURES_JSONL} report_md=${REPORT_MD}"
log "eval timeout_sec=${TIMEOUT_SEC}"
log "infer runtime device=${DEVICE} device_map_auto=${DEVICE_MAP_AUTO} kv_cache_min_length=${KV_CACHE_MIN_LENGTH} kv_cache_margin=${KV_CACHE_MARGIN}"
log "infer decode mode=${DECODE_MODE} attn_impl=${ATTN_IMPLEMENTATION} max_new_token=${MAX_NEW_TOKEN} total_token=${TOTAL_TOKEN} depth=${DEPTH} top_k=${TOP_K} max_samples=${MAX_SAMPLES}"
log "infer layered_probe samples=${LAYERED_PROBE_SAMPLES} device=${LAYERED_PROBE_DEVICE} dtype=${LAYERED_PROBE_DTYPE} output=${LAYERED_PROBE_OUTPUT:-auto} probe_only=${LAYERED_PROBE_ONLY}"
log "infer warmup warmup=${WARMUP} warmup_mode=${WARMUP_DECODE_MODE} warmup_probe=${WARMUP_PROBE} hard_reset_after_warmup=${HARD_RESET_AFTER_WARMUP}"
log "infer state hard_reset_before_probe=${HARD_RESET_BEFORE_PROBE} hard_reset_before_generate=${HARD_RESET_BEFORE_GENERATE} log_state_snapshot=${LOG_STATE_SNAPSHOT}"
log "infer strict checks head_load=${EAGLE_STRICT_HEAD_LOAD} draft_map=${EAGLE_STRICT_DRAFT_MAP}"
log "infer debug token_dump=${DEBUG_TOKEN_DUMP} logits_probe_samples=${LOGITS_PROBE_SAMPLES} diag_first=${DIAG_LOG_FIRST_SAMPLES} diag_invalid_max=${DIAG_LOG_INVALID_MAX} diag_preview_chars=${DIAG_RAW_PREVIEW_CHARS} abort_on_degenerate=${ABORT_ON_DEGENERATE} degenerate_window=${DEGENERATE_WINDOW} reset_kv_per_sample=${RESET_KV_PER_SAMPLE}"

log "[1/4] prepare BIRD dev json -> jsonl"
(
  cd "${EAGLE_DIR}"
  python -m eagle.text2sql.bird.prep_bird \
    --input-json "${BIRD_DEV_JSON}" \
    --db-root "${BIRD_DEV_DB_ROOT}" \
    --output-jsonl "${QUESTION_JSONL}"
)

log "[2/4] run EAGLE3 accelerated inference"
DEVICE="${DEVICE}" \
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO}" \
KV_CACHE_MIN_LENGTH="${KV_CACHE_MIN_LENGTH}" \
KV_CACHE_MARGIN="${KV_CACHE_MARGIN}" \
MAX_NEW_TOKEN="${MAX_NEW_TOKEN}" \
TOTAL_TOKEN="${TOTAL_TOKEN}" \
DEPTH="${DEPTH}" \
TOP_K="${TOP_K}" \
DECODE_MODE="${DECODE_MODE}" \
LAYERED_PROBE_SAMPLES="${LAYERED_PROBE_SAMPLES}" \
LAYERED_PROBE_DEVICE="${LAYERED_PROBE_DEVICE}" \
LAYERED_PROBE_DTYPE="${LAYERED_PROBE_DTYPE}" \
LAYERED_PROBE_OUTPUT="${LAYERED_PROBE_OUTPUT}" \
LAYERED_PROBE_ONLY="${LAYERED_PROBE_ONLY}" \
WARMUP="${WARMUP}" \
WARMUP_DECODE_MODE="${WARMUP_DECODE_MODE}" \
WARMUP_PROBE="${WARMUP_PROBE}" \
HARD_RESET_AFTER_WARMUP="${HARD_RESET_AFTER_WARMUP}" \
HARD_RESET_BEFORE_PROBE="${HARD_RESET_BEFORE_PROBE}" \
HARD_RESET_BEFORE_GENERATE="${HARD_RESET_BEFORE_GENERATE}" \
LOG_STATE_SNAPSHOT="${LOG_STATE_SNAPSHOT}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
EAGLE_STRICT_HEAD_LOAD="${EAGLE_STRICT_HEAD_LOAD}" \
EAGLE_STRICT_DRAFT_MAP="${EAGLE_STRICT_DRAFT_MAP}" \
DEBUG_TOKEN_DUMP="${DEBUG_TOKEN_DUMP}" \
LOGITS_PROBE_SAMPLES="${LOGITS_PROBE_SAMPLES}" \
DIAG_LOG_FIRST_SAMPLES="${DIAG_LOG_FIRST_SAMPLES}" \
DIAG_LOG_INVALID_MAX="${DIAG_LOG_INVALID_MAX}" \
DIAG_RAW_PREVIEW_CHARS="${DIAG_RAW_PREVIEW_CHARS}" \
ABORT_ON_DEGENERATE="${ABORT_ON_DEGENERATE}" \
DEGENERATE_WINDOW="${DEGENERATE_WINDOW}" \
RESET_KV_PER_SAMPLE="${RESET_KV_PER_SAMPLE}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
bash "${SCRIPT_DIR}/run_bird_eagle3_infer.sh" \
  "${BASE_MODEL_PATH}" \
  "${EA_MODEL_PATH}" \
  "${QUESTION_JSONL}" \
  "${PRED_JSONL}"

if [[ "${LAYERED_PROBE_ONLY}" == "1" || "${LAYERED_PROBE_ONLY}" == "true" ]]; then
  log "probe-only enabled, skip execution eval and summary aggregation"
  if [[ -n "${LAYERED_PROBE_OUTPUT}" ]]; then
    log "layered_probe_output=${LAYERED_PROBE_OUTPUT}"
  else
    log "layered_probe_output=${PRED_JSONL}.layered_probe.json"
  fi
  exit 0
fi

log "[3/4] run execution evaluation (EX + acceptance aggregation)"
bash "${SCRIPT_DIR}/run_bird_eval_exec.sh" \
  "${QUESTION_JSONL}" \
  "${PRED_JSONL}" \
  "${BIRD_DEV_DB_ROOT}" \
  "${SUMMARY_JSON}" \
  "${FAILURES_JSONL}" \
  "${REPORT_MD}" \
  "${TIMEOUT_SEC}"

log "[4/4] key metrics"
python - "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
with summary_path.open("r", encoding="utf-8") as f:
    s = json.load(f)

out = {
    "exec_accuracy": s.get("exec_accuracy", 0.0),
    "acceptance_rate_mean": s.get("acceptance_rate_mean", 0.0),
    "acceptance_rate_token_weighted": s.get("acceptance_rate_token_weighted", 0.0),
    "pred_executable_rate": s.get("pred_executable_rate", 0.0),
    "accepted_tokens_sum": s.get("accepted_tokens_sum", 0),
    "proposed_tokens_sum": s.get("proposed_tokens_sum", 0),
    "summary_json": str(summary_path),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

log "DONE"
log "question_jsonl=${QUESTION_JSONL}"
log "pred_jsonl=${PRED_JSONL}"
log "summary_json=${SUMMARY_JSON}"
log "failures_jsonl=${FAILURES_JSONL}"
log "report_md=${REPORT_MD}"
