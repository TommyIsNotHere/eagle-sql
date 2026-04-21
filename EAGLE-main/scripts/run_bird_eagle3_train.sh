#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------
# Config (override via env vars)
# -----------------------------
ROOT_DIR="${ROOT_DIR:-/mnt/nj-aigc/usr/wangtong2/eagle_sql}"
EAGLE_DIR="${EAGLE_DIR:-${ROOT_DIR}/EAGLE-main}"

# Qwen2.5-Coder-14B-Instruct 与 prepare.sh 的模型目录保持同层级
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT_DIR}/model/Qwen2.5-Coder-14B-Instruct}"

BIRD_TRAIN_JSON="${BIRD_TRAIN_JSON:-${ROOT_DIR}/bird/train/train.json}"
BIRD_TRAIN_DB_ROOT="${BIRD_TRAIN_DB_ROOT:-${ROOT_DIR}/bird/train/train_databases}"

WORKDIR="${WORKDIR:-${ROOT_DIR}/artifacts/bird_train_qwen25}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/model/Qwen2.5-Coder-14B-Instruct_eagle3_head}"
DS_CONFIG="${DS_CONFIG:-${EAGLE_DIR}/eagle/traineagle3/ds_config.json}"
DS_RUNTIME_CONFIG="${DS_RUNTIME_CONFIG:-${WORKDIR}/ds_config.runtime.json}"

EVAL_RATIO="${EVAL_RATIO:-0.02}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
MAX_LEN="${MAX_LEN:-2048}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREPROCESS_NUM_PROC="${PREPROCESS_NUM_PROC:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
SAVE_DS_EVERY_EPOCHS="${SAVE_DS_EVERY_EPOCHS:-10}"
AUTO_FIX_PYDANTIC="${AUTO_FIX_PYDANTIC:-0}"
REQUIRE_QWEN_KV="${REQUIRE_QWEN_KV:-1}"
EAGLE_USE_PADDING_ATTN_MASK="${EAGLE_USE_PADDING_ATTN_MASK:-0}"
EAGLE_DEBUG_NUMERICS_STEPS="${EAGLE_DEBUG_NUMERICS_STEPS:-3}"
EAGLE_FORCE_TEACHER_ATTN_MASK="${EAGLE_FORCE_TEACHER_ATTN_MASK:-1}"
EAGLE_DEBUG_TEACHER_NUMERICS_STEPS="${EAGLE_DEBUG_TEACHER_NUMERICS_STEPS:-3}"
EAGLE_TEACHER_MASK_FALLBACK_STEPS="${EAGLE_TEACHER_MASK_FALLBACK_STEPS:-1000000}"
# Keep teacher forward path aligned with inference runtime by default.
EAGLE_QWEN_TEACHER_IMPL="${EAGLE_QWEN_TEACHER_IMPL:-kv}"
EAGLE_STRICT_TEACHER_IMPL="${EAGLE_STRICT_TEACHER_IMPL:-1}"
EAGLE_TEACHER_HIDDEN_SELECTOR="${EAGLE_TEACHER_HIDDEN_SELECTOR:-paper}" # legacy/paper/custom
EAGLE_TEACHER_HIDDEN_CUSTOM="${EAGLE_TEACHER_HIDDEN_CUSTOM:-}"            # custom mode only, e.g. "1,24,48"
EAGLE_TRAIN_SYSTEM_PROMPT="${EAGLE_TRAIN_SYSTEM_PROMPT:-bird}" # auto/bird/sql/generic or custom prompt text
EAGLE_GOLD_CE_WEIGHT="${EAGLE_GOLD_CE_WEIGHT:-0.35}"
EAGLE_DISTILL_ONLY_IN_DRAFT="${EAGLE_DISTILL_ONLY_IN_DRAFT:-1}"
EAGLE_LOSS_MODE="${EAGLE_LOSS_MODE:-hybrid}" # hybrid/paper
# Paper-aligned default: enable training-time test via self rollout.
EAGLE_INPUT_ROLLOUT_MODE="${EAGLE_INPUT_ROLLOUT_MODE:-pred}" # teacher/pred/scheduled
EAGLE_INPUT_ROLLOUT_RATIO_START="${EAGLE_INPUT_ROLLOUT_RATIO_START:-0.0}"
EAGLE_INPUT_ROLLOUT_RATIO_END="${EAGLE_INPUT_ROLLOUT_RATIO_END:-0.3}"
EAGLE_INPUT_ROLLOUT_ALIGN_TARGET="${EAGLE_INPUT_ROLLOUT_ALIGN_TARGET:-1}"
EAGLE_TRACE_NONFINITE_STEPS="${EAGLE_TRACE_NONFINITE_STEPS:-8}"
EAGLE_CHECK_TARGET_PARAM_FINITE="${EAGLE_CHECK_TARGET_PARAM_FINITE:-0}"
EAGLE_FAIL_FAST_ON_MODEL_NONFINITE="${EAGLE_FAIL_FAST_ON_MODEL_NONFINITE:-1}"
# Default to fresh training start to avoid accidentally resuming from corrupted checkpoints.
# Set to 0 when you explicitly want auto-resume from SAVE_DIR.
EAGLE_DISABLE_AUTORESUME="${EAGLE_DISABLE_AUTORESUME:-1}"

# Runtime guardrails / diagnostics
EAGLE_DIAG_MODE="${EAGLE_DIAG_MODE:-0}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-1}"
EMPTY_CACHE_EVERY_STEPS="${EMPTY_CACHE_EVERY_STEPS:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-0}"
GLOBAL_STALL_SECONDS="${GLOBAL_STALL_SECONDS:-1200}"
ABORT_ON_GLOBAL_STALL="${ABORT_ON_GLOBAL_STALL:-1}"
STALL_DIAGNOSTICS="${STALL_DIAGNOSTICS:-1}"
STALL_DIAGNOSTICS_COOLDOWN="${STALL_DIAGNOSTICS_COOLDOWN:-300}"

# DeepSpeed ZeRO communication safety knobs.
# overlap_comm=true can improve throughput but is a common source of late-stage NCCL hangs.
DS_OVERLAP_COMM="${DS_OVERLAP_COMM:-0}"
DS_CONTIGUOUS_GRADIENTS="${DS_CONTIGUOUS_GRADIENTS:-1}"
DS_REDUCE_BUCKET_SIZE="${DS_REDUCE_BUCKET_SIZE:-}"
DS_ALLGATHER_BUCKET_SIZE="${DS_ALLGATHER_BUCKET_SIZE:-}"

# Distributed runtime debug knobs.
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"
NCCL_TRACE_COLL="${NCCL_TRACE_COLL:-0}"
PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

if [[ "${EAGLE_DIAG_MODE}" == "1" ]]; then
  NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  EAGLE_LOG_ALL_RANKS="${EAGLE_LOG_ALL_RANKS:-0}"
  HEARTBEAT_STEPS="${HEARTBEAT_STEPS:-20}"
  SLOW_STAGE_SECONDS="${SLOW_STAGE_SECONDS:-10}"
  CUDA_SYNC_HEARTBEAT_STEPS="${CUDA_SYNC_HEARTBEAT_STEPS:-20}"
else
  NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
  EAGLE_LOG_ALL_RANKS="${EAGLE_LOG_ALL_RANKS:-0}"
  HEARTBEAT_STEPS="${HEARTBEAT_STEPS:-100}"
  SLOW_STAGE_SECONDS="${SLOW_STAGE_SECONDS:-30}"
  CUDA_SYNC_HEARTBEAT_STEPS="${CUDA_SYNC_HEARTBEAT_STEPS:-0}"
fi

# 默认不打开 COLL 细粒度追踪，避免 NCCL INFO AllReduce 大量刷屏。
# 需要时显式设置：NCCL_TRACE_COLL=1
if [[ "${NCCL_TRACE_COLL}" == "1" && "${NCCL_DEBUG_SUBSYS}" != *"COLL"* ]]; then
  NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS},COLL"
fi

TRAIN_PREP_JSONL="${TRAIN_PREP_JSONL:-${WORKDIR}/bird_train_prep.jsonl}"
TRAIN_CHAT_JSONL="${TRAIN_CHAT_JSONL:-${WORKDIR}/bird_train_chat.jsonl}"
EVAL_CHAT_JSONL="${EVAL_CHAT_JSONL:-${WORKDIR}/bird_eval_chat.jsonl}"

LOG_FILE="${LOG_FILE:-${WORKDIR}/run_bird_eagle3_train.log}"

# -----------------------------
# Helpers
# -----------------------------
log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

die() {
  echo "[$(date '+%F %T')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
  exit 1
}

# -----------------------------
# Preflight
# -----------------------------
mkdir -p "${WORKDIR}" "${SAVE_DIR}"
: > "${LOG_FILE}"

for cmd in python bash deepspeed tee wc; do
  command -v "${cmd}" >/dev/null 2>&1 || die "missing command: ${cmd}"
done

# Ensure Triton cache points to a writable location to avoid noisy cache checks.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKDIR}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}/autotune"

# Reduce noisy PyTorch extension warning by pinning CUDA arch list when possible.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST_DETECTED="$(
    python - <<'PY' 2>/dev/null || true
import torch

if not torch.cuda.is_available():
    raise SystemExit(0)

seen = []
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    arch = f"{major}.{minor}"
    if arch not in seen:
        seen.append(arch)

print(";".join(seen))
PY
  )"
  if [[ -n "${TORCH_CUDA_ARCH_LIST_DETECTED}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_DETECTED}"
  fi
fi

# Ensure absolute package imports like `eagle.*` work under deepspeed subdir launch.
export PYTHONPATH="${EAGLE_DIR}:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export NCCL_BLOCKING_WAIT
export NCCL_DEBUG
export NCCL_DEBUG_SUBSYS
export TORCH_DISTRIBUTED_DEBUG
export PYTHONFAULTHANDLER
export EAGLE_LOG_ALL_RANKS

log "paths root=${ROOT_DIR} eagle=${EAGLE_DIR} base_model=${BASE_MODEL_PATH}"
log "data train_json=${BIRD_TRAIN_JSON} train_db_root=${BIRD_TRAIN_DB_ROOT}"
log "artifacts workdir=${WORKDIR} save_dir=${SAVE_DIR} log=${LOG_FILE}"
log "train epochs=${NUM_EPOCHS} max_len=${MAX_LEN} workers=${NUM_WORKERS} preprocess_num_proc=${PREPROCESS_NUM_PROC} eval_ratio=${EVAL_RATIO} seed=${SEED}"
log "optim warmup_ratio=${WARMUP_RATIO} save_every_epochs=${SAVE_EVERY_EPOCHS} save_ds_every_epochs=${SAVE_DS_EVERY_EPOCHS}"
log "runtime diag=${EAGLE_DIAG_MODE} heartbeat=${HEARTBEAT_STEPS} slow_stage_s=${SLOW_STAGE_SECONDS} dataloader_timeout_s=${DATALOADER_TIMEOUT} pin_memory=${PIN_MEMORY} persistent_workers=${PERSISTENT_WORKERS} prefetch_factor=${PREFETCH_FACTOR} in_order=${DATALOADER_IN_ORDER}"
log "runtime extras max_train_steps=${MAX_TRAIN_STEPS} max_eval_steps=${MAX_EVAL_STEPS} empty_cache_every_steps=${EMPTY_CACHE_EVERY_STEPS} cuda_sync_heartbeat_steps=${CUDA_SYNC_HEARTBEAT_STEPS} log_all_ranks=${EAGLE_LOG_ALL_RANKS}"
log "runtime stall_guard global_stall_seconds=${GLOBAL_STALL_SECONDS} abort_on_global_stall=${ABORT_ON_GLOBAL_STALL} stall_diagnostics=${STALL_DIAGNOSTICS} stall_diag_cooldown=${STALL_DIAGNOSTICS_COOLDOWN}"
log "deepspeed zero overlap_comm=${DS_OVERLAP_COMM} contiguous_gradients=${DS_CONTIGUOUS_GRADIENTS} reduce_bucket=${DS_REDUCE_BUCKET_SIZE:-<keep-default>} allgather_bucket=${DS_ALLGATHER_BUCKET_SIZE:-<keep-default>}"
log "distributed nccl_debug=${NCCL_DEBUG} nccl_debug_subsys=${NCCL_DEBUG_SUBSYS} nccl_trace_coll=${NCCL_TRACE_COLL} torch_dist_debug=${TORCH_DISTRIBUTED_DEBUG} blocking_wait=${NCCL_BLOCKING_WAIT}"
log "safety fail_fast_nonfinite=${EAGLE_FAIL_FAST_ON_MODEL_NONFINITE} disable_autoresume=${EAGLE_DISABLE_AUTORESUME} require_qwen_kv=${REQUIRE_QWEN_KV} use_padding_attn_mask=${EAGLE_USE_PADDING_ATTN_MASK}"
log "alignment teacher_impl=${EAGLE_QWEN_TEACHER_IMPL} strict_teacher_impl=${EAGLE_STRICT_TEACHER_IMPL} teacher_hidden_selector=${EAGLE_TEACHER_HIDDEN_SELECTOR} teacher_hidden_custom=${EAGLE_TEACHER_HIDDEN_CUSTOM:-<none>} train_system_prompt=${EAGLE_TRAIN_SYSTEM_PROMPT}"
log "alignment loss_mode=${EAGLE_LOSS_MODE} gold_ce_weight=${EAGLE_GOLD_CE_WEIGHT} distill_only_in_draft=${EAGLE_DISTILL_ONLY_IN_DRAFT} rollout_mode=${EAGLE_INPUT_ROLLOUT_MODE} rollout_ratio=(${EAGLE_INPUT_ROLLOUT_RATIO_START}->${EAGLE_INPUT_ROLLOUT_RATIO_END}) rollout_align_target=${EAGLE_INPUT_ROLLOUT_ALIGN_TARGET}"

[[ -d "${EAGLE_DIR}" ]] || die "EAGLE_DIR not found: ${EAGLE_DIR}"
[[ -d "${BASE_MODEL_PATH}" ]] || die "BASE_MODEL_PATH not found: ${BASE_MODEL_PATH}"

# Auto-fallback to repository-local BIRD train layout when ROOT_DIR/bird is absent.
if [[ ! -f "${BIRD_TRAIN_JSON}" ]]; then
  ALT_TRAIN_JSON="${EAGLE_DIR}/eagle/bird/train/train.json"
  if [[ -f "${ALT_TRAIN_JSON}" ]]; then
    BIRD_TRAIN_JSON="${ALT_TRAIN_JSON}"
    log "fallback BIRD_TRAIN_JSON=${BIRD_TRAIN_JSON}"
  fi
fi
if [[ ! -d "${BIRD_TRAIN_DB_ROOT}" ]]; then
  ALT_TRAIN_DB_ROOT="${EAGLE_DIR}/eagle/bird/train/train_databases"
  if [[ -d "${ALT_TRAIN_DB_ROOT}" ]]; then
    BIRD_TRAIN_DB_ROOT="${ALT_TRAIN_DB_ROOT}"
    log "fallback BIRD_TRAIN_DB_ROOT=${BIRD_TRAIN_DB_ROOT}"
  fi
fi

[[ -f "${BIRD_TRAIN_JSON}" ]] || die "BIRD_TRAIN_JSON not found: ${BIRD_TRAIN_JSON}"
[[ -d "${BIRD_TRAIN_DB_ROOT}" ]] || die "BIRD_TRAIN_DB_ROOT not found: ${BIRD_TRAIN_DB_ROOT}"
[[ -f "${DS_CONFIG}" ]] || die "deepspeed config not found: ${DS_CONFIG}"

# Build runtime DeepSpeed config so we can enforce safer ZeRO comm defaults
# and tune communication buckets without mutating source ds_config.json.
DS_CONFIG_EFFECTIVE="${DS_RUNTIME_CONFIG}"
log "step0: build runtime deepspeed config -> ${DS_CONFIG_EFFECTIVE}"
DS_CONFIG="${DS_CONFIG}" \
DS_RUNTIME_CONFIG="${DS_RUNTIME_CONFIG}" \
DS_OVERLAP_COMM="${DS_OVERLAP_COMM}" \
DS_CONTIGUOUS_GRADIENTS="${DS_CONTIGUOUS_GRADIENTS}" \
DS_REDUCE_BUCKET_SIZE="${DS_REDUCE_BUCKET_SIZE}" \
DS_ALLGATHER_BUCKET_SIZE="${DS_ALLGATHER_BUCKET_SIZE}" \
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}" || die "failed to build runtime deepspeed config"
import json
import os
from pathlib import Path


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


src = Path(os.environ["DS_CONFIG"])
dst = Path(os.environ["DS_RUNTIME_CONFIG"])

with src.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

zero = cfg.setdefault("zero_optimization", {})
zero["overlap_comm"] = to_bool(os.environ.get("DS_OVERLAP_COMM", "0"))
zero["contiguous_gradients"] = to_bool(os.environ.get("DS_CONTIGUOUS_GRADIENTS", "1"))

reduce_bucket = os.environ.get("DS_REDUCE_BUCKET_SIZE", "").strip()
if reduce_bucket:
    zero["reduce_bucket_size"] = float(reduce_bucket)

allgather_bucket = os.environ.get("DS_ALLGATHER_BUCKET_SIZE", "").strip()
if allgather_bucket:
    zero["allgather_bucket_size"] = float(allgather_bucket)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(
    "[runtime-ds-config] "
    f"src={src} dst={dst} "
    f"overlap_comm={zero.get('overlap_comm')} "
    f"contiguous_gradients={zero.get('contiguous_gradients')} "
    f"reduce_bucket_size={zero.get('reduce_bucket_size')} "
    f"allgather_bucket_size={zero.get('allgather_bucket_size')}"
)
PY
[[ -f "${DS_CONFIG_EFFECTIVE}" ]] || die "runtime deepspeed config missing: ${DS_CONFIG_EFFECTIVE}"
log "DS_CONFIG_EFFECTIVE=${DS_CONFIG_EFFECTIVE}"

# Python dependency preflight for DeepSpeed runtime.
CHECK_ERR_MSG=""
if ! CHECK_ERR_MSG="$(python - <<'PY' 2>&1
try:
    import pydantic
except Exception as e:
    raise SystemExit(f"pydantic import failed: {e}")

version = getattr(pydantic, "__version__", "0.0.0")
major = int(version.split(".", 1)[0])
if major < 2:
    raise SystemExit(
        f"Incompatible pydantic version: {version}. "
        "DeepSpeed in this environment requires pydantic>=2 "
        "(missing symbol: model_validator)."
    )

if not hasattr(pydantic, "model_validator"):
    raise SystemExit(
        f"pydantic {version} is missing model_validator. "
        "Please install a pydantic v2 release."
    )

try:
    import deepspeed  # noqa: F401
except Exception as e:
    raise SystemExit(f"deepspeed import failed: {e}")

print(f"[preflight] pydantic={version} OK, deepspeed import OK")
PY
 )"; then
  echo "${CHECK_ERR_MSG}" | tee -a "${LOG_FILE}" >&2
  if [[ "${AUTO_FIX_PYDANTIC}" == "1" ]]; then
    log "auto-fix: upgrading pydantic to v2 for deepspeed compatibility"
    python -m pip install -U "pydantic>=2,<3" 2>&1 | tee -a "${LOG_FILE}" \
      || die "auto-fix failed: unable to install pydantic>=2,<3"
    python - <<'PY' 2>&1 | tee -a "${LOG_FILE}" || die "post-fix preflight still failing"
import pydantic
import deepspeed  # noqa: F401
print(f"[preflight-after-fix] pydantic={pydantic.__version__}, deepspeed import OK")
PY
  else
    die "python dependency preflight failed. Fix with: python -m pip install -U \"pydantic>=2,<3\"; then rerun (or set AUTO_FIX_PYDANTIC=1)."
  fi
else
  echo "${CHECK_ERR_MSG}" | tee -a "${LOG_FILE}"
fi
cd "${EAGLE_DIR}"

# -----------------------------
# Run
# -----------------------------
log "step1: prepare BIRD train json -> normalized jsonl"
python -m eagle.text2sql.bird.prep_bird \
  --input-json "${BIRD_TRAIN_JSON}" \
  --db-root "${BIRD_TRAIN_DB_ROOT}" \
  --output-jsonl "${TRAIN_PREP_JSONL}" 2>&1 | tee -a "${LOG_FILE}"

[[ -f "${TRAIN_PREP_JSONL}" ]] || die "prepared train jsonl missing: ${TRAIN_PREP_JSONL}"
TRAIN_PREP_LINES="$(wc -l < "${TRAIN_PREP_JSONL}" | tr -d ' ')"
[[ "${TRAIN_PREP_LINES}" -gt 0 ]] || die "prepared train jsonl is empty: ${TRAIN_PREP_JSONL}"
log "prepared rows=${TRAIN_PREP_LINES}"

log "step2: build EAGLE3 SFT chat jsonl (train/eval split)"
python -m eagle.text2sql.bird.build_bird_eagle3_sft \
  --input-jsonl "${TRAIN_PREP_JSONL}" \
  --train-output-jsonl "${TRAIN_CHAT_JSONL}" \
  --eval-output-jsonl "${EVAL_CHAT_JSONL}" \
  --eval-ratio "${EVAL_RATIO}" \
  --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

[[ -f "${TRAIN_CHAT_JSONL}" ]] || die "train chat jsonl missing: ${TRAIN_CHAT_JSONL}"
[[ -f "${EVAL_CHAT_JSONL}" ]] || die "eval chat jsonl missing: ${EVAL_CHAT_JSONL}"
TRAIN_CHAT_LINES="$(wc -l < "${TRAIN_CHAT_JSONL}" | tr -d ' ')"
EVAL_CHAT_LINES="$(wc -l < "${EVAL_CHAT_JSONL}" | tr -d ' ')"
[[ "${TRAIN_CHAT_LINES}" -gt 0 ]] || die "train chat jsonl is empty: ${TRAIN_CHAT_JSONL}"
log "train chat rows=${TRAIN_CHAT_LINES}, eval chat rows=${EVAL_CHAT_LINES}"

log "step2.5: sanity-check SFT labels (SQL start/fence/backtick stats)"
TRAIN_CHAT_JSONL="${TRAIN_CHAT_JSONL}" \
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}" || die "SFT sanity check failed"
import json
import os
import re
from collections import Counter
from pathlib import Path

path = Path(os.environ["TRAIN_CHAT_JSONL"])
rows = 0
bad_start = 0
triple_fence = 0
backtick_rows = 0
first_counter = Counter()
start_re = re.compile(r"(?is)^\s*(select|with|insert|update|delete)\b")

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        conv = rec.get("conversations") or []
        if len(conv) < 2:
            continue
        sql = str(conv[1].get("value", "")).strip()
        rows += 1
        if "```" in sql:
            triple_fence += 1
        if "`" in sql:
            backtick_rows += 1
        m = start_re.search(sql)
        if not m:
            bad_start += 1
        else:
            first_counter[m.group(1).upper()] += 1

print(
    "[sft-sanity] "
    f"rows={rows} bad_start={bad_start} triple_fence={triple_fence} "
    f"backtick_rows={backtick_rows} first_tokens={dict(first_counter)}"
)

if rows <= 0:
    raise SystemExit("no usable rows found")
if bad_start > 0:
    raise SystemExit(f"non-SQL leading labels detected: {bad_start}/{rows}")
if triple_fence > 0:
    raise SystemExit(f"triple-backtick labels detected: {triple_fence}/{rows}")
PY

log "step3: train EAGLE3 head (DeepSpeed)"
(
  cd eagle/traineagle3
  EAGLE_REQUIRE_QWEN_KV="${REQUIRE_QWEN_KV}" \
  EAGLE_USE_PADDING_ATTN_MASK="${EAGLE_USE_PADDING_ATTN_MASK}" \
  EAGLE_DEBUG_NUMERICS_STEPS="${EAGLE_DEBUG_NUMERICS_STEPS}" \
  EAGLE_FORCE_TEACHER_ATTN_MASK="${EAGLE_FORCE_TEACHER_ATTN_MASK}" \
  EAGLE_DEBUG_TEACHER_NUMERICS_STEPS="${EAGLE_DEBUG_TEACHER_NUMERICS_STEPS}" \
  EAGLE_TEACHER_MASK_FALLBACK_STEPS="${EAGLE_TEACHER_MASK_FALLBACK_STEPS}" \
  EAGLE_QWEN_TEACHER_IMPL="${EAGLE_QWEN_TEACHER_IMPL}" \
  EAGLE_STRICT_TEACHER_IMPL="${EAGLE_STRICT_TEACHER_IMPL}" \
  EAGLE_TEACHER_HIDDEN_SELECTOR="${EAGLE_TEACHER_HIDDEN_SELECTOR}" \
  EAGLE_TEACHER_HIDDEN_CUSTOM="${EAGLE_TEACHER_HIDDEN_CUSTOM}" \
  EAGLE_GOLD_CE_WEIGHT="${EAGLE_GOLD_CE_WEIGHT}" \
  EAGLE_DISTILL_ONLY_IN_DRAFT="${EAGLE_DISTILL_ONLY_IN_DRAFT}" \
  EAGLE_LOSS_MODE="${EAGLE_LOSS_MODE}" \
  EAGLE_INPUT_ROLLOUT_MODE="${EAGLE_INPUT_ROLLOUT_MODE}" \
  EAGLE_INPUT_ROLLOUT_RATIO_START="${EAGLE_INPUT_ROLLOUT_RATIO_START}" \
  EAGLE_INPUT_ROLLOUT_RATIO_END="${EAGLE_INPUT_ROLLOUT_RATIO_END}" \
  EAGLE_INPUT_ROLLOUT_ALIGN_TARGET="${EAGLE_INPUT_ROLLOUT_ALIGN_TARGET}" \
  EAGLE_TRACE_NONFINITE_STEPS="${EAGLE_TRACE_NONFINITE_STEPS}" \
  EAGLE_CHECK_TARGET_PARAM_FINITE="${EAGLE_CHECK_TARGET_PARAM_FINITE}" \
  EAGLE_FAIL_FAST_ON_MODEL_NONFINITE="${EAGLE_FAIL_FAST_ON_MODEL_NONFINITE}" \
  EAGLE_DISABLE_AUTORESUME="${EAGLE_DISABLE_AUTORESUME}" \
  deepspeed main.py \
    --basepath "${BASE_MODEL_PATH}" \
    --trainpath "${TRAIN_CHAT_JSONL}" \
    --testpath "${EVAL_CHAT_JSONL}" \
    --savedir "${SAVE_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --max-len "${MAX_LEN}" \
    --num-workers "${NUM_WORKERS}" \
    --preprocess-num-proc "${PREPROCESS_NUM_PROC}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
    --save-ds-every-epochs "${SAVE_DS_EVERY_EPOCHS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --pin-memory "${PIN_MEMORY}" \
    --persistent-workers "${PERSISTENT_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --dataloader-in-order "${DATALOADER_IN_ORDER}" \
    --heartbeat-steps "${HEARTBEAT_STEPS}" \
    --slow-stage-seconds "${SLOW_STAGE_SECONDS}" \
    --empty-cache-every-steps "${EMPTY_CACHE_EVERY_STEPS}" \
    --cuda-sync-heartbeat-steps "${CUDA_SYNC_HEARTBEAT_STEPS}" \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    --max-eval-steps "${MAX_EVAL_STEPS}" \
    --global-stall-seconds "${GLOBAL_STALL_SECONDS}" \
    --abort-on-global-stall "${ABORT_ON_GLOBAL_STALL}" \
    --stall-diagnostics "${STALL_DIAGNOSTICS}" \
    --stall-diagnostics-cooldown "${STALL_DIAGNOSTICS_COOLDOWN}" \
    --system-prompt "${EAGLE_TRAIN_SYSTEM_PROMPT}" \
    --deepspeed_config "${DS_CONFIG_EFFECTIVE}"
) 2>&1 | tee -a "${LOG_FILE}"

log "SUCCESS"
log "train chat jsonl: ${TRAIN_CHAT_JSONL}"
log "eval chat jsonl:  ${EVAL_CHAT_JSONL}"
log "checkpoint dir:   ${SAVE_DIR}"
