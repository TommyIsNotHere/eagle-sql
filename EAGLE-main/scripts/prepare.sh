#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------
# Config (override via env vars)
# -----------------------------
ROOT_DIR="${ROOT_DIR:-/mnt/nj-aigc/usr/wangtong2/eagle_sql}"
EAGLE_DIR="${EAGLE_DIR:-${ROOT_DIR}/EAGLE-main}"

INPUT_JSON="${INPUT_JSON:-${ROOT_DIR}/bird/dev_20240627/dev.json}"
DB_ROOT="${DB_ROOT:-${ROOT_DIR}/bird/dev_20240627/dev_databases}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT_DIR}/model/Qwen3-14B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-${ROOT_DIR}/model/Qwen3-14B_eagle3}"
MAX_DB_DESC_CHARS="${MAX_DB_DESC_CHARS:-6000}"

WORKDIR="${WORKDIR:-${ROOT_DIR}/artifacts/bird_smoke}"
INFER_JSONL="${INFER_JSONL:-${WORKDIR}/bird_dev_infer.jsonl}"
PRED_JSONL="${PRED_JSONL:-${WORKDIR}/pred_dev.jsonl}"

RUN_SCRIPT="${RUN_SCRIPT:-${EAGLE_DIR}/scripts/run_bird_eagle3_infer.sh}"
GEN_SCRIPT="${GEN_SCRIPT:-${EAGLE_DIR}/eagle/evaluation/gen_ea_answer_qwen3_bird.py}"

# Optional: set GEN_SHA256 to enforce exact file integrity
# export GEN_SHA256="xxxxxxxx..."
GEN_SHA256="${GEN_SHA256:-}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${WORKDIR}/prepare_${TS}.log}"
LOCK_DIR="${LOCK_DIR:-${WORKDIR}/.prepare.lock}"

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

cleanup() {
  rm -rf "${LOCK_DIR}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# -----------------------------
# Preflight
# -----------------------------
mkdir -p "${WORKDIR}"
: > "${LOG_FILE}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  die "another prepare process is running (lock: ${LOCK_DIR})"
fi

for cmd in python bash rg wc tee sha256sum; do
  command -v "${cmd}" >/dev/null 2>&1 || die "missing command: ${cmd}"
done

log "ROOT_DIR=${ROOT_DIR}"
log "EAGLE_DIR=${EAGLE_DIR}"
log "INPUT_JSON=${INPUT_JSON}"
log "DB_ROOT=${DB_ROOT}"
log "MAX_DB_DESC_CHARS=${MAX_DB_DESC_CHARS}"
log "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
log "EA_MODEL_PATH=${EA_MODEL_PATH}"
log "INFER_JSONL=${INFER_JSONL}"
log "PRED_JSONL=${PRED_JSONL}"
log "LOG_FILE=${LOG_FILE}"

[[ -d "${EAGLE_DIR}" ]] || die "EAGLE_DIR not found: ${EAGLE_DIR}"
# Auto-fallback to repository-local BIRD data layout when ROOT_DIR/bird is absent.
if [[ ! -f "${INPUT_JSON}" ]]; then
  ALT_INPUT_JSON="${EAGLE_DIR}/eagle/bird/dev_20240627/dev.json"
  if [[ -f "${ALT_INPUT_JSON}" ]]; then
    INPUT_JSON="${ALT_INPUT_JSON}"
    log "fallback INPUT_JSON=${INPUT_JSON}"
  fi
fi
if [[ ! -d "${DB_ROOT}" ]]; then
  ALT_DB_ROOT="${EAGLE_DIR}/eagle/bird/dev_20240627/dev_databases"
  if [[ -d "${ALT_DB_ROOT}" ]]; then
    DB_ROOT="${ALT_DB_ROOT}"
    log "fallback DB_ROOT=${DB_ROOT}"
  fi
fi

[[ -f "${INPUT_JSON}" ]] || die "INPUT_JSON not found: ${INPUT_JSON}"
[[ -f "${RUN_SCRIPT}" ]] || die "run script not found: ${RUN_SCRIPT}"
[[ -f "${GEN_SCRIPT}" ]] || die "gen script not found: ${GEN_SCRIPT}"

# Guard against accidental overwrite (wrong file content)
rg -q 'def main|if __name__ == "__main__"' "${GEN_SCRIPT}" \
  || die "GEN script seems broken (missing main entry): ${GEN_SCRIPT}"

# Optional strict integrity check
if [[ -n "${GEN_SHA256}" ]]; then
  ACTUAL_SHA="$(sha256sum "${GEN_SCRIPT}" | awk '{print $1}')"
  [[ "${ACTUAL_SHA}" == "${GEN_SHA256}" ]] \
    || die "GEN script sha256 mismatch. expected=${GEN_SHA256}, actual=${ACTUAL_SHA}"
fi

# -----------------------------
# Run
# -----------------------------
cd "${EAGLE_DIR}"

log "step1: build infer jsonl"
PREP_ARGS=(
  --input-json "${INPUT_JSON}"
  --output-jsonl "${INFER_JSONL}"
  --max-db-desc-chars "${MAX_DB_DESC_CHARS}"
)
if [[ -d "${DB_ROOT}" ]]; then
  PREP_ARGS+=(--db-root "${DB_ROOT}")
else
  log "WARN: DB_ROOT not found (${DB_ROOT}); schema_context may be empty."
fi
python -m eagle.text2sql.bird.prep_bird "${PREP_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

[[ -f "${INFER_JSONL}" ]] || die "infer jsonl missing: ${INFER_JSONL}"
INFER_LINES="$(wc -l < "${INFER_JSONL}" | tr -d ' ')"
[[ "${INFER_LINES}" -gt 0 ]] || die "infer jsonl is empty: ${INFER_JSONL}"
log "infer records=${INFER_LINES}"

log "step2: run eagle3 infer"
bash "${RUN_SCRIPT}" \
  "${BASE_MODEL_PATH}" \
  "${EA_MODEL_PATH}" \
  "${INFER_JSONL}" \
  "${PRED_JSONL}" 2>&1 | tee -a "${LOG_FILE}"

[[ -f "${PRED_JSONL}" ]] || die "prediction file missing: ${PRED_JSONL}"
PRED_LINES="$(wc -l < "${PRED_JSONL}" | tr -d ' ')"
[[ "${PRED_LINES}" -gt 0 ]] || die "prediction file empty: ${PRED_JSONL}"

log "SUCCESS: pred lines=${PRED_LINES}"
log "Smoke test output: ${PRED_JSONL}"
