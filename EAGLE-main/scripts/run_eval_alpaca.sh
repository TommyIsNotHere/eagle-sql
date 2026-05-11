#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGLE_DIR="${EAGLE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ ! -d "${EAGLE_DIR}/eagle" ]]; then
  echo "ERROR: invalid EAGLE_DIR (missing eagle package): ${EAGLE_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${EAGLE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_ROOT="/mnt/nj-aigc/usr/wangtong2/eagle_sql"
BASE_MODEL_PATH="${1:-${PROJECT_ROOT}/model/Qwen2.5-Coder-14B-Instruct}"
EA_MODEL_PATH="${2:-${PROJECT_ROOT}/artifacts/eagle2/infer}"
ANSWER_FILE="${3:-${PROJECT_ROOT}/artifacts/eval_alpaca/eval_alpaca.jsonl}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# Chain config (top_k=1) — primary metric is α (per-token acceptance rate).
# Stage A ShareGPT validation: α = 0.883, speedup 2.09x at this setting.
# To switch to tree config, override: TOTAL_TOKEN=50 DEPTH=6 TOP_K=10.
TOTAL_TOKEN="${TOTAL_TOKEN:-4}"
DEPTH="${DEPTH:-3}"
TOP_K="${TOP_K:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"

(
  cd "${EAGLE_DIR}"
  python -m eagle.evaluation.eval_bench \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$EA_MODEL_PATH" \
    --bench-name alpaca \
    --answer-file "$ANSWER_FILE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --total-token "$TOTAL_TOKEN" \
    --depth "$DEPTH" \
    --top-k "$TOP_K" \
    --max-samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    --max-prompt-len "$MAX_PROMPT_LEN" \
    --warmup \
    --bf16
)
