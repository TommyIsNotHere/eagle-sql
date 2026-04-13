#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <base_model_path> <ea_model_path> <question_file.jsonl> <answer_file.jsonl>"
  exit 1
fi

BASE_MODEL_PATH="$1"
EA_MODEL_PATH="$2"
QUESTION_FILE="$3"
ANSWER_FILE="$4"

python -m eagle.evaluation.gen_ea_answer_qwen3_bird \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$EA_MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --answer-file "$ANSWER_FILE" \
  --temperature 0 \
  --bf16
