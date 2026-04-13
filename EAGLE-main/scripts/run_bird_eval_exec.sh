#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <question_file.jsonl> <pred_file.jsonl> <db_root> <summary.json> <failures.jsonl> <report.md> [timeout_sec]"
  exit 1
fi

QUESTION_FILE="$1"
PRED_FILE="$2"
DB_ROOT="$3"
SUMMARY_JSON="$4"
FAILURES_JSONL="$5"
REPORT_MD="$6"
TIMEOUT_SEC="${7:-15}"

python -m eagle.text2sql.bird.eval_exec \
  --question-jsonl "$QUESTION_FILE" \
  --pred-jsonl "$PRED_FILE" \
  --db-root "$DB_ROOT" \
  --output-summary-json "$SUMMARY_JSON" \
  --output-failures-jsonl "$FAILURES_JSONL" \
  --output-report-md "$REPORT_MD" \
  --timeout-sec "$TIMEOUT_SEC" \
  --ignore-row-order
