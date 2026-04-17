from __future__ import annotations

import argparse
import glob
import json
import math
import signal
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class QueryTimeoutError(RuntimeError):
    pass


@dataclass
class ExecResult:
    ok: bool
    status: str
    rows: list[tuple[Any, ...]] | None = None
    error: str | None = None
    elapsed_sec: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_questions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        qid = str(r.get("question_id"))
        out[qid] = r
    return out


def _find_db_path(sample: dict[str, Any], db_root: Path | None) -> Path | None:
    explicit = (
        sample.get("db_path")
        or sample.get("database_path")
        or sample.get("sqlite_path")
        or sample.get("db_file")
    )
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    db_id = sample.get("db_id")
    if not db_root or not db_id:
        return None

    candidates = [
        db_root / str(db_id) / f"{db_id}.sqlite",
        db_root / f"{db_id}.sqlite",
        db_root / str(db_id) / "database.sqlite",
    ]
    for c in candidates:
        if c.exists():
            return c

    dynamic = glob.glob(str(db_root / str(db_id) / "*.sqlite"))
    if dynamic:
        return Path(dynamic[0])
    return None


def _canonicalize_rows(rows: list[tuple[Any, ...]], ignore_row_order: bool) -> list[tuple[Any, ...]]:
    if not ignore_row_order:
        return rows
    return sorted(rows, key=lambda x: repr(x))


def _classify_sql_error(msg: str) -> str:
    m = msg.lower()
    if "interrupted" in m or "timeout" in m:
        return "timeout"
    if "syntax error" in m or "near" in m:
        return "syntax_error"
    return "runtime_error"


def _run_sql(db_path: Path, sql: str, timeout_sec: float, ignore_row_order: bool) -> ExecResult:
    sql = (sql or "").strip()
    if not sql:
        return ExecResult(ok=False, status="empty_sql", error="empty sql")

    conn = None
    start = time.time()
    old_handler = None
    try:
        # UNIX-only alarm timeout; sufficient for the current Linux/Web IDE workflow.
        def _timeout_handler(signum, frame):
            raise QueryTimeoutError("query timeout")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = None
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        rows = _canonicalize_rows(rows, ignore_row_order=ignore_row_order)
        return ExecResult(ok=True, status="ok", rows=rows, elapsed_sec=time.time() - start)
    except QueryTimeoutError as e:
        return ExecResult(ok=False, status="timeout", error=str(e), elapsed_sec=time.time() - start)
    except sqlite3.Error as e:
        c = _classify_sql_error(str(e))
        return ExecResult(ok=False, status=c, error=str(e), elapsed_sec=time.time() - start)
    except Exception as e:
        return ExecResult(ok=False, status="runtime_error", error=str(e), elapsed_sec=time.time() - start)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        if conn is not None:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution evaluation for BIRD Text-to-SQL predictions")
    parser.add_argument("--question-jsonl", type=str, required=True, help="Prepared question jsonl with gold_sql")
    parser.add_argument("--pred-jsonl", type=str, required=True, help="Prediction jsonl")
    parser.add_argument("--db-root", type=str, default="", help="Root directory of sqlite db files")
    parser.add_argument("--output-summary-json", type=str, required=True)
    parser.add_argument("--output-failures-jsonl", type=str, required=True)
    parser.add_argument("--output-report-md", type=str, required=True)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--ignore-row-order", action="store_true")
    parser.add_argument("--strict-row-order", action="store_true", help="If set, keep row order for result compare")
    args = parser.parse_args()

    if args.strict_row_order:
        args.ignore_row_order = False

    q_path = Path(args.question_jsonl)
    p_path = Path(args.pred_jsonl)
    db_root = Path(args.db_root) if args.db_root else None

    questions = _index_questions(_read_jsonl(q_path))
    preds = _read_jsonl(p_path)

    summary = {
        "total_preds": 0,
        "matched_questions": 0,
        "db_found": 0,
        "gold_exec_ok": 0,
        "pred_exec_ok": 0,
        "ex_matches": 0,
        "ex_denominator": 0,
        "exec_accuracy": 0.0,
        "pred_executable_rate": 0.0,
        "acceptance_samples": 0,
        "acceptance_rate_mean": 0.0,
        "acceptance_rate_token_weighted": 0.0,
        "accepted_tokens_sum": 0,
        "proposed_tokens_sum": 0,
        "wall_time_samples": 0,
        "wall_time_avg_sec": 0.0,
        "failure_breakdown": {},
        "settings": {
            "timeout_sec": args.timeout_sec,
            "ignore_row_order": args.ignore_row_order,
            "strict_row_order": args.strict_row_order,
        },
    }

    failures: list[dict[str, Any]] = []
    acceptance_rate_sum = 0.0
    wall_time_sum = 0.0

    for pr in preds:
        summary["total_preds"] += 1
        acceptance_rate = _safe_float(pr.get("acceptance_rate"), default=-1.0)
        if acceptance_rate >= 0.0:
            summary["acceptance_samples"] += 1
            acceptance_rate_sum += acceptance_rate

        accepted_tokens = max(0, _safe_int(pr.get("accepted_tokens"), default=0))
        proposed_tokens = max(0, _safe_int(pr.get("proposed_tokens"), default=0))
        if proposed_tokens > 0:
            summary["accepted_tokens_sum"] += accepted_tokens
            summary["proposed_tokens_sum"] += proposed_tokens

        wall_time = _safe_float(pr.get("wall_time"), default=-1.0)
        if wall_time >= 0.0:
            summary["wall_time_samples"] += 1
            wall_time_sum += wall_time

        qid = str(pr.get("question_id"))
        sample = questions.get(qid)
        if sample is None:
            reason = "question_not_found"
            summary["failure_breakdown"][reason] = summary["failure_breakdown"].get(reason, 0) + 1
            failures.append({"question_id": qid, "reason": reason, "pred_sql": pr.get("pred_sql", "")})
            continue

        summary["matched_questions"] += 1
        db_path = _find_db_path(sample, db_root)
        if db_path is None:
            reason = "db_not_found"
            summary["failure_breakdown"][reason] = summary["failure_breakdown"].get(reason, 0) + 1
            failures.append({
                "question_id": qid,
                "db_id": sample.get("db_id"),
                "reason": reason,
                "pred_sql": pr.get("pred_sql", ""),
                "gold_sql": sample.get("gold_sql", ""),
            })
            continue

        summary["db_found"] += 1
        gold_sql = (sample.get("gold_sql") or "").strip()
        pred_sql = (pr.get("pred_sql") or "").strip()

        if not gold_sql:
            reason = "gold_sql_empty"
            summary["failure_breakdown"][reason] = summary["failure_breakdown"].get(reason, 0) + 1
            failures.append({
                "question_id": qid,
                "db_id": sample.get("db_id"),
                "reason": reason,
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
            })
            continue

        g = _run_sql(db_path, gold_sql, timeout_sec=args.timeout_sec, ignore_row_order=args.ignore_row_order)
        p = _run_sql(db_path, pred_sql, timeout_sec=args.timeout_sec, ignore_row_order=args.ignore_row_order)

        if g.ok:
            summary["gold_exec_ok"] += 1
        if p.ok:
            summary["pred_exec_ok"] += 1

        # EX denominator: cases where gold is executable and db exists.
        if g.ok:
            summary["ex_denominator"] += 1
            if p.ok and p.rows == g.rows:
                summary["ex_matches"] += 1
            else:
                reason = "exec_mismatch" if p.ok else f"pred_{p.status}"
                summary["failure_breakdown"][reason] = summary["failure_breakdown"].get(reason, 0) + 1
                failures.append({
                    "question_id": qid,
                    "db_id": sample.get("db_id"),
                    "reason": reason,
                    "pred_sql": pred_sql,
                    "gold_sql": gold_sql,
                    "pred_error": p.error,
                    "gold_error": g.error,
                    "pred_rows_preview": p.rows[:5] if p.rows else [],
                    "gold_rows_preview": g.rows[:5] if g.rows else [],
                    "pred_exec_sec": p.elapsed_sec,
                    "gold_exec_sec": g.elapsed_sec,
                })
        else:
            reason = f"gold_{g.status}"
            summary["failure_breakdown"][reason] = summary["failure_breakdown"].get(reason, 0) + 1
            failures.append({
                "question_id": qid,
                "db_id": sample.get("db_id"),
                "reason": reason,
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "pred_error": p.error,
                "gold_error": g.error,
            })

    if summary["ex_denominator"] > 0:
        summary["exec_accuracy"] = summary["ex_matches"] / summary["ex_denominator"]
    if summary["db_found"] > 0:
        summary["pred_executable_rate"] = summary["pred_exec_ok"] / summary["db_found"]
    if summary["acceptance_samples"] > 0:
        summary["acceptance_rate_mean"] = acceptance_rate_sum / summary["acceptance_samples"]
    if summary["proposed_tokens_sum"] > 0:
        summary["acceptance_rate_token_weighted"] = (
            summary["accepted_tokens_sum"] / summary["proposed_tokens_sum"]
        )
    if summary["wall_time_samples"] > 0:
        summary["wall_time_avg_sec"] = wall_time_sum / summary["wall_time_samples"]

    out_summary = Path(args.output_summary_json)
    out_fail = Path(args.output_failures_jsonl)
    out_report = Path(args.output_report_md)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_fail.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with out_fail.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_failures = sorted(summary["failure_breakdown"].items(), key=lambda x: x[1], reverse=True)
    with out_report.open("w", encoding="utf-8") as f:
        f.write("# BIRD EX Evaluation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total predictions: {summary['total_preds']}\n")
        f.write(f"- Matched questions: {summary['matched_questions']}\n")
        f.write(f"- DB found: {summary['db_found']}\n")
        f.write(f"- Gold executable: {summary['gold_exec_ok']}\n")
        f.write(f"- Pred executable: {summary['pred_exec_ok']}\n")
        f.write(f"- EX denominator: {summary['ex_denominator']}\n")
        f.write(f"- EX matches: {summary['ex_matches']}\n")
        f.write(f"- Execution Accuracy (EX): {summary['exec_accuracy']:.4f}\n")
        f.write(f"- Pred executable rate: {summary['pred_executable_rate']:.4f}\n")
        f.write(f"- Acceptance rate (sample mean): {summary['acceptance_rate_mean']:.4f}\n")
        f.write(
            f"- Acceptance rate (token-weighted): "
            f"{summary['acceptance_rate_token_weighted']:.4f}\n"
        )
        f.write(
            f"- Accepted / Proposed tokens: "
            f"{summary['accepted_tokens_sum']} / {summary['proposed_tokens_sum']}\n"
        )
        f.write(f"- Avg wall time per sample (sec): {summary['wall_time_avg_sec']:.4f}\n")

        f.write("\n## Settings\n\n")
        f.write(f"- timeout_sec: {summary['settings']['timeout_sec']}\n")
        f.write(f"- ignore_row_order: {summary['settings']['ignore_row_order']}\n")

        f.write("\n## Failure Breakdown\n\n")
        if not top_failures:
            f.write("- No failures recorded.\n")
        else:
            for k, v in top_failures:
                f.write(f"- {k}: {v}\n")

        f.write("\n## Artifacts\n\n")
        f.write(f"- Summary JSON: `{out_summary}`\n")
        f.write(f"- Failures JSONL: `{out_fail}`\n")

    print(json.dumps({
        "exec_accuracy": summary["exec_accuracy"],
        "pred_executable_rate": summary["pred_executable_rate"],
        "acceptance_rate_mean": summary["acceptance_rate_mean"],
        "acceptance_rate_token_weighted": summary["acceptance_rate_token_weighted"],
        "accepted_tokens_sum": summary["accepted_tokens_sum"],
        "proposed_tokens_sum": summary["proposed_tokens_sum"],
        "wall_time_avg_sec": summary["wall_time_avg_sec"],
        "ex_matches": summary["ex_matches"],
        "ex_denominator": summary["ex_denominator"],
        "summary_json": str(out_summary),
        "failures_jsonl": str(out_fail),
        "report_md": str(out_report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
