from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _create_demo_db(db_root: Path) -> Path:
    db_dir = db_root / "demo"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "demo.sqlite"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            [(1, "alice"), (2, "bob"), (3, "cindy")],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _run_eval(
    question_jsonl: Path,
    pred_jsonl: Path,
    db_root: Path,
    out_dir: Path,
    strict_row_order: bool,
) -> dict[str, Any]:
    tag = "strict" if strict_row_order else "ignore"
    summary_json = out_dir / f"eval_{tag}_summary.json"
    failures_jsonl = out_dir / f"eval_{tag}_failures.jsonl"
    report_md = out_dir / f"eval_{tag}_report.md"

    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")

    cmd = [
        sys.executable,
        "-m",
        "eagle.text2sql.bird.eval_exec",
        "--question-jsonl",
        str(question_jsonl),
        "--pred-jsonl",
        str(pred_jsonl),
        "--db-root",
        str(db_root),
        "--output-summary-json",
        str(summary_json),
        "--output-failures-jsonl",
        str(failures_jsonl),
        "--output-report-md",
        str(report_md),
        "--timeout-sec",
        "5",
    ]
    if strict_row_order:
        cmd.append("--strict-row-order")
    else:
        cmd.append("--ignore-row-order")

    subprocess.run(cmd, check=True, env=env)

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary


def _assert_close(actual: float, expected: float, eps: float = 1e-9) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"float mismatch: actual={actual}, expected={expected}, eps={eps}")


def _assert_eq(actual: Any, expected: Any, key: str) -> None:
    if actual != expected:
        raise AssertionError(f"{key} mismatch: actual={actual}, expected={expected}")


def _build_cases(db_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions = [
        {"question_id": "q1", "db_id": "demo", "gold_sql": "SELECT name FROM users ORDER BY id;"},
        {"question_id": "q2", "db_id": "demo", "gold_sql": "SELECT name FROM users ORDER BY id;"},
        {"question_id": "q3", "db_id": "demo", "gold_sql": "SELECT name FROM users ORDER BY id;"},
        {"question_id": "q4", "db_id": "missing_db", "gold_sql": "SELECT 1;"},
        {"question_id": "q5", "db_id": "demo", "gold_sql": "SELECT name FROM users ORDER BY id ASC;"},
        {
            "question_id": "q6",
            "db_id": "wrong_db_id_but_explicit_path",
            "db_path": str(db_path),
            "gold_sql": "SELECT COUNT(*) FROM users;",
        },
    ]

    preds = [
        {"question_id": "q1", "db_id": "demo", "pred_sql": "SELECT name FROM users ORDER BY id;"},
        {"question_id": "q2", "db_id": "demo", "pred_sql": "SELECT name FROM users WHERE id > 1 ORDER BY id;"},
        {"question_id": "q3", "db_id": "demo", "pred_sql": "SELECT FROM users;"},
        {"question_id": "q4", "db_id": "missing_db", "pred_sql": "SELECT 1;"},
        {"question_id": "q5", "db_id": "demo", "pred_sql": "SELECT name FROM users ORDER BY id DESC;"},
        {"question_id": "q6", "db_id": "demo", "pred_sql": "SELECT COUNT(*) FROM users;"},
        {"question_id": "q404", "db_id": "demo", "pred_sql": "SELECT 1;"},
    ]
    return questions, preds


def _check_ignore_mode(summary: dict[str, Any]) -> None:
    _assert_eq(summary["total_preds"], 7, "total_preds")
    _assert_eq(summary["matched_questions"], 6, "matched_questions")
    _assert_eq(summary["db_found"], 5, "db_found")
    _assert_eq(summary["gold_exec_ok"], 5, "gold_exec_ok")
    _assert_eq(summary["pred_exec_ok"], 4, "pred_exec_ok")
    _assert_eq(summary["ex_denominator"], 5, "ex_denominator")
    _assert_eq(summary["ex_matches"], 3, "ex_matches")
    _assert_close(summary["exec_accuracy"], 0.6)
    _assert_close(summary["pred_executable_rate"], 0.8)

    fb = summary.get("failure_breakdown", {})
    _assert_eq(fb.get("exec_mismatch", 0), 1, "failure.exec_mismatch")
    _assert_eq(fb.get("pred_syntax_error", 0), 1, "failure.pred_syntax_error")
    _assert_eq(fb.get("db_not_found", 0), 1, "failure.db_not_found")
    _assert_eq(fb.get("question_not_found", 0), 1, "failure.question_not_found")


def _check_strict_mode(summary: dict[str, Any]) -> None:
    _assert_eq(summary["total_preds"], 7, "total_preds")
    _assert_eq(summary["matched_questions"], 6, "matched_questions")
    _assert_eq(summary["db_found"], 5, "db_found")
    _assert_eq(summary["gold_exec_ok"], 5, "gold_exec_ok")
    _assert_eq(summary["pred_exec_ok"], 4, "pred_exec_ok")
    _assert_eq(summary["ex_denominator"], 5, "ex_denominator")
    _assert_eq(summary["ex_matches"], 2, "ex_matches")
    _assert_close(summary["exec_accuracy"], 0.4)
    _assert_close(summary["pred_executable_rate"], 0.8)

    fb = summary.get("failure_breakdown", {})
    _assert_eq(fb.get("exec_mismatch", 0), 2, "failure.exec_mismatch")
    _assert_eq(fb.get("pred_syntax_error", 0), 1, "failure.pred_syntax_error")
    _assert_eq(fb.get("db_not_found", 0), 1, "failure.db_not_found")
    _assert_eq(fb.get("question_not_found", 0), 1, "failure.question_not_found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal strict validation for eval_exec covering DB resolution, SQL execution, "
            "result comparison, and failure classification."
        )
    )
    parser.add_argument("--output-dir", type=str, default="", help="Optional persistent output directory")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
        work_dir = out_dir
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="eval_exec_validate_")
        work_dir = Path(temp_ctx.name)

    db_root = work_dir / "db_root"
    db_path = _create_demo_db(db_root)
    questions, preds = _build_cases(db_path=db_path)

    question_jsonl = work_dir / "questions.jsonl"
    pred_jsonl = work_dir / "preds.jsonl"
    _write_jsonl(question_jsonl, questions)
    _write_jsonl(pred_jsonl, preds)

    summary_ignore = _run_eval(
        question_jsonl=question_jsonl,
        pred_jsonl=pred_jsonl,
        db_root=db_root,
        out_dir=work_dir,
        strict_row_order=False,
    )
    _check_ignore_mode(summary_ignore)

    summary_strict = _run_eval(
        question_jsonl=question_jsonl,
        pred_jsonl=pred_jsonl,
        db_root=db_root,
        out_dir=work_dir,
        strict_row_order=True,
    )
    _check_strict_mode(summary_strict)

    report = {
        "status": "PASS",
        "work_dir": str(work_dir),
        "ignore_row_order": {
            "exec_accuracy": summary_ignore["exec_accuracy"],
            "failure_breakdown": summary_ignore.get("failure_breakdown", {}),
        },
        "strict_row_order": {
            "exec_accuracy": summary_strict["exec_accuracy"],
            "failure_breakdown": summary_strict.get("failure_breakdown", {}),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if temp_ctx is not None:
        temp_ctx.cleanup()


if __name__ == "__main__":
    main()

