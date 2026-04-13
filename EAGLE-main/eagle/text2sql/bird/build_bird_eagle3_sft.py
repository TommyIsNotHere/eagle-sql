from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path


def _first_nonempty_text(*values) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            joined = "\n".join(str(x).strip() for x in value if str(x).strip())
            if joined:
                return joined
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _fallback_build_user_prompt(sample: dict) -> str:
    question = _first_nonempty_text(sample.get("question"), sample.get("nl"), sample.get("utterance"))
    evidence = _first_nonempty_text(
        sample.get("evidence"),
        sample.get("external_knowledge"),
        sample.get("hint"),
        sample.get("hints"),
    )
    schema_context = _first_nonempty_text(sample.get("schema_context"), sample.get("schema"))
    database_description = _first_nonempty_text(
        sample.get("database_description"),
        sample.get("db_description"),
        sample.get("db_desc"),
    )
    db_id = _first_nonempty_text(sample.get("db_id"), sample.get("database_id"))

    parts = [
        "[DB_ID]\n" + (db_id or "N/A"),
        "[SCHEMA]\n" + (schema_context or "N/A"),
        "[DATABASE DESCRIPTION]\n" + (database_description or "N/A"),
        "[EVIDENCE]\n" + (evidence or "N/A"),
        "[QUESTION]\n" + question,
        "[OUTPUT]\nReturn exactly one SQLite SQL statement ending with ';'.",
    ]
    return "\n\n".join(parts)


def _load_build_user_prompt():
    module_candidates = [
        "eagle.text2sql.bird.prompt_builder",
        ".prompt_builder",
    ]
    fn_candidates = [
        "build_bird_user_prompt",
        "build_user_prompt",
        "build_prompt",
    ]

    for module_name in module_candidates:
        try:
            if module_name.startswith("."):
                mod = importlib.import_module(module_name, package=__package__)
            else:
                mod = importlib.import_module(module_name)
        except Exception:
            continue

        for fn_name in fn_candidates:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn

    return _fallback_build_user_prompt


def _load_extract_sql():
    module_candidates = [
        "eagle.text2sql.bird.postprocess_sql",
        ".postprocess_sql",
    ]
    fn_candidates = [
        "extract_sql",
        "postprocess_sql",
        "normalize_sql",
    ]

    for module_name in module_candidates:
        try:
            if module_name.startswith("."):
                mod = importlib.import_module(module_name, package=__package__)
            else:
                mod = importlib.import_module(module_name)
        except Exception:
            continue
        for fn_name in fn_candidates:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn

    def _id_sql(text: str) -> str:
        return str(text).strip()

    return _id_sql


build_bird_user_prompt = _load_build_user_prompt()
extract_sql = _load_extract_sql()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_chat_record(sample: dict, fallback_id: int) -> dict | None:
    user_prompt = build_bird_user_prompt(sample)
    gold_sql = (
        sample.get("gold_sql")
        or sample.get("SQL")
        or sample.get("sql")
        or sample.get("query")
        or ""
    )
    gold_sql = extract_sql(str(gold_sql))
    if not gold_sql:
        return None

    question_id = sample.get("question_id", fallback_id)
    return {
        "id": str(question_id),
        "question_id": question_id,
        "db_id": sample.get("db_id", ""),
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": gold_sql},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EAGLE3 training jsonl from prepared BIRD jsonl")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Prepared jsonl (output of prep_bird.py)")
    parser.add_argument("--train-output-jsonl", type=str, required=True)
    parser.add_argument("--eval-output-jsonl", type=str, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.02, help="Eval split ratio in [0,1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    if not (0.0 <= args.eval_ratio < 1.0):
        raise ValueError("--eval-ratio must be in [0, 1)")

    rows = _load_jsonl(Path(args.input_jsonl))
    chat_rows: list[dict] = []
    for idx, row in enumerate(rows):
        rec = _to_chat_record(row, fallback_id=idx)
        if rec is not None:
            chat_rows.append(rec)

    if args.max_samples > 0:
        chat_rows = chat_rows[: args.max_samples]

    rnd = random.Random(args.seed)
    rnd.shuffle(chat_rows)

    eval_size = int(len(chat_rows) * args.eval_ratio)
    eval_rows = chat_rows[:eval_size]
    train_rows = chat_rows[eval_size:]

    _write_jsonl(Path(args.train_output_jsonl), train_rows)
    _write_jsonl(Path(args.eval_output_jsonl), eval_rows)

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "usable_rows": len(chat_rows),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_output_jsonl": str(Path(args.train_output_jsonl).resolve()),
                "eval_output_jsonl": str(Path(args.eval_output_jsonl).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
