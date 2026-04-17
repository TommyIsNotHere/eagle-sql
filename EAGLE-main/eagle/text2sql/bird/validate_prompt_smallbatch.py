from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


SQL_START_RE = re.compile(r"(?is)\b(select|with|insert|update|delete)\b")
DEFAULT_SYSTEM_PROMPT = (
    "You are a SQLite Text-to-SQL generator for the BIRD benchmark.\n"
    "Given [SCHEMA], [DATABASE DESCRIPTION], [EVIDENCE], and [QUESTION], return exactly one executable SQLite SQL statement.\n"
    "Return SQL only. No explanation, no markdown, no list/bullet/number sequence.\n"
    "The SQL must start with SELECT or WITH and use only tables/columns from [SCHEMA].\n"
    "Use [EVIDENCE] and [DATABASE DESCRIPTION] to resolve ambiguous terms."
)


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


def _fallback_build_user_prompt(sample: dict[str, Any]) -> str:
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


def _load_prompt_components() -> tuple[str, Callable[[dict[str, Any]], str]]:
    module_candidates = [
        "eagle.text2sql.bird.prompt_builder",
        ".prompt_builder",
    ]
    prompt_candidates = [
        "SYSTEM_PROMPT",
        "SYSTEM",
        "PROMPT_SYSTEM",
        "system_prompt",
    ]
    fn_candidates = [
        "build_bird_user_prompt",
        "build_user_prompt",
        "build_prompt",
    ]

    default_prompt = DEFAULT_SYSTEM_PROMPT
    default_fn: Callable[[dict[str, Any]], str] = _fallback_build_user_prompt

    for module_name in module_candidates:
        try:
            if module_name.startswith("."):
                mod = importlib.import_module(module_name, package=__package__)
            else:
                mod = importlib.import_module(module_name)
        except Exception:
            continue

        system_prompt = default_prompt
        for key in prompt_candidates:
            val = getattr(mod, key, None)
            if isinstance(val, str) and val.strip():
                system_prompt = val
                break

        for fn_name in fn_candidates:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return system_prompt, fn

    return default_prompt, default_fn


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


SYSTEM_PROMPT, build_bird_user_prompt = _load_prompt_components()
extract_sql = _load_extract_sql()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _classify_invalid_reason(raw_text: str, pred_sql: str) -> str:
    if pred_sql:
        return ""
    text = (raw_text or "").strip()
    if not text:
        return "empty_output"
    if re.fullmatch(r"[-\d,\.\s]+", text):
        return "numeric_garbage"
    if not SQL_START_RE.search(text):
        return "missing_sql_keyword"
    return "postprocess_failed"


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32

    # auto
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def _model_input_device(model, fallback: torch.device) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return fallback


def _build_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype(args.dtype)
    runtime_device = _resolve_device(args.device)
    use_device_map = bool(args.device_map_auto) and runtime_device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.unk_token_id

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_device_map:
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True

    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
    except Exception:
        load_kwargs.pop("device_map", None)
        load_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
        model = model.to(runtime_device)

    model.eval()
    return model, tokenizer, runtime_device, dtype


def _run_eval_exec(
    question_jsonl: Path,
    pred_jsonl: Path,
    db_root: Path,
    output_dir: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    summary_json = output_dir / "prompt_eval_summary.json"
    failures_jsonl = output_dir / "prompt_eval_failures.jsonl"
    report_md = output_dir / "prompt_eval_report.md"

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
        str(timeout_sec),
        "--ignore-row-order",
    ]
    subprocess.run(cmd, check=True, env=env)
    with summary_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Small-batch prompt validation: run direct Qwen generation with current BIRD prompt "
            "and optionally evaluate SQL execution quality."
        )
    )
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--question-jsonl", type=str, required=True, help="Prepared BIRD jsonl with gold_sql")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--device-map-auto", dest="device_map_auto", action="store_true")
    parser.add_argument("--no-device-map-auto", dest="device_map_auto", action="store_false")
    parser.set_defaults(device_map_auto=True)
    parser.add_argument("--run-eval", dest="run_eval", action="store_true")
    parser.add_argument("--no-run-eval", dest="run_eval", action="store_false")
    parser.set_defaults(run_eval=True)
    parser.add_argument("--db-root", type=str, default="")
    parser.add_argument("--eval-timeout-sec", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "prompt_smallbatch_pred.jsonl"
    sample_question_jsonl = out_dir / "prompt_smallbatch_questions.jsonl"
    summary_json = out_dir / "prompt_smallbatch_summary.json"

    rows = _read_jsonl(Path(args.question_jsonl))
    if not rows:
        raise RuntimeError(f"No rows found: {args.question_jsonl}")

    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(rows), generator=rng).tolist()
    rows = [rows[i] for i in order[: max(1, min(args.num_samples, len(rows)))]]
    _write_jsonl(sample_question_jsonl, rows)

    model, tokenizer, runtime_device, dtype = _load_model_and_tokenizer(args)
    model_infer_device = _model_input_device(model, runtime_device)
    print(
        "[prompt-validate] model loaded: "
        f"runtime_device={runtime_device}, infer_device={model_infer_device}, "
        f"dtype={dtype}, cuda_available={torch.cuda.is_available()}"
    )

    records: list[dict[str, Any]] = []
    start_all = time.time()
    valid_sql_count = 0

    for idx, sample in enumerate(rows, start=1):
        user_prompt = build_bird_user_prompt(sample)
        msgs = _build_messages(user_prompt)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model_infer_device) for k, v in enc.items()}
        prompt_tokens = int(enc["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if args.temperature > 1e-5:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = float(args.top_p)
        else:
            gen_kwargs["do_sample"] = False

        t0 = time.time()
        with torch.inference_mode():
            out_ids = model.generate(**enc, **gen_kwargs)
        wall_time = time.time() - t0

        gen_ids = out_ids[0][prompt_tokens:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_sql = extract_sql(raw_text)
        invalid_reason = _classify_invalid_reason(raw_text, pred_sql)
        is_valid_sql = bool(pred_sql)
        if is_valid_sql:
            valid_sql_count += 1

        rec = {
            "question_id": sample.get("question_id"),
            "db_id": sample.get("db_id"),
            "model_id": "qwen-direct-prompt-validate",
            "pred_sql": pred_sql,
            "raw_output": raw_text,
            "new_tokens": int(gen_ids.shape[-1]),
            "prompt_tokens": prompt_tokens,
            "wall_time": float(wall_time),
            "is_valid_sql": is_valid_sql,
            "invalid_reason": invalid_reason,
            "tstamp": time.time(),
        }
        records.append(rec)

        print(
            f"[prompt-validate] {idx}/{len(rows)} qid={sample.get('question_id')} "
            f"valid_sql={int(is_valid_sql)} wall={wall_time:.3f}s"
        )

    _write_jsonl(pred_jsonl, records)

    invalid_breakdown: dict[str, int] = {}
    for r in records:
        reason = r.get("invalid_reason") or ""
        if reason:
            invalid_breakdown[reason] = invalid_breakdown.get(reason, 0) + 1

    summary: dict[str, Any] = {
        "total_samples": len(records),
        "valid_sql_count": valid_sql_count,
        "valid_sql_rate": (valid_sql_count / len(records)) if records else 0.0,
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in records) / max(1, len(records)),
        "avg_new_tokens": sum(r["new_tokens"] for r in records) / max(1, len(records)),
        "avg_wall_time_sec": sum(r["wall_time"] for r in records) / max(1, len(records)),
        "invalid_breakdown": invalid_breakdown,
        "artifacts": {
            "questions_jsonl": str(sample_question_jsonl),
            "pred_jsonl": str(pred_jsonl),
        },
        "runtime": {
            "requested_device": str(args.device),
            "requested_dtype": str(args.dtype),
            "resolved_runtime_device": str(runtime_device),
            "model_input_device": str(model_infer_device),
            "resolved_dtype": str(dtype),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
        "elapsed_total_sec": float(time.time() - start_all),
    }

    if args.run_eval:
        if not args.db_root:
            raise RuntimeError("--run-eval requires --db-root")
        eval_summary = _run_eval_exec(
            question_jsonl=sample_question_jsonl,
            pred_jsonl=pred_jsonl,
            db_root=Path(args.db_root),
            output_dir=out_dir,
            timeout_sec=float(args.eval_timeout_sec),
        )
        summary["eval_summary"] = {
            "exec_accuracy": eval_summary.get("exec_accuracy", 0.0),
            "pred_executable_rate": eval_summary.get("pred_executable_rate", 0.0),
            "ex_matches": eval_summary.get("ex_matches", 0),
            "ex_denominator": eval_summary.get("ex_denominator", 0),
            "failure_breakdown": eval_summary.get("failure_breakdown", {}),
        }
        summary["artifacts"]["eval_summary_json"] = str(out_dir / "prompt_eval_summary.json")
        summary["artifacts"]["eval_failures_jsonl"] = str(out_dir / "prompt_eval_failures.jsonl")
        summary["artifacts"]["eval_report_md"] = str(out_dir / "prompt_eval_report.md")

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
