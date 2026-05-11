"""Generate Text-to-SQL predictions for BIRD with EAGLE + Qwen models."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from ..model.ea_model import EaModel
except Exception:
    from eagle.model.ea_model import EaModel


SQL_START_RE = re.compile(r"(?is)\b(select|with|insert|update|delete)\b")
DEFAULT_SYSTEM_PROMPT = (
    "You are a SQLite Text-to-SQL generator for the BIRD benchmark.\n"
    "Given [SCHEMA], [DATABASE DESCRIPTION], [EVIDENCE], and [QUESTION], return exactly one executable SQLite SQL statement.\n"
    "Return SQL only. No explanation, no markdown, no list/bullet/number sequence.\n"
    "The SQL must start with SELECT or WITH and use only tables/columns from [SCHEMA].\n"
    "Use [EVIDENCE] and [DATABASE DESCRIPTION] to resolve ambiguous terms."
)


# ─── Prompt Building ─────────────────���────────────────────────────────────────

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
    question = _first_nonempty_text(sample.get("question"), sample.get("nl"))
    evidence = _first_nonempty_text(sample.get("evidence"), sample.get("external_knowledge"), sample.get("hint"))
    schema_context = _first_nonempty_text(sample.get("schema_context"), sample.get("schema"))
    database_description = _first_nonempty_text(sample.get("database_description"), sample.get("db_description"))
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
    import importlib
    for module_name in ["eagle.text2sql.bird.prompt_builder", "..text2sql.bird.prompt_builder"]:
        try:
            if module_name.startswith("."):
                mod = importlib.import_module(module_name, package=__package__)
            else:
                mod = importlib.import_module(module_name)
        except Exception:
            continue
        for key in ["SYSTEM_PROMPT", "SYSTEM", "system_prompt"]:
            val = getattr(mod, key, None)
            if isinstance(val, str) and val.strip():
                system_prompt = val
                break
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        for fn_name in ["build_bird_user_prompt", "build_user_prompt", "build_prompt"]:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return system_prompt, fn
    return DEFAULT_SYSTEM_PROMPT, _fallback_build_user_prompt


def _load_extract_sql():
    import importlib
    for module_name in ["eagle.text2sql.bird.postprocess_sql", "..text2sql.bird.postprocess_sql"]:
        try:
            if module_name.startswith("."):
                mod = importlib.import_module(module_name, package=__package__)
            else:
                mod = importlib.import_module(module_name)
        except Exception:
            continue
        for fn_name in ["extract_sql", "postprocess_sql", "normalize_sql"]:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
    return lambda text: str(text).strip()


SYSTEM_PROMPT, build_bird_user_prompt = _load_prompt_components()
extract_sql = _load_extract_sql()


# ─── Utilities ────────────────��────────────────────────────────────────���──────

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_invalid_reason(raw_text: str, pred_sql: str) -> str:
    if pred_sql:
        return ""
    text = (raw_text or "").strip()
    if not text:
        return "empty_output"
    if not SQL_START_RE.search(text):
        return "missing_sql_keyword"
    return "postprocess_failed"


def build_messages(user_prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_retry_messages(user_prompt: str, raw_output: str):
    repair_prompt = (
        f"{user_prompt}\n\n"
        "[RETRY_REQUIREMENTS]\n"
        "Your previous answer was invalid because it was not SQL.\n"
        "Now output exactly one valid SQLite SQL statement.\n"
        "The first token must be SELECT or WITH.\n"
        "Use only tables/columns from [SCHEMA].\n"
        "Do not output explanation, markdown, list, or any extra text."
    )
    return build_messages(repair_prompt)


# ─── Generation ───────────────────────────────────────────────────────────────

def _drop_kv_cache(model) -> None:
    for name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, name):
            delattr(model, name)


def generate_once(model, tokenizer, msgs, args, device_getter):
    if args.reset_kv_per_sample:
        _drop_kv_cache(model)

    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.tolist()
    prompt_tokens = int(enc["input_ids"].shape[-1])

    max_length = prompt_tokens + args.max_new_token + args.total_token + 256
    max_length = max(3072, max_length)

    input_device = device_getter()
    enc = {k: v.to(input_device) for k, v in enc.items()}

    if input_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    if args.decode_mode == "hf":
        output_ids = model.base_model.generate(
            **enc,
            max_new_tokens=int(args.max_new_token),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
            do_sample=bool(args.temperature > 1e-5),
        )
        new_token = int(output_ids[0].shape[-1] - prompt_tokens)
        stats = {
            "accepted_tokens": new_token, "proposed_tokens": new_token,
            "acceptance_rate": 1.0, "mean_accepted_length": 0.0, "tree_steps": 0,
        }
    elif args.decode_mode == "naive":
        output_ids, new_token, _ = model.naivegenerate(
            torch.as_tensor(input_ids, device=input_device),
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            max_length=max_length,
            log=True,
        )
        stats = {
            "accepted_tokens": new_token, "proposed_tokens": new_token,
            "acceptance_rate": 1.0, "mean_accepted_length": 0.0, "tree_steps": 0,
        }
    else:
        output_ids, new_token, _, stats = model.eagenerate(
            torch.as_tensor(input_ids, device=input_device),
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            max_length=max_length,
            log=True,
            return_stats=True,
        )

    if input_device.type == "cuda":
        torch.cuda.synchronize()
    cost = time.time() - start

    gen_ids = output_ids[0][len(input_ids[0]):]
    raw_text = tokenizer.decode(gen_ids, spaces_between_special_tokens=False)
    return raw_text, int(new_token), stats, cost, prompt_tokens


# ─── Main Inference Loop ─────────────���───────────────────────────��────────────

@torch.inference_mode()
def run_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_kwargs = {
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "total_token": args.total_token,
        "depth": args.depth,
        "top_k": args.top_k,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "use_eagle3": args.use_eagle3,
    }

    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except Exception:
        has_accelerate = False

    if args.device_map_auto and has_accelerate:
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["device_map"] = "auto"

    model = EaModel.from_pretrained(**load_kwargs)
    if not (args.device_map_auto and has_accelerate):
        model = model.to(device)

    tokenizer = model.get_tokenizer()
    model.eval()

    def model_input_device():
        try:
            return model.base_model.model.embed_tokens.weight.device
        except Exception:
            return device

    samples = load_jsonl(args.question_file)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    out_path = Path(args.answer_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[config] eagle{'3' if args.use_eagle3 else '2'} mode={args.decode_mode} samples={len(samples)}")
    print(f"[config] ea_model={args.ea_model_path}")
    print(f"[config] output={out_path.resolve()}")

    # Warmup
    if samples and args.warmup:
        warm_prompt = build_bird_user_prompt(samples[0])
        generate_once(model, tokenizer, build_messages(warm_prompt), args, model_input_device)
        if args.reset_kv_per_sample:
            _drop_kv_cache(model)
        print("[warmup] done")

    valid_count = 0
    retry_count = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(tqdm(samples), start=1):
            user_prompt = build_bird_user_prompt(sample)
            raw_text, new_token, stats, cost, prompt_tokens = generate_once(
                model, tokenizer, build_messages(user_prompt), args, model_input_device,
            )
            pred_sql = extract_sql(raw_text)
            invalid_reason = classify_invalid_reason(raw_text, pred_sql)
            retry_used = False

            if invalid_reason and args.retry_invalid_sql:
                retry_used = True
                retry_count += 1
                retry_raw, retry_new, retry_stats, retry_cost, retry_prompt = generate_once(
                    model, tokenizer, build_retry_messages(user_prompt, raw_text), args, model_input_device,
                )
                retry_sql = extract_sql(retry_raw)
                if retry_sql:
                    pred_sql, raw_text, new_token, stats, cost, prompt_tokens = (
                        retry_sql, retry_raw, retry_new, retry_stats, retry_cost, retry_prompt
                    )
                    invalid_reason = ""

            is_valid = bool(pred_sql)
            if is_valid:
                valid_count += 1

            rec = {
                "question_id": sample.get("question_id"),
                "db_id": sample.get("db_id"),
                "pred_sql": pred_sql,
                "raw_output": raw_text,
                "new_tokens": new_token,
                "accepted_tokens": int(stats.get("accepted_tokens", new_token)),
                "proposed_tokens": int(stats.get("proposed_tokens", new_token)),
                "tree_steps": int(stats.get("tree_steps", 0)),
                "mean_accepted_length": float(stats.get("mean_accepted_length", 0.0)),
                "acceptance_rate": float(stats.get("acceptance_rate", 1.0)),
                "wall_time": cost,
                "prompt_tokens": prompt_tokens,
                "is_valid_sql": is_valid,
                "invalid_reason": invalid_reason,
                "retry_used": retry_used,
                "decode_mode": args.decode_mode,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if i % args.progress_log_every == 0 or i == len(samples):
                print(f"[progress] {i}/{len(samples)} valid={valid_count} ({valid_count/i:.3f}) retry={retry_count}")

    print(f"[done] valid={valid_count}/{len(samples)} ({valid_count/len(samples):.3f}) retry={retry_count}")


# ─── CLI ───────────────────────────────────────────────────────���──────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-token", type=int, default=128)
    # Tree config: defaults follow EAGLE-2 paper Appendix A for 13B class
    # (also reasonable for 14B). Use --total-token -1 to enable auto-tune.
    parser.add_argument("--total-token", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--decode-mode", type=str, default="eagle", choices=["eagle", "naive", "hf"])
    parser.add_argument("--warmup", dest="warmup", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--progress-log-every", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--reset-kv-per-sample", dest="reset_kv_per_sample", action="store_true")
    parser.add_argument("--no-reset-kv-per-sample", dest="reset_kv_per_sample", action="store_false")
    parser.set_defaults(reset_kv_per_sample=True)
    parser.add_argument("--retry-invalid-sql", dest="retry_invalid_sql", action="store_true")
    parser.add_argument("--no-retry-invalid-sql", dest="retry_invalid_sql", action="store_false")
    parser.set_defaults(retry_invalid_sql=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use-eagle3", dest="use_eagle3", action="store_true")
    parser.add_argument("--use-eagle2", dest="use_eagle3", action="store_false")
    parser.set_defaults(use_eagle3=False)
    args = parser.parse_args()

    print(f"[args] {vars(args)}")
    run_infer(args)


if __name__ == "__main__":
    main()
