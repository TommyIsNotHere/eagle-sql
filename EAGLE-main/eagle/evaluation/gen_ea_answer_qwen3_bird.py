"""Generate Text-to-SQL predictions for BIRD with EAGLE3 + Qwen models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path

import shortuuid
import torch
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..text2sql.bird.prompt_builder import SYSTEM_PROMPT, build_bird_user_prompt
    from ..text2sql.bird.postprocess_sql import extract_sql
except Exception:
    from eagle.model.ea_model import EaModel
    from eagle.text2sql.bird.prompt_builder import SYSTEM_PROMPT, build_bird_user_prompt
    from eagle.text2sql.bird.postprocess_sql import extract_sql


SQL_START_RE = re.compile(r"(?is)\b(select|with|insert|update|delete)\b")


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
    if re.fullmatch(r"[-\d,\.\s]+", text):
        return "numeric_garbage"
    if not SQL_START_RE.search(text):
        return "missing_sql_keyword"
    return "postprocess_failed"


def build_messages(user_prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_retry_messages(user_prompt: str, raw_output: str):
    # Do NOT feed the invalid previous output back to the model. In practice,
    # this can amplify degenerate patterns (e.g. repeated numeric tokens).
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


def generate_once(model, tokenizer, msgs, args, device_getter):
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([prompt]).input_ids
    input_device = device_getter()
    if input_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    output_ids, new_token, _, stats = model.eagenerate(
        torch.as_tensor(input_ids, device=input_device),
        temperature=args.temperature,
        max_new_tokens=args.max_new_token,
        log=True,
        return_stats=True,
    )
    if input_device.type == "cuda":
        torch.cuda.synchronize()
    cost = time.time() - start
    gen_ids = output_ids[0][len(input_ids[0]):]
    raw_text = tokenizer.decode(gen_ids, spaces_between_special_tokens=False)
    return raw_text, int(new_token), stats, cost, len(input_ids[0])


@torch.inference_mode()
def run_infer(args):
    fallback_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs = {
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "total_token": args.total_token,
        "depth": args.depth,
        "top_k": args.top_k,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "use_eagle3": True,
    }
    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except Exception:
        has_accelerate = False
    if has_accelerate:
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["device_map"] = "auto"
    else:
        print("accelerate not found; fallback to single-device model loading.")

    try:
        model = EaModel.from_pretrained(**load_kwargs)
    except ValueError as e:
        msg = str(e)
        if "requires `accelerate`" in msg and "device_map" in msg:
            # Defensive retry for environments where accelerate probing is
            # inconsistent with actual runtime imports.
            load_kwargs.pop("device_map", None)
            load_kwargs.pop("low_cpu_mem_usage", None)
            print("retrying model load without device_map/low_cpu_mem_usage (accelerate unavailable)")
            model = EaModel.from_pretrained(**load_kwargs)
        else:
            raise
    # Without accelerate/device_map, the model stays on CPU by default.
    # Move model to the selected runtime device to keep input/model devices aligned.
    if not has_accelerate:
        model = model.to(fallback_device)
    tokenizer = model.get_tokenizer()
    model.eval()

    def model_input_device():
        # For device_map='auto' (accelerate), different submodules may be split
        # across devices. Inputs must follow embedding weight device.
        try:
            return model.base_model.model.embed_tokens.weight.device
        except Exception:
            return fallback_device

    samples = load_jsonl(args.question_file)
    out_path = Path(args.answer_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Create/truncate output file before warmup so path is visible even if warmup fails.
    with out_path.open("w", encoding="utf-8"):
        pass
    print(f"writing predictions to: {out_path.resolve()}")

    # warmup
    if samples:
        warm_prompt = build_bird_user_prompt(samples[0])
        msgs = build_messages(warm_prompt)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([prompt]).input_ids
        _ = model.eagenerate(
            torch.as_tensor(input_ids, device=model_input_device()),
            temperature=args.temperature,
            log=False,
        )

    valid_count = 0
    retry_count = 0
    debug_written = 0
    debug_prompt_file = Path(args.debug_prompt_file) if args.debug_prompt_file else None
    if debug_prompt_file is not None:
        debug_prompt_file.parent.mkdir(parents=True, exist_ok=True)
        with debug_prompt_file.open("w", encoding="utf-8"):
            pass
    with out_path.open("a", encoding="utf-8") as fout:
        for i, sample in enumerate(tqdm(samples), start=1):
            user_prompt = build_bird_user_prompt(sample)
            if debug_prompt_file is not None and debug_written < args.debug_prompt_max_samples:
                with debug_prompt_file.open("a", encoding="utf-8") as df:
                    df.write(
                        json.dumps(
                            {
                                "question_id": sample.get("question_id"),
                                "db_id": sample.get("db_id"),
                                "user_prompt": user_prompt,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                debug_written += 1
            raw_text, new_token, stats, cost, prompt_tokens = generate_once(
                model=model,
                tokenizer=tokenizer,
                msgs=build_messages(user_prompt),
                args=args,
                device_getter=model_input_device,
            )
            pred_sql = extract_sql(raw_text)
            invalid_reason = classify_invalid_reason(raw_text, pred_sql)
            retry_used = False
            final_attempt = 1

            if invalid_reason and args.retry_invalid_sql:
                retry_used = True
                retry_count += 1
                retry_raw_text, retry_new_token, retry_stats, retry_cost, retry_prompt_tokens = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    msgs=build_retry_messages(user_prompt, raw_text),
                    args=args,
                    device_getter=model_input_device,
                )
                retry_pred_sql = extract_sql(retry_raw_text)
                retry_invalid_reason = classify_invalid_reason(retry_raw_text, retry_pred_sql)
                if retry_pred_sql:
                    pred_sql = retry_pred_sql
                    raw_text = retry_raw_text
                    new_token = retry_new_token
                    stats = retry_stats
                    cost = retry_cost
                    prompt_tokens = retry_prompt_tokens
                    invalid_reason = ""
                    final_attempt = 2
                else:
                    invalid_reason = f"{invalid_reason}|retry:{retry_invalid_reason or 'invalid'}"
                    final_attempt = 2

            is_valid_sql = bool(pred_sql)
            if is_valid_sql:
                valid_count += 1

            rec = {
                "question_id": sample.get("question_id"),
                "db_id": sample.get("db_id"),
                "answer_id": shortuuid.uuid(),
                "model_id": args.model_id,
                "pred_sql": pred_sql,
                "raw_output": raw_text,
                "new_tokens": int(new_token),
                "tree_steps": int(stats["tree_steps"]),
                "accepted_tokens": int(stats["accepted_tokens"]),
                "proposed_tokens": int(stats["proposed_tokens"]),
                "acceptance_rate": float(stats["acceptance_rate"]),
                "wall_time": cost,
                "prompt_tokens": int(prompt_tokens),
                "is_valid_sql": is_valid_sql,
                "invalid_reason": invalid_reason,
                "retry_used": retry_used,
                "final_attempt": final_attempt,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % args.progress_log_every == 0:
                print(
                    f"[progress] done={i}/{len(samples)} valid={valid_count} "
                    f"valid_rate={valid_count / i:.4f} retry={retry_count}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="qwen-eagle3-bird")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-token", type=int, default=128)
    # max-new-token:128-256
    parser.add_argument("--total-token", type=int, default=16)
    # total-token:16-32
    parser.add_argument("--depth", type=int, default=3)
    # depth:2-6
    parser.add_argument("--top-k", type=int, default=3)
    # top-k:2-6
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--progress-log-every", type=int, default=20)
    parser.add_argument("--debug-prompt-file", type=str, default="")
    parser.add_argument("--debug-prompt-max-samples", type=int, default=5)
    parser.add_argument("--retry-invalid-sql", dest="retry_invalid_sql", action="store_true")
    parser.add_argument("--no-retry-invalid-sql", dest="retry_invalid_sql", action="store_false")
    parser.set_defaults(retry_invalid_sql=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    run_infer(args)


if __name__ == "__main__":
    main()
