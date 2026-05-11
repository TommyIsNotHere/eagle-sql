"""Stage A validation: verify EAGLE2 speculative decoding on general conversation.

Compares eagle vs naive decoding on a set of prompts, reports:
- Output consistency (text match)
- Acceptance rate / speedup
- Generation quality (no garbled / repetitive output)

Usage:
    python -m eagle.evaluation.eval_sharegpt \
        --ea-model-path /path/to/eagle2/infer \
        --base-model-path /path/to/Qwen2.5-Coder-14B-Instruct \
        [--question-file /path/to/sharegpt_test.jsonl] \
        [--answer-file /path/to/output.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:
    from ..model.ea_model import EaModel
except Exception:
    from eagle.model.ea_model import EaModel


BUILTIN_PROMPTS = [
    [
        {"role": "user", "content": "Explain the difference between a stack and a queue in data structures."},
    ],
    [
        {"role": "user", "content": "Write a Python function to check if a string is a palindrome."},
    ],
    [
        {"role": "user", "content": "What are the main differences between TCP and UDP?"},
    ],
    [
        {"role": "user", "content": "Summarize the key ideas of the transformer architecture in deep learning."},
    ],
    [
        {"role": "user", "content": "How does garbage collection work in Java?"},
    ],
    [
        {"role": "user", "content": "What is the CAP theorem in distributed systems? Give a brief explanation."},
    ],
    [
        {"role": "user", "content": "Write a SQL query to find the top 3 customers by total order amount from an orders table."},
    ],
    [
        {"role": "user", "content": "Explain what a closure is in JavaScript with a simple example."},
    ],
]


ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def load_sharegpt_prompts(path: str, max_samples: int = 0, seed: int = 42):
    """Load prompts from ShareGPT-format JSONL. Only keep the first user turn.

    When max_samples > 0, all prompts are first read into memory, then a random
    subset of size max_samples is drawn using the given seed. Fixed seed makes
    the sampled subset deterministic across runs (for fair model comparison).
    """
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            messages = []
            for turn in convs:
                role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
                value = turn.get("value", "").strip()
                if not value:
                    continue
                messages.append({"role": role, "content": value})
                if role == "user":
                    break
            if messages and messages[-1]["role"] == "user":
                prompts.append(messages)

    if max_samples > 0 and len(prompts) > max_samples:
        rng = random.Random(seed)
        prompts = rng.sample(prompts, max_samples)
    return prompts


def _drop_kv_cache(model) -> None:
    for name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, name):
            delattr(model, name)
    if hasattr(model, "ea_layer"):
        model.ea_layer.reset_kv()
    torch.cuda.empty_cache()


def generate(model, tokenizer, messages, mode, max_new_tokens, total_token, device_fn):
    """Run generation in eagle or naive mode. Returns (text, new_tokens, stats, wall_time)."""
    _drop_kv_cache(model)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.tolist()
    prompt_tokens = int(enc["input_ids"].shape[-1])

    max_length = max(3072, prompt_tokens + max_new_tokens + total_token + 256)
    input_device = device_fn()
    enc = {k: v.to(input_device) for k, v in enc.items()}

    if input_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    if mode == "eagle":
        output_ids, new_token, _, stats = model.eagenerate(
            torch.as_tensor(input_ids, device=input_device),
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
            return_stats=True,
        )
    elif mode == "naive":
        output_ids, new_token, _ = model.naivegenerate(
            torch.as_tensor(input_ids, device=input_device),
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
        )
        stats = {
            "accepted_tokens": new_token, "proposed_tokens": new_token,
            "acceptance_rate": 1.0, "mean_accepted_length": 0.0, "tree_steps": 0,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if input_device.type == "cuda":
        torch.cuda.synchronize()
    wall_time = time.time() - start

    gen_ids = output_ids[0][len(input_ids[0]):]
    text = tokenizer.decode(gen_ids, spaces_between_special_tokens=False)
    return text, int(new_token), stats, wall_time, prompt_tokens


def check_output_quality(text: str) -> list[str]:
    """Basic quality checks for generated text."""
    issues = []
    if not text.strip():
        issues.append("empty_output")
    words = text.split()
    if len(words) > 10:
        from collections import Counter
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        counts = Counter(trigrams)
        most_common_count = counts.most_common(1)[0][1] if counts else 0
        if most_common_count > max(5, len(trigrams) * 0.3):
            issues.append(f"repetitive(top_trigram_count={most_common_count})")
    return issues


@torch.inference_mode()
def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_kwargs = {
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "total_token": args.total_token,
        "depth": args.depth,
        "top_k": args.top_k,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "use_eagle3": False,
    }

    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except Exception:
        has_accelerate = False

    if args.device_map_auto and has_accelerate:
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["device_map"] = "auto"

    print("[loading] base model + EAGLE2 head ...")
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

    if args.question_file and os.path.exists(args.question_file):
        print(f"[data] loading prompts from {args.question_file} (seed={args.seed})")
        prompts = load_sharegpt_prompts(
            args.question_file, max_samples=args.max_samples, seed=args.seed,
        )
    else:
        print("[data] using built-in test prompts")
        prompts = BUILTIN_PROMPTS
        if args.max_samples > 0:
            prompts = prompts[:args.max_samples]

    print(f"[config] mode=eagle+naive, prompts={len(prompts)}, max_new_tokens={args.max_new_tokens}")

    # Warmup
    if prompts and args.warmup:
        generate(model, tokenizer, prompts[0], "eagle", 32, args.total_token, model_input_device)
        _drop_kv_cache(model)
        generate(model, tokenizer, prompts[0], "naive", 32, args.total_token, model_input_device)
        _drop_kv_cache(model)
        print("[warmup] done")

    out_path = None
    fout = None
    if args.answer_file:
        out_path = Path(args.answer_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fout = out_path.open("w", encoding="utf-8")

    max_prompt_len = args.max_prompt_len

    results = []
    for i, messages in enumerate(prompts):
        user_text = messages[-1]["content"][:80]
        print(f"\n[{i+1}/{len(prompts)}] {user_text}...")

        # Pre-check prompt length to avoid OOM on extremely long inputs
        _check_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _check_len = len(tokenizer.encode(_check_prompt, add_special_tokens=False))
        if _check_len > max_prompt_len:
            print(f"  SKIP: prompt too long ({_check_len} tokens > {max_prompt_len} limit)")
            continue

        eagle_text, eagle_tokens, eagle_stats, eagle_time, prompt_len = generate(
            model, tokenizer, messages, "eagle", args.max_new_tokens, args.total_token, model_input_device,
        )

        naive_text, naive_tokens, naive_stats, naive_time, _ = generate(
            model, tokenizer, messages, "naive", args.max_new_tokens, args.total_token, model_input_device,
        )

        # Prefix-based match: bf16 speculative decoding has small numerical drift
        # vs naive, so strict equality is too harsh. Count chars from the start
        # that agree, normalize by min length. ≥0.95 counts as a pass.
        e_strip, n_strip = eagle_text.strip(), naive_text.strip()
        common_chars = 0
        for c1, c2 in zip(e_strip, n_strip):
            if c1 != c2:
                break
            common_chars += 1
        min_len = min(len(e_strip), len(n_strip))
        prefix_ratio = common_chars / min_len if min_len > 0 else 0.0
        text_match = prefix_ratio >= 0.95
        exact_match = e_strip == n_strip

        speedup = naive_time / eagle_time if eagle_time > 0 else 0.0
        acceptance_rate = float(eagle_stats.get("acceptance_rate", 0.0))
        mean_accepted_len = float(eagle_stats.get("mean_accepted_length", 0.0))
        eagle_issues = check_output_quality(eagle_text)

        print(f"  eagle: {eagle_tokens} tokens, {eagle_time:.2f}s, τ={mean_accepted_len:.2f}, accept={acceptance_rate:.3f}")
        print(f"  naive: {naive_tokens} tokens, {naive_time:.2f}s")
        print(f"  speedup: {speedup:.2f}x, prefix={prefix_ratio:.3f} match={text_match} exact={exact_match}")
        if eagle_issues:
            print(f"  quality issues: {eagle_issues}")

        # Show first 200 chars of eagle output
        preview = eagle_text[:200].replace("\n", "\\n")
        print(f"  output: {preview}")

        rec = {
            "idx": i,
            "user_prompt": messages[-1]["content"],
            "prompt_tokens": prompt_len,
            "eagle_tokens": eagle_tokens,
            "naive_tokens": naive_tokens,
            "eagle_time": round(eagle_time, 4),
            "naive_time": round(naive_time, 4),
            "speedup": round(speedup, 4),
            "acceptance_rate": round(acceptance_rate, 4),
            "mean_accepted_length": round(mean_accepted_len, 4),
            "accepted_tokens": int(eagle_stats.get("accepted_tokens", 0)),
            "proposed_tokens": int(eagle_stats.get("proposed_tokens", 0)),
            "tree_steps": int(eagle_stats.get("tree_steps", 0)),
            "prefix_ratio": round(prefix_ratio, 4),
            "common_prefix_chars": common_chars,
            "text_match": text_match,
            "exact_match": exact_match,
            "eagle_issues": eagle_issues,
            "eagle_output": eagle_text,
            "naive_output": naive_text,
        }
        results.append(rec)

        if fout:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    if fout:
        fout.close()

    # Summary
    n = len(results)
    if n == 0:
        print("\n[summary] no results")
        return

    avg_accept = sum(r["acceptance_rate"] for r in results) / n
    avg_tau = sum(r["mean_accepted_length"] for r in results) / n
    avg_speedup = sum(r["speedup"] for r in results) / n
    avg_prefix = sum(r["prefix_ratio"] for r in results) / n
    match_count = sum(1 for r in results if r["text_match"])
    exact_count = sum(1 for r in results if r["exact_match"])
    issue_count = sum(1 for r in results if r["eagle_issues"])
    total_eagle_time = sum(r["eagle_time"] for r in results)
    total_naive_time = sum(r["naive_time"] for r in results)
    wall_speedup = total_naive_time / total_eagle_time if total_eagle_time > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"[summary] {n} prompts")
    print(f"  avg acceptance length (τ): {avg_tau:.2f}")
    print(f"  avg acceptance rate (tree): {avg_accept:.4f}")
    print(f"  avg per-sample speedup: {avg_speedup:.2f}x")
    print(f"  wall-clock speedup: {wall_speedup:.2f}x")
    print(f"  avg prefix consistency: {avg_prefix:.4f}")
    print(f"  prefix match (≥0.95): {match_count}/{n}")
    print(f"  exact match: {exact_count}/{n}")
    print(f"  quality issues: {issue_count}/{n}")
    print(f"{'='*60}")

    if out_path:
        summary = {
            "num_prompts": n,
            "avg_acceptance_length": round(avg_tau, 4),
            "avg_acceptance_rate": round(avg_accept, 4),
            "avg_speedup": round(avg_speedup, 4),
            "wall_speedup": round(wall_speedup, 4),
            "avg_prefix_ratio": round(avg_prefix, 4),
            "prefix_match_ratio": round(match_count / n, 4),
            "exact_match_ratio": round(exact_count / n, 4),
            "quality_issue_ratio": round(issue_count / n, 4),
        }
        summary_path = out_path.parent / "eval_sharegpt_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[output] detail: {out_path}")
        print(f"[output] summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answer-file", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # Tree config: defaults follow EAGLE-2 paper Appendix A for 13B class
    # (also reasonable for 14B). Use --total-token -1 to enable auto-tune.
    parser.add_argument("--total-token", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for random sampling from question file (reproducible)")
    parser.add_argument("--max-prompt-len", type=int, default=8192,
                        help="Skip prompts longer than this (tokens) to avoid OOM")
    parser.add_argument("--warmup", dest="warmup", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    print(f"[args] {vars(args)}")
    run_eval(args)


if __name__ == "__main__":
    main()
