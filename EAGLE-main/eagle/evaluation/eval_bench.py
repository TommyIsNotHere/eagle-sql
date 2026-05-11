"""Unified speculative-decoding evaluation for Alpaca / HumanEval / MT-Bench.

Reports speedup metrics under the EAGLE tree configuration:
- Accept length τ (mean tokens accepted per draft step)
- Throughput (tokens / second) for eagle and naive decoding
- Wall-clock speedup (naive_time / eagle_time)

No fastchat dependency: prompts are built via tokenizer.apply_chat_template(),
which matches the base model's actual chat template (Qwen2.5-Coder uses the
chatml format embedded in its tokenizer_config.json).

Usage:
    python -m eagle.evaluation.eval_bench \
        --ea-model-path /path/to/eagle2/infer \
        --base-model-path /path/to/Qwen2.5-Coder-14B-Instruct \
        --bench-name alpaca \
        --answer-file /path/to/output.jsonl \
        --bf16
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

try:
    from ..model.ea_model import EaModel
except Exception:
    from eagle.model.ea_model import EaModel


# Bench data: <EAGLE-main>/eagle/data/<bench>/question.jsonl
BENCH_DIR = Path(__file__).resolve().parent.parent / "data"

SUPPORTED_BENCHES = ("alpaca", "humaneval", "mt_bench")


# ─── Data loading ────────────────────────────────────────────────────────────

def load_bench_questions(path: str, max_samples: int = 0, seed: int = 42):
    """Load questions from bench JSONL.

    Schema (alpaca / humaneval / mt_bench):
        {"question_id": int, "category": str, "turns": [str, ...],
         "reference": [str, ...]  (optional)}

    When max_samples > 0, randomly sample max_samples questions with the given
    seed so subsets are reproducible across runs and across modes.
    """
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    if max_samples > 0 and len(questions) > max_samples:
        rng = random.Random(seed)
        questions = rng.sample(questions, max_samples)
        questions.sort(key=lambda q: q.get("question_id", 0))
    return questions


# ─── Generation ──────────────────────────────────────────────────────────────

def _drop_kv_cache(model) -> None:
    for name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, name):
            delattr(model, name)
    if hasattr(model, "ea_layer"):
        model.ea_layer.reset_kv()
    torch.cuda.empty_cache()


def generate(model, tokenizer, messages, mode, max_new_tokens, total_token, device_fn):
    """Run a single generation in `eagle` or `naive` mode.

    Returns: (text, new_tokens, stats, wall_time, prompt_tokens)
    """
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


def strip_special_tokens(text: str, tokenizer) -> str:
    """Remove tokenizer special tokens (e.g. <|im_end|>) from generated text."""
    for special in tokenizer.special_tokens_map.values():
        if isinstance(special, list):
            for tok in special:
                text = text.replace(tok, "")
        else:
            text = text.replace(special, "")
    return text.strip()


# ─── Main eval loop ──────────────────────────────────────────────────────────

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

    print(f"[loading] base model + EAGLE{'3' if args.use_eagle3 else '2'} head ...")
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

    # Resolve question file
    if args.question_file:
        q_path = args.question_file
    else:
        q_path = str(BENCH_DIR / args.bench_name / "question.jsonl")
    print(f"[data] loading questions from {q_path} (seed={args.seed})")
    questions = load_bench_questions(q_path, max_samples=args.max_samples, seed=args.seed)

    print(f"[config] bench={args.bench_name}, samples={len(questions)}, max_new_tokens={args.max_new_tokens}")
    print(f"[config] tree: total_token={args.total_token} depth={args.depth} top_k={args.top_k}")

    # Warmup with first sample (32 tokens each side, both modes)
    if questions and args.warmup:
        warm_msgs = [{"role": "user", "content": questions[0]["turns"][0]}]
        generate(model, tokenizer, warm_msgs, "eagle", 32, args.total_token, model_input_device)
        _drop_kv_cache(model)
        generate(model, tokenizer, warm_msgs, "naive", 32, args.total_token, model_input_device)
        _drop_kv_cache(model)
        print("[warmup] done")

    out_path = Path(args.answer_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    results = []
    for qi, question in enumerate(questions):
        qid = question.get("question_id", qi)
        turns = question.get("turns", [])
        print(f"\n[{qi+1}/{len(questions)}] qid={qid} turns={len(turns)}")

        # Eagle and naive maintain separate histories: bf16 numerical drift
        # may cause divergent continuations on later turns. This matches how
        # MT-Bench evaluates per-model conversations.
        messages_eagle = []
        messages_naive = []

        eagle_turns = []
        naive_turns = []

        for tj, qs in enumerate(turns):
            messages_eagle.append({"role": "user", "content": qs})
            messages_naive.append({"role": "user", "content": qs})

            # Skip if prompt would blow up the budget (rare for these benches)
            check_prompt = tokenizer.apply_chat_template(messages_eagle, tokenize=False, add_generation_prompt=True)
            check_len = len(tokenizer.encode(check_prompt, add_special_tokens=False))
            if check_len > args.max_prompt_len:
                print(f"  SKIP turn{tj+1}: prompt {check_len} tok > {args.max_prompt_len}")
                break

            eagle_text, eagle_tokens, eagle_stats, eagle_time, eagle_prompt_len = generate(
                model, tokenizer, messages_eagle, "eagle",
                args.max_new_tokens, args.total_token, model_input_device,
            )
            naive_text, naive_tokens, naive_stats, naive_time, naive_prompt_len = generate(
                model, tokenizer, messages_naive, "naive",
                args.max_new_tokens, args.total_token, model_input_device,
            )

            eagle_clean = strip_special_tokens(eagle_text, tokenizer)
            naive_clean = strip_special_tokens(naive_text, tokenizer)

            messages_eagle.append({"role": "assistant", "content": eagle_clean})
            messages_naive.append({"role": "assistant", "content": naive_clean})

            speedup = naive_time / eagle_time if eagle_time > 0 else 0.0
            eagle_tput = eagle_tokens / eagle_time if eagle_time > 0 else 0.0
            naive_tput = naive_tokens / naive_time if naive_time > 0 else 0.0
            tau = float(eagle_stats.get("mean_accepted_length", 0.0))
            accept_rate = float(eagle_stats.get("acceptance_rate", 0.0))

            print(f"  turn{tj+1}: eagle {eagle_tokens}tok/{eagle_time:.2f}s={eagle_tput:.1f}tok/s τ={tau:.2f}")
            print(f"          naive {naive_tokens}tok/{naive_time:.2f}s={naive_tput:.1f}tok/s")
            print(f"          speedup={speedup:.2f}x")

            eagle_turns.append({
                "turn_idx": tj,
                "output": eagle_clean,
                "new_tokens": eagle_tokens,
                "wall_time": round(eagle_time, 4),
                "throughput": round(eagle_tput, 2),
                "mean_accepted_length": round(tau, 4),
                "acceptance_rate": round(accept_rate, 4),
                "accepted_tokens": int(eagle_stats.get("accepted_tokens", 0)),
                "proposed_tokens": int(eagle_stats.get("proposed_tokens", 0)),
                "tree_steps": int(eagle_stats.get("tree_steps", 0)),
                "prompt_tokens": eagle_prompt_len,
            })
            naive_turns.append({
                "turn_idx": tj,
                "output": naive_clean,
                "new_tokens": naive_tokens,
                "wall_time": round(naive_time, 4),
                "throughput": round(naive_tput, 2),
                "prompt_tokens": naive_prompt_len,
            })

        rec = {
            "question_id": qid,
            "category": question.get("category", ""),
            "bench_name": args.bench_name,
            "num_turns": len(eagle_turns),
            "eagle": eagle_turns,
            "naive": naive_turns,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        results.append(rec)

    fout.close()

    # ─── Aggregate summary ───────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("\n[summary] no results")
        return

    turn_records = [t for r in results for t in r["eagle"]]
    naive_records = [t for r in results for t in r["naive"]]
    num_turn_records = len(turn_records)

    total_eagle_time = sum(t["wall_time"] for t in turn_records)
    total_naive_time = sum(t["wall_time"] for t in naive_records)
    total_eagle_tokens = sum(t["new_tokens"] for t in turn_records)
    total_naive_tokens = sum(t["new_tokens"] for t in naive_records)
    total_accepted = sum(t["accepted_tokens"] for t in turn_records)
    total_proposed = sum(t["proposed_tokens"] for t in turn_records)

    avg_tau = sum(t["mean_accepted_length"] for t in turn_records) / num_turn_records
    # α (per-token acceptance rate): primary metric for chain config (top_k=1).
    # Weighted-global form is more accurate than averaging per-turn rates
    # when turns have unequal lengths.
    alpha_global = (total_accepted / total_proposed) if total_proposed > 0 else 0.0
    alpha_mean = sum(t["acceptance_rate"] for t in turn_records) / num_turn_records
    eagle_tput = total_eagle_tokens / total_eagle_time if total_eagle_time > 0 else 0.0
    naive_tput = total_naive_tokens / total_naive_time if total_naive_time > 0 else 0.0
    wall_speedup = total_naive_time / total_eagle_time if total_eagle_time > 0 else 0.0

    is_chain = (args.top_k == 1)

    summary = {
        "bench_name": args.bench_name,
        "num_questions": n,
        "num_turn_records": num_turn_records,
        "decode_shape": "chain" if is_chain else "tree",
        "tree_config": {
            "total_token": args.total_token,
            "depth": args.depth,
            "top_k": args.top_k,
        },
        "acceptance_rate_global": round(alpha_global, 4),
        "acceptance_rate_mean": round(alpha_mean, 4),
        "mean_accepted_length": round(avg_tau, 4),
        "eagle_throughput_tokens_per_s": round(eagle_tput, 2),
        "naive_throughput_tokens_per_s": round(naive_tput, 2),
        "wall_speedup": round(wall_speedup, 4),
        "total_eagle_time_s": round(total_eagle_time, 2),
        "total_naive_time_s": round(total_naive_time, 2),
        "total_eagle_tokens": total_eagle_tokens,
        "total_naive_tokens": total_naive_tokens,
        "total_accepted_tokens": total_accepted,
        "total_proposed_tokens": total_proposed,
    }

    print(f"\n{'='*60}")
    print(f"[summary] {args.bench_name}  questions={n}  turns={num_turn_records}  shape={summary['decode_shape']}")
    print(f"  α (per-token accept rate): {alpha_global:.4f}   <- primary, token-weighted")
    print(f"  α (per-turn mean):         {alpha_mean:.4f}   <- reference, turn-equal-weight")
    print(f"  τ (mean accept length):    {avg_tau:.2f}")
    print(f"  Throughput (eagle): {eagle_tput:.1f} tok/s")
    print(f"  Throughput (naive): {naive_tput:.1f} tok/s")
    print(f"  Wall speedup:       {wall_speedup:.2f}x")
    print(f"{'='*60}")

    summary_path = out_path.parent / f"eval_{args.bench_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[output] detail: {out_path}")
    print(f"[output] summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="alpaca",
                        choices=list(SUPPORTED_BENCHES),
                        help="Benchmark name; resolves to eagle/data/<name>/question.jsonl")
    parser.add_argument("--question-file", type=str, default="",
                        help="Override the auto-resolved bench path")
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # Tree defaults aligned with eval_sharegpt.py (EAGLE-2 paper Appendix A,
    # 13B class; also reasonable for 14B). Set total-token to -1 for auto-tune.
    parser.add_argument("--total-token", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sampling questions (reproducible subsets)")
    parser.add_argument("--max-prompt-len", type=int, default=8192,
                        help="Skip turns whose prompt would exceed this length")
    parser.add_argument("--warmup", dest="warmup", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use-eagle3", action="store_true")
    args = parser.parse_args()

    print(f"[args] {vars(args)}")
    run_eval(args)


if __name__ == "__main__":
    main()
