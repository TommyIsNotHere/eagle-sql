"""Generate Text-to-SQL predictions for BIRD with EAGLE3 + Qwen models."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from collections import Counter, deque
from typing import Any, Callable

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    import shortuuid
except Exception:
    shortuuid = None

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


def _fmt_float(value: Any, digits: int = 4, na: str = "na") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return na


def _preview_text(text: str, max_chars: int = 120) -> str:
    s = str(text or "").replace("\n", "\\n").replace("\r", "\\r").strip()
    max_chars = max(16, int(max_chars))
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."


def _counter_summary(counter: Counter, topn: int = 3) -> str:
    if not counter:
        return "none"
    parts = []
    for key, val in counter.most_common(max(1, int(topn))):
        parts.append(f"{key}:{int(val)}")
    return ",".join(parts)


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
        "..text2sql.bird.prompt_builder",
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
        "..text2sql.bird.postprocess_sql",
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


def _new_answer_id() -> str:
    if shortuuid is not None:
        return shortuuid.uuid()
    return uuid.uuid4().hex


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


def _resolve_required_max_length(prompt_tokens: int, args) -> int:
    # EaModel.eagenerate internally subtracts (total_tokens + 10) from max_length
    # for its decoding loop, so we reserve enough room here for both prompt and
    # requested generation budget.
    required = (
        int(prompt_tokens)
        + int(args.max_new_token)
        + int(args.total_token)
        + int(args.kv_cache_margin)
        + 32
    )
    return max(int(args.kv_cache_min_length), required)


def _get_kv_cache_capacity(model) -> int:
    try:
        past = getattr(model, "past_key_values", None)
        if not past:
            return 0
        key_cache = past[0][0]
        data = getattr(key_cache, "data", None)
        if data is None:
            return 0
        return int(data.shape[2])
    except Exception:
        return 0


def _drop_kv_cache(model) -> None:
    for name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, name):
            delattr(model, name)


def _reset_tree_state_on_base_model(base_model) -> None:
    try:
        base_core = getattr(base_model, "model", None)
        if base_core is None:
            return
        if hasattr(base_core, "tree_mask"):
            base_core.tree_mask = None
        if hasattr(base_core, "tree_mode"):
            base_core.tree_mode = None
    except Exception:
        pass


def _reset_tree_state(model) -> None:
    try:
        base_model = getattr(model, "base_model", None)
        _reset_tree_state_on_base_model(base_model)
    except Exception:
        pass


def _reset_ea_layer_kv(model) -> None:
    try:
        ea_layer = getattr(model, "ea_layer", None)
        if ea_layer is not None and hasattr(ea_layer, "reset_kv"):
            ea_layer.reset_kv()
    except Exception:
        pass


def _hard_reset_decode_state(model, *, drop_kv: bool = True, reset_tree_state: bool = True, reset_ea_kv: bool = True) -> None:
    if reset_tree_state:
        _reset_tree_state(model)
    if reset_ea_kv:
        _reset_ea_layer_kv(model)
    if drop_kv:
        _drop_kv_cache(model)


def _state_snapshot(model) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "kv_cache_capacity": int(_get_kv_cache_capacity(model)),
        "has_past_key_values": int(bool(getattr(model, "past_key_values", None))),
        "tree_mask_active": 0,
        "tree_mask_last_dim": 0,
        "tree_mode_active": "",
    }
    try:
        base_model = getattr(model, "base_model", None)
        base_core = getattr(base_model, "model", None)
        if base_core is not None:
            tree_mask = getattr(base_core, "tree_mask", None)
            if tree_mask is not None:
                snap["tree_mask_active"] = 1
                if hasattr(tree_mask, "size"):
                    snap["tree_mask_last_dim"] = int(tree_mask.size(-1))
            tree_mode = getattr(base_core, "tree_mode", None)
            if tree_mode is not None:
                snap["tree_mode_active"] = str(tree_mode)
    except Exception:
        pass
    return snap


def _prefix_state(prefix: str, snap: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in snap.items():
        out[f"{prefix}{k}"] = v
    return out


def _build_gen_diag(gen_ids, max_dump: int = 16) -> dict[str, Any]:
    if isinstance(gen_ids, torch.Tensor):
        ids = gen_ids.detach().tolist()
    else:
        ids = list(gen_ids)
    ids = [int(x) for x in ids]
    n = len(ids)
    if n == 0:
        return {
            "gen_token_count": 0,
            "gen_unique_token_count": 0,
            "gen_top1_token_id": -1,
            "gen_top1_ratio": 0.0,
            "gen_longest_run_len": 0,
            "gen_longest_run_ratio": 0.0,
            "gen_token_ids_head": [],
            "gen_token_ids_tail": [],
        }
    c = Counter(ids)
    top1_token_id, top1_count = c.most_common(1)[0]
    longest_run = 1
    cur_run = 1
    for idx in range(1, n):
        if ids[idx] == ids[idx - 1]:
            cur_run += 1
            if cur_run > longest_run:
                longest_run = cur_run
        else:
            cur_run = 1
    dump = max(0, int(max_dump))
    return {
        "gen_token_count": int(n),
        "gen_unique_token_count": int(len(c)),
        "gen_top1_token_id": int(top1_token_id),
        "gen_top1_ratio": float(top1_count / n),
        "gen_longest_run_len": int(longest_run),
        "gen_longest_run_ratio": float(longest_run / n),
        "gen_token_ids_head": ids[:dump],
        "gen_token_ids_tail": ids[-dump:] if dump > 0 else [],
    }


def _is_degenerate_record(rec: dict[str, Any]) -> bool:
    if rec.get("is_valid_sql"):
        return False
    try:
        top1_ratio = float(rec.get("gen_top1_ratio", 0.0))
        unique_tokens = int(rec.get("gen_unique_token_count", 0))
        new_tokens = int(rec.get("new_tokens", 0))
    except Exception:
        return False
    raw = (rec.get("raw_output") or "").strip()
    repeated_single_char = bool(raw) and (len(set(raw)) == 1)
    return (
        new_tokens >= 8
        and top1_ratio >= 0.98
        and unique_tokens <= 2
        and repeated_single_char
    )


def _ensure_kv_cache_capacity(model, required_max_length: int) -> None:
    current = _get_kv_cache_capacity(model)
    if current > 0 and current < required_max_length:
        print(f"[kv-cache] reallocating cache: {current} -> {required_max_length}")
        _drop_kv_cache(model)


def _probe_first_step_logits_from_forward_model(
    forward_model,
    tokenizer,
    enc: dict[str, torch.Tensor],
    *,
    use_cache: bool = False,
) -> dict[str, Any]:
    try:
        with torch.inference_mode():
            out = forward_model(**enc, use_cache=use_cache, return_dict=True)
        logits = out.logits[:, -1, :].float()
        finite_mask = torch.isfinite(logits)
        finite_ratio = float(finite_mask.float().mean().item())
        nan_count = int(torch.isnan(logits).sum().item())
        inf_count = int(torch.isinf(logits).sum().item())
        safe_logits = torch.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
        top1_id = int(torch.argmax(safe_logits, dim=-1).item())
        top1_logit = float(safe_logits[0, top1_id].item())
        probs = torch.nn.functional.softmax(safe_logits, dim=-1)
        top1_prob = float(probs[0, top1_id].item())
        entropy = float((-(probs * torch.log(probs + 1e-12)).sum(dim=-1)).item())
        k = min(5, int(safe_logits.shape[-1]))
        topk_vals, topk_ids = torch.topk(safe_logits, k=k, dim=-1)
        topk_ids_list = [int(x) for x in topk_ids[0].tolist()]
        topk_probs_list = [float(x) for x in probs[0, topk_ids[0]].tolist()]
        topk_toks = []
        for tok_id in topk_ids_list:
            try:
                tok_txt = tokenizer.convert_ids_to_tokens(tok_id)
            except Exception:
                tok_txt = tokenizer.decode(
                    [tok_id],
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
            topk_toks.append(str(tok_txt))
        try:
            top1_tok = tokenizer.convert_ids_to_tokens(top1_id)
        except Exception:
            top1_tok = tokenizer.decode(
                [top1_id],
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
        return {
            "probe_logits_finite_ratio": finite_ratio,
            "probe_logits_nan_count": nan_count,
            "probe_logits_inf_count": inf_count,
            "probe_top1_token_id": top1_id,
            "probe_top1_token_text": str(top1_tok),
            "probe_top1_logit": top1_logit,
            "probe_top1_prob": top1_prob,
            "probe_top2_prob": float(topk_probs_list[1]) if len(topk_probs_list) > 1 else 0.0,
            "probe_top1_top2_margin": (
                float(topk_probs_list[0] - topk_probs_list[1]) if len(topk_probs_list) > 1 else float(topk_probs_list[0])
            ),
            "probe_topk_token_ids": topk_ids_list,
            "probe_topk_token_texts": topk_toks,
            "probe_topk_probs": topk_probs_list,
            "probe_entropy": entropy,
        }
    except Exception as e:
        return {
            "probe_error": f"{type(e).__name__}: {e}",
        }


def _probe_first_step_logits(model, tokenizer, enc: dict[str, torch.Tensor]) -> dict[str, Any]:
    base = getattr(model, "base_model", None)
    return _probe_first_step_logits_from_forward_model(base, tokenizer, enc, use_cache=False)


def _resolve_layered_probe_device(args, runtime_device: torch.device) -> torch.device:
    choice = str(args.layered_probe_device).strip().lower()
    if choice == "same":
        return runtime_device
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("layered_probe_device=cuda but CUDA is unavailable")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return runtime_device


def _resolve_layered_probe_dtype(args, probe_device: torch.device) -> torch.dtype:
    choice = str(args.layered_probe_dtype).strip().lower()
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "fp32":
        return torch.float32
    # auto
    if probe_device.type == "cpu":
        return torch.float32
    if args.bf16:
        return torch.bfloat16
    return torch.float16


def _model_input_device_from_model(forward_model, fallback: torch.device) -> torch.device:
    try:
        return forward_model.model.embed_tokens.weight.device
    except Exception:
        return fallback


def _load_kv_rewrite_model(base_model_path: str, *, torch_dtype: torch.dtype, attn_implementation: str):
    model_type = AutoConfig.from_pretrained(base_model_path).architectures[0]
    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    if model_type == "LlamaForCausalLM":
        from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM

        return KVLlamaForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    if model_type == "Qwen2ForCausalLM":
        from eagle.model.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM

        return KVQwen2ForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    if model_type == "Qwen3ForCausalLM":
        qwen3_mod = importlib.import_module("eagle.model.modeling_qwen3_kv")
        if not hasattr(qwen3_mod, "Qwen3ForCausalLM"):
            available = [n for n in dir(qwen3_mod) if "Qwen3" in n][:20]
            raise ImportError(
                "Qwen3ForCausalLM not found in eagle.model.modeling_qwen3_kv. "
                f"available_qwen3_symbols={available}"
            )
        cls = getattr(qwen3_mod, "Qwen3ForCausalLM")
        return cls.from_pretrained(base_model_path, **load_kwargs)

    from eagle.model.modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM

    return KVMixtralForCausalLM.from_pretrained(base_model_path, **load_kwargs)


def _prepare_layered_probe_inputs(samples: list[dict[str, Any]], tokenizer, n_samples: int) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    n = max(0, int(n_samples))
    if n <= 0:
        return inputs
    for i, sample in enumerate(samples[:n], start=1):
        user_prompt = build_bird_user_prompt(sample)
        msgs = build_messages(user_prompt)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        enc_cpu = {k: v.cpu() for k, v in enc.items()}
        inputs.append(
            {
                "sample_index": int(i),
                "question_id": sample.get("question_id"),
                "db_id": sample.get("db_id"),
                "prompt_tokens": int(enc_cpu["input_ids"].shape[-1]),
                "enc_cpu": enc_cpu,
            }
        )
    return inputs


def _summarize_layered_probe_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_layer.setdefault(str(r.get("layer", "unknown")), []).append(r)

    for layer, rows in by_layer.items():
        n = len(rows)
        finite_vals: list[float] = []
        nan_sum = 0
        inf_sum = 0
        err = 0
        for r in rows:
            probe = r.get("probe", {})
            if probe.get("probe_error"):
                err += 1
                continue
            try:
                finite_vals.append(float(probe.get("probe_logits_finite_ratio", 1.0)))
                nan_sum += int(probe.get("probe_logits_nan_count", 0))
                inf_sum += int(probe.get("probe_logits_inf_count", 0))
            except Exception:
                continue
        out[layer] = {
            "n": int(n),
            "error_count": int(err),
            "finite_min": min(finite_vals) if finite_vals else float("nan"),
            "finite_mean": (sum(finite_vals) / len(finite_vals)) if finite_vals else float("nan"),
            "nan_sum": int(nan_sum),
            "inf_sum": int(inf_sum),
        }
    return out


def _layer_is_numerically_bad(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    finite_min = float(summary.get("finite_min", 1.0))
    return (
        int(summary.get("nan_sum", 0)) > 0
        or int(summary.get("inf_sum", 0)) > 0
        or finite_min < 0.999999
    )


def _diagnose_layered_probe_stack(layer_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hf = layer_summary.get("hf_native", {})
    kv = layer_summary.get("kv_rewrite", {})
    eagle = layer_summary.get("eagle_wrapped", {})
    hf_bad = _layer_is_numerically_bad(hf)
    kv_bad = _layer_is_numerically_bad(kv)
    eagle_bad = _layer_is_numerically_bad(eagle)

    likely = "insufficient_signal"
    confidence = "low"
    if hf_bad and kv_bad and eagle_bad:
        likely = "upstream_runtime_or_base_weight_numerical_instability"
        confidence = "high"
    elif (not hf_bad) and kv_bad and eagle_bad:
        likely = "kv_rewrite_layer_numerical_instability"
        confidence = "high"
    elif (not hf_bad) and (not kv_bad) and eagle_bad:
        likely = "eagle_wrapper_decode_chain_numerical_instability"
        confidence = "high"
    elif hf_bad and (not kv_bad):
        likely = "hf_native_only_issue_or_probe_config_mismatch"
        confidence = "medium"
    elif (not hf_bad) and (not kv_bad) and (not eagle_bad):
        likely = "no_first_step_numerical_instability_detected"
        confidence = "medium"

    return {
        "likely_bad_layer": likely,
        "confidence": confidence,
        "hf_bad": bool(hf_bad),
        "kv_bad": bool(kv_bad),
        "eagle_bad": bool(eagle_bad),
    }


def _run_layer_probe_for_model(
    *,
    layer_name: str,
    forward_model,
    tokenizer,
    probe_inputs: list[dict[str, Any]],
    fallback_device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in probe_inputs:
        input_device = _model_input_device_from_model(forward_model, fallback_device)
        enc = {k: v.to(input_device) for k, v in item["enc_cpu"].items()}
        _reset_tree_state_on_base_model(forward_model)
        probe = _probe_first_step_logits_from_forward_model(forward_model, tokenizer, enc, use_cache=False)
        rec = {
            "layer": layer_name,
            "sample_index": int(item["sample_index"]),
            "question_id": item.get("question_id"),
            "db_id": item.get("db_id"),
            "prompt_tokens": int(item.get("prompt_tokens", 0)),
            "probe": probe,
        }
        records.append(rec)
    return records


def _cleanup_forward_model(forward_model) -> None:
    try:
        del forward_model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_layered_probe_stack(
    *,
    args,
    model,
    runtime_device: torch.device,
    samples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    n_samples = max(0, int(args.layered_probe_samples))
    if n_samples <= 0:
        return None
    if not samples:
        return None

    probe_device = _resolve_layered_probe_device(args, runtime_device)
    probe_dtype = _resolve_layered_probe_dtype(args, probe_device)
    probe_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    probe_inputs = _prepare_layered_probe_inputs(samples, probe_tokenizer, n_samples)
    if not probe_inputs:
        return None

    print(
        "[layered-probe] "
        f"enabled=1 n_samples={len(probe_inputs)} "
        f"probe_device={probe_device} probe_dtype={probe_dtype}"
    )

    records: list[dict[str, Any]] = []

    # Probe EAGLE wrapped path first (current serving path).
    try:
        _hard_reset_decode_state(model, drop_kv=True, reset_tree_state=True, reset_ea_kv=True)
    except Exception:
        pass
    records.extend(
        _run_layer_probe_for_model(
            layer_name="eagle_wrapped",
            forward_model=model.base_model,
            tokenizer=probe_tokenizer,
            probe_inputs=probe_inputs,
            fallback_device=runtime_device,
        )
    )

    # Probe HF native path.
    hf_native = None
    try:
        print("[layered-probe] loading hf_native model...")
        load_kwargs: dict[str, Any] = {"torch_dtype": probe_dtype}
        if args.attn_implementation:
            load_kwargs["attn_implementation"] = args.attn_implementation
        hf_native = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
        if probe_device.type != "cpu":
            hf_native = hf_native.to(probe_device)
        hf_native.eval()
        records.extend(
            _run_layer_probe_for_model(
                layer_name="hf_native",
                forward_model=hf_native,
                tokenizer=probe_tokenizer,
                probe_inputs=probe_inputs,
                fallback_device=probe_device,
            )
        )
    except Exception as e:
        records.append(
            {
                "layer": "hf_native",
                "sample_index": -1,
                "question_id": None,
                "db_id": None,
                "prompt_tokens": 0,
                "probe": {"probe_error": f"{type(e).__name__}: {e}"},
            }
        )
    finally:
        _cleanup_forward_model(hf_native)

    # Probe KV rewritten path.
    kv_rewrite = None
    try:
        print("[layered-probe] loading kv_rewrite model...")
        kv_rewrite = _load_kv_rewrite_model(
            args.base_model_path,
            torch_dtype=probe_dtype,
            attn_implementation=args.attn_implementation,
        )
        if probe_device.type != "cpu":
            kv_rewrite = kv_rewrite.to(probe_device)
        kv_rewrite.eval()
        records.extend(
            _run_layer_probe_for_model(
                layer_name="kv_rewrite",
                forward_model=kv_rewrite,
                tokenizer=probe_tokenizer,
                probe_inputs=probe_inputs,
                fallback_device=probe_device,
            )
        )
    except Exception as e:
        records.append(
            {
                "layer": "kv_rewrite",
                "sample_index": -1,
                "question_id": None,
                "db_id": None,
                "prompt_tokens": 0,
                "probe": {"probe_error": f"{type(e).__name__}: {e}"},
            }
        )
    finally:
        _cleanup_forward_model(kv_rewrite)

    summary_by_layer = _summarize_layered_probe_records(records)
    diagnosis = _diagnose_layered_probe_stack(summary_by_layer)
    for layer, sm in summary_by_layer.items():
        print(
            "[layered-probe-summary] "
            f"layer={layer} n={sm.get('n')} err={sm.get('error_count')} "
            f"finite_min={_fmt_float(sm.get('finite_min'), 6)} "
            f"finite_mean={_fmt_float(sm.get('finite_mean'), 6)} "
            f"nan_sum={sm.get('nan_sum')} inf_sum={sm.get('inf_sum')}"
        )
    print(
        "[layered-probe-diagnosis] "
        f"likely_bad_layer={diagnosis.get('likely_bad_layer')} "
        f"confidence={diagnosis.get('confidence')} "
        f"hf_bad={int(bool(diagnosis.get('hf_bad')))} "
        f"kv_bad={int(bool(diagnosis.get('kv_bad')))} "
        f"eagle_bad={int(bool(diagnosis.get('eagle_bad')))}"
    )

    out_path = Path(args.layered_probe_output).expanduser() if args.layered_probe_output else Path(
        str(args.answer_file) + ".layered_probe.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": len(probe_inputs),
        "probe_device": str(probe_device),
        "probe_dtype": str(probe_dtype),
        "summary_by_layer": summary_by_layer,
        "diagnosis": diagnosis,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[layered-probe] wrote: {out_path.resolve()}")
    return payload


def generate_once(model, tokenizer, msgs, args, device_getter, probe: bool = False):
    if args.hard_reset_before_probe:
        _hard_reset_decode_state(model, drop_kv=True, reset_tree_state=True, reset_ea_kv=True)
    elif args.reset_kv_per_sample:
        _drop_kv_cache(model)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.tolist()
    prompt_tokens = int(enc["input_ids"].shape[-1])
    required_max_length = _resolve_required_max_length(prompt_tokens, args)
    _ensure_kv_cache_capacity(model, required_max_length)
    input_device = device_getter()
    enc = {k: v.to(input_device) for k, v in enc.items()}
    state_before_probe = _state_snapshot(model) if args.log_state_snapshot else {}
    probe_diag = _probe_first_step_logits(model, tokenizer, enc) if probe else {}
    if args.hard_reset_before_generate:
        _hard_reset_decode_state(model, drop_kv=True, reset_tree_state=True, reset_ea_kv=True)
    state_before_generate = _state_snapshot(model) if args.log_state_snapshot else {}
    if input_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    if args.decode_mode == "hf":
        gen_kwargs = {
            "max_new_tokens": int(args.max_new_token),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            # Keep plain-HF diagnostic path independent from custom KV cache behavior.
            "use_cache": False,
            "do_sample": bool(args.temperature > 1e-5),
        }
        if args.temperature > 1e-5:
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = 1.0
        output_ids = model.base_model.generate(**enc, **gen_kwargs)
        new_token = int(output_ids[0].shape[-1] - prompt_tokens)
        stats = {
            "accepted_tokens": int(new_token),
            "proposed_tokens": int(new_token),
            "acceptance_rate": 1.0 if new_token > 0 else 0.0,
            "tree_steps": int(new_token),
        }
    elif args.decode_mode == "naive":
        output_ids, new_token, _ = model.naivegenerate(
            torch.as_tensor(input_ids, device=input_device),
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            max_length=required_max_length,
            log=True,
        )
        stats = {
            "accepted_tokens": int(new_token),
            "proposed_tokens": int(new_token),
            "acceptance_rate": 1.0 if new_token > 0 else 0.0,
            "tree_steps": int(new_token),
        }
    else:
        output_ids, new_token, _, stats = model.eagenerate(
            torch.as_tensor(input_ids, device=input_device),
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            max_length=required_max_length,
            log=True,
            return_stats=True,
        )
    if input_device.type == "cuda":
        torch.cuda.synchronize()
    cost = time.time() - start
    state_after_generate = _state_snapshot(model) if args.log_state_snapshot else {}
    gen_ids = output_ids[0][len(input_ids[0]):]
    raw_text = tokenizer.decode(gen_ids, spaces_between_special_tokens=False)
    gen_diag = _build_gen_diag(gen_ids, max_dump=args.debug_token_dump)
    top1_id = int(gen_diag.get("gen_top1_token_id", -1))
    if top1_id >= 0:
        try:
            top1_tok = tokenizer.convert_ids_to_tokens(top1_id)
        except Exception:
            top1_tok = tokenizer.decode(
                [top1_id],
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
    else:
        top1_tok = ""
    gen_diag["gen_top1_token_text"] = str(top1_tok)
    gen_diag.update(probe_diag)
    if args.log_state_snapshot:
        gen_diag.update(_prefix_state("state_before_probe_", state_before_probe))
        gen_diag.update(_prefix_state("state_before_generate_", state_before_generate))
        gen_diag.update(_prefix_state("state_after_generate_", state_after_generate))
    return raw_text, int(new_token), stats, cost, prompt_tokens, gen_diag


def _classify_collapse_hint(rec: dict[str, Any]) -> str:
    if rec.get("is_valid_sql"):
        return "none"

    probe_error = str(rec.get("probe_error", "") or "")
    if probe_error:
        return "numerical_or_forward_error"

    try:
        finite_ratio = float(rec.get("probe_logits_finite_ratio", 1.0))
        nan_count = int(rec.get("probe_logits_nan_count", 0))
        inf_count = int(rec.get("probe_logits_inf_count", 0))
    except Exception:
        finite_ratio, nan_count, inf_count = 1.0, 0, 0

    if nan_count > 0 or inf_count > 0 or finite_ratio < 0.999999:
        return "numerical_instability"

    # Clean numerics but tree mask remains active across sample boundaries often
    # indicates decode-state contamination instead of weight-level instability.
    try:
        tm_before_probe = int(rec.get("state_before_probe_tree_mask_active", 0))
        tm_before_gen = int(rec.get("state_before_generate_tree_mask_active", 0))
    except Exception:
        tm_before_probe, tm_before_gen = 0, 0
    if (tm_before_probe > 0 or tm_before_gen > 0) and not rec.get("is_valid_sql"):
        return "decode_state_contamination_suspected"

    is_degenerate = bool(rec.get("is_degenerate", False))
    decode_mode = str(rec.get("decode_mode", "") or "")
    try:
        top1_ratio = float(rec.get("gen_top1_ratio", 0.0))
        unique_tokens = int(rec.get("gen_unique_token_count", 0))
        accept_rate = float(rec.get("acceptance_rate", 0.0))
    except Exception:
        top1_ratio, unique_tokens, accept_rate = 0.0, 0, 0.0
    repeated_single_char = bool(rec.get("raw_single_char_repetition", False))

    if (
        is_degenerate
        and decode_mode == "eagle"
        and top1_ratio >= 0.98
        and unique_tokens <= 2
        and repeated_single_char
        and accept_rate >= 0.99
    ):
        return "decode_path_logic_collapse_suspected"

    if is_degenerate:
        return "degenerate_generation_no_numeric_signal"

    invalid_reason = str(rec.get("invalid_reason", "") or "")
    if invalid_reason == "numeric_garbage":
        return "numeric_output_without_sql"
    if invalid_reason:
        return "invalid_sql_non_collapse"
    return "unknown_invalid"


def _format_sample_diag(i: int, total: int, rec: dict[str, Any]) -> str:
    state_info = ""
    if "state_before_probe_tree_mask_active" in rec:
        state_info = (
            f" state_tm={rec.get('state_before_probe_tree_mask_active', 'na')}"
            f"->{rec.get('state_before_generate_tree_mask_active', 'na')}"
            f"->{rec.get('state_after_generate_tree_mask_active', 'na')}"
            f" state_kv={rec.get('state_before_probe_kv_cache_capacity', 'na')}"
            f"->{rec.get('state_before_generate_kv_cache_capacity', 'na')}"
            f"->{rec.get('state_after_generate_kv_cache_capacity', 'na')}"
        )
    return (
        "[diag] "
        f"i={i}/{total} qid={rec.get('question_id')} db={rec.get('db_id')} "
        f"mode={rec.get('decode_mode')} valid={int(bool(rec.get('is_valid_sql')))} "
        f"invalid={rec.get('invalid_reason') or 'none'} collapse={rec.get('collapse_hint')} "
        f"new={rec.get('new_tokens')} acc={_fmt_float(rec.get('acceptance_rate'), 3)} "
        f"gen_top1={rec.get('gen_top1_token_id')}:{rec.get('gen_top1_token_text')} "
        f"top1r={_fmt_float(rec.get('gen_top1_ratio'), 3)} "
        f"uniq={rec.get('gen_unique_token_count')} "
        f"run={rec.get('gen_longest_run_len')}/{rec.get('gen_token_count')} "
        f"probe_fin={_fmt_float(rec.get('probe_logits_finite_ratio'), 6)} "
        f"probe_nan={rec.get('probe_logits_nan_count', 'na')} "
        f"probe_inf={rec.get('probe_logits_inf_count', 'na')} "
        f"probe_H={_fmt_float(rec.get('probe_entropy'), 4)} "
        f"probe_top1={rec.get('probe_top1_token_id', 'na')}:{rec.get('probe_top1_token_text', 'na')} "
        f"p1={_fmt_float(rec.get('probe_top1_prob'), 4)} "
        f"p2={_fmt_float(rec.get('probe_top2_prob'), 4)} "
        f"raw='{rec.get('raw_preview', '')}'"
        f"{state_info}"
    )


def _summarize_probe_recent(records: list[dict[str, Any]]) -> str:
    probe_records = [r for r in records if "probe_logits_finite_ratio" in r or "probe_error" in r]
    if not probe_records:
        return "probe_recent=n/a"

    finite_vals = []
    nan_sum = 0
    inf_sum = 0
    err_count = 0
    entropy_vals = []
    for r in probe_records:
        if r.get("probe_error"):
            err_count += 1
            continue
        try:
            finite_vals.append(float(r.get("probe_logits_finite_ratio", 1.0)))
            nan_sum += int(r.get("probe_logits_nan_count", 0))
            inf_sum += int(r.get("probe_logits_inf_count", 0))
            entropy_vals.append(float(r.get("probe_entropy", 0.0)))
        except Exception:
            continue
    finite_min = min(finite_vals) if finite_vals else 1.0
    entropy_min = min(entropy_vals) if entropy_vals else 0.0
    return (
        f"probe_recent=n={len(probe_records)} err={err_count} "
        f"finite_min={_fmt_float(finite_min, 6)} nan_sum={nan_sum} inf_sum={inf_sum} "
        f"entropy_min={_fmt_float(entropy_min, 4)}"
    )


@torch.inference_mode()
def run_infer(args):
    fallback_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device) if args.device != "auto" else fallback_device
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    load_kwargs = {
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "total_token": args.total_token,
        "depth": args.depth,
        "top_k": args.top_k,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "use_eagle3": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except Exception:
        has_accelerate = False
    use_device_map = bool(args.device_map_auto) and has_accelerate and runtime_device.type == "cuda"
    if use_device_map:
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["device_map"] = "auto"
    elif not has_accelerate and bool(args.device_map_auto):
        print("accelerate not found; ignore --device-map-auto and fallback to single-device model loading.")

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
    if not use_device_map:
        model = model.to(runtime_device)
    tokenizer = model.get_tokenizer()
    model.eval()
    print(
        "[runtime] "
        f"requested_device={args.device} resolved_device={runtime_device} "
        f"device_map_auto={int(bool(args.device_map_auto))} use_device_map={int(use_device_map)} "
        f"accelerate={int(has_accelerate)}"
    )

    def model_input_device():
        # For device_map='auto' (accelerate), different submodules may be split
        # across devices. Inputs must follow embedding weight device.
        try:
            return model.base_model.model.embed_tokens.weight.device
        except Exception:
            return runtime_device

    samples = load_jsonl(args.question_file)
    if args.max_samples > 0:
        samples = samples[: int(args.max_samples)]
    out_path = Path(args.answer_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Create/truncate output file before warmup so path is visible even if warmup fails.
    with out_path.open("w", encoding="utf-8"):
        pass
    print(f"writing predictions to: {out_path.resolve()}")

    layered_probe_result = _run_layered_probe_stack(
        args=args,
        model=model,
        runtime_device=runtime_device,
        samples=samples,
    )
    if layered_probe_result is not None and args.layered_probe_only:
        print("[layered-probe] probe-only enabled, skip generation.")
        return

    # warmup
    if samples and args.warmup:
        warm_prompt = build_bird_user_prompt(samples[0])
        msgs = build_messages(warm_prompt)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        warm_enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        warm_input_device = model_input_device()
        warm_enc = {k: v.to(warm_input_device) for k, v in warm_enc.items()}
        input_ids = warm_enc["input_ids"].tolist()
        warm_prompt_tokens = len(input_ids[0])
        warm_max_length = _resolve_required_max_length(warm_prompt_tokens, args)
        _ensure_kv_cache_capacity(model, warm_max_length)
        warmup_mode = args.decode_mode if args.warmup_decode_mode == "auto" else args.warmup_decode_mode
        print(
            "[warmup] "
            f"enabled=1 mode={warmup_mode} prompt_tokens={warm_prompt_tokens} "
            f"max_length={warm_max_length} probe={int(bool(args.warmup_probe))}"
        )
        if args.warmup_probe:
            warm_probe_before = _probe_first_step_logits(model, tokenizer, warm_enc)
            print(
                "[warmup-probe-before] "
                f"fin={_fmt_float(warm_probe_before.get('probe_logits_finite_ratio'), 6)} "
                f"nan={warm_probe_before.get('probe_logits_nan_count', 'na')} "
                f"inf={warm_probe_before.get('probe_logits_inf_count', 'na')} "
                f"top1={warm_probe_before.get('probe_top1_token_id', 'na')}:{warm_probe_before.get('probe_top1_token_text', 'na')} "
                f"p1={_fmt_float(warm_probe_before.get('probe_top1_prob'), 4)}"
            )
        if warmup_mode == "hf":
            _ = model.base_model.generate(
                torch.as_tensor(input_ids, device=warm_input_device),
                max_new_tokens=max(8, int(args.max_new_token)),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=False,
            )
        elif warmup_mode == "naive":
            _ = model.naivegenerate(
                torch.as_tensor(input_ids, device=warm_input_device),
                temperature=args.temperature,
                max_length=warm_max_length,
                log=False,
            )
        else:
            _ = model.eagenerate(
                torch.as_tensor(input_ids, device=warm_input_device),
                temperature=args.temperature,
                max_length=warm_max_length,
                log=False,
            )
        if args.warmup_probe:
            warm_probe_after = _probe_first_step_logits(model, tokenizer, warm_enc)
            print(
                "[warmup-probe-after] "
                f"fin={_fmt_float(warm_probe_after.get('probe_logits_finite_ratio'), 6)} "
                f"nan={warm_probe_after.get('probe_logits_nan_count', 'na')} "
                f"inf={warm_probe_after.get('probe_logits_inf_count', 'na')} "
                f"top1={warm_probe_after.get('probe_top1_token_id', 'na')}:{warm_probe_after.get('probe_top1_token_text', 'na')} "
                f"p1={_fmt_float(warm_probe_after.get('probe_top1_prob'), 4)}"
            )
        if args.hard_reset_after_warmup:
            print("[warmup] hard reset decode state after warmup")
            _hard_reset_decode_state(model, drop_kv=True, reset_tree_state=True, reset_ea_kv=True)
        elif args.reset_kv_per_sample:
            _drop_kv_cache(model)
    elif samples:
        print("[warmup] enabled=0 (skip)")

    valid_count = 0
    retry_count = 0
    probe_remaining = max(0, int(args.logits_probe_samples))
    degenerate_window = deque(maxlen=max(1, int(args.degenerate_window)))
    top1_id_window = deque(maxlen=max(1, int(args.degenerate_window)))
    collapse_hint_window = deque(maxlen=max(1, int(args.degenerate_window)))
    invalid_reason_window = deque(maxlen=max(1, int(args.degenerate_window)))
    recent_records_window = deque(maxlen=max(1, int(args.degenerate_window)))
    debug_written = 0
    diag_logged_invalid = 0
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
            raw_text, new_token, stats, cost, prompt_tokens, gen_diag = generate_once(
                model=model,
                tokenizer=tokenizer,
                msgs=build_messages(user_prompt),
                args=args,
                device_getter=model_input_device,
                probe=(probe_remaining > 0),
            )
            if probe_remaining > 0:
                probe_remaining -= 1
            pred_sql = extract_sql(raw_text)
            invalid_reason = classify_invalid_reason(raw_text, pred_sql)
            retry_used = False
            final_attempt = 1

            if invalid_reason and args.retry_invalid_sql:
                retry_used = True
                retry_count += 1
                retry_raw_text, retry_new_token, retry_stats, retry_cost, retry_prompt_tokens, retry_gen_diag = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    msgs=build_retry_messages(user_prompt, raw_text),
                    args=args,
                    device_getter=model_input_device,
                    probe=False,
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
                    gen_diag = retry_gen_diag
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
                "answer_id": _new_answer_id(),
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
                "decode_mode": args.decode_mode,
                **gen_diag,
            }
            raw_trim = str(rec.get("raw_output", "") or "").strip()
            rec["raw_char_set_size"] = int(len(set(raw_trim))) if raw_trim else 0
            rec["raw_single_char_repetition"] = bool(raw_trim) and (len(set(raw_trim)) == 1)
            rec["raw_preview"] = _preview_text(rec.get("raw_output", ""), max_chars=args.diag_raw_preview_chars)
            rec["is_degenerate"] = _is_degenerate_record(rec)
            rec["collapse_hint"] = _classify_collapse_hint(rec)
            degenerate_window.append(bool(rec["is_degenerate"]))
            top1_id_window.append(int(rec.get("gen_top1_token_id", -1)))
            collapse_hint_window.append(str(rec.get("collapse_hint", "")))
            invalid_reason_window.append(str(rec.get("invalid_reason", "") or ""))
            recent_records_window.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            should_diag = False
            if i <= int(args.diag_log_first_samples):
                should_diag = True
            elif rec["is_degenerate"]:
                should_diag = True
            elif (not rec["is_valid_sql"]) and diag_logged_invalid < int(args.diag_log_invalid_max):
                should_diag = True
                diag_logged_invalid += 1
            if should_diag:
                print(_format_sample_diag(i, len(samples), rec))

            if i % args.progress_log_every == 0:
                recent_deg = int(sum(degenerate_window))
                recent_n = int(len(degenerate_window))
                recent_ratio = float(recent_deg / recent_n) if recent_n > 0 else 0.0
                if top1_id_window:
                    c = Counter(top1_id_window)
                    top1_id_mode, top1_id_count = c.most_common(1)[0]
                    top1_id_mode_ratio = float(top1_id_count / len(top1_id_window))
                else:
                    top1_id_mode, top1_id_mode_ratio = -1, 0.0
                print(
                    f"[progress] done={i}/{len(samples)} valid={valid_count} "
                    f"valid_rate={valid_count / i:.4f} retry={retry_count} "
                    f"deg_recent={recent_deg}/{recent_n}({recent_ratio:.2f}) "
                    f"top1_id_mode={top1_id_mode}({top1_id_mode_ratio:.2f})"
                )
                collapse_recent = Counter([x for x in collapse_hint_window if x and x != "none"])
                invalid_recent = Counter([x for x in invalid_reason_window if x])
                print(
                    "[progress-diag] "
                    f"collapse_recent={_counter_summary(collapse_recent, topn=4)} "
                    f"invalid_recent={_counter_summary(invalid_recent, topn=4)} "
                    f"{_summarize_probe_recent(list(recent_records_window))}"
                )
            if (
                args.abort_on_degenerate
                and len(degenerate_window) == degenerate_window.maxlen
                and all(degenerate_window)
            ):
                c = Counter(top1_id_window)
                top1_id_mode, top1_id_count = c.most_common(1)[0] if c else (-1, 0)
                top1_id_mode_ratio = float(top1_id_count / len(top1_id_window)) if top1_id_window else 0.0
                collapse_mode = Counter([x for x in collapse_hint_window if x]).most_common(1)
                collapse_mode_text = collapse_mode[0][0] if collapse_mode else "unknown"
                invalid_mode = Counter([x for x in invalid_reason_window if x]).most_common(1)
                invalid_mode_text = invalid_mode[0][0] if invalid_mode else "none"
                probe_recent_text = _summarize_probe_recent(list(recent_records_window))
                raise RuntimeError(
                    "Detected sustained degenerate generations "
                    f"for last {len(degenerate_window)} samples "
                    "(single-token collapse, e.g. repeated punctuation). "
                    f"top1_id_mode={top1_id_mode} ({top1_id_mode_ratio:.2f}). "
                    f"collapse_hint_mode={collapse_mode_text} invalid_mode={invalid_mode_text}. "
                    f"{probe_recent_text}. "
                    "Abort early. Suggestion: if probe indicates clean numerics, suspect decode-path logic; "
                    "if probe shows NaN/Inf/finite<1, suspect numerical instability."
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
    parser.add_argument("--decode-mode", type=str, default="eagle", choices=["eagle", "naive", "hf"])
    parser.add_argument("--layered-probe-samples", type=int, default=0)
    parser.add_argument("--layered-probe-device", type=str, default="cpu", choices=["same", "auto", "cuda", "cpu"])
    parser.add_argument("--layered-probe-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--layered-probe-output", type=str, default="")
    parser.add_argument("--layered-probe-only", action="store_true")
    parser.add_argument("--warmup", dest="warmup", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    parser.add_argument("--warmup-decode-mode", type=str, default="auto", choices=["auto", "eagle", "naive", "hf"])
    parser.add_argument("--warmup-probe", action="store_true")
    parser.add_argument("--hard-reset-after-warmup", action="store_true")
    parser.add_argument("--hard-reset-before-probe", action="store_true")
    parser.add_argument("--hard-reset-before-generate", action="store_true")
    parser.add_argument("--log-state-snapshot", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--device-map-auto", dest="device_map_auto", action="store_true")
    parser.add_argument("--no-device-map-auto", dest="device_map_auto", action="store_false")
    parser.set_defaults(device_map_auto=False)
    parser.add_argument("--progress-log-every", type=int, default=20)
    parser.add_argument("--debug-prompt-file", type=str, default="")
    parser.add_argument("--debug-prompt-max-samples", type=int, default=5)
    parser.add_argument("--debug-token-dump", type=int, default=16)
    parser.add_argument("--diag-log-first-samples", type=int, default=3)
    parser.add_argument("--diag-log-invalid-max", type=int, default=20)
    parser.add_argument("--diag-raw-preview-chars", type=int, default=120)
    parser.add_argument("--abort-on-degenerate", action="store_true")
    parser.add_argument("--degenerate-window", type=int, default=20)
    parser.add_argument("--retry-invalid-sql", dest="retry_invalid_sql", action="store_true")
    parser.add_argument("--no-retry-invalid-sql", dest="retry_invalid_sql", action="store_false")
    parser.set_defaults(retry_invalid_sql=True)
    parser.add_argument("--kv-cache-min-length", type=int, default=3072)
    parser.add_argument("--kv-cache-margin", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--logits-probe-samples", type=int, default=0)
    parser.add_argument("--reset-kv-per-sample", dest="reset_kv_per_sample", action="store_true")
    parser.add_argument("--no-reset-kv-per-sample", dest="reset_kv_per_sample", action="store_false")
    parser.set_defaults(reset_kv_per_sample=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    run_infer(args)


if __name__ == "__main__":
    main()
