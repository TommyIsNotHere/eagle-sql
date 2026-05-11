#!/usr/bin/env python3
"""Summarize 4-group BIRD decode diagnostics and infer likely root cause.

Example:
  python scripts/summarize_bird_decode_diagnosis.py \
    --group A=/path/to/diag_hf_A_baseline \
    --group B=/path/to/diag_hf_B_clean \
    --group C=/path/to/diag_hf_C_crosswarm \
    --group D=/path/to/diag_naive_D_control
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def _as_bool_text(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return "1"
    if s in {"0", "false", "no", "n", "off"}:
        return "0"
    return str(v)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _resolve_pred_jsonl(raw_path: str) -> Path:
    p = Path(raw_path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        direct = p / "pred_dev.jsonl"
        if direct.is_file():
            return direct
        candidates = sorted(p.glob("*.jsonl"))
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(f"cannot resolve pred jsonl from: {raw_path}")


def _parse_run_log(group_dir: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    log_candidates = [
        group_dir / "run_bird_eagle3_full_eval.log",
        group_dir / "run_bird_eagle3_infer.log",
    ]
    log_path = next((x for x in log_candidates if x.is_file()), None)
    if log_path is None:
        return cfg

    warmup_re = re.compile(
        r"infer warmup warmup=(\S+) warmup_mode=(\S+) warmup_probe=(\S+) hard_reset_after_warmup=(\S+)"
    )
    state_re = re.compile(
        r"infer state hard_reset_before_probe=(\S+) hard_reset_before_generate=(\S+) log_state_snapshot=(\S+)"
    )
    decode_re = re.compile(r"infer decode mode=(\S+)")

    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                m = warmup_re.search(line)
                if m:
                    cfg["warmup"] = _as_bool_text(m.group(1))
                    cfg["warmup_mode"] = m.group(2)
                    cfg["warmup_probe"] = _as_bool_text(m.group(3))
                    cfg["hard_reset_after_warmup"] = _as_bool_text(m.group(4))
                m = state_re.search(line)
                if m:
                    cfg["hard_reset_before_probe"] = _as_bool_text(m.group(1))
                    cfg["hard_reset_before_generate"] = _as_bool_text(m.group(2))
                    cfg["log_state_snapshot"] = _as_bool_text(m.group(3))
                m = decode_re.search(line)
                if m:
                    cfg["decode_mode"] = m.group(1)
    except Exception:
        return cfg
    return cfg


def _summarize_group(label: str, raw_path: str) -> dict[str, Any]:
    pred_path = _resolve_pred_jsonl(raw_path)
    group_dir = pred_path.parent
    rows = _load_jsonl(pred_path)
    total = len(rows)
    if total == 0:
        return {
            "label": label,
            "path": str(group_dir),
            "pred_jsonl": str(pred_path),
            "error": "empty_or_unreadable_jsonl",
            "total": 0,
            "config": _parse_run_log(group_dir),
        }

    valid_count = 0
    probe_count = 0
    probe_error_count = 0
    probe_finite_values: list[float] = []
    probe_entropy_values: list[float] = []
    probe_nan_sum = 0
    probe_inf_sum = 0
    top1_ratio_values: list[float] = []
    tau_values: list[float] = []
    wall_time_values: list[float] = []
    tree_mask_before_probe_values: list[int] = []
    tree_mask_before_gen_values: list[int] = []
    single_char_repeat_count = 0

    collapse_counter: Counter[str] = Counter()
    invalid_counter: Counter[str] = Counter()
    decode_mode_counter: Counter[str] = Counter()

    numerical_instability_count = 0
    decode_state_contam_count = 0
    decode_logic_collapse_count = 0
    invalid_sql_non_collapse_count = 0

    for r in rows:
        if bool(r.get("is_valid_sql")):
            valid_count += 1

        decode_mode_counter[str(r.get("decode_mode", "") or "unknown")] += 1
        collapse_hint = str(r.get("collapse_hint", "") or "none")
        invalid_reason = str(r.get("invalid_reason", "") or "none")
        collapse_counter[collapse_hint] += 1
        invalid_counter[invalid_reason] += 1

        if bool(r.get("raw_single_char_repetition", False)):
            single_char_repeat_count += 1

        try:
            top1_ratio_values.append(float(r.get("gen_top1_ratio", 0.0)))
        except Exception:
            pass
        try:
            v = float(r.get("mean_accepted_length", 0.0))
            if v > 0:
                tau_values.append(v)
        except Exception:
            pass
        try:
            wall_time_values.append(float(r.get("wall_time", 0.0)))
        except Exception:
            pass

        if "state_before_probe_tree_mask_active" in r:
            try:
                tree_mask_before_probe_values.append(int(r.get("state_before_probe_tree_mask_active", 0)))
            except Exception:
                pass
        if "state_before_generate_tree_mask_active" in r:
            try:
                tree_mask_before_gen_values.append(int(r.get("state_before_generate_tree_mask_active", 0)))
            except Exception:
                pass

        has_probe = (
            "probe_logits_finite_ratio" in r
            or "probe_logits_nan_count" in r
            or "probe_logits_inf_count" in r
            or "probe_error" in r
        )
        if has_probe:
            probe_count += 1
        probe_error = str(r.get("probe_error", "") or "")
        if probe_error:
            probe_error_count += 1
        if has_probe and not probe_error:
            try:
                finite = float(r.get("probe_logits_finite_ratio", 1.0))
                probe_finite_values.append(finite)
                nan_c = int(r.get("probe_logits_nan_count", 0))
                inf_c = int(r.get("probe_logits_inf_count", 0))
                probe_nan_sum += nan_c
                probe_inf_sum += inf_c
                if "probe_entropy" in r:
                    probe_entropy_values.append(float(r.get("probe_entropy", 0.0)))
                if nan_c > 0 or inf_c > 0 or finite < 0.999999:
                    numerical_instability_count += 1
            except Exception:
                pass

        if collapse_hint == "numerical_instability":
            numerical_instability_count += 1
        elif collapse_hint == "decode_state_contamination_suspected":
            decode_state_contam_count += 1
        elif collapse_hint == "decode_path_logic_collapse_suspected":
            decode_logic_collapse_count += 1
        elif collapse_hint == "invalid_sql_non_collapse":
            invalid_sql_non_collapse_count += 1

    # Avoid double-counting when both probe and collapse label mark numerical instability.
    numerical_instability_count = min(
        total,
        max(
            collapse_counter.get("numerical_instability", 0),
            numerical_instability_count,
        ),
    )

    config = _parse_run_log(group_dir)
    summary = {
        "label": label,
        "path": str(group_dir),
        "pred_jsonl": str(pred_path),
        "total": total,
        "valid_count": valid_count,
        "valid_rate": _safe_ratio(valid_count, total),
        "invalid_count": total - valid_count,
        "decode_mode_counter": dict(decode_mode_counter),
        "collapse_hint_counter": dict(collapse_counter),
        "invalid_reason_counter": dict(invalid_counter),
        "numerical_instability_count": numerical_instability_count,
        "numerical_instability_rate": _safe_ratio(numerical_instability_count, total),
        "decode_state_contam_count": decode_state_contam_count,
        "decode_state_contam_rate": _safe_ratio(decode_state_contam_count, total),
        "decode_logic_collapse_count": decode_logic_collapse_count,
        "decode_logic_collapse_rate": _safe_ratio(decode_logic_collapse_count, total),
        "invalid_sql_non_collapse_count": invalid_sql_non_collapse_count,
        "invalid_sql_non_collapse_rate": _safe_ratio(invalid_sql_non_collapse_count, total),
        "single_char_repeat_count": single_char_repeat_count,
        "single_char_repeat_rate": _safe_ratio(single_char_repeat_count, total),
        "probe_count": probe_count,
        "probe_coverage_rate": _safe_ratio(probe_count, total),
        "probe_error_count": probe_error_count,
        "probe_error_rate": _safe_ratio(probe_error_count, max(1, probe_count)),
        "probe_finite_min": min(probe_finite_values) if probe_finite_values else math.nan,
        "probe_finite_mean": statistics.mean(probe_finite_values) if probe_finite_values else math.nan,
        "probe_nan_sum": probe_nan_sum,
        "probe_inf_sum": probe_inf_sum,
        "probe_entropy_min": min(probe_entropy_values) if probe_entropy_values else math.nan,
        "avg_gen_top1_ratio": statistics.mean(top1_ratio_values) if top1_ratio_values else math.nan,
        "avg_tau": statistics.mean(tau_values) if tau_values else math.nan,
        "avg_wall_time_sec": statistics.mean(wall_time_values) if wall_time_values else math.nan,
        "state_before_probe_tree_mask_active_rate": (
            _safe_ratio(sum(tree_mask_before_probe_values), len(tree_mask_before_probe_values))
            if tree_mask_before_probe_values
            else math.nan
        ),
        "state_before_generate_tree_mask_active_rate": (
            _safe_ratio(sum(tree_mask_before_gen_values), len(tree_mask_before_gen_values))
            if tree_mask_before_gen_values
            else math.nan
        ),
        "config": config,
    }
    return summary


def _fmt_pct(v: float) -> str:
    if isinstance(v, float) and not math.isnan(v):
        return f"{v * 100:.1f}%"
    return "n/a"


def _is_high(v: float, th: float = 0.5) -> bool:
    return (not math.isnan(v)) and v >= th


def _is_low(v: float, th: float = 0.2) -> bool:
    return (not math.isnan(v)) and v <= th


def _diagnose(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = set(groups.keys())
    has_abcd = {"A", "B", "C", "D"}.issubset(labels)

    def g(label: str) -> dict[str, Any] | None:
        return groups.get(label)

    evidence: list[str] = []
    likely_cause = "mixed_or_insufficient_signal"
    confidence = "low"
    rule = "fallback"

    if has_abcd:
        a, b, c, d = g("A"), g("B"), g("C"), g("D")
        assert a and b and c and d
        ar = float(a.get("numerical_instability_rate", math.nan))
        br = float(b.get("numerical_instability_rate", math.nan))
        cr = float(c.get("numerical_instability_rate", math.nan))
        dr = float(d.get("numerical_instability_rate", math.nan))
        d_non_sql = float(d.get("invalid_sql_non_collapse_rate", math.nan))

        evidence.append(f"A numerical_instability={_fmt_pct(ar)}")
        evidence.append(f"B numerical_instability={_fmt_pct(br)}")
        evidence.append(f"C numerical_instability={_fmt_pct(cr)}")
        evidence.append(f"D numerical_instability={_fmt_pct(dr)}")

        if _is_high(ar) and _is_low(br) and _is_high(cr):
            likely_cause = "warmup_or_decode_state_contamination"
            rule = "A high, B low, C high"
            confidence = "high" if _is_low(dr) else "medium"
            evidence.append(
                "Pattern matches warmup/state contamination: baseline fails, clean-reset recovers, cross-warmup relapses."
            )
        elif _is_high(ar) and _is_high(br) and _is_high(cr):
            if _is_low(dr):
                likely_cause = "hf_eagle_path_specific_numerical_instability"
                rule = "A/B/C high, D low"
                confidence = "medium"
                evidence.append("Naive path remains numerically stable while hf/eagle stay unstable.")
            else:
                likely_cause = "global_numerical_instability"
                rule = "A/B/C/D all high"
                confidence = "high"
                evidence.append("All paths show numerical instability; likely lower-level numeric/runtime issue.")
        elif _is_low(ar) and _is_low(br) and _is_low(cr) and _is_low(dr):
            likely_cause = "non_numeric_decode_or_prompt_chain_issue"
            rule = "all groups low numerical instability"
            confidence = "medium"
            evidence.append("No dominant NaN/Inf signal across groups.")
        elif _is_low(dr) and _is_high(d_non_sql, 0.5):
            likely_cause = "naive_prompt_or_format_chain_issue"
            rule = "D low numeric instability but high invalid_sql_non_collapse"
            confidence = "medium"
            evidence.append("Naive mainly fails as non-SQL formatting, not numeric collapse.")
    else:
        # Generic diagnosis without strict A/B/C/D mapping.
        items = list(groups.values())
        num_rates = [float(x.get("numerical_instability_rate", math.nan)) for x in items]
        avg_num = statistics.mean([x for x in num_rates if not math.isnan(x)]) if num_rates else math.nan
        if not math.isnan(avg_num) and avg_num >= 0.6:
            likely_cause = "numerical_instability_dominant"
            confidence = "medium"
            rule = "average numerical instability high"
        elif not math.isnan(avg_num) and avg_num <= 0.2:
            likely_cause = "non_numeric_decode_chain_issue"
            confidence = "medium"
            rule = "average numerical instability low"
        evidence.append(f"avg numerical_instability={_fmt_pct(avg_num)}")

    return {
        "likely_cause": likely_cause,
        "confidence": confidence,
        "rule": rule,
        "evidence": evidence,
    }


def _text_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# BIRD Decode Diagnosis Summary")
    lines.append("")
    diag = result["diagnosis"]
    lines.append(f"- Likely cause: `{diag['likely_cause']}`")
    lines.append(f"- Confidence: `{diag['confidence']}`")
    lines.append(f"- Rule: `{diag['rule']}`")
    lines.append("")
    lines.append("## Evidence")
    for e in diag["evidence"]:
        lines.append(f"- {e}")
    lines.append("")
    lines.append("## Group Metrics")
    for label in sorted(result["groups"].keys()):
        g = result["groups"][label]
        lines.append(f"### {label}")
        lines.append(f"- path: `{g.get('path')}`")
        lines.append(f"- total: {g.get('total')}, valid_rate: {_fmt_pct(float(g.get('valid_rate', math.nan)))}")
        lines.append(f"- numerical_instability_rate: {_fmt_pct(float(g.get('numerical_instability_rate', math.nan)))}")
        lines.append(f"- invalid_sql_non_collapse_rate: {_fmt_pct(float(g.get('invalid_sql_non_collapse_rate', math.nan)))}")
        lines.append(f"- probe_coverage_rate: {_fmt_pct(float(g.get('probe_coverage_rate', math.nan)))}")
        lines.append(
            "- state_before_probe_tree_mask_active_rate: "
            f"{_fmt_pct(float(g.get('state_before_probe_tree_mask_active_rate', math.nan)))}"
        )
        lines.append(f"- config: {json.dumps(g.get('config', {}), ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BIRD 4-group decode diagnosis.")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Group input in LABEL=PATH format, e.g. A=/path/to/workdir",
    )
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-md", type=str, default="")
    args = parser.parse_args()

    groups: dict[str, dict[str, Any]] = {}
    for item in args.group:
        if "=" not in item:
            raise ValueError(f"invalid --group value: {item} (expect LABEL=PATH)")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"invalid --group value: {item}")
        groups[label] = _summarize_group(label, raw_path)

    diagnosis = _diagnose(groups)
    result = {
        "groups": groups,
        "diagnosis": diagnosis,
    }

    if args.output_json:
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_text = _text_report(result)
    if args.output_md:
        out_md = Path(args.output_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text, encoding="utf-8")

    print(md_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
