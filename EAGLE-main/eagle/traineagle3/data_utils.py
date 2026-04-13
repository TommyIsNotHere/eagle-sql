from __future__ import annotations

from typing import Any

import torch


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
    "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please don't share false information."
)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def _normalize_turns(source: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not source:
        return []

    turns: list[dict[str, str]] = []
    for row in source:
        if not isinstance(row, dict):
            continue
        raw_role = str(row.get("from", row.get("role", ""))).strip().lower()
        role = ROLE_MAP.get(raw_role)
        if role is None:
            continue
        content = str(row.get("value", row.get("content", ""))).strip()
        if not content:
            continue
        turns.append({"role": role, "content": content})

    # Enforce user -> assistant alternation and drop malformed turns.
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    if not turns:
        return []

    filtered: list[dict[str, str]] = []
    expected = "user"
    for t in turns:
        if t["role"] != expected:
            continue
        filtered.append(t)
        expected = "assistant" if expected == "user" else "user"

    # Keep complete dialogue pairs only.
    if filtered and filtered[-1]["role"] != "assistant":
        filtered = filtered[:-1]
    return filtered


def build_tokenized_sample(
    tokenizer,
    source: list[dict[str, Any]] | None,
    max_len: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, torch.Tensor] | None:
    turns = _normalize_turns(source)
    if not turns:
        return None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.unk_token_id

    system_msg = {"role": "system", "content": system_prompt}
    full_messages = [system_msg] + turns
    conversation = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = tokenizer(
        conversation,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    if len(input_ids) > max_len:
        return None

    # Build assistant supervision spans in a template-agnostic manner:
    # compare token length before/after each assistant turn.
    running_messages = [system_msg]
    spans: list[tuple[int, int]] = []
    for turn in turns:
        if turn["role"] == "assistant":
            prefix_text = tokenizer.apply_chat_template(
                running_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                running_messages + [turn],
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            if len(full_ids) > len(prefix_ids):
                spans.append((len(prefix_ids), len(full_ids)))
        running_messages.append(turn)

    loss_mask = torch.zeros_like(input_ids)
    for start, end in spans:
        start = min(max(start, 0), len(input_ids))
        end = min(max(end, 0), len(input_ids))
        if end > start:
            loss_mask[start:end] = 1

    if int(loss_mask.sum().item()) == 0:
        return None

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids[None, :],
        "attention_mask": attention_mask[None, :],
        "loss_mask": loss_mask[None, :],
    }
