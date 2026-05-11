"""
Extract hidden states from Qwen2.5 base model for EAGLE2 training.

Input:  ShareGPT-format JSONL (conversations with "from"/"value" turns)
Output: Per-sample .pt files containing {hidden_state, input_ids, loss_mask}

Usage:
    python -m eagle.ge_data.ge_data_qwen25 \
        --basepath /path/to/Qwen2.5-Coder-14B-Instruct \
        --input-jsonl /path/to/train_chat.jsonl \
        --output-dir /path/to/data/sharegpt \
        --max-len 2048 \
        --num-proc 4
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def log(msg):
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    return parser.parse_args()


ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def build_chat_messages(conversations, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in conversations:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    return messages


def _tokenize_chat(tokenizer, messages, add_generation_prompt=False):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def build_loss_mask(tokenizer, messages, input_ids, max_len):
    """Build loss mask: 1 for assistant tokens, 0 for user/system tokens."""
    loss_mask = torch.zeros(len(input_ids), dtype=torch.int64)

    prefix_messages = []
    for msg in messages:
        prefix_messages.append(msg)
        if msg["role"] == "assistant":
            prefix_ids = _tokenize_chat(tokenizer, prefix_messages, add_generation_prompt=False)
            prev = prefix_messages[:-1]
            if not prev:
                continue
            without_last = _tokenize_chat(tokenizer, prev, add_generation_prompt=True)
            start = min(len(without_last), max_len - 1)
            end = min(len(prefix_ids), max_len)
            if start < end:
                loss_mask[start:end] = 1

    return loss_mask


class ConversationDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conversations = sample["conversations"]
        messages = build_chat_messages(conversations)

        input_ids = _tokenize_chat(self.tokenizer, messages, add_generation_prompt=False)
        input_ids = input_ids[:self.max_len]
        loss_mask = build_loss_mask(self.tokenizer, messages, input_ids, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": loss_mask,
            "idx": idx,
        }


def collate_fn(batch):
    return batch


@torch.no_grad()
def extract_hidden_states(model, input_ids, device):
    input_ids_tensor = input_ids.unsqueeze(0).to(device)
    outputs = model(input_ids_tensor, output_hidden_states=True)
    hidden_state = outputs.hidden_states[-1][0].cpu()
    return hidden_state


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log(f"Loading tokenizer from {args.basepath}")
    tokenizer = AutoTokenizer.from_pretrained(args.basepath, use_fast=True)
    log(f"Tokenizer loaded. vocab_size={len(tokenizer)}")

    log(f"Loading model from {args.basepath} (this may take several minutes for large models)")
    log(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")

    try:
        import accelerate  # noqa: F401
        log(f"accelerate version: {accelerate.__version__}")
        model = AutoModelForCausalLM.from_pretrained(
            args.basepath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except ImportError:
        log("WARNING: accelerate not installed, loading model to single GPU")
        model = AutoModelForCausalLM.from_pretrained(
            args.basepath,
            torch_dtype=torch.bfloat16,
        ).cuda()

    model.eval()
    log(f"Model loaded. dtype={model.dtype}")

    log(f"Reading samples from {args.input_jsonl}")
    samples = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    end_idx = args.end_idx if args.end_idx > 0 else len(samples)
    samples = samples[args.start_idx:end_idx]
    log(f"Processing {len(samples)} samples (idx {args.start_idx} to {end_idx})")

    dataset = ConversationDataset(samples, tokenizer, args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_proc,
    )

    device = next(model.parameters()).device
    saved = 0
    skipped = 0

    for batch in tqdm(loader, desc="Extracting hidden states"):
        for item in batch:
            input_ids = item["input_ids"]
            loss_mask = item["loss_mask"]
            idx = item["idx"]

            if loss_mask.sum() == 0:
                skipped += 1
                continue

            hidden_state = extract_hidden_states(model, input_ids, device)

            out_path = os.path.join(args.output_dir, f"{args.start_idx + idx}.pt")
            torch.save({
                "hidden_state": hidden_state,
                "input_ids": input_ids,
                "loss_mask": loss_mask,
            }, out_path)
            saved += 1

    log(f"Done. Saved: {saved}, Skipped (no assistant tokens): {skipped}")
    log(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
