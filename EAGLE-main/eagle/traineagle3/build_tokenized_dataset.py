from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from .data_utils import build_tokenized_sample, resolve_system_prompt
except ImportError:
    from data_utils import build_tokenized_sample, resolve_system_prompt


def build_input_signature(path_str: str) -> str:
    path = Path(path_str).resolve()
    if not path.exists():
        return f"{path}|missing"
    st = path.stat()
    return f"{path}|{int(st.st_size)}|{int(st.st_mtime_ns)}"


def build_dataset(tokenizer, datapath: str, max_len: int, num_proc: int, seed: int, system_prompt: str):
    ds = load_dataset("json", data_files=datapath)["train"]
    ds = ds.shuffle(seed=seed)
    original_columns = ds.column_names

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": [],
        }
        conversations = examples.get("conversations") or []
        for source in conversations:
            sample = build_tokenized_sample(
                tokenizer=tokenizer,
                source=source,
                max_len=max_len,
                system_prompt=system_prompt,
            )
            if sample is None:
                continue
            new_examples["input_ids"].append(sample["input_ids"])
            new_examples["loss_mask"].append(sample["loss_mask"])
            new_examples["attention_mask"].append(sample["attention_mask"])
        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=max(1, int(num_proc)),
        remove_columns=original_columns,
        load_from_cache_file=False,
    )
    if len(ds) <= 0:
        raise RuntimeError(f"tokenized dataset is empty for datapath={datapath}")
    return ds


def build_one(
    *,
    split_name: str,
    datapath: str,
    output_dir: str,
    tokenizer,
    max_len: int,
    num_proc: int,
    seed: int,
    system_prompt: str,
    force_rebuild: bool,
):
    output_path = Path(output_dir)
    meta_path = output_path / "_eagle_tokenized_meta.json"
    input_signature = build_input_signature(datapath)
    expected_meta = {
        "input_signature": input_signature,
        "tokenizer_path": str(tokenizer.name_or_path),
        "max_len": int(max_len),
        "preprocess_num_proc": int(max(1, int(num_proc))),
        "seed": int(seed),
        "system_prompt": system_prompt,
    }

    if output_path.exists() and not force_rebuild:
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    old_meta = json.load(f)
                if all(old_meta.get(k) == v for k, v in expected_meta.items()):
                    print(f"[tokenized-build] reuse {split_name}: {output_path}")
                    return
                print(f"[tokenized-build] metadata drift detected; rebuilding split={split_name}: {output_path}")
            except Exception:
                print(f"[tokenized-build] metadata parse failed; rebuilding split={split_name}: {output_path}")
        else:
            print(f"[tokenized-build] metadata missing; rebuilding split={split_name}: {output_path}")

    t0 = time.time()
    dataset = build_dataset(
        tokenizer=tokenizer,
        datapath=datapath,
        max_len=max_len,
        num_proc=num_proc,
        seed=seed,
        system_prompt=system_prompt,
    )

    tmp_path = Path(f"{output_path}.tmp.{os.getpid()}")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split_name,
        "rows": len(dataset),
        "dataset_fingerprint": str(getattr(dataset, "_fingerprint", "unknown")),
        "input_path": datapath,
        "input_signature": input_signature,
        "tokenizer_path": str(tokenizer.name_or_path),
        "max_len": int(max_len),
        "preprocess_num_proc": int(max(1, int(num_proc))),
        "seed": int(seed),
        "system_prompt": system_prompt,
    }
    with (tmp_path / "_eagle_tokenized_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, output_path)

    print(
        f"[tokenized-build] done split={split_name} rows={len(dataset)} "
        f"fingerprint={meta['dataset_fingerprint']} elapsed={time.time() - t0:.1f}s path={output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build offline tokenized datasets for EAGLE3 training")
    parser.add_argument("--basepath", type=str, required=True)
    parser.add_argument("--trainpath", type=str, required=True)
    parser.add_argument("--testpath", type=str, required=True)
    parser.add_argument("--train-output-dir", type=str, required=True)
    parser.add_argument("--test-output-dir", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--preprocess-num-proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=os.environ.get("EAGLE_TRAIN_SYSTEM_PROMPT", "auto"),
        help="System prompt mode for tokenization: auto|bird|sql|generic or custom prompt text",
    )
    parser.add_argument("--force-rebuild", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.basepath)
    system_prompt = resolve_system_prompt(args.system_prompt)
    print(
        "[tokenized-build] "
        f"tokenizer={args.basepath} max_len={args.max_len} num_proc={args.preprocess_num_proc} seed={args.seed} "
        f"prompt_mode={args.system_prompt} prompt_chars={len(system_prompt)}"
    )

    build_one(
        split_name="train",
        datapath=args.trainpath,
        output_dir=args.train_output_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
        num_proc=args.preprocess_num_proc,
        seed=args.seed,
        system_prompt=system_prompt,
        force_rebuild=bool(int(args.force_rebuild)),
    )
    build_one(
        split_name="eval",
        datapath=args.testpath,
        output_dir=args.test_output_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
        num_proc=args.preprocess_num_proc,
        seed=args.seed,
        system_prompt=system_prompt,
        force_rebuild=bool(int(args.force_rebuild)),
    )


if __name__ == "__main__":
    main()
