import argparse
import deepspeed
import math
import os
import time
import inspect
import subprocess

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/llama31chat/8B/')
parser.add_argument('--trainpath', type=str,
                    default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/l318b.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/0318.json")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument('--num-epochs', type=int, default=40)
parser.add_argument('--max-len', type=int, default=2048)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--preprocess-num-proc', type=int, default=8)
parser.add_argument('--warmup-ratio', type=float, default=0.03)
parser.add_argument('--save-every-epochs', type=int, default=1)
parser.add_argument('--save-ds-every-epochs', type=int, default=10)
parser.add_argument('--dataloader-timeout', type=float, default=120.0)
parser.add_argument('--pin-memory', type=int, default=0)
parser.add_argument('--persistent-workers', type=int, default=1)
parser.add_argument('--prefetch-factor', type=int, default=1)
parser.add_argument('--dataloader-in-order', type=int, default=1)
parser.add_argument('--heartbeat-steps', type=int, default=100)
parser.add_argument('--slow-stage-seconds', type=float, default=30.0)
parser.add_argument('--empty-cache-every-steps', type=int, default=0)
parser.add_argument('--cuda-sync-heartbeat-steps', type=int, default=0)
parser.add_argument('--max-train-steps', type=int, default=0)
parser.add_argument('--max-eval-steps', type=int, default=0)
parser.add_argument('--global-stall-seconds', type=float, default=1200.0)
parser.add_argument('--abort-on-global-stall', type=int, default=1)
parser.add_argument('--stall-diagnostics', type=int, default=1)
parser.add_argument('--stall-diagnostics-cooldown', type=float, default=300.0)
parser.add_argument(
    '--system-prompt',
    type=str,
    default=os.environ.get("EAGLE_TRAIN_SYSTEM_PROMPT", "auto"),
    help="System prompt mode for tokenization: auto|bird|sql|generic or a literal custom prompt",
)
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": args.num_epochs,
    "num_workers": args.num_workers,
    "max_len": args.max_len,
    "preprocess_num_proc": args.preprocess_num_proc,
    "warmup_ratio": args.warmup_ratio,
    "save_every_epochs": args.save_every_epochs,
    "save_ds_every_epochs": args.save_ds_every_epochs,
    "dataloader_timeout": args.dataloader_timeout,
    "pin_memory": bool(int(args.pin_memory)),
    "persistent_workers": bool(int(args.persistent_workers)),
    "prefetch_factor": int(args.prefetch_factor),
    "dataloader_in_order": bool(int(args.dataloader_in_order)),
    "heartbeat_steps": int(args.heartbeat_steps),
    "slow_stage_seconds": float(args.slow_stage_seconds),
    "empty_cache_every_steps": int(args.empty_cache_every_steps),
    "cuda_sync_heartbeat_steps": int(args.cuda_sync_heartbeat_steps),
    "max_train_steps": int(args.max_train_steps),
    "max_eval_steps": int(args.max_eval_steps),
    "global_stall_seconds": float(args.global_stall_seconds),
    "abort_on_global_stall": bool(int(args.abort_on_global_stall)),
    "stall_diagnostics": bool(int(args.stall_diagnostics)),
    "stall_diagnostics_cooldown": float(args.stall_diagnostics_cooldown),
    "config_path": "config.json",
    "gradient_checkpoint": True,
    "gradient_checkpointing": True,
}

from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from data_utils import build_tokenized_sample, resolve_system_prompt



def build_dataset_rank(
        tokenizer, datapath, system_prompt
):

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = max(1, int(train_config["preprocess_num_proc"]))

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            source = examples['conversations'][i]
            sample = build_tokenized_sample(
                tokenizer=tokenizer,
                source=source,
                max_len=train_config["max_len"],
                system_prompt=system_prompt,
            )
            if sample is None:
                continue
            new_examples["input_ids"].append(sample["input_ids"])
            new_examples["loss_mask"].append(sample["loss_mask"])
            new_examples["attention_mask"].append(sample["attention_mask"])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = int(pad_token_id)

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N, fill_value=0):
        B, n = intensors.shape
        padding_tensor = torch.full((B, N - n), fill_value=fill_value, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item['input_ids'], max_length, fill_value=self.pad_token_id) for item in features]
        )
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length, fill_value=0) for item in features]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length, fill_value=0) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
train_system_prompt = resolve_system_prompt(args.system_prompt)
print(
    "[prompt] "
    f"mode={args.system_prompt} "
    f"chars={len(train_system_prompt)} "
    f"preview={train_system_prompt[:120].replace(chr(10), ' ')}"
)
traindataset = build_dataset_rank(tokenizer, args.trainpath, train_system_prompt)
testdataset = build_dataset_rank(tokenizer, args.testpath, train_system_prompt)

world_size_env = max(1, int(os.environ.get("WORLD_SIZE", "1")))
global_batch = (
    int(ds_config.get("train_micro_batch_size_per_gpu", 1))
    * world_size_env
    * int(ds_config.get("gradient_accumulation_steps", 1))
)
optimizer_steps_per_epoch = max(1, math.ceil(len(traindataset) / max(1, global_batch)))
total_optimizer_steps = optimizer_steps_per_epoch * int(train_config["num_epochs"])
warmup_steps = int(total_optimizer_steps * float(train_config["warmup_ratio"]))
warmup_steps = max(1, min(warmup_steps, max(1, total_optimizer_steps - 1)))

if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
    ds_config["scheduler"]["params"]["total_num_steps"] = int(total_optimizer_steps)
    ds_config["scheduler"]["params"]["warmup_num_steps"] = int(warmup_steps)

print(
    f"[schedule] world_size={world_size_env}, global_batch={global_batch}, "
    f"steps/epoch={optimizer_steps_per_epoch}, total_steps={total_optimizer_steps}, warmup_steps={warmup_steps}"
)

config = EConfig.from_pretrained(train_config["config_path"])
base_cfg = AutoConfig.from_pretrained(args.basepath)

# Keep EAGLE head structural hyper-params in sync with the selected base model
# (critical for Qwen2.5-14B: hidden size / vocab / heads differ from default llama config.json).
sync_keys = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "hidden_act",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "pretraining_tp",
]
for key in sync_keys:
    if hasattr(base_cfg, key):
        setattr(config, key, getattr(base_cfg, key))

# Keep draft vocab bounded by real vocab.
draft_vocab_size = int(getattr(config, "draft_vocab_size", 32000))
config.draft_vocab_size = min(draft_vocab_size, int(config.vocab_size))

print(
    f"[train-config] hidden={config.hidden_size}, inter={config.intermediate_size}, "
    f"heads={config.num_attention_heads}/{config.num_key_value_heads}, "
    f"vocab={config.vocab_size}, draft_vocab={config.draft_vocab_size}"
)
_stage_t0 = time.time()
print("[stage] building Model() ...")
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)
print(f"[stage] Model() done in {time.time() - _stage_t0:.1f}s")

_stage_t0 = time.time()
print("[stage] model.scandata() ...")
model.scandata(
    args.trainpath,
    args.basepath,
    tokenized_dataset=traindataset,
    cache_context={
        "max_len": int(train_config["max_len"]),
        "system_prompt_mode": str(args.system_prompt),
        "system_prompt_text": train_system_prompt,
        "draft_vocab_size": int(config.draft_vocab_size),
        "teacher_hidden_selector": os.environ.get("EAGLE_TEACHER_HIDDEN_SELECTOR", "paper"),
    },
)
print(f"[stage] model.scandata() done in {time.time() - _stage_t0:.1f}s")


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

_stage_t0 = time.time()
print("[stage] deepspeed.initialize() ...")
# Avoid passing DeepSpeed config through both CLI args and function kwargs.
# We keep the in-memory `ds_config` (already patched with runtime scheduler steps).
if hasattr(args, "deepspeed_config"):
    args.deepspeed_config = None
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config,
                                                     )
print(f"[stage] deepspeed.initialize() done in {time.time() - _stage_t0:.1f}s")

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
device = model_engine.device


def rank_log(message: str, force_all_ranks: bool = False):
    log_all = os.environ.get("EAGLE_LOG_ALL_RANKS", "0") == "1"
    if force_all_ranks or log_all or global_rank == 0:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}][rank{global_rank}] {message}", flush=True)


wandb = None
wandb_enabled = os.environ.get("ENABLE_WANDB", "0") == "1"
if global_rank == 0 and wandb_enabled:
    try:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="l382", entity="yuhui-li", config=ds_config)
    except Exception as e:
        print(f"WARN: wandb disabled due to init failure: {e}")
        wandb = None

os.makedirs(args.savedir, exist_ok=True)

collator = DataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id or 0)


def build_loader(dataset, sampler):
    loader_kwargs = {
        "batch_size": train_config["bs"],
        "sampler": sampler,
        "num_workers": train_config["num_workers"],
        "pin_memory": train_config["pin_memory"],
        "collate_fn": collator,
    }
    if train_config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = train_config["persistent_workers"]
        if train_config["prefetch_factor"] > 0:
            loader_kwargs["prefetch_factor"] = train_config["prefetch_factor"]
        if train_config["dataloader_timeout"] > 0:
            loader_kwargs["timeout"] = train_config["dataloader_timeout"]
        if "in_order" in inspect.signature(DataLoader).parameters:
            loader_kwargs["in_order"] = train_config["dataloader_in_order"]
    elif train_config["dataloader_timeout"] > 0:
        rank_log(
            "WARN: dataloader_timeout>0 is ignored because num_workers=0. "
            "Set num_workers>0 to enable DataLoader timeout."
        )
    return DataLoader(dataset, **loader_kwargs)


sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = build_loader(testdataset, sampler)

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = build_loader(traindataset, train_sampler)


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


def sanitize_nonfinite_trainable_params(module, verbose_prefix=""):
    fixed_params = 0
    total_bad = 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        bad_mask = ~torch.isfinite(p.data)
        bad_count = int(bad_mask.sum().item())
        if bad_count > 0:
            total_bad += bad_count
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)
            fixed_params += 1
            print(f"{verbose_prefix}sanitize param `{name}` bad_values={bad_count}")
    if fixed_params > 0:
        print(f"{verbose_prefix}sanitized trainable params: {fixed_params}, total_bad_values={total_bad}")
    return fixed_params, total_bad


disable_autoresume = os.environ.get("EAGLE_DISABLE_AUTORESUME", "0") == "1"
checkpoint_path, start_epoch = (None, 0) if disable_autoresume else find_max_state_with_file(args.savedir)
if disable_autoresume and global_rank == 0:
    print("EAGLE_DISABLE_AUTORESUME=1 -> start from epoch 0 (skip checkpoint auto-resume)")
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)
    sanitize_nonfinite_trainable_params(model_engine.module, verbose_prefix="[checkpoint] ")



heartbeat_steps = max(1, int(train_config["heartbeat_steps"]))
slow_stage_seconds = max(0.0, float(train_config["slow_stage_seconds"]))
empty_cache_every_steps = max(0, int(train_config["empty_cache_every_steps"]))
cuda_sync_heartbeat_steps = max(0, int(train_config["cuda_sync_heartbeat_steps"]))
max_train_steps = max(0, int(train_config["max_train_steps"]))
max_eval_steps = max(0, int(train_config["max_eval_steps"]))
global_stall_seconds = max(0.0, float(train_config["global_stall_seconds"]))
abort_on_global_stall = bool(train_config["abort_on_global_stall"])
stall_diagnostics = bool(train_config["stall_diagnostics"])
stall_diagnostics_cooldown = max(0.0, float(train_config["stall_diagnostics_cooldown"]))
last_stall_diag_ts = 0.0


def _run_cmd_for_diag(cmd):
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return out
    except Exception as e:
        return f"<diag-failed: {e}>"


def maybe_dump_stall_diag(tag: str, epoch: int, batch_idx: int, wait_s: float):
    global last_stall_diag_ts
    if not stall_diagnostics:
        return
    now = time.time()
    if now - last_stall_diag_ts < stall_diagnostics_cooldown:
        return
    last_stall_diag_ts = now
    rank_log(
        f"STALL-DIAG trigger tag={tag} epoch={epoch} batch={batch_idx} wait={wait_s:.3f}s",
        force_all_ranks=True,
    )
    self_ps = _run_cmd_for_diag(
        ["ps", "-o", "pid,ppid,stat,pcpu,pmem,etime,wchan:24,comm", "-p", str(os.getpid())]
    )
    rank_log("STALL-DIAG ps-self:\n" + self_ps, force_all_ranks=True)
    gpu_q = _run_cmd_for_diag(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    rank_log("STALL-DIAG nvidia-smi query:\n" + gpu_q, force_all_ranks=True)
    dmesg_tail = _run_cmd_for_diag(["bash", "-lc", "dmesg -T | tail -n 30"])
    rank_log("STALL-DIAG dmesg-tail:\n" + dmesg_tail, force_all_ranks=True)

if global_rank == 0:
    rank_log(
        "runtime-debug-config "
        f"num_workers={train_config['num_workers']} pin_memory={int(train_config['pin_memory'])} "
        f"persistent_workers={int(train_config['persistent_workers'])} "
        f"prefetch_factor={train_config['prefetch_factor']} "
        f"dataloader_in_order={int(train_config['dataloader_in_order'])} "
        f"dataloader_timeout={train_config['dataloader_timeout']} "
        f"heartbeat_steps={heartbeat_steps} slow_stage_seconds={slow_stage_seconds} "
        f"empty_cache_every_steps={empty_cache_every_steps} "
        f"cuda_sync_heartbeat_steps={cuda_sync_heartbeat_steps} "
        f"max_train_steps={max_train_steps} max_eval_steps={max_eval_steps} "
        f"global_stall_seconds={global_stall_seconds} "
        f"abort_on_global_stall={int(abort_on_global_stall)} "
        f"stall_diagnostics={int(stall_diagnostics)} "
        f"stall_diag_cooldown={stall_diagnostics_cooldown}"
    )

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch + 1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]
    skipped_train_batches = 0
    processed_train_batches = 0
    prev_train_loop_end = time.perf_counter()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        if max_train_steps > 0 and batch_idx >= max_train_steps:
            if global_rank == 0:
                rank_log(f"Reached max_train_steps={max_train_steps}; stop epoch early for diagnostics.")
            break

        processed_train_batches += 1
        batch_begin = time.perf_counter()
        data_wait_sec = batch_begin - prev_train_loop_end
        if data_wait_sec > slow_stage_seconds:
            rank_log(
                f"SLOW stage=dataloader_wait epoch={epoch} batch={batch_idx} wait={data_wait_sec:.3f}s",
                force_all_ranks=True,
            )
            maybe_dump_stall_diag("train_dataloader_wait", epoch, batch_idx, data_wait_sec)
            if global_stall_seconds > 0 and data_wait_sec >= global_stall_seconds:
                global_stall_flag = torch.tensor(1, dtype=torch.int32, device=device)
                deepspeed.comm.all_reduce(global_stall_flag, op=deepspeed.comm.ReduceOp.MIN)
                if global_stall_flag.item() == 1:
                    msg = (
                        "suspected global process pause/preemption: "
                        f"all ranks saw dataloader_wait >= {global_stall_seconds}s "
                        f"(epoch={epoch}, batch={batch_idx}, wait={data_wait_sec:.3f}s)"
                    )
                    rank_log("ERROR " + msg, force_all_ranks=True)
                    if abort_on_global_stall:
                        raise RuntimeError(msg)

        stage_t0 = time.perf_counter()
        model_engine.zero_grad()
        zero_grad_sec = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        input_ids = data["input_ids"].to(device, non_blocking=True)
        attention_mask = data["attention_mask"].to(device, non_blocking=True)
        loss_mask = data["loss_mask"]
        h2d_sec = time.perf_counter() - stage_t0

        try:
            stage_t0 = time.perf_counter()
            plosses, vlosses, acces = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            forward_sec = time.perf_counter() - stage_t0
        except Exception as e:
            rank_log(
                f"ERROR stage=forward epoch={epoch} batch={batch_idx} "
                f"wait={data_wait_sec:.3f}s h2d={h2d_sec:.3f}s err={repr(e)}",
                force_all_ranks=True,
            )
            raise

        model_nonfinite_flag = torch.tensor(
            int(getattr(model_engine.module, "_nonfinite_found_this_step", 0)),
            dtype=torch.int32,
            device=device,
        )
        stage_t0 = time.perf_counter()
        deepspeed.comm.all_reduce(model_nonfinite_flag, op=deepspeed.comm.ReduceOp.MAX)
        allreduce_nonfinite_sec = time.perf_counter() - stage_t0
        if model_nonfinite_flag.item() > 0:
            stage = getattr(model_engine.module, "_nonfinite_stage_this_step", "unknown")
            if global_rank == 0:
                print(
                    f"ERROR: non-finite tensor detected in model forward "
                    f"(epoch={epoch}, batch={batch_idx}, first_stage={stage})."
                )
            if os.environ.get("EAGLE_FAIL_FAST_ON_MODEL_NONFINITE", "1") == "1":
                raise RuntimeError(
                    "non-finite tensor detected in model forward; aborting early to avoid NCCL timeout."
                )

        ploss_tensor = torch.stack([p.float() for p in plosses])
        bad_flag = (~torch.isfinite(ploss_tensor)).any().to(dtype=torch.int32)
        stage_t0 = time.perf_counter()
        deepspeed.comm.all_reduce(bad_flag, op=deepspeed.comm.ReduceOp.MAX)
        allreduce_badflag_sec = time.perf_counter() - stage_t0
        if bad_flag.item() > 0:
            if global_rank == 0:
                print(f"WARN: non-finite ploss detected at epoch={epoch}, batch={batch_idx}; skip this batch")
            model_engine.zero_grad()
            skipped_train_batches += 1
            prev_train_loop_end = time.perf_counter()
            continue

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss

        stage_t0 = time.perf_counter()
        model_engine.backward(loss)
        backward_sec = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        model_engine.step()
        step_sec = time.perf_counter() - stage_t0
        iter_total_sec = time.perf_counter() - batch_begin

        if cuda_sync_heartbeat_steps > 0 and ((batch_idx + 1) % cuda_sync_heartbeat_steps == 0):
            stage_t0 = time.perf_counter()
            torch.cuda.synchronize(device)
            sync_sec = time.perf_counter() - stage_t0
            if sync_sec > slow_stage_seconds:
                rank_log(
                    f"SLOW stage=cuda_sync epoch={epoch} batch={batch_idx} sync={sync_sec:.3f}s",
                    force_all_ranks=True,
                )

        if empty_cache_every_steps > 0 and ((batch_idx + 1) % empty_cache_every_steps == 0):
            torch.cuda.empty_cache()

        if (
            h2d_sec > slow_stage_seconds
            or forward_sec > slow_stage_seconds
            or allreduce_nonfinite_sec > slow_stage_seconds
            or allreduce_badflag_sec > slow_stage_seconds
            or backward_sec > slow_stage_seconds
            or step_sec > slow_stage_seconds
        ):
            rank_log(
                f"SLOW epoch={epoch} batch={batch_idx} wait={data_wait_sec:.3f}s zero={zero_grad_sec:.3f}s "
                f"h2d={h2d_sec:.3f}s fwd={forward_sec:.3f}s "
                f"ar_nonfinite={allreduce_nonfinite_sec:.3f}s ar_bad={allreduce_badflag_sec:.3f}s "
                f"bwd={backward_sec:.3f}s step={step_sec:.3f}s iter={iter_total_sec:.3f}s",
                force_all_ranks=True,
            )

        if batch_idx % heartbeat_steps == 0:
            mem_alloc_mb = 0.0
            mem_reserved_mb = 0.0
            if torch.cuda.is_available():
                mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            lr = optimizer.optimizer.param_groups[0]["lr"]
            rank_log(
                f"HB epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                f"wait={data_wait_sec:.3f}s h2d={h2d_sec:.3f}s fwd={forward_sec:.3f}s "
                f"ar_nonfinite={allreduce_nonfinite_sec:.3f}s ar_bad={allreduce_badflag_sec:.3f}s "
                f"bwd={backward_sec:.3f}s step={step_sec:.3f}s iter={iter_total_sec:.3f}s "
                f"lr={lr:.8f} mem_alloc={mem_alloc_mb:.1f}MB mem_reserved={mem_reserved_mb:.1f}MB"
            )

        prev_train_loop_end = time.perf_counter()

        if global_rank == 0 and wandb is not None:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    if global_rank == 0 and skipped_train_batches > 0:
        print(f"WARN: skipped_train_batches={skipped_train_batches}/{processed_train_batches}")
    if processed_train_batches == 0:
        raise RuntimeError("No training batches were processed in this epoch; training is invalid.")
    if skipped_train_batches >= processed_train_batches:
        raise RuntimeError("All training batches were skipped due non-finite losses; training is invalid.")

    for i in range(len(epoch_acces)):
        if len(epoch_acces[i]) == 0:
            acc_i = torch.tensor(0.0, device=device)
        else:
            acc_vals = torch.tensor(epoch_acces[i], dtype=torch.float32, device=device)
            acc_vals = torch.nan_to_num(acc_vals, nan=0.0, posinf=0.0, neginf=0.0)
            acc_i = acc_vals.mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            if wandb is not None:
                wandb.log({f"train/epochacc_{i}": acc_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.4f}")

    for i in range(len(epoch_plosses)):
        if len(epoch_plosses[i]) == 0:
            loss_i = torch.tensor(0.0, device=device)
        else:
            loss_vals = torch.tensor(epoch_plosses[i], dtype=torch.float32, device=device)
            loss_vals = torch.nan_to_num(loss_vals, nan=0.0, posinf=0.0, neginf=0.0)
            loss_i = loss_vals.mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            if wandb is not None:
                wandb.log({f"train/epochploss_{i}": loss_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.6f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]
    skipped_eval_batches = 0
    processed_eval_batches = 0
    prev_eval_loop_end = time.perf_counter()

    for batch_idx, data in enumerate(tqdm(test_loader)):
        if max_eval_steps > 0 and batch_idx >= max_eval_steps:
            if global_rank == 0:
                rank_log(f"Reached max_eval_steps={max_eval_steps}; stop eval early for diagnostics.")
            break
        processed_eval_batches += 1
        batch_begin = time.perf_counter()
        data_wait_sec = batch_begin - prev_eval_loop_end
        if data_wait_sec > slow_stage_seconds:
            rank_log(
                f"SLOW stage=eval_dataloader_wait epoch={epoch} batch={batch_idx} wait={data_wait_sec:.3f}s",
                force_all_ranks=True,
            )
            maybe_dump_stall_diag("eval_dataloader_wait", epoch, batch_idx, data_wait_sec)
            if global_stall_seconds > 0 and data_wait_sec >= global_stall_seconds:
                global_stall_flag = torch.tensor(1, dtype=torch.int32, device=device)
                deepspeed.comm.all_reduce(global_stall_flag, op=deepspeed.comm.ReduceOp.MIN)
                if global_stall_flag.item() == 1:
                    msg = (
                        "suspected global process pause/preemption in eval: "
                        f"all ranks saw dataloader_wait >= {global_stall_seconds}s "
                        f"(epoch={epoch}, batch={batch_idx}, wait={data_wait_sec:.3f}s)"
                    )
                    rank_log("ERROR " + msg, force_all_ranks=True)
                    if abort_on_global_stall:
                        raise RuntimeError(msg)

        with torch.no_grad():
            stage_t0 = time.perf_counter()
            input_ids = data["input_ids"].to(device, non_blocking=True)
            attention_mask = data["attention_mask"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"]
            h2d_sec = time.perf_counter() - stage_t0

            stage_t0 = time.perf_counter()
            plosses, vlosses, acces = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            forward_sec = time.perf_counter() - stage_t0
            model_nonfinite_flag = torch.tensor(
                int(getattr(model_engine.module, "_nonfinite_found_this_step", 0)),
                dtype=torch.int32,
                device=device,
            )
            stage_t0 = time.perf_counter()
            deepspeed.comm.all_reduce(model_nonfinite_flag, op=deepspeed.comm.ReduceOp.MAX)
            allreduce_nonfinite_sec = time.perf_counter() - stage_t0
            if model_nonfinite_flag.item() > 0:
                stage = getattr(model_engine.module, "_nonfinite_stage_this_step", "unknown")
                if global_rank == 0:
                    print(
                        f"ERROR: non-finite tensor detected in eval forward "
                        f"(epoch={epoch}, batch={batch_idx}, first_stage={stage})."
                    )
                if os.environ.get("EAGLE_FAIL_FAST_ON_MODEL_NONFINITE", "1") == "1":
                    raise RuntimeError(
                        "non-finite tensor detected in eval forward; aborting early to avoid NCCL timeout."
                    )
            ploss_tensor = torch.stack([p.float() for p in plosses])
            bad_flag = (~torch.isfinite(ploss_tensor)).any().to(dtype=torch.int32)
            stage_t0 = time.perf_counter()
            deepspeed.comm.all_reduce(bad_flag, op=deepspeed.comm.ReduceOp.MAX)
            allreduce_badflag_sec = time.perf_counter() - stage_t0
            if bad_flag.item() > 0:
                if global_rank == 0:
                    print(f"WARN: non-finite eval ploss at epoch={epoch}, batch={batch_idx}; skip this batch")
                skipped_eval_batches += 1
                prev_eval_loop_end = time.perf_counter()
                continue
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

            iter_total_sec = time.perf_counter() - batch_begin
            if (
                h2d_sec > slow_stage_seconds
                or forward_sec > slow_stage_seconds
                or allreduce_nonfinite_sec > slow_stage_seconds
                or allreduce_badflag_sec > slow_stage_seconds
            ):
                rank_log(
                    f"SLOW eval epoch={epoch} batch={batch_idx} wait={data_wait_sec:.3f}s "
                    f"h2d={h2d_sec:.3f}s fwd={forward_sec:.3f}s "
                    f"ar_nonfinite={allreduce_nonfinite_sec:.3f}s ar_bad={allreduce_badflag_sec:.3f}s "
                    f"iter={iter_total_sec:.3f}s",
                    force_all_ranks=True,
                )

            if batch_idx % heartbeat_steps == 0:
                rank_log(
                    f"HB-EVAL epoch={epoch} batch={batch_idx}/{len(test_loader)} "
                    f"wait={data_wait_sec:.3f}s h2d={h2d_sec:.3f}s fwd={forward_sec:.3f}s "
                    f"ar_nonfinite={allreduce_nonfinite_sec:.3f}s ar_bad={allreduce_badflag_sec:.3f}s "
                    f"iter={iter_total_sec:.3f}s"
                )
            prev_eval_loop_end = time.perf_counter()

    if global_rank == 0 and skipped_eval_batches > 0:
        print(f"WARN: skipped_eval_batches={skipped_eval_batches}/{processed_eval_batches}")
    if processed_eval_batches == 0:
        raise RuntimeError("No eval batches were processed in this epoch; eval is invalid.")
    if skipped_eval_batches >= processed_eval_batches:
        raise RuntimeError("All eval batches were skipped due non-finite losses; eval is invalid.")

    for i in range(len(epoch_acces)):
        if len(epoch_acces[i]) == 0:
            acc_i = torch.tensor(0.0, device=device)
        else:
            acc_vals = torch.tensor(epoch_acces[i], dtype=torch.float32, device=device)
            acc_vals = torch.nan_to_num(acc_vals, nan=0.0, posinf=0.0, neginf=0.0)
            acc_i = acc_vals.mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            if wandb is not None:
                wandb.log({f"test/epochacc_{i}": acc_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.4f}")

    for i in range(len(epoch_plosses)):
        if len(epoch_plosses[i]) == 0:
            loss_i = torch.tensor(0.0, device=device)
        else:
            loss_vals = torch.tensor(epoch_plosses[i], dtype=torch.float32, device=device)
            loss_vals = torch.nan_to_num(loss_vals, nan=0.0, posinf=0.0, neginf=0.0)
            loss_i = loss_vals.mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            if wandb is not None:
                wandb.log({f"test/epochploss_{i}": loss_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.6f}")

    # clear out the redundance cache after each epoch
    torch.cuda.empty_cache()

    save_every = max(1, int(train_config["save_every_epochs"]))
    save_ds_every = max(1, int(train_config["save_ds_every_epochs"]))

    if ((epoch + 1) % save_every == 0) or (epoch == num_epochs - 1):
        model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if (epoch + 1) % save_ds_every == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
