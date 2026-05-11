"""
EAGLE2 training for Qwen2.5-Coder-14B-Instruct.

Trains a lightweight draft head (cnets1.Model) on pre-extracted hidden states.
Data format: directory of .pt files, each containing {hidden_state, input_ids, loss_mask}.

Usage:
    accelerate launch -m eagle.train2.main \
        --basepath /path/to/Qwen2.5-Coder-14B-Instruct \
        --datadir /path/to/data/*.pt \
        --cpdir /path/to/checkpoints \
        --lr 3e-5 --bs 4 --num-epochs 20
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors import safe_open
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--cpdir", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--num-warmup-steps", type=int, default=2000)
    parser.add_argument("--total-steps", type=int, default=800000)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--v-w", type=float, default=1.0)
    parser.add_argument("--p-w", type=float, default=0.1)
    parser.add_argument("--data-noise", action="store_true", default=True)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--configpath", type=str, default=None,
                        help="EAGLE head config.json. If None, auto-generate from basepath.")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".pt"):
                datapath.append(os.path.join(root, file))
    datapath.sort()
    return datapath


def load_lm_head(basepath):
    """Load frozen lm_head from base model."""
    config = AutoConfig.from_pretrained(basepath)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    try:
        index_path = os.path.join(basepath, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            head_file = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(basepath, head_file), framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except (FileNotFoundError, KeyError):
        index_path = os.path.join(basepath, "pytorch_model.bin.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            head_file = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(basepath, head_file), map_location="cpu")
        tensor = weights["lm_head.weight"].float()

    head.weight.data = tensor
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head


def generate_eagle_config(basepath, outpath):
    """Auto-generate EAGLE2 head config from base model config.

    Aligns with the official yuhuili/EAGLE-Qwen2 release: when the base model
    is a Qwen2/Qwen2.5 family, emit `model_type: "qwen2"` and `qkv_bias: true`
    so the head's attention layer keeps the QKV bias terms that Qwen relies on.
    For LLaMA-family bases, omit `qkv_bias` (cnets1.py defaults to bias=False).
    """
    base_config = AutoConfig.from_pretrained(basepath)
    base_model_type = str(getattr(base_config, "model_type", "")).lower()
    is_qwen = base_model_type.startswith("qwen")

    eagle_config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_act": getattr(base_config, "hidden_act", "silu"),
        "hidden_size": base_config.hidden_size,
        "intermediate_size": base_config.intermediate_size,
        "num_attention_heads": base_config.num_attention_heads,
        "num_key_value_heads": getattr(base_config, "num_key_value_heads", base_config.num_attention_heads),
        "num_hidden_layers": 1,
        "rms_norm_eps": getattr(base_config, "rms_norm_eps", 1e-6),
        "max_position_embeddings": getattr(base_config, "max_position_embeddings", 2048),
        "vocab_size": base_config.vocab_size,
        "pad_token_id": getattr(base_config, "pad_token_id", 0) or 0,
        "bos_token_id": getattr(base_config, "bos_token_id", 1),
        "eos_token_id": getattr(base_config, "eos_token_id", 2),
        "model_type": "qwen2" if is_qwen else "llama",
        "tie_word_embeddings": False,
        "use_cache": True,
        "initializer_range": 0.02,
        "torch_dtype": "bfloat16",
        "rope_theta": getattr(base_config, "rope_theta", 10000.0),
        "rope_scaling": getattr(base_config, "rope_scaling", None),
    }
    if is_qwen:
        # Qwen2/Qwen2.5 attention has bias on Q/K/V (not on O). cnets1.py reads
        # `qkv_bias` via hasattr() — presence of this field switches Q/K/V to
        # bias=True. Mirrors yuhuili/EAGLE-Qwen2-7B-Instruct/config.json.
        eagle_config["qkv_bias"] = bool(getattr(base_config, "attention_bias", True))

    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(eagle_config, f, indent=2)
    return outpath


class AddUniformNoise:
    def __init__(self, std=0.2):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        data["hidden_state_big"] = tensor + noise
        return data


class HiddenStateDataset(Dataset):
    def __init__(self, filepaths, max_len, transform=None):
        self.filepaths = filepaths
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        data = torch.load(self.filepaths[index], map_location="cpu")
        hidden_state = data["hidden_state"][:self.max_len][None, :]
        input_ids = data["input_ids"][:self.max_len][None, :]
        loss_mask = data["loss_mask"][:self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        input_ids_target = torch.cat((input_ids_target, torch.tensor([[0]])), dim=1)

        target = hidden_state[:, 1:, :]
        target = torch.cat((target, torch.zeros(1, 1, target.shape[2])), dim=1)
        loss_mask[-1] = 0

        new_data = {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "target": target,
            "hidden_state_big": hidden_state,
            "input_ids": input_ids_target,
        }

        if self.transform:
            new_data = self.transform(new_data)
        return new_data


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        return torch.cat((intensors, padding_tensor), dim=1)

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features):
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        batch = {
            "input_ids": torch.cat([self.paddingtensor2D(item["input_ids"], max_length) for item in features]),
            "hidden_states": torch.cat([self.paddingtensor(item["hidden_state_big"], max_length) for item in features]),
            "target": torch.cat([self.paddingtensor(item["target"], max_length) for item in features]),
            "loss_mask": torch.tensor(
                [item["loss_mask"] + [0] * (max_length - len(item["loss_mask"])) for item in features]
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] + [0] * (max_length - len(item["attention_mask"])) for item in features]
            ),
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in topk]


def compute_loss(target, target_p, predict, loss_mask, head, criterion):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0] * max_length
    correct = [0] * max_length
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden = hidden_states[:, :pre_len]
        pre_ids = input_ids[:, :pre_len]

        past_key_values = None
        gen_input = pre_hidden
        gen_ids = pre_ids
        generated = []
        for _ in range(max_length):
            if past_key_values is not None:
                out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values, use_cache=True)
            else:
                out_hidden, past_key_values = model(gen_input, input_ids=gen_ids, use_cache=True)
            last_hidden = out_hidden[:, -1:]
            token = head(last_hidden).argmax(dim=-1)
            generated.append(token)
            gen_ids = torch.cat((gen_ids, token), dim=1)

        generate_ids = torch.cat(generated, dim=1)
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0 or pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    return [correct[i] / max(total[i], 1) for i in range(max_length)]


def main():
    args = parse_args()

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    set_seed(0)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    wandb = None
    if accelerator.is_main_process and not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(project="eagle2-qwen25", config=vars(args))
            wandb = _wandb
        except ImportError:
            pass

    # Load frozen lm_head
    head = load_lm_head(args.basepath)

    # EAGLE2 head config
    if args.configpath and os.path.exists(args.configpath):
        configpath = args.configpath
    else:
        configpath = os.path.join(args.cpdir, "eagle2_config.json")
        generate_eagle_config(args.basepath, configpath)
        if accelerator.is_main_process:
            print(f"Auto-generated EAGLE2 config: {configpath}")

    from eagle.model.cnets1 import Model
    from eagle.model.configs import EConfig

    config = EConfig.from_pretrained(configpath)
    model = Model(config, load_emb=True, path=args.basepath)

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps
    )

    # Data
    datapath = list_files(args.datadir)
    if not datapath:
        raise FileNotFoundError(f"No .pt files found in {args.datadir}")
    if accelerator.is_main_process:
        print(f"Found {len(datapath)} samples in {args.datadir}")

    split = int(len(datapath) * 0.95)
    traindatapath = datapath[:split]
    testdatapath = datapath[split:]

    aug = AddUniformNoise(std=args.noise_std) if args.data_noise else None
    train_dataset = HiddenStateDataset(traindatapath, args.max_len, transform=aug)
    test_dataset = HiddenStateDataset(testdatapath, args.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        collate_fn=DataCollatorWithPadding(), num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False,
        collate_fn=DataCollatorWithPadding(), num_workers=args.num_workers, pin_memory=True,
    )

    os.makedirs(args.cpdir, exist_ok=True)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for data in train_loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                with torch.no_grad():
                    target_head = head(data["target"])
                    target_p = nn.Softmax(dim=2)(target_head).detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask, head, criterion)
                loss = args.v_w * vloss + args.p_w * ploss
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target_ids = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target_ids) * loss_mask.squeeze()).sum().item()
                total += ct
                correct += cc
                epoch_loss += loss.item()
                num_batches += 1

            if accelerator.is_main_process and wandb and num_batches % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/vloss": vloss.item(),
                    "train/ploss": ploss.item(),
                    "train/acc": cc / max(ct, 1),
                    "train/lr": optimizer.param_groups[0]["lr"] if hasattr(optimizer, 'param_groups') else args.lr,
                })

        # Epoch summary
        correct_t = torch.tensor(correct).cuda()
        total_t = torch.tensor(total).cuda()
        correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
        if accelerator.is_main_process:
            c, t = correct_t.sum().item(), total_t.sum().item()
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Loss: {epoch_loss/max(num_batches,1):.4f} Acc: {100*c/max(t,1):.2f}%")

        # Eval + save
        if (epoch + 1) % args.save_freq == 0:
            model.eval()
            eval_correct = 0
            eval_total = 0
            k_accs = []

            for batch_idx, data in enumerate(test_loader):
                with torch.no_grad():
                    if batch_idx < 10:
                        k_accs.append(getkacc(model, data, head, max_length=5))
                    predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                    target_head = head(data["target"])
                    target_p = nn.Softmax(dim=2)(target_head).detach()
                    loss_mask = data["loss_mask"][:, :, None]
                    _, _, out_head = compute_loss(data["target"], target_p, predict, loss_mask, head, criterion)
                    _, predicted = torch.max(out_head, 2)
                    _, target_ids = torch.max(target_head, 2)
                    ct = loss_mask.sum().item()
                    cc = ((predicted == target_ids) * loss_mask.squeeze()).sum().item()
                    eval_correct += cc
                    eval_total += ct

            ec = torch.tensor(eval_correct).cuda()
            et = torch.tensor(eval_total).cuda()
            ec, et = accelerator.gather_for_metrics((ec, et))
            if accelerator.is_main_process:
                c, t = ec.sum().item(), et.sum().item()
                print(f"  Eval Acc: {100*c/max(t,1):.2f}%")
                if k_accs:
                    mean_k = np.array(k_accs).mean(axis=0)
                    tau_est = 1.0
                    prod = 1.0
                    for a in mean_k:
                        prod *= a
                        tau_est += prod
                    print(f"  K-acc: {['%.3f'%x for x in mean_k]}  τ_est={tau_est:.2f}")
                else:
                    tau_est = 0.0
                if wandb:
                    log_d = {"test/acc": c / max(t, 1), "epoch": epoch + 1}
                    if tau_est > 0:
                        log_d["test/tau_est"] = tau_est
                    wandb.log(log_d)

            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch+1}")
            if accelerator.is_main_process:
                print(f"  Checkpoint saved: {args.cpdir}/state_{epoch+1}")


if __name__ == "__main__":
    main()
