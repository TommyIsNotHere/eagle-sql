# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from modeling_llama_kv import LlamaForCausalLM
from configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
import re
import time
import hashlib
import json
import importlib
import importlib.util
from pathlib import Path

from data_utils import build_tokenized_sample

def _load_kv_qwen2_cls():
    errors = []

    # 1) Canonical package import.
    try:
        mod = importlib.import_module("eagle.model.modeling_qwen2_kv")
        cls = getattr(mod, "Qwen2ForCausalLM", None)
        if cls is not None:
            return cls, None
        errors.append("Qwen2ForCausalLM not found in eagle.model.modeling_qwen2_kv")
    except Exception as e:
        errors.append(f"package import failed: {e}")

    # 2) Absolute file-path import fallback (independent of PYTHONPATH/cwd).
    try:
        module_path = Path(__file__).resolve().parents[1] / "model" / "modeling_qwen2_kv.py"
        spec = importlib.util.spec_from_file_location("eagle_modeling_qwen2_kv_fallback", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to build import spec from {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, "Qwen2ForCausalLM", None)
        if cls is None:
            raise ImportError("Qwen2ForCausalLM not found in file import")
        return cls, None
    except Exception as e:
        errors.append(f"file import failed: {e}")

    return None, " | ".join(errors)


KVQwen2ForCausalLM, _qwen2_kv_import_error = _load_kv_qwen2_cls()

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        base_theta = float(getattr(self.config, "rope_theta", 10000.0))
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base_theta,
            )
        else:
            # Cross-model compatibility:
            # LLaMA uses {"type": "...", "factor": ...}
            # Some Qwen configs may use {"rope_type": "...", ...} and/or omit "type".
            if not isinstance(rope_scaling, dict):
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=base_theta,
                )
                return

            scaling_type = rope_scaling.get("type") or rope_scaling.get("rope_type") or rope_scaling.get("name")
            scaling_factor = float(rope_scaling.get("factor", 1.0))

            if scaling_factor <= 1.0:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=base_theta,
                )
                return

            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=base_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=base_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                # Unsupported rope type (e.g. yarn). Fall back to base RoPE for stability.
                print(f"WARN: unsupported rope scaling type '{scaling_type}', fallback to base RoPE")
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=base_theta,
                )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            cache_hidden: Optional[List[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        # cache_k = [self.k_proj(hidden) for hidden in cache_hidden]
        # cache_v = [self.v_proj(hidden) for hidden in cache_hidden]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        # query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids + lck)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid modify hidden cache inplace which will cause in-place modification error when enable gradient checkpoint. 
        # Return the updated hidden cache instead.
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)
            
        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        lck = len(cache_k)


        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = cache_k[i]

            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # Return the updated hidden cache.
        new_past_key_value = [local_cache_k,local_cache_v]
        return attn_output, new_past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.last = last
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, last=last)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            cache_hidden: Optional[List[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        if cache_hidden is None:
            cache_hidden = [[], []]

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        return_hidden = hidden_states

        # cache_hidden.append(hidden_states)

        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states


        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)


        return outputs, latest_hidden_cache


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def process_data(data_chunk):

    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids= input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1

    return token_dict


def merge_dicts(dicts):
    """合并多个 Counter 字典"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


class Model(nn.Module):
    def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None):
        super().__init__() 
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, index=index) for index in range(config.num_hidden_layers)])
        self.train_config = training_config
        # Settng dschf to allow efficient ZeRO-3 usage between hf and ds.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        self.midlayer = LlamaDecoderLayeremb(config)
        if isinstance(self.train_config, dict):
            self.max_len = int(self.train_config.get("max_len", 2048))
            self.preprocess_num_proc = int(
                self.train_config.get(
                    "preprocess_num_proc",
                    min(8, max(1, os.cpu_count() or 1)),
                )
            )
            self.gradient_checkpointing = bool(
                self.train_config.get(
                    "gradient_checkpointing",
                    self.train_config.get("gradient_checkpoint", True),
                )
            )
        else:
            self.max_len = int(getattr(self.train_config, "max_len", 2048))
            self.preprocess_num_proc = int(
                getattr(
                    self.train_config,
                    "preprocess_num_proc",
                    min(8, max(1, os.cpu_count() or 1)),
                )
            )
            self.gradient_checkpointing = bool(
                getattr(
                    self.train_config,
                    "gradient_checkpointing",
                    getattr(self.train_config, "gradient_checkpoint", True),
                )
            )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.length = 7
        # Custom attention path here can become numerically unstable with explicit padding masks.
        # Default to causal-only; enable padding mask only for controlled ablations.
        self.use_padding_attn_mask = os.environ.get("EAGLE_USE_PADDING_ATTN_MASK", "0") == "1"
        self._debug_numerics_steps = int(os.environ.get("EAGLE_DEBUG_NUMERICS_STEPS", "3"))
        self.force_teacher_attn_mask = os.environ.get("EAGLE_FORCE_TEACHER_ATTN_MASK", "1") == "1"
        self._debug_teacher_numerics_steps = int(
            os.environ.get("EAGLE_DEBUG_TEACHER_NUMERICS_STEPS", "3")
        )
        self._teacher_mask_fallback_steps = int(
            os.environ.get("EAGLE_TEACHER_MASK_FALLBACK_STEPS", "1000000")
        )
        # Keep teacher forward path aligned with inference runtime by default.
        self.qwen_teacher_impl = os.environ.get("EAGLE_QWEN_TEACHER_IMPL", "kv").strip().lower()
        self.strict_teacher_impl = os.environ.get("EAGLE_STRICT_TEACHER_IMPL", "1") == "1"
        self.teacher_hidden_selector = os.environ.get("EAGLE_TEACHER_HIDDEN_SELECTOR", "paper").strip().lower()
        self.teacher_hidden_custom = os.environ.get("EAGLE_TEACHER_HIDDEN_CUSTOM", "").strip()
        if self.teacher_hidden_selector not in {"legacy", "paper", "custom"}:
            print(
                "WARN: invalid EAGLE_TEACHER_HIDDEN_SELECTOR="
                f"{self.teacher_hidden_selector}, fallback to legacy"
            )
            self.teacher_hidden_selector = "legacy"
        # Paper-aligned default for training-time test: feed self predictions
        # back during training rollout.
        self.input_rollout_mode = os.environ.get("EAGLE_INPUT_ROLLOUT_MODE", "pred").strip().lower()
        if self.input_rollout_mode not in {"teacher", "pred", "scheduled"}:
            print(
                "WARN: invalid EAGLE_INPUT_ROLLOUT_MODE="
                f"{self.input_rollout_mode}, fallback to teacher"
            )
            self.input_rollout_mode = "teacher"
        try:
            _rollout_ratio_start = float(os.environ.get("EAGLE_INPUT_ROLLOUT_RATIO_START", "0.0"))
        except Exception:
            _rollout_ratio_start = 0.0
        try:
            _rollout_ratio_end = float(os.environ.get("EAGLE_INPUT_ROLLOUT_RATIO_END", "0.3"))
        except Exception:
            _rollout_ratio_end = 0.3
        self.input_rollout_ratio_start = min(1.0, max(0.0, _rollout_ratio_start))
        self.input_rollout_ratio_end = min(1.0, max(0.0, _rollout_ratio_end))
        self.input_rollout_align_target = os.environ.get("EAGLE_INPUT_ROLLOUT_ALIGN_TARGET", "1") == "1"
        self.loss_mode = os.environ.get("EAGLE_LOSS_MODE", "hybrid").strip().lower()
        if self.loss_mode not in {"hybrid", "paper"}:
            print(f"WARN: invalid EAGLE_LOSS_MODE={self.loss_mode}, fallback to hybrid")
            self.loss_mode = "hybrid"
        # Anchor distillation with supervised CE on gold draft-token ids.
        try:
            _gold_w = float(os.environ.get("EAGLE_GOLD_CE_WEIGHT", "0.35"))
        except Exception:
            _gold_w = 0.35
        self.gold_ce_weight = min(1.0, max(0.0, _gold_w))
        self.distill_only_in_draft = os.environ.get("EAGLE_DISTILL_ONLY_IN_DRAFT", "1") == "1"
        self._trace_nonfinite_steps = int(os.environ.get("EAGLE_TRACE_NONFINITE_STEPS", "8"))
        self._check_target_param_finite = os.environ.get("EAGLE_CHECK_TARGET_PARAM_FINITE", "0") == "1"
        self._nonfinite_found_this_step = 0
        self._nonfinite_stage_this_step = ""
        base_cfg = AutoConfig.from_pretrained(path)
        cfg_dtype = getattr(base_cfg, "torch_dtype", None)
        if cfg_dtype is None:
            cfg_dtype = getattr(base_cfg, "dtype", None)
        dtype_name = str(cfg_dtype).lower() if cfg_dtype is not None else ""
        if cfg_dtype == torch.bfloat16 or "bfloat16" in dtype_name:
            target_dtype = torch.bfloat16
        elif cfg_dtype == torch.float16 or "float16" in dtype_name:
            target_dtype = torch.float16
        else:
            # Keep bf16 as preferred fallback with current DeepSpeed config.
            target_dtype = torch.bfloat16
        arch = (getattr(base_cfg, "architectures", None) or [""])[0]
        require_qwen_kv = os.environ.get("EAGLE_REQUIRE_QWEN_KV", "0") == "1"
        print(
            "[train-loss-config] "
            f"gold_ce_weight={self.gold_ce_weight:.3f} "
            f"distill_only_in_draft={int(self.distill_only_in_draft)} "
            f"loss_mode={self.loss_mode} "
            f"teacher_hidden_selector={self.teacher_hidden_selector} "
            f"input_rollout_mode={self.input_rollout_mode} "
            f"input_rollout_align_target={int(self.input_rollout_align_target)} "
            f"input_rollout_ratio=({self.input_rollout_ratio_start:.2f}->{self.input_rollout_ratio_end:.2f}) "
            f"strict_teacher_impl={int(self.strict_teacher_impl)} "
            f"teacher_impl={self.qwen_teacher_impl}"
        )

        def _load_with_dtype(model_cls, model_path, dtype):
            # transformers>=4.56 prefers `dtype`; older versions may only support `torch_dtype`.
            try:
                return model_cls.from_pretrained(model_path, dtype=dtype)
            except TypeError:
                return model_cls.from_pretrained(model_path, torch_dtype=dtype)

        if arch == "Qwen2ForCausalLM":
            if self.strict_teacher_impl and self.qwen_teacher_impl != "kv":
                raise ValueError(
                    "Qwen2 training requires EAGLE_QWEN_TEACHER_IMPL=kv when "
                    "EAGLE_STRICT_TEACHER_IMPL=1 to keep teacher/runtime path aligned."
                )
            if self.qwen_teacher_impl == "hf":
                print("INFO: EAGLE_QWEN_TEACHER_IMPL=hf -> use transformers.AutoModelForCausalLM as teacher")
                self.target_model = _load_with_dtype(AutoModelForCausalLM, path, target_dtype)
            elif self.qwen_teacher_impl == "kv":
                if KVQwen2ForCausalLM is None:
                    err_msg = (
                        "Qwen2ForCausalLM base model detected, but "
                        "custom KV class `eagle.model.modeling_qwen2_kv.Qwen2ForCausalLM` is unavailable. "
                        f"detail: {_qwen2_kv_import_error}"
                    )
                    if require_qwen_kv:
                        raise ImportError(
                            err_msg
                            + " | strict mode enabled by EAGLE_REQUIRE_QWEN_KV=1; aborting to avoid silent fallback."
                        )
                    print(f"WARN: {err_msg}")
                    print("WARN: fallback to transformers.AutoModelForCausalLM")
                    self.target_model = _load_with_dtype(AutoModelForCausalLM, path, target_dtype)
                else:
                    print("INFO: EAGLE_QWEN_TEACHER_IMPL=kv -> use custom KVQwen2ForCausalLM as teacher")
                    self.target_model = _load_with_dtype(KVQwen2ForCausalLM, path, target_dtype)
            else:
                raise ValueError(
                    f"invalid EAGLE_QWEN_TEACHER_IMPL={self.qwen_teacher_impl}, expect one of: kv, hf"
                )
        else:
            self.target_model = _load_with_dtype(LlamaForCausalLM, path, target_dtype)
        self.target_model.eval()
        self.fc=nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        for param in self.target_model.parameters():
            param.requires_grad = False
        if self._check_target_param_finite:
            bad_params = 0
            bad_values = 0
            for name, p in self.target_model.named_parameters():
                bad = int((~torch.isfinite(p.data)).sum().item())
                if bad > 0:
                    bad_params += 1
                    bad_values += bad
                    print(f"[target-param-check] non-finite param `{name}` bad_values={bad}")
            print(
                "[target-param-check] "
                f"checked target params, bad_params={bad_params}, bad_values={bad_values}"
            )

        if not load_emb:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        else:

            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, _weight=tensor)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def _count_tokens_from_tokenized_dataset(self, tokenized_dataset, rank: int):
        token_counts = torch.zeros(self.vocab_size, dtype=torch.long)
        total_rows = len(tokenized_dataset)
        t0 = time.time()
        next_pct = 25

        for idx in range(total_rows):
            sample = tokenized_dataset[idx]
            ids = sample["input_ids"]
            mask = sample["loss_mask"]

            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)

            if ids.dim() > 1:
                ids = ids[0]
            if mask.dim() > 1:
                mask = mask[0]

            ids = ids.long()
            mask = mask.bool()

            if mask.any():
                selected = ids[mask]
                token_counts += torch.bincount(selected, minlength=self.vocab_size)

            pct = int((idx + 1) / max(1, total_rows) * 100)
            if pct >= next_pct:
                print(f"[scandata][rank {rank}] token count {pct}% ({idx + 1}/{total_rows})")
                next_pct = pct + 25

        print(f"[scandata][rank {rank}] token counting done rows={total_rows} in {time.time() - t0:.1f}s")
        return token_counts

    def _build_cache_from_counts(self, token_counts: torch.Tensor, N: int, rank: int):
        if token_counts.numel() != self.vocab_size:
            raise ValueError(
                f"token_counts size mismatch: got {token_counts.numel()}, expect vocab_size={self.vocab_size}"
            )

        total_frequency = int(token_counts.sum().item())
        if total_frequency <= 0:
            raise RuntimeError("scandata failed: no supervised tokens found in tokenized dataset")

        k = min(int(N), int(self.vocab_size))
        top_vals, top_idx = torch.topk(token_counts, k=k)
        top_sum = int(top_vals.sum().item())
        top_ratio = top_sum / max(1, total_frequency)
        print(f"[scandata][rank {rank}] top {k} token frequency ratio: {top_ratio:.2%}")

        used_tokens = torch.sort(top_idx).values.long()
        d2t = used_tokens - torch.arange(used_tokens.numel(), dtype=torch.long)
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        t2d[used_tokens] = True
        return d2t, t2d

    def _build_t2d_index(self, t2d: torch.Tensor):
        # Map full-vocab token id -> draft vocab id (or -1 when token is out of draft vocab).
        t2d = t2d.bool()
        t2d_index = torch.full((self.vocab_size,), -1, dtype=torch.long)
        used_tokens = torch.nonzero(t2d, as_tuple=False).squeeze(1)
        if used_tokens.numel() > 0:
            t2d_index[used_tokens] = torch.arange(used_tokens.numel(), dtype=torch.long)
        return t2d_index

    def _normalize_teacher_hidden_indices(self, indices, total_hidden_states: int):
        if total_hidden_states <= 0:
            raise ValueError("teacher hidden states must be non-empty")

        out = []
        for idx in indices:
            try:
                i = int(idx)
            except Exception:
                continue
            i = min(max(i, 0), total_hidden_states - 1)
            out.append(i)

        if not out:
            out = [0]
        if len(out) >= 3:
            return out[:3]
        while len(out) < 3:
            out.append(out[-1])
        return out

    def _resolve_teacher_hidden_indices(self, total_hidden_states: int):
        if self.teacher_hidden_selector == "legacy":
            return self._normalize_teacher_hidden_indices([0, 1, 2], total_hidden_states)

        if self.teacher_hidden_selector == "custom":
            raw = self.teacher_hidden_custom
            if raw:
                custom = [x.strip() for x in raw.split(",") if x.strip()]
                return self._normalize_teacher_hidden_indices(custom, total_hidden_states)
            print("WARN: EAGLE_TEACHER_HIDDEN_SELECTOR=custom but EAGLE_TEACHER_HIDDEN_CUSTOM is empty")
            return self._normalize_teacher_hidden_indices([0, 1, 2], total_hidden_states)

        # paper: low/mid/high from decoder stack (skip embedding at 0 when possible)
        last = total_hidden_states - 1
        low = 1 if total_hidden_states > 1 else 0
        mid = (low + last) // 2
        return self._normalize_teacher_hidden_indices([low, mid, last], total_hidden_states)

    def _scheduled_rollout_ratio(self, step_idx: int):
        if self.length <= 1:
            return self.input_rollout_ratio_end
        alpha = float(step_idx) / float(max(1, self.length - 1))
        ratio = self.input_rollout_ratio_start + alpha * (
            self.input_rollout_ratio_end - self.input_rollout_ratio_start
        )
        return min(1.0, max(0.0, ratio))

    def _rollout_next_input_ids(self, input_ids, pred_draft_ids, next_loss_mask, step_idx: int):
        teacher_next = padding(input_ids, left=False)
        if self.input_rollout_mode == "teacher":
            return teacher_next

        if not hasattr(self, "draft_to_token"):
            return teacher_next
        if self.draft_to_token.numel() <= 0:
            return teacher_next

        draft_to_token = self.draft_to_token.to(input_ids.device)
        pred_draft_ids = pred_draft_ids.long().clamp(min=0, max=draft_to_token.numel() - 1)
        pred_token_ids = draft_to_token[pred_draft_ids]

        replace_mask = next_loss_mask.squeeze(-1) > 0.5
        if self.input_rollout_mode == "pred":
            return torch.where(replace_mask, pred_token_ids, teacher_next)

        ratio = self._scheduled_rollout_ratio(step_idx)
        if ratio <= 0.0:
            return teacher_next
        if ratio >= 1.0:
            return torch.where(replace_mask, pred_token_ids, teacher_next)
        sample_mask = torch.rand_like(teacher_next.float()) < ratio
        mix_mask = replace_mask & sample_mask
        return torch.where(mix_mask, pred_token_ids, teacher_next)

    @torch.no_grad()
    def _compute_teacher_distill_target(self, input_ids, attention_mask):
        teacher_attention_mask = (
            attention_mask if (self.use_padding_attn_mask or self.force_teacher_attn_mask) else None
        )

        def _run_teacher_logits(mask):
            outs_local = self.target_model(
                input_ids=input_ids,
                attention_mask=mask,
                output_hidden_states=False,
                return_dict=True,
            )
            return outs_local.logits

        target = _run_teacher_logits(teacher_attention_mask)
        if (
            teacher_attention_mask is None
            and attention_mask is not None
            and self._teacher_mask_fallback_steps > 0
        ):
            primary_bad = int((~torch.isfinite(target)).sum().item())
            if primary_bad > 0:
                target_masked = _run_teacher_logits(attention_mask)
                masked_bad = int((~torch.isfinite(target_masked)).sum().item())
                if masked_bad <= primary_bad:
                    target = target_masked
                self._teacher_mask_fallback_steps -= 1

        return padding(target, left=False)

    def _trace_nonfinite(self, stage: str, tensor: torch.Tensor):
        bad_mask = ~torch.isfinite(tensor)
        bad_count = int(bad_mask.sum().item())
        if bad_count <= 0:
            return 0

        self._nonfinite_found_this_step = 1
        if not self._nonfinite_stage_this_step:
            self._nonfinite_stage_this_step = stage

        if self._trace_nonfinite_steps > 0:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            if rank == 0:
                total = int(tensor.numel())
                bad_pos = torch.nonzero(bad_mask, as_tuple=False)
                first_bad = bad_pos[0].tolist() if bad_pos.numel() > 0 else []
                finite_count = total - bad_count
                finite_abs_max = float(
                    torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
                )
                print(
                    "[nonfinite] "
                    f"stage={stage}, bad={bad_count}/{total}, finite={finite_count}, "
                    f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
                    f"first_bad_idx={first_bad}, finite_abs_max={finite_abs_max:.6e}"
                )
            self._trace_nonfinite_steps -= 1
        return bad_count

    def _estimate_in_draft_ratio(self, tokenized_dataset, t2d: torch.Tensor, rank: int):
        sample_rows = int(os.environ.get("EAGLE_SCANDATA_COVERAGE_SAMPLE_ROWS", "2048"))
        sample_rows = max(1, min(sample_rows, len(tokenized_dataset)))
        supervised = 0
        in_draft = 0
        t2d = t2d.bool()
        for idx in range(sample_rows):
            sample = tokenized_dataset[idx]
            ids = sample["input_ids"]
            mask = sample["loss_mask"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)
            if ids.dim() > 1:
                ids = ids[0]
            if mask.dim() > 1:
                mask = mask[0]
            ids = ids.long()
            mask = mask.bool()
            if mask.any():
                supervised += int(mask.sum().item())
                in_draft += int(t2d[ids[mask]].sum().item())
        ratio = float(in_draft) / float(max(1, supervised))
        print(
            f"[scandata][rank {rank}] sample in-draft coverage: "
            f"{in_draft}/{supervised} ({ratio:.2%}) over {sample_rows} rows"
        )
        if supervised > 0 and in_draft == 0:
            print(
                f"[scandata][rank {rank}] WARN: zero in-draft supervised tokens in sample. "
                "This can degenerate accuracy metrics."
            )
        return ratio

    def scandata(self, datapath, tokenizerpath, tokenized_dataset=None, cache_context=None):
        N = self.draft_vocab_size
        tokenizer_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(tokenizerpath).strip("/"))
        datapath_sig = str(datapath)
        try:
            datapath_path = Path(datapath).resolve()
            if datapath_path.exists():
                st = datapath_path.stat()
                datapath_sig = f"{datapath_path}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
            else:
                datapath_sig = str(datapath_path)
        except Exception:
            datapath_sig = str(datapath)
        datapath_key = hashlib.md5(datapath_sig.encode("utf-8")).hexdigest()[:12]
        cache_context_payload = "none"
        if cache_context is not None:
            try:
                cache_context_payload = json.dumps(cache_context, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            except Exception:
                cache_context_payload = repr(cache_context)
        cache_context_key = hashlib.md5(cache_context_payload.encode("utf-8")).hexdigest()[:8]
        cache_path = f"cache_{tokenizer_tag}_{datapath_key}_{cache_context_key}_{N}.pt"
        lock_path = f"{cache_path}.lock"
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        lock_stale_sec = int(os.environ.get("EAGLE_SCANDATA_LOCK_STALE_SEC", "10800"))
        lock_unknown_owner_stale_sec = int(
            os.environ.get("EAGLE_SCANDATA_LOCK_UNKNOWN_OWNER_STALE_SEC", "120")
        )

        def _pid_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

        def _read_lock_owner_pid(lock_file: str):
            try:
                with open(lock_file, "r") as f:
                    first = f.readline().strip()
                if first.isdigit():
                    return int(first)
            except Exception:
                return None
            return None

        lock_fd = None

        # In distributed launches, multiple ranks may enter here concurrently.
        # Use a lock file so only one rank builds cache while others wait and reuse it.
        if not os.path.exists(cache_path):
            waited = 0
            while True:
                try:
                    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    try:
                        os.write(lock_fd, f"{os.getpid()}\n{int(time.time())}\n".encode("utf-8"))
                    except Exception:
                        pass
                    print(f"[scandata][rank {rank}] acquired build lock: {cache_path}")
                    break
                except FileExistsError:
                    if os.path.exists(cache_path):
                        print(f"[scandata][rank {rank}] cache already built by another rank: {cache_path}")
                        break

                    # Handle stale lock left by crashed workers.
                    lock_age = None
                    owner_pid = None
                    owner_alive = None
                    try:
                        lock_age = time.time() - os.path.getmtime(lock_path)
                        owner_pid = _read_lock_owner_pid(lock_path)
                        owner_alive = _pid_alive(owner_pid) if owner_pid is not None else None

                        remove_reason = None
                        if owner_pid is None and lock_age > lock_unknown_owner_stale_sec:
                            remove_reason = (
                                f"unknown_owner_age>{lock_unknown_owner_stale_sec}s"
                            )
                        elif owner_alive is False:
                            remove_reason = "owner_process_dead"
                        elif lock_age > lock_stale_sec:
                            remove_reason = f"age>{lock_stale_sec}s"

                        if remove_reason is not None:
                            print(
                                f"[scandata][rank {rank}] removing stale lock "
                                f"(reason={remove_reason}, age={int(lock_age)}s, "
                                f"owner_pid={owner_pid}, owner_alive={owner_alive})"
                            )
                            os.remove(lock_path)
                            continue
                    except FileNotFoundError:
                        # Another worker removed the lock; retry acquiring.
                        continue
                    except Exception:
                        pass

                    waited += 1
                    if waited % 30 == 0:
                        age_text = "n/a" if lock_age is None else str(int(lock_age))
                        print(
                            f"[scandata][rank {rank}] waiting for cache builder ... {waited}s "
                            f"(lock_age={age_text}s, owner_pid={owner_pid}, owner_alive={owner_alive})"
                        )
                    time.sleep(1.0)

        if not os.path.exists(cache_path):
            try:
                if tokenized_dataset is not None:
                    print(
                        f"[scandata][rank {rank}] building cache from tokenized dataset "
                        f"(rows={len(tokenized_dataset)}, vocab={self.vocab_size}, draft_vocab={N})"
                    )
                    token_counts = self._count_tokens_from_tokenized_dataset(tokenized_dataset, rank)
                    d2t, t2d = self._build_cache_from_counts(token_counts, N, rank)
                    torch.save({"d2t": d2t, "t2d": t2d}, cache_path)
                    print(f"[scandata][rank {rank}] cache saved: {cache_path}")
                else:
                    t0 = time.time()
                    tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
                    dataset = load_dataset('json', data_files=datapath)
                    dataset = dataset['train']
                    original_columns1 = dataset.column_names
                    num_proc = max(1, min(self.preprocess_num_proc, max(1, os.cpu_count() or 1)))
                    print(f"[scandata] building cache at {cache_path}, num_proc={num_proc}, rows={len(dataset)}")

                    def preprocess_function(examples):
                        new_examples = {
                            "input_ids": [],
                            "loss_mask": []
                        }
                        conversations = examples.get("conversations") or []
                        for source in conversations:
                            sample = build_tokenized_sample(
                                tokenizer=tokenizer,
                                source=source,
                                max_len=self.max_len,
                            )
                            if sample is None:
                                continue
                            new_examples["input_ids"].append(sample["input_ids"])
                            new_examples["loss_mask"].append(sample["loss_mask"])
                        return new_examples

                    dataset = dataset.map(
                        preprocess_function,
                        batched=True,
                        num_proc=num_proc,
                        remove_columns=original_columns1,
                        load_from_cache_file=False
                    )
                    print(f"[scandata][rank {rank}] dataset.map done in {time.time() - t0:.1f}s")

                    # Avoid another large multiprocessing fanout here (can trigger OOM kill in multi-rank runs).
                    t1 = time.time()
                    token_dict = Counter()
                    for i in range(len(dataset)):
                        ids = dataset[i]["input_ids"][0]
                        mask = dataset[i]["loss_mask"][0]
                        for j in range(len(ids)):
                            if mask[j] == 1:
                                token_dict[ids[j]] += 1
                    print(f"[scandata][rank {rank}] token counting done in {time.time() - t1:.1f}s")

                    total_frequency = sum(token_dict.values())
                    top_N = token_dict.most_common(N)
                    top_N_frequency_sum = sum(freq for key, freq in top_N)
                    top_N_ratio = top_N_frequency_sum / total_frequency
                    print(f"top {N} token frequency ratio: {top_N_ratio:.2%}")
                    used_tokens = [key for key, freq in top_N]
                    used_tokens.sort()
                    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
                    t2d = [i in used_tokens for i in range(self.vocab_size)]
                    d2t = torch.tensor(d2t)
                    t2d = torch.tensor(t2d)
                    cache = {
                        "d2t": d2t,
                        "t2d": t2d
                    }
                    torch.save(cache, cache_path)
                    print(f"[scandata][rank {rank}] cache saved: {cache_path}")
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                if os.path.exists(lock_path):
                    os.remove(lock_path)

        if not os.path.exists(cache_path):
            raise RuntimeError(f"scandata cache missing after build/wait: {cache_path}")

        print(f"[scandata][rank {rank}] loading cache: {cache_path}")
        cache = torch.load(cache_path)
        d2t = cache["d2t"]
        t2d = cache["t2d"]
        t2d_index = self._build_t2d_index(t2d)
        draft_to_token = torch.arange(d2t.numel(), dtype=torch.long) + d2t.long()
        if tokenized_dataset is not None and rank == 0:
            self._estimate_in_draft_ratio(tokenized_dataset, t2d, rank)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        self.register_buffer("t2d_index", t2d_index)
        self.register_buffer("draft_to_token", draft_to_token)
        self.l1smooth = nn.SmoothL1Loss(reduction="none")

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        teacher_attention_mask = (
            attention_mask if (self.use_padding_attn_mask or self.force_teacher_attn_mask) else None
        )

        def _run_teacher(mask):
            outs_local = self.target_model(
                input_ids=input_ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states_all = outs_local.hidden_states
            hidden_idx = self._resolve_teacher_hidden_indices(len(hidden_states_all))
            hs_chunks = [hidden_states_all[i] for i in hidden_idx]
            for i, hs_i in zip(hidden_idx, hs_chunks):
                self._trace_nonfinite(f"teacher/hidden_raw_l{i}", hs_i)
            hs = torch.cat(hs_chunks, dim=-1)
            tg = outs_local.logits
            return hs, tg

        hidden_states, target = _run_teacher(teacher_attention_mask)

        # Auto-fallback: if causal-only teacher forward is numerically unstable, retry once
        # with explicit attention mask (limited to early steps for overhead control).
        if (
            teacher_attention_mask is None
            and attention_mask is not None
            and self._teacher_mask_fallback_steps > 0
        ):
            primary_bad = int((~torch.isfinite(hidden_states)).sum().item()) + int(
                (~torch.isfinite(target)).sum().item()
            )
            if primary_bad > 0:
                hidden_states_masked, target_masked = _run_teacher(attention_mask)
                masked_bad = int((~torch.isfinite(hidden_states_masked)).sum().item()) + int(
                    (~torch.isfinite(target_masked)).sum().item()
                )
                if masked_bad <= primary_bad:
                    hidden_states, target = hidden_states_masked, target_masked
                    if self._debug_teacher_numerics_steps > 0:
                        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                        if rank == 0:
                            print(
                                "[teacher-numerics] fallback to explicit attention_mask "
                                f"reduced non-finite values: {primary_bad} -> {masked_bad}"
                            )
                self._teacher_mask_fallback_steps -= 1

        if self._debug_teacher_numerics_steps > 0:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            if rank == 0:
                hidden_bad = int((~torch.isfinite(hidden_states)).sum().item())
                target_bad = int((~torch.isfinite(target)).sum().item())
                print(
                    "[teacher-numerics] "
                    f"hidden_bad={hidden_bad}, target_bad={target_bad}, "
                    f"use_attn_mask={teacher_attention_mask is not None}"
                )
            self._debug_teacher_numerics_steps -= 1

        self._trace_nonfinite("teacher/hidden_concat_raw", hidden_states)
        self._trace_nonfinite("teacher/logits_raw", target)

        # Keep distillation tensors numerically safe.
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=30.0, neginf=-30.0)
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    def forward(
            self,
            # hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            loss_mask: Optional[torch.Tensor] = None,

    ):
        self._nonfinite_found_this_step = 0
        self._nonfinite_stage_this_step = ""
        hidden_states, target, loss_mask, input_ids = self.dataprepare(input_ids, attention_mask, loss_mask)
        rollout_attention_mask = padding(attention_mask, left=False) if attention_mask is not None else None

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # with torch.no_grad():
        #     inputs_embeds = self.embed_tokens(input_ids)
        #     inputs_embeds = inputs_embeds.detach()

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        hidden_states=self.fc(hidden_states)
        self._trace_nonfinite("student/fc_out_raw", hidden_states)
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        if not self.use_padding_attn_mask:
            attention_mask = None
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        plosses = []
        vlosses = []
        acces = []
        cache_hidden = [[], []]

        for idx in range(self.length):
            last = idx == self.length - 1
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None, output_attentions)

                    return custom_forward

                layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),
                    inputs_embeds,
                    hidden_states,
                    cache_hidden,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:

                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=True,
                )

            hidden_states_out = layer_outputs[0]
            self._trace_nonfinite(f"student/layer{idx}_out_pre_norm_raw", hidden_states_out)
            # cache_hidden.append(layer_outputs[1])
            # kv_cahce = layer_outputs[-1]

            with torch.no_grad():
                # Keep buffers immutable in forward, and build local device copies.
                t2d_mask = self.t2d.to(target.device)
                t2d_index = self.t2d_index.to(target.device)
                label_token_ids = input_ids.long()
                label_in_draft = t2d_mask[label_token_ids]
                label_draft_ids = t2d_index[label_token_ids]

                # Distill teacher distribution on draft vocab and anchor with
                # supervised CE on gold draft-token ids.
                supervised_mask = loss_mask.float()
                in_draft_position_mask = label_in_draft[..., None].float() * supervised_mask
                position_mask = supervised_mask
                if self.distill_only_in_draft:
                    # Use in-draft positions for distillation when possible. Fallback
                    # to all supervised positions to avoid fully-zero gradient batches.
                    if float(in_draft_position_mask.sum().item()) > 0.0:
                        position_mask = in_draft_position_mask
                target_head = target[..., t2d_mask].float()
                self._trace_nonfinite(f"teacher/layer{idx}_target_head_raw", target_head)
                target_row_is_finite = torch.isfinite(target_head).all(dim=2, keepdim=True)
                target_head = torch.nan_to_num(target_head, nan=0.0, posinf=30.0, neginf=-30.0)
                target_p = F.softmax(target_head, dim=2)
                target_p = torch.nan_to_num(target_p, nan=0.0, posinf=0.0, neginf=0.0)
                target_p = target_p / target_p.sum(dim=2, keepdim=True).clamp_min(1e-12)
                target_p = target_p.detach()



            hidden_states = hidden_states_out

            hidden_states_out = torch.nan_to_num(
                hidden_states_out, nan=0.0, posinf=1e4, neginf=-1e4
            )
            hidden_states_out = self.norm(hidden_states_out)
            self._trace_nonfinite(f"student/layer{idx}_out_post_norm_raw", hidden_states_out)

            raw_logits = self.lm_head(hidden_states_out).float()
            self._trace_nonfinite(f"student/layer{idx}_raw_logits", raw_logits)
            logits_row_is_finite = torch.isfinite(raw_logits).all(dim=2)
            logits = torch.nan_to_num(raw_logits, nan=0.0, posinf=30.0, neginf=-30.0)
            out_logp = F.log_softmax(logits, dim=2)
            distill_per_token_loss = -(target_p * out_logp).sum(dim=2)
            distill_per_token_loss = torch.nan_to_num(distill_per_token_loss, nan=0.0, posinf=0.0, neginf=0.0)
            token_mask = position_mask.squeeze(-1).float()
            raw_valid_tokens = token_mask.sum()
            valid_tokens = raw_valid_tokens.clamp_min(1.0)
            distill_loss = (distill_per_token_loss * token_mask).sum() / valid_tokens

            # Gold anchor loss: only on supervised tokens whose gold ids are inside draft vocab.
            ignore_index = -100
            supervised_2d = supervised_mask.squeeze(-1) > 0.5
            gold_target = label_draft_ids.clone()
            invalid_gold = (~label_in_draft) | (~supervised_2d) | (label_draft_ids < 0)
            gold_target = gold_target.masked_fill(invalid_gold, ignore_index)
            gold_valid_mask = (gold_target != ignore_index).float()
            gold_valid_tokens = gold_valid_mask.sum()
            if self.loss_mode == "paper":
                gold_loss = distill_loss.new_zeros(())
                loss = distill_loss
            elif float(gold_valid_tokens.item()) > 0.0 and self.gold_ce_weight > 0.0:
                flat_logp = out_logp.reshape(-1, out_logp.size(-1))
                flat_target = gold_target.reshape(-1)
                gold_nll = F.nll_loss(
                    flat_logp,
                    flat_target,
                    reduction="none",
                    ignore_index=ignore_index,
                ).reshape_as(gold_target)
                gold_nll = torch.nan_to_num(gold_nll.float(), nan=0.0, posinf=0.0, neginf=0.0)
                gold_loss = (gold_nll * gold_valid_mask).sum() / gold_valid_tokens.clamp_min(1.0)
                loss = (1.0 - self.gold_ce_weight) * distill_loss + self.gold_ce_weight * gold_loss
            else:
                gold_loss = distill_loss.new_zeros(())
                loss = distill_loss
            plosses.append(loss)
            with torch.no_grad():
                pred_ids = logits.argmax(-1)
                acc_mask = token_mask * label_in_draft.float() * (label_draft_ids >= 0).float()
                acc_valid_tokens = acc_mask.sum().clamp_min(1.0)
                correct = ((pred_ids == label_draft_ids).float() * acc_mask).sum()
                acces.append((correct / acc_valid_tokens).item())

                if idx == 0 and self._debug_numerics_steps > 0:
                    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                    if rank == 0:
                        supervised_tokens = float(loss_mask.sum().item())
                        in_draft_tokens = float(in_draft_position_mask.sum().item())
                        distill_tokens = float(raw_valid_tokens.item())
                        gold_tokens = float(gold_valid_tokens.item())
                        finite_teacher_rows = float(target_row_is_finite.squeeze(-1).sum().item())
                        finite_student_rows = float(logits_row_is_finite.sum().item())
                        print(
                            "[numerics] "
                            f"supervised_tokens={supervised_tokens:.0f}, "
                            f"in_draft_tokens={in_draft_tokens:.0f}, "
                            f"distill_tokens={distill_tokens:.0f}, "
                            f"gold_tokens={gold_tokens:.0f}, "
                            f"finite_teacher_rows={finite_teacher_rows:.0f}, "
                            f"finite_student_rows={finite_student_rows:.0f}, "
                            f"valid_tokens={float(valid_tokens.item()):.0f}, "
                            f"distill_loss={float(distill_loss.item()):.6f}, "
                            f"gold_loss={float(gold_loss.item()):.6f}, "
                            f"batch_acc={acces[-1]:.6f}, batch_ploss={float(loss.item()):.6f}"
                        )
                        if supervised_tokens > 0 and float(raw_valid_tokens.item()) <= 0.0:
                            print(
                                "WARN: zero valid tokens after masking; "
                                "metrics/loss for this batch are degenerate."
                            )
                    self._debug_numerics_steps -= 1

            if not last:
                next_loss_mask = padding(loss_mask, left=False)
                next_input_ids = self._rollout_next_input_ids(
                    input_ids=input_ids,
                    pred_draft_ids=pred_ids,
                    next_loss_mask=next_loss_mask,
                    step_idx=idx,
                )
                next_rollout_attention_mask = (
                    padding(rollout_attention_mask, left=False)
                    if rollout_attention_mask is not None
                    else None
                )
                if self.input_rollout_align_target and self.input_rollout_mode != "teacher":
                    next_target = self._compute_teacher_distill_target(next_input_ids, next_rollout_attention_mask)
                else:
                    next_target = padding(target, left=False)
                input_ids = next_input_ids
                target = next_target
                loss_mask = next_loss_mask
                rollout_attention_mask = next_rollout_attention_mask



        return plosses, vlosses, acces




