import copy
import json
import time
import importlib

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        def _try_load_with_optional_prefix_strip(state_dict):
            target_keys = set(self.ea_layer.state_dict().keys())

            def _load(sd):
                res = self.ea_layer.load_state_dict(sd, strict=False)
                loaded = len(target_keys) - len(res.missing_keys)
                return res, loaded

            # First try raw keys.
            best_sd = state_dict
            best_res, best_loaded = _load(best_sd)

            # If too few keys are loaded, try common wrapper prefixes.
            # This handles checkpoints saved from wrapped engines (e.g. deepspeed/module).
            if best_loaded < max(8, int(0.2 * max(1, len(target_keys)))):
                candidates = (
                    "module.",
                    "model.",
                    "module.model.",
                    "ea_layer.",
                    "module.ea_layer.",
                    "model.ea_layer.",
                )
                for prefix in candidates:
                    remapped = {}
                    hit = 0
                    for k, v in state_dict.items():
                        nk = k[len(prefix):] if k.startswith(prefix) else k
                        remapped[nk] = v
                        if nk in target_keys:
                            hit += 1
                    if hit == 0:
                        continue
                    res, loaded = _load(remapped)
                    if loaded > best_loaded:
                        best_sd = remapped
                        best_res = res
                        best_loaded = loaded
            return best_res, best_loaded, len(target_keys)

        load_, loaded_keys, total_keys = _try_load_with_optional_prefix_strip(ea_layer_state_dict)

        raw_missing_keys = list(load_.missing_keys)
        raw_unexpected_keys = list(load_.unexpected_keys)
        missing_n = len(raw_missing_keys)
        unexpected_n = len(raw_unexpected_keys)
        loaded_ratio = float(loaded_keys / max(1, total_keys))

        # Some checkpoint/key mismatches are expected across training/inference variants:
        # - embed_tokens.weight can be omitted when frozen params are excluded from save.
        # - t2d_index is a train-side helper buffer introduced in newer checkpoints.
        benign_missing = {"embed_tokens.weight"}
        benign_unexpected = {"t2d_index"}
        effective_missing_keys = [k for k in raw_missing_keys if k not in benign_missing]
        effective_unexpected_keys = [k for k in raw_unexpected_keys if k not in benign_unexpected]
        effective_loaded = total_keys - len(effective_missing_keys)
        effective_loaded_ratio = float(effective_loaded / max(1, total_keys))
        if missing_n > 0 or unexpected_n > 0:
            print(
                "[ea-load] "
                f"loaded={loaded_keys}/{total_keys} ({loaded_ratio:.1%}), "
                f"missing={missing_n}, unexpected={unexpected_n}"
            )
            if missing_n > 0:
                print(f"[ea-load] missing(sample)={raw_missing_keys[:12]}")
            if unexpected_n > 0:
                print(f"[ea-load] unexpected(sample)={raw_unexpected_keys[:12]}")
            if effective_missing_keys or effective_unexpected_keys:
                print(
                    "[ea-load] "
                    f"effective_missing={len(effective_missing_keys)}, "
                    f"effective_unexpected={len(effective_unexpected_keys)}, "
                    f"effective_loaded={effective_loaded}/{total_keys} ({effective_loaded_ratio:.1%})"
                )
        else:
            print(f"[ea-load] loaded={loaded_keys}/{total_keys} (100.0%)")

        critical_candidates = (
            "fc.weight",
            "lm_head.weight",
            "midlayer.self_attn.q_proj.weight",
            "midlayer.self_attn.k_proj.weight",
            "midlayer.self_attn.v_proj.weight",
            "midlayer.mlp.down_proj.weight",
        )
        missing_set = set(effective_missing_keys)
        critical_missing = [k for k in critical_candidates if k in missing_set]
        if critical_missing:
            msg = f"critical ea_layer weights missing after load: {critical_missing}"
            if os.environ.get("EAGLE_STRICT_HEAD_LOAD", "0") == "1":
                raise RuntimeError(msg)
            print(f"[ea-load][warn] {msg}")

        # Optional debug stats for critical tensors to detect silent corruption
        # (e.g., all-zero/NaN weights that can cause repetitive token collapse).
        if os.environ.get("EAGLE_DEBUG_HEAD_STATS", "1") == "1":
            sd = self.ea_layer.state_dict()
            for name in critical_candidates:
                if name not in sd:
                    continue
                w = sd[name].detach()
                finite = torch.isfinite(w)
                finite_ratio = float(finite.float().mean().item())
                if finite.any():
                    wf = w[finite].float()
                    mean = float(wf.mean().item())
                    std = float(wf.std(unbiased=False).item())
                    absmax = float(wf.abs().max().item())
                else:
                    mean = float("nan")
                    std = float("nan")
                    absmax = float("nan")
                print(
                    "[ea-load] tensor "
                    f"name={name} shape={tuple(w.shape)} finite_ratio={finite_ratio:.6f} "
                    f"mean={mean:.6e} std={std:.6e} absmax={absmax:.6e}"
                )

        # Optional strict guard for production debugging.
        if (
            os.environ.get("EAGLE_STRICT_HEAD_LOAD", "0") == "1"
            and (
                effective_loaded_ratio < 0.95
                or len(effective_missing_keys) > 0
                or len(effective_unexpected_keys) > 0
            )
        ):
            raise RuntimeError(
                "EAGLE_STRICT_HEAD_LOAD=1 and ea_layer checkpoint load is incomplete: "
                f"effective_loaded={effective_loaded}/{total_keys}, "
                f"effective_missing={len(effective_missing_keys)}, "
                f"effective_unexpected={len(effective_unexpected_keys)}, "
                f"raw_missing={missing_n}, raw_unexpected={unexpected_n}"
            )

        # Quick health checks for draft vocab mapping. If these look degenerate,
        # generation often collapses to repeated punctuation/special tokens.
        strict_draft_map = os.environ.get("EAGLE_STRICT_DRAFT_MAP", "1") == "1"
        ea_draft_vocab = getattr(self.ea_layer.config, 'draft_vocab_size', self.config.vocab_size)
        if self.config.vocab_size != ea_draft_vocab:
            issues = []
            selected = None
            d2t_nonzero = None
            map_min = None
            map_max = None
            map_unique = None
            if not hasattr(self.ea_layer, "t2d"):
                issues.append("missing buffer t2d")
            if not hasattr(self.ea_layer, "d2t"):
                issues.append("missing buffer d2t")
            if hasattr(self.ea_layer, "t2d"):
                try:
                    selected = int(self.ea_layer.t2d.sum().item())
                    if selected <= 0:
                        issues.append("t2d has zero selected tokens")
                    elif selected < max(8, int(0.1 * int(self.ea_layer.config.draft_vocab_size))):
                        issues.append(
                            f"t2d selected tokens too small: {selected} "
                            f"(draft_vocab_size={int(self.ea_layer.config.draft_vocab_size)})"
                        )
                except Exception as e:
                    issues.append(f"t2d check failed: {e}")
            if hasattr(self.ea_layer, "d2t"):
                try:
                    d2t_nonzero = int((self.ea_layer.d2t != 0).sum().item())
                    if d2t_nonzero <= 0:
                        issues.append("d2t appears all-zero")
                    d2t = self.ea_layer.d2t.long()
                    idx = torch.arange(d2t.numel(), device=d2t.device, dtype=torch.long)
                    mapped = idx + d2t
                    map_min = int(mapped.min().item())
                    map_max = int(mapped.max().item())
                    map_unique = int(torch.unique(mapped).numel())
                    if map_min < 0 or map_max >= int(self.config.vocab_size):
                        issues.append(
                            f"d2t mapped ids out of range: min={map_min}, max={map_max}, "
                            f"vocab={int(self.config.vocab_size)}"
                        )
                    if map_unique != int(self.ea_layer.config.draft_vocab_size):
                        issues.append(
                            f"d2t mapped ids not unique: unique={map_unique}, "
                            f"draft_vocab={int(self.ea_layer.config.draft_vocab_size)}"
                        )
                    if hasattr(self.ea_layer, "t2d"):
                        in_mask = self.ea_layer.t2d[mapped].all().item()
                        if not bool(in_mask):
                            issues.append("d2t mapped ids are not all marked by t2d")
                except Exception as e:
                    issues.append(f"d2t check failed: {e}")

            print(
                "[ea-load] draft-map "
                f"strict={int(strict_draft_map)} selected_t2d={selected} d2t_nonzero={d2t_nonzero} "
                f"map_min={map_min} map_max={map_max} map_unique={map_unique} "
                f"draft_vocab={int(self.ea_layer.config.draft_vocab_size)} vocab={int(self.config.vocab_size)}"
            )

            if issues:
                msg = (
                    "invalid/degenerate draft vocab mapping after checkpoint load: "
                    + "; ".join(issues)
                    + ". This usually means ea head checkpoint did not load mapping buffers "
                      "(e.g. key prefix mismatch or missing buffers in saved state)."
                )
                if strict_draft_map:
                    raise RuntimeError(msg)
                print(f"[ea-load][warn] {msg}")
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            # Tree config: EAGLE-2 paper Appendix A recommends total_token=50,
            # depth=6, top_k=10 for 13B class models (also reasonable for 14B).
            # Pass total_token=-1 to enable runtime auto-tune (see lines below).
            total_token=50,
            depth=6,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            # Keep this import lazy so environments that run Qwen2 do not
            # fail at module import time due to optional/new Transformers APIs
            # used by the Qwen3 implementation.
            qwen3_mod = importlib.import_module(".modeling_qwen3_kv", package=__package__)
            if not hasattr(qwen3_mod, "Qwen3ForCausalLM"):
                available = [n for n in dir(qwen3_mod) if "Qwen3" in n][:20]
                raise ImportError(
                    "Qwen3ForCausalLM not found in modeling_qwen3_kv. "
                    f"module={getattr(qwen3_mod, '__file__', 'unknown')}, "
                    f"available_qwen3_symbols={available}"
                )
            KVQwen3ForCausalLM = getattr(qwen3_mod, "Qwen3ForCausalLM")
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=bool(self.use_eagle3),
                return_dict=True,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            return_stats=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        accepted_tokens = 0
        proposed_tokens = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        idx = -1
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            # Count UNIQUE proposed tokens in the tree (before padding is appended).
            # Earlier we counted entries across retrieve_indices paths, which
            # double-counts tokens shared by multiple branches and grossly
            # inflates the denominator of acceptance_rate. The tree actually
            # costs base-model one forward over draft_tokens.shape[1]-1 unique
            # candidates (excluding the alignment anchor at position 0).
            unique_tree_proposed = int(draft_tokens.shape[1] - 1)
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            proposed_tokens += unique_tree_proposed
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accepted_tokens += int(accept_length + 1)
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            if return_stats:
                tree_steps = int(idx + 1) if idx >= 0 else 0
                stats = {
                    "accepted_tokens": int(accepted_tokens),
                    "proposed_tokens": int(proposed_tokens),
                    "acceptance_rate": float(accepted_tokens / proposed_tokens) if proposed_tokens > 0 else 0.0,
                    "mean_accepted_length": float(accepted_tokens / tree_steps) if tree_steps > 0 else 0.0,
                    "tree_steps": tree_steps,
                }
                return input_ids, new_token, idx, stats
            return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
