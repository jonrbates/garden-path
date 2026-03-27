from __future__ import annotations

import copy
import inspect
import math
import random
import types
import numpy as np
import torch
from typing import Any, Optional
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Model:
    def __init__(self, model_id: str, quantize: bool = True, dtype: str = "fp16"):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )

        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

        if quantize:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

        self._ablated_idx: Optional[int] = None
        self.model.config.use_cache = False

        print("device_map: ", getattr(self.model, "hf_device_map", "N/A"))

    def first_token_id(self, txt: str, prepend_space: bool = True) -> int:
        """Return id of the first token encoding 'txt', inserting a leading space if needed."""
        if prepend_space:
            txt = " " + txt.lstrip()
        return self.tokenizer(txt, add_special_tokens=False).input_ids[0]

    @staticmethod
    def _ablated_forward(
        self_layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """Monkey-patched forward with two ablation modes per block:

        - ablate_*_resid: drop the residual, keep only block output
        - skip_*: drop the block output, keep only residual (skip the block)
        """
        residual = hidden_states

        # Self Attention
        if getattr(self_layer, "_skip_attn", False):
            hidden_states = residual
            attn_cache = None
        else:
            hidden_states = self_layer.input_layernorm(hidden_states)
            # Build attn kwargs dynamically — different architectures
            # (Llama, Qwen, etc.) accept different argument sets.
            attn_kwargs = dict(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            # Only pass args the attention module actually accepts
            attn_params = inspect.signature(self_layer.self_attn.forward).parameters
            if "past_key_values" in attn_params:
                attn_kwargs["past_key_values"] = past_key_values
            if "use_cache" in attn_params:
                attn_kwargs["use_cache"] = use_cache
            if "cache_position" in attn_params:
                attn_kwargs["cache_position"] = cache_position
            if "position_embeddings" in attn_params:
                attn_kwargs["position_embeddings"] = position_embeddings
            attn_kwargs.update(kwargs)
            attn_out = self_layer.self_attn(**attn_kwargs)
            hidden_states = attn_out[0]
            attn_cache = attn_out[1] if len(attn_out) > 1 else None
            if not getattr(self_layer, "_ablate_attn_resid", False):
                hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        if getattr(self_layer, "_skip_mlp", False):
            hidden_states = residual
        else:
            hidden_states = self_layer.post_attention_layernorm(hidden_states)
            hidden_states = self_layer.mlp(hidden_states)
            if not getattr(self_layer, "_ablate_mlp_resid", False):
                hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (attn_cache,)
        return outputs

    _ABLATION_ATTRS = (
        "_ablate_attn_resid",
        "_ablate_mlp_resid",
        "_skip_attn",
        "_skip_mlp",
    )

    def _make_ablated_copy(
        self,
        layer,
        *,
        ablate_attn=False,
        ablate_mlp=False,
        skip_attn=False,
        skip_mlp=False,
    ):
        """Deepcopy the (possibly quantized) layer and patch its forward."""
        new_layer = copy.deepcopy(layer)
        new_layer._ablate_attn_resid = bool(ablate_attn)
        new_layer._ablate_mlp_resid = bool(ablate_mlp)
        new_layer._skip_attn = bool(skip_attn)
        new_layer._skip_mlp = bool(skip_mlp)
        new_layer._orig_forward = new_layer.forward
        new_layer.forward = types.MethodType(self._ablated_forward, new_layer)
        return new_layer

    def _restore_layer(self, idx: int):
        layer = self.model.model.layers[idx]
        if hasattr(layer, "_orig_forward"):
            layer.forward = layer._orig_forward
            del layer._orig_forward
        for attr in self._ABLATION_ATTRS:
            if hasattr(layer, attr):
                delattr(layer, attr)

    def set_ablation(
        self,
        layer_idx: Optional[int],
        *,
        ablate_attn=False,
        ablate_mlp=False,
        skip_attn=False,
        skip_mlp=False,
    ):
        """Idempotently sets ablation.

        Modes (per block):
        - ablate_attn/ablate_mlp: drop the residual, keep only block output
        - skip_attn/skip_mlp: drop the block output, keep only residual
        """
        any_active = ablate_attn or ablate_mlp or skip_attn or skip_mlp
        if not any_active:
            if self._ablated_idx is not None:
                self._restore_layer(self._ablated_idx)
                self._ablated_idx = None
            return

        if layer_idx is None:
            raise ValueError("layer_idx must be provided when enabling ablation.")

        if self._ablated_idx is not None and self._ablated_idx != layer_idx:
            self._restore_layer(self._ablated_idx)
            self._ablated_idx = None

        base = self.model.model.layers[layer_idx]
        patched = self._make_ablated_copy(
            base,
            ablate_attn=ablate_attn,
            ablate_mlp=ablate_mlp,
            skip_attn=skip_attn,
            skip_mlp=skip_mlp,
        )
        self.model.model.layers[layer_idx] = patched
        self._ablated_idx = layer_idx

    # ------------------------------------------------------------------ #
    #  Activation patching (causal tracing)                                #
    # ------------------------------------------------------------------ #

    def _build_input_ids(self, sentence: str) -> torch.Tensor:
        """Tokenize with BOS, return (1, seq_len) tensor on model device."""
        ids = self.tokenizer(sentence, add_special_tokens=False).input_ids
        bos = self.tokenizer.bos_token_id
        prefix = [bos] if bos is not None else []
        return torch.tensor([prefix + ids], device=self.model.device)

    def _get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from input_ids."""
        return self.model.model.embed_tokens(input_ids)

    def cache_layer_outputs(
        self, input_ids: torch.Tensor, noise_std: float = 0.0
    ) -> dict[int, torch.Tensor]:
        """Forward pass, optionally noising embeddings, caching each layer's output.

        Returns {layer_idx: hidden_states} where hidden_states is (1, seq, dim).
        """
        embeds = self._get_embeds(input_ids)
        if noise_std > 0:
            embeds = embeds + torch.randn_like(embeds) * noise_std

        cache = {}
        hooks = []
        for idx in range(len(self.model.model.layers)):
            def _make_hook(i):
                def hook(module, inp, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    cache[i] = hs.detach().clone()
                    return output
                return hook
            h = self.model.model.layers[idx].register_forward_hook(_make_hook(idx))
            hooks.append(h)

        with torch.inference_mode():
            self.model(inputs_embeds=embeds)

        for h in hooks:
            h.remove()
        return cache

    def forward_patched(
        self,
        input_ids: torch.Tensor,
        noise_std: float,
        clean_cache: dict[int, torch.Tensor],
        restore_layers: set[int],
        restore_position: int | None = None,
    ) -> torch.Tensor:
        """Forward with noised embeddings, restoring clean activations at specified layers.

        If restore_position is given, only that token position is restored (Meng et al.
        style). Otherwise the entire hidden state is replaced (less informative).

        Returns logits (seq_len, vocab).
        """
        embeds = self._get_embeds(input_ids)
        if noise_std > 0:
            embeds = embeds + torch.randn_like(embeds) * noise_std

        hooks = []
        for idx in restore_layers:
            replacement = clean_cache[idx]
            def _make_hook(repl, pos):
                def hook(module, inp, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    if pos is not None:
                        # Restore only the critical token position
                        hs = hs.clone()
                        hs[:, pos, :] = repl[:, pos, :]
                    else:
                        hs = repl
                    if isinstance(output, tuple):
                        return (hs,) + output[1:]
                    return hs
                return hook
            h = self.model.model.layers[idx].register_forward_hook(
                _make_hook(replacement, restore_position)
            )
            hooks.append(h)

        with torch.inference_mode():
            logits = self.model(inputs_embeds=embeds).logits[0]

        for h in hooks:
            h.remove()
        return logits

    def surprisal_at(
        self, logits: torch.Tensor, input_ids: torch.Tensor, position: int
    ) -> float:
        """Compute surprisal of the token at `position` given logits at position-1."""
        probs = softmax(logits[position - 1], dim=-1).to(torch.float32)
        p = probs[input_ids[0, position]].clamp(min=1e-10).item()
        return -math.log2(p)

    def topk_at(
        self, logits: torch.Tensor, k: int = 5
    ) -> list[tuple[str, float]]:
        """Top-k tokens from logits at the last position."""
        probs = softmax(logits[-1], dim=-1).to(torch.float32)
        top_p, top_i = probs.topk(k)
        toks = self.tokenizer.convert_ids_to_tokens(top_i.tolist())
        return list(zip(toks, top_p.tolist()))

    def compute_sentence_metrics(
        self, sentence: str, k: int = 3, add_first: bool = True
    ):
        """Compute sentence metrics.

        Next token probability, distribution entropy, continuation probability,
        surprisal,

        If add_first=True, the first record (step 0) is the model's
        prediction for the *first* word in the sentence.
        """
        # ------------------------------------------------------------- #
        #  build input:  <s>  + sentence tokens  (no EOS)               #
        # ------------------------------------------------------------- #
        ids_no_bos = self.tokenizer(sentence, add_special_tokens=False).input_ids
        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None:
            input_ids = torch.tensor(
                [[bos_id] + ids_no_bos], device=self.model.device
            )
        else:
            input_ids = torch.tensor(
                [ids_no_bos], device=self.model.device
            )

        with torch.inference_mode():
            logits = self.model(input_ids).logits[0]  # (seq_len+1, vocab)
        probs_all = softmax(logits, dim=-1).to(torch.float32)
        seq_len = input_ids.size(1)

        # clip probabilities
        probs_all = probs_all.clamp(min=1e-10)

        # token-level next-token probabilities
        bos_offset = 1 if bos_id is not None else 0
        token_probs = [1.0] * bos_offset  # dummy for BOS if present
        for i in range(bos_offset, seq_len):
            p = probs_all[i - 1, input_ids[0, i]].item()
            token_probs.append(p)

        # continuation log-probs
        cont_logp = [0.0] * seq_len
        running = 0.0
        for i in range(seq_len - 1, bos_offset - 1 if bos_offset else 0, -1):
            running += math.log(token_probs[i])
            cont_logp[i - 1] = running

        # assemble records
        recs = []
        start_j = 0 if add_first else bos_offset
        for j in range(start_j, seq_len - 1):  # skip final EOS-pos
            probs = probs_all[j]
            entropy = -(probs * probs.log2()).sum().item()

            true_id = input_ids[0, j + 1].item()
            true_tok = self.tokenizer.convert_ids_to_tokens([true_id])[0]
            p_true = probs[true_id].item()
            rank = (probs > p_true).sum().item() + 1

            top_p, top_i = probs.topk(k)
            top_toks = self.tokenizer.convert_ids_to_tokens(top_i.tolist())
            topk = list(zip(top_toks, top_p.tolist()))

            recs.append(
                dict(
                    step=j,  # 0-based when add_first=True
                    prefix=self.tokenizer.decode(
                        input_ids[0, bos_offset : j + 1]
                    ),  # skip the BOS in decode
                    next_token=true_tok,
                    prob=p_true,
                    rank=rank,
                    entropy=entropy,
                    surprisal=-math.log2(p_true),
                    topk=topk,
                    cont_prob=math.exp(cont_logp[j]),
                )
            )
        return recs
