#
# Copyright 2016 The BigDL Authors.
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
#

# IPEX-LLM forward optimizations for the Qwen3.5 (dense) architecture.
#
# Qwen3.5 interleaves standard Gated-Attention layers with Gated DeltaNet
# (linear-attention) layers, switched per layer via the hybrid marker
# in the model config (e.g. ``config.layer_types`` or the llama.cpp-style
# ``full_attention_interval`` metadata exposed by HuggingFace transformers).
#
# For the full-attention layers we reuse the Qwen3 attention path: fused
# QKV Linear, partial IMRoPE (rotary applied only to the rotary-head slice),
# and the IPEX SDPA kernel.
#
# For the DeltaNet layers we intentionally keep the upstream HuggingFace
# reference forward in this first landing. Correctness first; a fused
# SYCL/IPEX kernel for the gated-delta recurrence is a follow-up.

import torch
from typing import Optional, List, Tuple

from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention


def _get_hf_qwen3_5_modules():
    """Locate the HuggingFace Qwen3.5 modeling module.

    Transformers used a transitional name ``qwen3_next`` during early
    development of the Qwen3.5 architecture before the final
    ``qwen3_5`` module landed. Support either so IPEX-LLM can be
    paired with a range of transformers versions.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
        return mod
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_next import modeling_qwen3_next as mod
        return mod
    except ImportError:
        pass
    return None


def _get_attention_class():
    mod = _get_hf_qwen3_5_modules()
    if mod is None:
        return None
    for name in ("Qwen3_5Attention", "Qwen3NextAttention"):
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls
    return None


def merge_qkv(module: torch.nn.Module):
    # Only fuse QKV on full-attention layers. DeltaNet blocks have no
    # q_proj/k_proj/v_proj; ``merge_qkv_base`` is a no-op for them thanks to
    # its ``isinstance`` guard.
    attn_cls = _get_attention_class()
    if attn_cls is not None:
        merge_qkv_base(module, attn_cls)


def qwen3_5_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> MoeModelOutputWithPast:
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    use_cache = True if device.type == "xpu" else use_cache
    # Only swap in IPEX-LLM's DynamicNormalCache when every layer is a
    # standard full-attention layer. Hybrid builds (with Gated DeltaNet
    # layers) rely on HuggingFace's own hybrid cache to carry the
    # per-layer conv / recurrent state, so we leave that pipeline alone.
    layer_types = getattr(self.config, "layer_types", None)
    is_pure_attention = (
        layer_types is None
        or all("attention" in str(t).lower() for t in layer_types)
    )
    if (
        is_pure_attention
        and use_cache
        and not isinstance(past_key_values, DynamicNormalCache)
    ):
        past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)

    mod = _get_hf_qwen3_5_modules()
    base_model_cls = None
    if mod is not None:
        for name in ("Qwen3_5Model", "Qwen3NextModel"):
            base_model_cls = getattr(mod, name, None)
            if base_model_cls is not None:
                break

    if base_model_cls is None:
        # Fall back to the bound HF forward if we cannot resolve the class.
        return self.__class__.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

    return base_model_cls.forward(
        self=self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **flash_attn_kwargs,
    )


def qwen3_5_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, -1, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split(
        [
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.num_key_value_heads,
        ],
        dim=1,
    )

    # Qwen3.5 keeps the per-head q_norm / k_norm from Qwen3.
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    # Partial IMRoPE: if the head dim of ``cos`` is smaller than the head
    # dim of ``query_states``, rotate only the leading rotary slice and
    # leave the tail untouched. Both the IPEX and HF paths below already
    # slice on the last dim, so the partial case falls out naturally when
    # the config builds ``cos`` / ``sin`` at the rotary dim.
    if device.type == "xpu":
        from ipex_llm.transformers.models.common import rotary_half_with_cache_inplaced

        rotary_half_with_cache_inplaced(query_states, key_states, cos, sin)
    else:
        # Route through whichever HF helper exists for this version.
        apply_rotary_pos_emb = None
        mod = _get_hf_qwen3_5_modules()
        if mod is not None:
            apply_rotary_pos_emb = getattr(mod, "apply_rotary_pos_emb", None)
        if apply_rotary_pos_emb is None:
            from transformers.models.qwen3.modeling_qwen3 import (
                apply_rotary_pos_emb as _fallback,
            )

            apply_rotary_pos_emb = _fallback
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len == key_states.size(2),
        self.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
