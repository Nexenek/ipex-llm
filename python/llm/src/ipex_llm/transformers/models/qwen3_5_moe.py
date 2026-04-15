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

# IPEX-LLM forward optimizations for the Qwen3.5 MoE architecture
# (e.g. Qwen3.5-35B-A3B, Qwen3.5-397B-A17B).
#
# Like the dense variant, hybrid layers interleave Gated DeltaNet blocks
# with standard Gated-Attention blocks. On top of that, the feed-forward
# sub-layer is a sparse Mixture-of-Experts (top-k with optional
# ``norm_topk_prob``) that we can accelerate with the fused ``xe_linear``
# MoE kernel when it is available.

import torch
from typing import Optional, List

from transformers.processing_utils import Unpack
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.utils import use_fuse_moe


def _get_hf_qwen3_5_moe_modules():
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
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
    mod = _get_hf_qwen3_5_moe_modules()
    if mod is None:
        return None
    for name in ("Qwen3_5MoeAttention", "Qwen3NextAttention"):
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls
    return None


def merge_qkv(module: torch.nn.Module):
    attn_cls = _get_attention_class()
    if attn_cls is not None:
        merge_qkv_base(module, attn_cls)


def qwen3_5_moe_model_forward(
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

    mod = _get_hf_qwen3_5_moe_modules()
    base_model_cls = None
    if mod is not None:
        for name in ("Qwen3_5MoeModel", "Qwen3NextModel"):
            base_model_cls = getattr(mod, name, None)
            if base_model_cls is not None:
                break

    if base_model_cls is None:
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


def qwen3_5_moe_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Fused MoE forward, mirroring the Qwen3-MoE optimization.

    Falls back to a per-expert loop when the fused kernel is unavailable
    (CPU, large batches, or quantization types not yet covered by
    ``xe_linear.moe_forward_vec``).
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    norm_topk_prob = getattr(self, "norm_topk_prob", True)

    if router_logits.device == "xpu":
        import xe_addons

        selected_experts, routing_weights = xe_addons.moe_softmax_topk(
            router_logits, self.top_k, norm_topk_prob
        )
    else:
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

    if selected_experts.size(0) == 1:
        if use_fuse_moe(hidden_states, self.experts[0].down_proj.qtype):
            if getattr(self, "gates", None) is None:
                gate_addrs = [
                    expert.gate_proj.weight.data_ptr() for expert in self.experts
                ]
                up_addrs = [
                    expert.up_proj.weight.data_ptr() for expert in self.experts
                ]
                down_addrs = [
                    expert.down_proj.weight.data_ptr() for expert in self.experts
                ]
                gates = torch.tensor(
                    gate_addrs, dtype=torch.uint64, device=hidden_states.device
                )
                ups = torch.tensor(
                    up_addrs, dtype=torch.uint64, device=hidden_states.device
                )
                downs = torch.tensor(
                    down_addrs, dtype=torch.uint64, device=hidden_states.device
                )
                self.register_buffer("gates", gates, persistent=False)
                self.register_buffer("ups", ups, persistent=False)
                self.register_buffer("downs", downs, persistent=False)

            import xe_linear

            final_hidden_states = xe_linear.moe_forward_vec(
                hidden_states,
                selected_experts,
                routing_weights,
                self.gates,
                self.ups,
                self.downs,
                hidden_states.size(-1),
                self.experts[0].intermediate_size,
                self.experts[0].down_proj.qtype,
            )
        else:
            idxs = selected_experts.flatten().tolist()
            outputs = []
            for i in idxs:
                expert = self.experts[i]
                outputs.append(expert(hidden_states))
            outs = torch.cat(outputs, dim=0)
            reshaped_topk_weight = routing_weights.squeeze(0).unsqueeze(-1)
            final_hidden_states = (outs * reshaped_topk_weight).sum(dim=0, keepdim=True)
    else:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states, router_logits
