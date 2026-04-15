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

# GGUF loader for Qwen3.5 (dense) and Qwen3.5 MoE.
#
# GGUF files produced by llama.cpp (post PR #19468) carry
# ``general.architecture == "qwen35"`` for dense builds and
# ``general.architecture == "qwen35moe"`` for MoE builds.
# Metadata keys are prefixed with ``qwen3_5`` (with an underscore).
# Tensor names follow llama.cpp's ``blk.N.<name>`` scheme and include
# both standard-attention tensors and Gated DeltaNet (SSM-style)
# tensors for hybrid layers.
#
# The first-landing strategy here is intentionally conservative:
#   * Synthesize a HuggingFace ``Qwen3_5Config`` / ``Qwen3_5MoeConfig``
#     from the GGUF metadata and ``init_empty_weights`` a matching
#     ``AutoModelForCausalLM``.
#   * Stream GGUF tensors into the empty module using a forgiving
#     name-remap that handles attention, MLP, DeltaNet, and MoE expert
#     tensors. Unknown names are logged and skipped rather than raising
#     — this keeps the loader working across minor metadata drift.
#   * Quantize every Linear as it lands via
#     ``replace_with_low_bit_linear_for_module``.
#   * Reconstruct the tokenizer via ``AutoTokenizer`` from a sibling
#     directory alongside the ``.gguf`` file (the Unsloth-style layout),
#     falling back to a GPT-2 BPE rebuild from GGUF metadata.

import os
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from ..gguf import GGUFFileLoader
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.transformers.convert import replace_with_low_bit_linear_for_module
from ipex_llm.utils.common import invalidInputError


_ARCH_PREFIXES = ("qwen3_5", "qwen35", "qwen3.5", "qwen3_next", "qwen3next")


def _cfg(config: dict, key: str, default=None):
    """Look up a metadata key across known Qwen3.5 arch prefixes."""
    for prefix in _ARCH_PREFIXES:
        full = f"{prefix}.{key}"
        if full in config:
            return config[full]
    return default


def _hf_qwen3_5_classes(is_moe: bool):
    """Resolve (Config, CausalLM) classes from HF transformers, across
    the transitional ``qwen3_next`` name and the final ``qwen3_5`` one.
    """
    candidates = (
        ("qwen3_5_moe", "Qwen3_5MoeConfig", "Qwen3_5MoeForCausalLM") if is_moe
        else ("qwen3_5", "Qwen3_5Config", "Qwen3_5ForCausalLM"),
        ("qwen3_next", "Qwen3NextConfig", "Qwen3NextForCausalLM"),
    )
    from importlib import import_module

    for module_name, config_name, model_name in candidates:
        try:
            mod = import_module(f"transformers.models.{module_name}")
        except ImportError:
            continue
        cfg_cls = getattr(mod, config_name, None)
        model_cls = getattr(mod, model_name, None)
        if cfg_cls is not None and model_cls is not None:
            return cfg_cls, model_cls
    invalidInputError(
        False,
        "Your installed `transformers` version does not provide the "
        "Qwen3.5 / Qwen3-Next modelling classes. Please upgrade "
        "`transformers` to a release that supports Qwen3.5.",
    )


def _build_config(loader: GGUFFileLoader, is_moe: bool):
    config = loader.config
    cfg_cls, _ = _hf_qwen3_5_classes(is_moe)

    hf_kwargs = dict(
        vocab_size=len(config["tokenizer.ggml.tokens"]),
        hidden_size=_cfg(config, "embedding_length"),
        intermediate_size=_cfg(config, "feed_forward_length"),
        num_hidden_layers=_cfg(config, "block_count"),
        num_attention_heads=_cfg(config, "attention.head_count"),
        num_key_value_heads=_cfg(config, "attention.head_count_kv"),
        max_position_embeddings=_cfg(config, "context_length", 32768),
        rms_norm_eps=_cfg(config, "attention.layer_norm_rms_epsilon", 1e-6),
        rope_theta=_cfg(config, "rope.freq_base", 1_000_000.0),
        use_cache=True,
        tie_word_embeddings=_cfg(config, "tie_word_embeddings", False),
    )
    head_dim = _cfg(config, "attention.key_length")
    if head_dim is not None:
        hf_kwargs["head_dim"] = head_dim
    bos = config.get("tokenizer.ggml.bos_token_id")
    eos = config.get("tokenizer.ggml.eos_token_id")
    if bos is not None:
        hf_kwargs["bos_token_id"] = bos
    if eos is not None:
        hf_kwargs["eos_token_id"] = eos

    if is_moe:
        hf_kwargs.update(
            num_experts=_cfg(config, "expert_count"),
            num_experts_per_tok=_cfg(config, "expert_used_count"),
            moe_intermediate_size=_cfg(config, "expert_feed_forward_length"),
        )

    # Drop any keys whose value we couldn't read so that HF uses its
    # config defaults rather than blowing up on ``None``.
    hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}

    # Carry the hybrid-layer pattern through if present.
    full_attn_interval = _cfg(config, "attention.full_attention_interval")
    if full_attn_interval is not None:
        hf_kwargs["full_attention_interval"] = full_attn_interval

    return cfg_cls(**hf_kwargs)


def _remap_tensor_name(name: str):
    """Map a GGUF tensor name to a HuggingFace parameter path.

    Returns ``None`` for tensors we don't recognize so the caller can
    skip them with a warning instead of failing the load.
    """
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"

    parts = name.split(".")
    # Expected layout: blk.<layer_id>.<component>...
    if len(parts) < 3 or parts[0] != "blk":
        return None
    try:
        layer_id = int(parts[1])
    except ValueError:
        return None
    tail = ".".join(parts[2:])

    base = f"model.layers.{layer_id}"

    # Standard full-attention tensors
    attn_map = {
        "attn_norm.weight": f"{base}.input_layernorm.weight",
        "attn_q.weight": f"{base}.self_attn.q_proj.weight",
        "attn_q.bias": f"{base}.self_attn.q_proj.bias",
        "attn_k.weight": f"{base}.self_attn.k_proj.weight",
        "attn_k.bias": f"{base}.self_attn.k_proj.bias",
        "attn_v.weight": f"{base}.self_attn.v_proj.weight",
        "attn_v.bias": f"{base}.self_attn.v_proj.bias",
        "attn_output.weight": f"{base}.self_attn.o_proj.weight",
        "attn_q_norm.weight": f"{base}.self_attn.q_norm.weight",
        "attn_k_norm.weight": f"{base}.self_attn.k_norm.weight",
        "ffn_norm.weight": f"{base}.post_attention_layernorm.weight",
    }
    if tail in attn_map:
        return attn_map[tail]

    # Dense MLP
    mlp_map = {
        "ffn_gate.weight": f"{base}.mlp.gate_proj.weight",
        "ffn_up.weight": f"{base}.mlp.up_proj.weight",
        "ffn_down.weight": f"{base}.mlp.down_proj.weight",
    }
    if tail in mlp_map:
        return mlp_map[tail]

    # MoE expert tensors
    moe_map = {
        "ffn_gate_inp.weight": f"{base}.mlp.gate.weight",
        "ffn_gate_exps.weight": f"{base}.mlp.experts.gate_proj.weight",
        "ffn_up_exps.weight": f"{base}.mlp.experts.up_proj.weight",
        "ffn_down_exps.weight": f"{base}.mlp.experts.down_proj.weight",
    }
    if tail in moe_map:
        return moe_map[tail]

    # Gated DeltaNet / SSM-style tensors for hybrid layers.
    # Names track llama.cpp PR #19468 conventions; HF parameter paths
    # follow upstream ``modeling_qwen3_5.py`` / ``modeling_qwen3_next.py``.
    ssm_map = {
        "ssm_norm.weight": f"{base}.linear_attn.norm.weight",
        "ssm_in_proj.weight": f"{base}.linear_attn.in_proj.weight",
        "ssm_out_proj.weight": f"{base}.linear_attn.out_proj.weight",
        "ssm_conv1d.weight": f"{base}.linear_attn.conv1d.weight",
        "ssm_conv1d.bias": f"{base}.linear_attn.conv1d.bias",
        "ssm_dt.weight": f"{base}.linear_attn.dt_proj.weight",
        "ssm_dt.bias": f"{base}.linear_attn.dt_bias",
        "ssm_a_log": f"{base}.linear_attn.A_log",
        "ssm_a": f"{base}.linear_attn.A_log",
        "ssm_b": f"{base}.linear_attn.b",
        "ssm_z": f"{base}.linear_attn.z",
        "ssm_beta": f"{base}.linear_attn.beta",
    }
    if tail in ssm_map:
        return ssm_map[tail]

    return None


def _build_tokenizer(fpath: str, loader: GGUFFileLoader):
    """Best-effort tokenizer reconstruction.

    Priority order:
      1. If the directory containing the ``.gguf`` carries
         ``tokenizer.json`` / ``tokenizer_config.json`` (the layout
         Unsloth and the official Qwen GGUF releases ship), load via
         ``AutoTokenizer.from_pretrained``.
      2. Otherwise rebuild a GPT-2 style BPE tokenizer from the GGUF
         ``tokenizer.ggml.tokens`` + ``tokenizer.ggml.merges`` metadata.
    """
    from transformers import AutoTokenizer

    model_dir = os.path.dirname(os.path.abspath(fpath))
    candidates = ("tokenizer.json", "tokenizer_config.json")
    if any(os.path.isfile(os.path.join(model_dir, c)) for c in candidates):
        try:
            return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            pass

    # Fallback: rebuild from GGUF metadata via the fast-tokenizers library.
    from tempfile import TemporaryDirectory
    import json

    config = loader.config
    tokens = config.get("tokenizer.ggml.tokens")
    merges = config.get("tokenizer.ggml.merges")
    if tokens is None or merges is None:
        invalidInputError(
            False,
            "GGUF file does not carry a usable tokenizer and no "
            "tokenizer.json was found alongside it. Please place the "
            "matching HuggingFace tokenizer files in the same directory "
            "as the .gguf file.",
        )

    vocab = {tok: i for i, tok in enumerate(tokens)}
    # ``merges`` is stored as "<a> <b>" strings.
    merge_list = [tuple(m.split(" ", 1)) for m in merges if " " in m]

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
        from tokenizers.decoders import ByteLevel as ByteLevelDec
    except ImportError:
        invalidInputError(
            False,
            "Tokenizer reconstruction requires the `tokenizers` "
            "package. Install it, or place a tokenizer.json next to "
            "the .gguf file.",
        )

    bpe = BPE(vocab=vocab, merges=merge_list, fuse_unk=False)
    fast = Tokenizer(bpe)
    fast.pre_tokenizer = ByteLevelPre(add_prefix_space=False, use_regex=True)
    fast.decoder = ByteLevelDec()

    with TemporaryDirectory() as tmp:
        tok_json = os.path.join(tmp, "tokenizer.json")
        fast.save(tok_json)
        # Minimal tokenizer_config.json so AutoTokenizer picks Qwen2-style.
        with open(os.path.join(tmp, "tokenizer_config.json"), "w") as f:
            json.dump(
                {"tokenizer_class": "Qwen2TokenizerFast"},
                f,
            )
        return AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)


def _load_gguf_generic(loader: GGUFFileLoader, dtype: torch.dtype,
                      low_bit: str, is_moe: bool, fpath: str = ""):
    hf_config = _build_config(loader, is_moe=is_moe)
    _, model_cls = _hf_qwen3_5_classes(is_moe)

    qtype = ggml_tensor_qtype[low_bit]

    with init_empty_weights():
        model = model_cls(hf_config)

    skipped = []

    def process(name, tensor):
        nonlocal model
        module_name = _remap_tensor_name(name)
        if module_name is None:
            skipped.append(name)
            return
        try:
            set_module_tensor_to_device(
                model, module_name, "cpu", tensor, dtype=dtype
            )
        except (KeyError, AttributeError, ValueError):
            # Parameter path did not resolve (e.g. a DeltaNet tensor
            # whose HF attribute name drifted between transformers
            # versions). Record and continue — we prefer a partial load
            # to a hard crash because the HF reference forward will
            # warn loudly on any actually-used missing weight.
            skipped.append(f"{name} -> {module_name}")
            return
        # Only Linear-backed params get low-bit-converted; the call is a
        # no-op for norm / bias / embedding weights.
        model = replace_with_low_bit_linear_for_module(
            model, qtype=qtype, module_name=module_name
        )

    loader.tensor_loader.load_while_process(process)

    if skipped:
        import warnings

        preview = skipped[:8]
        more = "" if len(skipped) <= 8 else f" (+{len(skipped) - 8} more)"
        warnings.warn(
            "ipex-llm Qwen3.5 GGUF loader skipped unrecognized tensors: "
            f"{preview}{more}. This is expected for metadata that the HF "
            "reference forward does not consume; if inference is broken, "
            "file an issue with these names attached."
        )

    tokenizer = _build_tokenizer(fpath or getattr(loader.tensor_loader, "fpath", ""),
                                 loader)
    return model, tokenizer


def load_gguf_qwen3_5(loader: GGUFFileLoader,
                     dtype: torch.dtype = torch.float,
                     low_bit: str = "sym_int4"):
    return _load_gguf_generic(
        loader, dtype, low_bit, is_moe=False,
        fpath=getattr(loader.tensor_loader, "fpath", ""),
    )


def load_gguf_qwen3_5_moe(loader: GGUFFileLoader,
                         dtype: torch.dtype = torch.float,
                         low_bit: str = "sym_int4"):
    return _load_gguf_generic(
        loader, dtype, low_bit, is_moe=True,
        fpath=getattr(loader.tensor_loader, "fpath", ""),
    )
