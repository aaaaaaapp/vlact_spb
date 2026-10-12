"""Ulysses sequence-parallel monkey-patch for Qwen2.5-VL / Qwen3-VL text attention.

RoPE is applied on the local shard (absolute position_ids), then heads/seq
are exchanged via all-to-all — same convention as SpikeBrain SP patches.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from starVLA.model.modules.vlm.sp_ulysses import all_to_all_comm


def _find_language_model(vl):
    if hasattr(vl, "model") and hasattr(vl.model, "language_model"):
        return vl.model.language_model
    if hasattr(vl, "language_model"):
        return vl.language_model
    raise RuntimeError("Could not locate Qwen language_model")


def _is_qwen3_text_attn(attn: nn.Module) -> bool:
    name = type(attn).__name__
    return "Qwen3" in name


def _ulysses_a2a_qkv(attn, query_states, key_states, value_states, attention_mask):
    """query/key/value in [B, H, L, D]; return same layout after Ulysses a2a."""
    sp_group = getattr(attn, "sp_group", None)
    sp_size = int(getattr(attn, "sp_size", 1) or 1)
    if sp_group is None or sp_size <= 1:
        return query_states, key_states, value_states, attention_mask, query_states.size(2)

    num_heads = query_states.size(1)
    num_kv_heads = key_states.size(1)
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()
    if sp_size > num_kv_heads:
        rep = num_heads // num_kv_heads
        k = torch.repeat_interleave(k, rep, dim=2)
        v = torch.repeat_interleave(v, rep, dim=2)
    q = all_to_all_comm(q, process_group=sp_group, scatter_dim=2, gather_dim=1)
    k = all_to_all_comm(k, process_group=sp_group, scatter_dim=2, gather_dim=1)
    v = all_to_all_comm(v, process_group=sp_group, scatter_dim=2, gather_dim=1)
    query_states = q.transpose(1, 2).contiguous()
    key_states = k.transpose(1, 2).contiguous()
    value_states = v.transpose(1, 2).contiguous()
    q_len = query_states.size(2)
    # Full-sequence causal FA; drop 2D padding masks that assume local length.
    return query_states, key_states, value_states, None, q_len


def _ulysses_a2a_out(attn, attn_output, bsz, q_len, head_dim):
    """Normalize FA output to [B, L, H*D], optionally Ulysses-scatter seq / gather heads."""
    sp_group = getattr(attn, "sp_group", None)
    sp_size = int(getattr(attn, "sp_size", 1) or 1)

    # eager/FA typically return [B, L, H, D]; some paths may still be [B, H, L, D].
    if attn_output.dim() == 4:
        if attn_output.size(1) != q_len and attn_output.size(2) == q_len:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

    if sp_group is None or sp_size <= 1:
        return attn_output, q_len

    h_local = attn_output.shape[-1] // head_dim
    o = attn_output.view(bsz, q_len, h_local, head_dim)
    o = all_to_all_comm(o, process_group=sp_group, scatter_dim=1, gather_dim=2)
    q_len = o.size(1)
    return o.reshape(bsz, q_len, -1).contiguous(), q_len


def _patched_attn_forward_qwen25(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        eager_attention_forward,
    )

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    query_states, key_states, value_states, attention_mask, q_len = _ulysses_a2a_qkv(
        self, query_states, key_states, value_states, attention_mask
    )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output, q_len = _ulysses_a2a_out(self, attn_output, bsz, q_len, self.head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _patched_attn_forward_qwen3(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    bsz, q_len = input_shape
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    query_states, key_states, value_states, attention_mask, q_len = _ulysses_a2a_qkv(
        self, query_states, key_states, value_states, attention_mask
    )
    input_shape = (bsz, q_len)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output, q_len = _ulysses_a2a_out(self, attn_output, bsz, q_len, self.head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def install_qwen_ulysses_sp(vl_model: nn.Module) -> nn.Module:
    """Patch Qwen2.5-VL / Qwen3-VL text attentions + attach ``set_sp_group``."""
    lm = _find_language_model(vl_model)
    for layer in lm.layers:
        attn = layer.self_attn
        attn.sp_group = None
        attn.sp_size = 1

        def _set_sp(self, sp_group, sp_size):
            self.sp_group = sp_group
            self.sp_size = int(sp_size)

        attn.set_sp_group = _set_sp.__get__(attn, attn.__class__)
        if _is_qwen3_text_attn(attn):
            attn.forward = _patched_attn_forward_qwen3.__get__(attn, attn.__class__)
        else:
            attn.forward = _patched_attn_forward_qwen25.__get__(attn, attn.__class__)

    def _model_set_sp(self, sp_group, sp_size=1):
        for layer in _find_language_model(self).layers:
            if hasattr(layer.self_attn, "set_sp_group"):
                layer.self_attn.set_sp_group(sp_group, sp_size)

    vl_model.set_sp_group = _model_set_sp.__get__(vl_model, vl_model.__class__)
    return vl_model
