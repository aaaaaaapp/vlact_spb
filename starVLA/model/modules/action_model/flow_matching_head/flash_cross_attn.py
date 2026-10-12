"""Sana/PixArt-style DiT cross-attention via flash-attn.

Sana's ``MultiHeadCrossAttention`` (``cross_attn_type=flash``) uses xformers
``memory_efficient_attention`` with BlockDiagonalMask when available, else SDPA.
This env has **no xformers** but already ships **flash-attn 2.6.3** (same as
SpikeBrain-2.0-VL). We mirror Sana's Q(query tokens)/KV(condition) layout and
dispatch to ``flash_attn_func`` / ``flash_attn_varlen_func``.

Self-attention (interleaved DiT layers with ``encoder_hidden_states=None``)
stays on SDPA — do not switch those to linear-attn.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input

    _FLASH_ATTN_OK = True
    _FLASH_ATTN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    flash_attn_func = None  # type: ignore
    flash_attn_varlen_func = None  # type: ignore
    unpad_input = None  # type: ignore
    _FLASH_ATTN_OK = False
    _FLASH_ATTN_IMPORT_ERROR = exc


def flash_attn_available() -> bool:
    return bool(_FLASH_ATTN_OK)


def _as_bool_key_mask(
    attention_mask: torch.Tensor,
    batch_size: int,
    key_len: int,
) -> Optional[torch.Tensor]:
    """Normalize encoder mask to ``[B, key_len]`` bool (True = keep)."""
    if attention_mask is None:
        return None
    m = attention_mask
    if m.dtype != torch.bool:
        # 0/1 keep mask, or additive mask (0 keep, <0 drop)
        if torch.any(m < 0):
            m = m >= 0
        else:
            m = m != 0
    while m.ndim > 2:
        # [B, 1, 1, S] / [B, H, Q, S] → take key dim
        m = m[..., 0, :] if m.shape[-2] == 1 else m[:, 0, 0, :]
        if m.ndim > 2:
            m = m.reshape(batch_size, -1)[:, :key_len]
            break
    if m.ndim == 1:
        m = m.unsqueeze(0).expand(batch_size, -1)
    if m.shape[0] != batch_size or m.shape[-1] != key_len:
        # Best-effort crop / broadcast; caller may fall back to SDPA.
        if m.shape[-1] != key_len or m.shape[0] != batch_size:
            return None
    return m.to(dtype=torch.bool)


def _flash_cross_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_mask: Optional[torch.Tensor],
    dropout_p: float,
) -> torch.Tensor:
    """q/k/v: ``[B, L, H, D]`` (flash-attn layout). Returns same layout."""
    assert flash_attn_func is not None and flash_attn_varlen_func is not None

    # flash-attn requires fp16/bf16
    in_dtype = query.dtype
    if query.dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

    B, T, H, D = query.shape
    S = key.shape[1]

    use_varlen = False
    if key_mask is not None:
        # Skip varlen if mask is all-keep (common on RoboTwin nopack).
        if not bool(key_mask.all().item()):
            use_varlen = True

    if not use_varlen:
        out = flash_attn_func(
            query, key, value, dropout_p=dropout_p, softmax_scale=None, causal=False
        )
        return out.to(dtype=in_dtype)

    # Varlen KV (padding tokens removed). Query stays dense length T per sample.
    # unpad_input expects [B, S, ...] ; flash-attn 2.6 returns (tensor, indices, cu_seqlens, max_seqlen)
    key_u, _indices, cu_k, max_k = unpad_input(key, key_mask)[:4]
    value_u = unpad_input(value, key_mask)[0]
    q_flat = query.reshape(B * T, H, D)
    cu_q = torch.arange(0, (B + 1) * T, step=T, device=query.device, dtype=torch.int32)
    out_flat = flash_attn_varlen_func(
        q_flat,
        key_u,
        value_u,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=T,
        max_seqlen_k=int(max_k),
        dropout_p=dropout_p,
        softmax_scale=None,
        causal=False,
    )
    out = out_flat.reshape(B, T, H, D)
    return out.to(dtype=in_dtype)


class FlashCrossAttnProcessor:
    """Drop-in for ``AttnProcessor2_0``: flash cross-attn, SDPA self-attn.

    Follows Sana ``MultiHeadCrossAttention`` dataflow (Q from tokens, KV from
    condition) but calls Dao-AILab flash-attn instead of xformers.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if not is_cross:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # flash layout [B, L, H, D]; SDPA layout [B, H, L, D]
        query_f = query.view(batch_size, -1, attn.heads, head_dim)
        key_f = key.view(batch_size, -1, attn.heads, head_dim)
        value_f = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            # norm_q expects [B, H, L, D] in some versions; support both
            q_n = query_f.transpose(1, 2)
            q_n = attn.norm_q(q_n)
            query_f = q_n.transpose(1, 2)
        if attn.norm_k is not None:
            k_n = key_f.transpose(1, 2)
            k_n = attn.norm_k(k_n)
            key_f = k_n.transpose(1, 2)

        dropout_p = 0.0
        if hasattr(attn, "dropout") and attn.training:
            # Match AttnProcessor2_0 training default (0.0) unless attn.attention_dropout set
            dropout_p = float(getattr(attn, "attention_dropout", 0.0) or 0.0)

        if is_cross and flash_attn_available():
            key_mask = _as_bool_key_mask(
                attention_mask, batch_size, key_len=key_f.shape[1]
            )
            try:
                hidden_f = _flash_cross_attn(
                    query_f, key_f, value_f, key_mask=key_mask, dropout_p=dropout_p
                )
                hidden_states = hidden_f.reshape(batch_size, -1, attn.heads * head_dim)
            except Exception:
                # Safe fallback to SDPA (same as Sana when xformers missing)
                query_b = query_f.transpose(1, 2)
                key_b = key_f.transpose(1, 2)
                value_b = value_f.transpose(1, 2)
                mask = None
                if attention_mask is not None:
                    seq_len = encoder_hidden_states.shape[1]
                    mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
                    mask = mask.view(batch_size, attn.heads, -1, mask.shape[-1])
                hidden_states = F.scaled_dot_product_attention(
                    query_b, key_b, value_b, attn_mask=mask, dropout_p=dropout_p, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
        else:
            # Self-attn or flash unavailable → SDPA
            query_b = query_f.transpose(1, 2)
            key_b = key_f.transpose(1, 2)
            value_b = value_f.transpose(1, 2)
            mask = None
            if attention_mask is not None:
                seq_len = encoder_hidden_states.shape[1]
                mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
                mask = mask.view(batch_size, attn.heads, -1, mask.shape[-1])
            hidden_states = F.scaled_dot_product_attention(
                query_b, key_b, value_b, attn_mask=mask, dropout_p=dropout_p, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_dit_flash_cross_attn(dit) -> int:
    """Install ``FlashCrossAttnProcessor`` on DiT cross-attn layers. Returns count."""
    if not flash_attn_available():
        raise RuntimeError(
            "flash-attn is required for DiT flash cross-attn but import failed: "
            f"{_FLASH_ATTN_IMPORT_ERROR}"
        )
    n = 0
    for block in dit.transformer_blocks:
        if getattr(block, "cross_attention_dim", None) is not None:
            block.attn1.set_processor(FlashCrossAttnProcessor())
            n += 1
    return n
