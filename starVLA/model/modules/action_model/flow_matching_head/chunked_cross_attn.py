"""Chunked + sequence-parallel DiT cross-attention.

DiT cross-attn has short queries (T≈40) and long KV (VLM last_hidden, up to
hundreds of K tokens). Default SDPA materializes full K/V and OOMs.

This processor:
1. Projects KV in chunks (never full-seq K/V at once).
2. Merges chunk results with online softmax (numerically aligned with full attn).
3. Optionally merges across an SP process group so each rank only holds L/SP
   of ``encoder_hidden_states`` — no full-L gather.

Self-attention (encoder_hidden_states is None) falls back to SDPA.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _online_update(
    m: Optional[torch.Tensor],
    l: Optional[torch.Tensor],
    o: Optional[torch.Tensor],
    scores: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention-style online softmax update.

    Args:
        m, l, o: running max / sum-exp / weighted value, or None on first chunk.
        scores: [B, H, T, Lc]
        value:  [B, H, Lc, D]
    Returns:
        updated (m, l, o) with shapes [B,H,T,1], [B,H,T,1], [B,H,T,D]
    """
    row_max = scores.amax(dim=-1, keepdim=True)
    if m is None:
        m = row_max
        p = torch.exp(scores - m)
        l = p.sum(dim=-1, keepdim=True)
        o = torch.matmul(p, value)
        return m, l, o

    m_new = torch.maximum(m, row_max)
    alpha = torch.exp(m - m_new)
    p = torch.exp(scores - m_new)
    l = alpha * l + p.sum(dim=-1, keepdim=True)
    o = alpha * o + torch.matmul(p, value)
    return m_new, l, o


def _merge_stats(
    m_a: torch.Tensor,
    l_a: torch.Tensor,
    o_a: torch.Tensor,
    m_b: torch.Tensor,
    l_b: torch.Tensor,
    o_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = torch.maximum(m_a, m_b)
    a = torch.exp(m_a - m)
    b = torch.exp(m_b - m)
    l = a * l_a + b * l_b
    o = a * o_a + b * o_b
    return m, l, o


def _chunked_attn_local(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    chunk_size: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local chunked attention → (m, l, o). query/key/value: [B,H,S,D]."""
    m = l = o = None
    L = key.shape[2]
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        k = key[:, :, start:end, :]
        v = value[:, :, start:end, :]
        scores = torch.matmul(query, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            # mask: [B, H, T, L_full] or broadcastable; slice KV dim
            scores = scores + attention_mask[:, :, :, start:end]
        m, l, o = _online_update(m, l, o, scores, v)
    assert m is not None and l is not None and o is not None
    return m, l, o


def _all_gather_merge(
    m: torch.Tensor,
    l: torch.Tensor,
    o: torch.Tensor,
    process_group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge online-softmax stats across SP ranks (identical result on all).

    Uses differentiable ``all_gather`` when inputs need grad so training can
    backprop through DiT SP; falls back to ``dist.all_gather`` for inference.
    """
    world = dist.get_world_size(process_group)
    if world <= 1:
        return m, l, o

    needs_grad = m.requires_grad or l.requires_grad or o.requires_grad
    if needs_grad:
        from torch.distributed.nn.functional import all_gather as diff_all_gather

        m_list = diff_all_gather(m.contiguous(), group=process_group)
        l_list = diff_all_gather(l.contiguous(), group=process_group)
        o_list = diff_all_gather(o.contiguous(), group=process_group)
    else:
        m_list = [torch.empty_like(m) for _ in range(world)]
        l_list = [torch.empty_like(l) for _ in range(world)]
        o_list = [torch.empty_like(o) for _ in range(world)]
        dist.all_gather(m_list, m.contiguous(), group=process_group)
        dist.all_gather(l_list, l.contiguous(), group=process_group)
        dist.all_gather(o_list, o.contiguous(), group=process_group)

    m_acc, l_acc, o_acc = m_list[0], l_list[0], o_list[0]
    for i in range(1, world):
        m_acc, l_acc, o_acc = _merge_stats(
            m_acc, l_acc, o_acc, m_list[i], l_list[i], o_list[i]
        )
    return m_acc, l_acc, o_acc


class ChunkedSPCrossAttnProcessor:
    """Drop-in replacement for ``AttnProcessor2_0`` on DiT cross-attn layers."""

    def __init__(
        self,
        chunk_size: int = 8192,
        process_group=None,
    ):
        self.chunk_size = int(chunk_size)
        self.process_group = process_group

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

        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads
        scale = head_dim**-0.5

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)

        # Self-attn (short T): keep SDPA path.
        if not is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            mask = None
            if attention_mask is not None:
                seq_len = encoder_hidden_states.shape[1]
                mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
                mask = mask.view(batch_size, attn.heads, -1, mask.shape[-1])
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        else:
            # Cross-attn: chunked local KV (+ optional SP merge). Never full K/V.
            L_local = encoder_hidden_states.shape[1]
            m = l = o = None
            mask_full = None
            if attention_mask is not None:
                # prepare against *local* length; SP ranks each mask their shard
                mask_full = attn.prepare_attention_mask(
                    attention_mask, L_local, batch_size
                )
                mask_full = mask_full.view(batch_size, attn.heads, -1, mask_full.shape[-1])

            for start in range(0, L_local, self.chunk_size):
                end = min(start + self.chunk_size, L_local)
                enc = encoder_hidden_states[:, start:end, :]
                key = attn.to_k(enc)
                value = attn.to_v(enc)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                scores = torch.matmul(query, key.transpose(-2, -1)) * scale
                if mask_full is not None:
                    scores = scores + mask_full[:, :, :, start:end]
                m, l, o = _online_update(m, l, o, scores, value)

            assert m is not None and l is not None and o is not None

            if self.process_group is not None and dist.is_initialized():
                world = dist.get_world_size(self.process_group)
                if world > 1:
                    m, l, o = _all_gather_merge(m, l, o, self.process_group)

            hidden_states = o / l.clamp_min(1e-20)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
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


def install_dit_chunked_sp_cross_attn(
    dit,
    process_group=None,
    chunk_size: int = 8192,
) -> int:
    """Replace cross-attn processors on a ``DiT``; returns #layers patched."""
    n = 0
    for block in dit.transformer_blocks:
        if getattr(block, "cross_attention_dim", None) is not None:
            block.attn1.set_processor(
                ChunkedSPCrossAttnProcessor(
                    chunk_size=chunk_size, process_group=process_group
                )
            )
            n += 1
    return n


def restore_dit_default_attn_processors(dit) -> None:
    """Restore diffusers default AttnProcessor2_0 on all DiT blocks."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    for block in dit.transformer_blocks:
        block.attn1.set_processor(AttnProcessor2_0())


@torch.inference_mode()
def predict_action_sp(
    action_head,
    vl_embs_local: torch.Tensor,
    process_group=None,
    chunk_size: int = 8192,
    state: torch.Tensor | None = None,
    encoder_attention_mask=None,
    init_actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``predict_action`` with chunked (+ optional SP) full-L cross-attn.

    ``vl_embs_local`` is the per-rank KV shard ``(B, L/SP, H)`` when
    ``process_group`` is set; otherwise the full ``(B, L, H)``.
    All ranks must call this together when SP is enabled.
    """
    dit = action_head.model
    n = install_dit_chunked_sp_cross_attn(
        dit, process_group=process_group, chunk_size=chunk_size
    )
    if n == 0:
        raise RuntimeError("No DiT cross-attn layers found to patch")

    # Broadcast init noise so every rank denoises the same trajectory.
    if init_actions is None:
        device = vl_embs_local.device
        dtype = torch.float32
        b = vl_embs_local.shape[0]
        init_actions = torch.randn(
            size=(b, action_head.config.action_horizon, action_head.config.action_dim),
            dtype=dtype,
            device=device,
        )
        if process_group is not None and dist.is_initialized():
            if dist.get_world_size(process_group) > 1:
                # group_src=0 = SP-local leader; global src=0 breaks non-0 DP groups
                dist.broadcast(init_actions, group=process_group, group_src=0)

    try:
        return action_head.predict_action(
            vl_embs_local,
            state=state,
            encoder_attention_mask=encoder_attention_mask,
            init_actions=init_actions,
        )
    finally:
        restore_dit_default_attn_processors(dit)
