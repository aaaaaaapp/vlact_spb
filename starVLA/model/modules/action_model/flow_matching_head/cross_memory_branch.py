# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable cross-memory branches for DiT hybrid layers.

Interface: Q from action tokens, KV from VL register / memory tokens.
Switch backend via ``memory_type`` ∈ {none, gdn, sse, ttt_linear, ttt_mlp}.

``sse`` implements cross-SSEGDN (FLA mask path): shared GDN stream ∥ sparse
state partitions with top-k write (on memory) and top-k read (on action).
Requires ``expand_v=1``.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_FLA_IMPORT_ERROR: Optional[Exception] = None
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except Exception as e:  # pragma: no cover
    chunk_gated_delta_rule = None  # type: ignore
    _FLA_IMPORT_ERROR = e

try:
    from fla.ops.ttt import chunk_ttt_linear
except Exception:
    chunk_ttt_linear = None  # type: ignore

_SSE_IMPORT_ERROR: Optional[Exception] = None
try:
    from fla.ops.sse import softmax_and_mask
except Exception as e:  # pragma: no cover
    softmax_and_mask = None  # type: ignore
    _SSE_IMPORT_ERROR = e

MEMORY_TYPES = ("none", "gdn", "sse", "ttt_linear", "ttt_mlp")

# SpikeBrain-2.0-VL / Qwen2.5-VL image pad id (see configuration_spb2_vl.py).
DEFAULT_IMAGE_TOKEN_ID = 151655


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)


def consecutive_id_spans(ids_1d: torch.Tensor, token_id: int) -> list[tuple[int, int]]:
    """Return ``[start, end)`` spans of consecutive ``token_id`` in a 1D id tensor."""
    if ids_1d.ndim != 1:
        raise ValueError(f"ids_1d must be 1D, got shape {tuple(ids_1d.shape)}")
    mask = ids_1d == int(token_id)
    if not bool(mask.any()):
        return []
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    # Iterate in Python: image counts are small (≤ tens), spans are few.
    for i, is_tok in enumerate(mask.tolist()):
        if is_tok and start is None:
            start = i
        elif (not is_tok) and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, int(ids_1d.numel())))
    return spans


def image_token_spans_from_input_ids(
    input_ids: torch.Tensor,
    image_token_id: int = DEFAULT_IMAGE_TOKEN_ID,
) -> list[list[tuple[int, int]]]:
    """Per-batch-sample image-pad spans from ``input_ids`` ``[B, S]`` (or ``[S]``)."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B,S] or [S], got {tuple(input_ids.shape)}")
    return [consecutive_id_spans(input_ids[b], image_token_id) for b in range(input_ids.shape[0])]


def group_image_spans_by_timestep(
    image_spans: list[tuple[int, int]],
    images_per_timestep: int,
) -> list[list[tuple[int, int]]]:
    """Group ordered image spans into timesteps (oldest → newest).

    Image order is assumed ``[t0_view0, t0_view1, ..., t1_view0, ...]``.
    """
    if images_per_timestep <= 0:
        raise ValueError(f"images_per_timestep must be > 0, got {images_per_timestep}")
    n = len(image_spans)
    if n == 0:
        return []
    if n % images_per_timestep != 0:
        # Fall back: each image is its own timestep (still a temporal write axis).
        return [[sp] for sp in image_spans]
    t_steps = n // images_per_timestep
    return [
        image_spans[t * images_per_timestep : (t + 1) * images_per_timestep]
        for t in range(t_steps)
    ]


class VLRegisterBank(nn.Module):
    """Learnable registers conditioned on VL encoder states → memory KV.

    Without image spans (legacy): R registers attend over the full VL sequence
    → ``memory_kv [B, R, D]``.

    With image spans (temporal write): for each VL history timestep present in
    the input, the **same** R registers (default R=16) attend over that
    timestep's image tokens only; outputs concat oldest→newest →
    ``memory_kv [B, T*R, D]``. ``T`` is whatever history was provided — not a
    fixed context window. GDN/SSE/TTT scan along this write sequence.
    """

    def __init__(
        self,
        dim: int,
        num_register_tokens: int = 16,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_register_tokens <= 0:
            raise ValueError("num_register_tokens must be > 0")
        if dim % num_heads != 0:
            raise ValueError(f"register dim {dim} must be divisible by num_heads {num_heads}")
        self.num_register_tokens = num_register_tokens
        self.registers = nn.Parameter(torch.randn(1, num_register_tokens, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @staticmethod
    def _to_key_padding_mask(
        encoder_attention_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Convert common mask layouts to MHA key_padding_mask [B, S] (True=ignore)."""
        mask = encoder_attention_mask
        # Drop broadcast singleton dims: [B,1,1,S] / [B,1,S] → [B,S]
        while mask.ndim > 2 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.ndim == 3 and mask.shape[1] == mask.shape[2]:
            # [B,S,S] attention mask → row-any keep
            mask = mask.select(1, 0)
        if mask.ndim != 2:
            raise ValueError(
                f"encoder_attention_mask must reduce to [B,S], got {tuple(encoder_attention_mask.shape)}"
            )
        if mask.shape[-1] != seq_len:
            raise ValueError(
                f"mask seq_len {mask.shape[-1]} != encoder seq_len {seq_len}"
            )
        if mask.dtype == torch.bool:
            # Assume True=keep (HF style); MHA wants True=ignore
            return ~mask
        # numeric: >0 keep
        return mask <= 0

    def _compress_kv(
        self,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend shared registers over ``kv`` ``[B, S, D]`` → ``[B, R, D]``."""
        B = kv.shape[0]
        q = self.norm_q(self.registers.expand(B, -1, -1))
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(
            q, kv_n, kv_n, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out + self.registers.expand(B, -1, -1)

    def _compress_timestep_groups(
        self,
        encoder_hidden_states_1: torch.Tensor,
        groups: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        """Compress one sample's VL timestep groups → ``[1, T*R, D]``.

        ``T`` is whatever history was provided (not a fixed context length).
        Each timestep is compressed to ``num_register_tokens`` (default 16).
        """
        if not groups:
            return self._compress_kv(encoder_hidden_states_1)
        per_t: list[torch.Tensor] = []
        for spans_t in groups:
            chunks = [
                encoder_hidden_states_1[0, s:e, :]
                for (s, e) in spans_t
                if e > s
            ]
            if chunks:
                kv_t = torch.cat(chunks, dim=0).unsqueeze(0)
            else:
                kv_t = encoder_hidden_states_1.new_zeros(
                    (1, 1, encoder_hidden_states_1.shape[-1])
                )
            per_t.append(self._compress_kv(kv_t))
        return torch.cat(per_t, dim=1)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_token_spans: Optional[list[list[tuple[int, int]]]] = None,
        images_per_timestep: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [B, S, D] VL tokens
            encoder_attention_mask: optional [B, S] / [B,1,1,S] with 1=keep / 0=pad
            image_token_spans: per-sample list of ``[start, end)`` image-pad spans,
                ordered as images appear in the chat template (oldest→newest when
                the dataloader stacks history that way).
            images_per_timestep: views per VL timestep (e.g. 2 = primary+wrist).
                Spans are grouped into timesteps; each timestep → R registers
                (R=``num_register_tokens``, typically 16); concat → ``[B, T*R, D]``.
                ``T`` is inferred from the provided images (variable history OK).
        Returns:
            memory_kv: [B, T*R, D] with T = #VL history steps in the sample
        """
        B, S, _ = encoder_hidden_states.shape
        use_temporal = (
            image_token_spans is not None
            and len(image_token_spans) == B
            and images_per_timestep > 0
            and any(len(sp) > 0 for sp in image_token_spans)
        )
        if not use_temporal:
            key_padding_mask = None
            if encoder_attention_mask is not None:
                key_padding_mask = self._to_key_padding_mask(encoder_attention_mask, S)
            return self._compress_kv(encoder_hidden_states, key_padding_mask)

        # Per-sample path: T may differ across the batch (variable history).
        # Always compress **image tokens of each VL timestep** to R registers —
        # including T=1 (do not fall back to full-seq text+image compress).
        outs: list[torch.Tensor] = []
        max_write = 0
        for b in range(B):
            groups = group_image_spans_by_timestep(
                image_token_spans[b], images_per_timestep
            )
            mem_b = self._compress_timestep_groups(
                encoder_hidden_states[b : b + 1], groups
            )
            outs.append(mem_b)
            max_write = max(max_write, int(mem_b.shape[1]))

        # Pad write length across batch (right-pad zeros); GDN scan is content-driven
        # and trailing zero keys contribute negligibly when T is short.
        R = self.num_register_tokens
        if max_write % R != 0:
            # Should be T*R; keep raw length if grouping fell back oddly.
            pass
        stacked = encoder_hidden_states.new_zeros(B, max_write, encoder_hidden_states.shape[-1])
        for b, mem_b in enumerate(outs):
            stacked[b, : mem_b.shape[1]] = mem_b[0]
        return stacked


class CrossMemoryBranch(nn.Module):
    """Cross memory: Q=action, KV=memory registers. Backend selected by ``memory_type``."""

    def __init__(
        self,
        dim: int,
        memory_type: str = "gdn",
        num_heads: int = 12,
        num_v_heads: Optional[int] = None,
        head_dim: int = 64,
        expand_v: float = 2.0,
        mlp_expand: int = 4,
        mini_batch_size: int = 16,
        base_lr: float = 0.1,
        use_short_conv: bool = False,
        num_sparse_partition: int = 4,
        num_writer: int = 1,
        num_reader: int = 1,
        **_: Any,
    ):
        super().__init__()
        memory_type = (memory_type or "none").lower()
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {MEMORY_TYPES}, got {memory_type}")
        self.memory_type = memory_type
        self.dim = dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_dim = head_dim
        self.expand_v = float(expand_v)
        self.mlp_expand = mlp_expand
        self.mini_batch_size = mini_batch_size
        self.base_lr = base_lr
        self.key_dim = num_heads * head_dim
        self.value_dim = int(self.num_v_heads * head_dim * self.expand_v)
        self.last_aux_loss: Optional[torch.Tensor] = None

        if memory_type == "none":
            return

        if self.key_dim > dim:
            raise ValueError(f"num_heads*head_dim={self.key_dim} > dim={dim}")

        # Shared projections (action → Q, memory → K/V)
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

        if memory_type == "gdn":
            if chunk_gated_delta_rule is None:
                raise RuntimeError(f"FLA gated_delta_rule unavailable: {_FLA_IMPORT_ERROR}")
            # GDN write controls (A_log / dt_bias kept fp32 for numerical stability)
            self.b_proj = nn.Linear(dim, self.num_v_heads, bias=False)
            self.a_proj = nn.Linear(dim, self.num_v_heads, bias=False)
            self.A_log = nn.Parameter(torch.randn(self.num_v_heads, dtype=torch.float32) * 0.1)
            self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.float32))
        elif memory_type == "sse":
            if chunk_gated_delta_rule is None:
                raise RuntimeError(f"FLA gated_delta_rule unavailable: {_FLA_IMPORT_ERROR}")
            if softmax_and_mask is None:
                raise RuntimeError(f"FLA SSE softmax_and_mask unavailable: {_SSE_IMPORT_ERROR}")
            # FLA SSE mask kernels assume head_k == head_v (bench also uses expand_v=1).
            if not math.isclose(self.expand_v, 1.0):
                raise ValueError(
                    f"sse cross-memory requires expand_v=1 (got {self.expand_v}); "
                    "use gdn for expand_v=2."
                )
            if num_writer >= num_sparse_partition or num_reader >= num_sparse_partition:
                raise ValueError(
                    "num_writer and num_reader must be < num_sparse_partition "
                    f"(got writer={num_writer}, reader={num_reader}, N={num_sparse_partition})"
                )
            self.num_sparse_partition = int(num_sparse_partition)
            self.num_writer = int(num_writer)
            self.num_reader = int(num_reader)
            head_v_dim = int(self.head_dim * self.expand_v)
            # Dual stream (shared + sparse), matching FLA SSEGDN.
            self.lora_q_proj = nn.Sequential(
                nn.Linear(dim, head_v_dim, bias=False),
                nn.Linear(head_v_dim, self.key_dim, bias=False),
            )
            self.lora_k_proj = nn.Sequential(
                nn.Linear(dim, head_v_dim, bias=False),
                nn.Linear(head_v_dim, self.key_dim, bias=False),
            )
            self.b_proj = nn.Linear(dim, self.num_v_heads * 2, bias=False)
            self.a_proj = nn.Linear(dim, self.num_v_heads * 2, bias=False)
            self.A_log = nn.Parameter(torch.randn(self.num_v_heads * 2, dtype=torch.float32) * 0.1)
            self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads * 2, dtype=torch.float32))
            self.e_proj = nn.Linear(dim, self.num_sparse_partition, bias=False)  # write router
            self.e_read_proj = nn.Linear(dim, self.num_sparse_partition, bias=False)  # read router
        elif memory_type == "ttt_linear":
            if self.value_dim != self.key_dim:
                raise ValueError("ttt_linear requires expand_v=1 and num_v_heads==num_heads")
            self.lr_proj = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.lr_proj.weight)
            nn.init.zeros_(self.lr_proj.bias)
            self.ttt_norm_weight = nn.Parameter(torch.ones(num_heads, head_dim))
            self.ttt_norm_bias = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.W0 = nn.Parameter(torch.randn(num_heads, head_dim, head_dim) * (head_dim**-0.5))
            self.b0 = nn.Parameter(torch.zeros(num_heads, 1, head_dim))
        elif memory_type == "ttt_mlp":
            if self.value_dim != self.key_dim:
                raise ValueError("ttt_mlp requires expand_v=1 and num_v_heads==num_heads")
            inner = head_dim * mlp_expand
            self.lr_proj = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.lr_proj.weight)
            nn.init.zeros_(self.lr_proj.bias)
            self.W1_0 = nn.Parameter(torch.randn(num_heads, head_dim, inner) * 0.02)
            self.b1_0 = nn.Parameter(torch.zeros(num_heads, 1, inner))
            self.W2_0 = nn.Parameter(torch.randn(num_heads, inner, head_dim) * 0.02)
            self.b2_0 = nn.Parameter(torch.zeros(num_heads, 1, head_dim))

        self.use_short_conv = use_short_conv  # reserved; cross path skips short-conv

    def _apply(self, fn):
        out = super()._apply(fn)
        # Keep GDN stability params in fp32 across .to(dtype=bf16/fp16).
        for name in ("A_log", "dt_bias"):
            p = getattr(self, name, None)
            if isinstance(p, nn.Parameter) and p.data.dtype != torch.float32:
                p.data = p.data.float()
        return out

    def forward(
        self,
        query_states: torch.Tensor,
        memory_kv: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
        traj_chunk_parallel: bool = False,
        traj_n: Optional[int] = None,
    ):
        """
        Args:
            query_states: [B, L, D] action (post-norm). When ``traj_chunk_parallel``,
                ``B`` is ``traj_n * T`` (robot-time batch, traj-major contiguous).
            memory_kv: [B, R, D] VL registers / memory tokens (per robot step).
            initial_state: optional GDN state ``[N, H, K, V]`` to continue a prior write
                (RoboTTT-style trajectory recursion / TBPTT segment carry).
                For ``traj_chunk_parallel``, ``N=traj_n`` (one carry per trajectory).
            return_final_state: if True, return ``(output, final_state)`` for GDN;
                other backends return ``(output, None)``.
            traj_chunk_parallel: LLM-style one long GDN scan over the window:
                interleave per-step write tokens with read-only action queries and run
                a single ``chunk_gated_delta_rule`` (chunk-parallel along the long seq).
            traj_n: number of parallel trajectories in the flat batch (default 1).
        Returns:
            [B, L, D] or ``([B, L, D], Optional[state])`` when ``return_final_state``.
        """
        self.last_final_state = None
        if self.memory_type == "none":
            out = torch.zeros_like(query_states)
            return (out, None) if return_final_state else out

        if self.memory_type == "gdn":
            if traj_chunk_parallel:
                out, state = self._forward_gdn_traj_chunk_parallel(
                    query_states,
                    memory_kv,
                    initial_state=initial_state,
                    traj_n=traj_n,
                )
            else:
                out, state = self._forward_gdn_cross(
                    query_states,
                    memory_kv,
                    initial_state=initial_state,
                    return_final_state=True,
                )
            self.last_final_state = state
            return (out, state) if return_final_state else out
        if self.memory_type == "sse":
            out = self._forward_sse_cross(query_states, memory_kv)
            return (out, None) if return_final_state else out
        if self.memory_type == "ttt_linear":
            out = self._forward_ttt_linear_cross(query_states, memory_kv)
            return (out, None) if return_final_state else out
        if self.memory_type == "ttt_mlp":
            out = self._forward_ttt_mlp_cross(query_states, memory_kv)
            return (out, None) if return_final_state else out
        raise RuntimeError(f"unhandled memory_type={self.memory_type}")

    def _reshape_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = q.shape
        R = k.shape[1]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, R, self.num_heads, self.head_dim)
        v = v.view(B, R, self.num_v_heads, -1)
        return q, k, v

    def _forward_gdn_cross(
        self,
        query_states: torch.Tensor,
        memory_kv: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False,
    ):
        # Write state from memory KV (optionally continuing ``initial_state``),
        # then read with action Q.
        q = F.silu(self.q_proj(query_states))
        k = F.silu(self.k_proj(memory_kv))
        v = F.silu(self.v_proj(memory_kv))
        q, k, v = self._reshape_qkv(q, k, v)

        # GDN decay / beta from memory tokens
        beta = self.b_proj(memory_kv).sigmoid()  # [B,R,Hv]
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(memory_kv).float() + self.dt_bias)
        g = g.to(dtype=q.dtype)

        # Align head counts: if Hv != Hk, repeat k/q heads or beta
        if self.num_v_heads != self.num_heads:
            repeat = self.num_v_heads // self.num_heads
            if self.num_v_heads % self.num_heads != 0:
                raise ValueError("num_v_heads must be multiple of num_heads for cross-GDN")
            k = k.repeat_interleave(repeat, dim=2)
            q = q.repeat_interleave(repeat, dim=2)
            # beta/g already [B,R,Hv]

        # Write: scan memory with q_write=k; carry state across robot timesteps / TBPTT segs.
        init = initial_state
        if init is not None and init.dtype != k.dtype:
            init = init.to(dtype=k.dtype)
        _, state = chunk_gated_delta_rule(
            k,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # state: [B, Hv, K, V]; q: [B, T, Hv, K]
        o = torch.einsum("bthk,bhkv->bthv", q.to(state.dtype), state).to(q.dtype)
        B, T, _, _ = o.shape
        o = o.reshape(B, T, self.value_dim)
        out = self.o_proj(o)
        if return_final_state:
            return out, state
        return out

    def _forward_gdn_traj_chunk_parallel(
        self,
        query_states: torch.Tensor,
        memory_kv: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        traj_n: Optional[int] = None,
    ):
        """LLM-style: one long GDN scan over a trajectory window (optionally N trajs).

        Flat input is traj-major contiguous: ``[ex0_t0..ex0_tT), ex1_t0.., ...]``.
        Per robot step ``t``: write ``memory_kv[t]`` (``R`` tokens, ``beta>0``), then
        read with ``query_states[t]`` (``L`` tokens, ``beta=0`` so state is unchanged).
        Each trajectory is an independent FLA batch row.

        Args:
            query_states: ``[N*T, L, D]`` (or ``[T, L, D]`` when ``traj_n=1``)
            memory_kv: ``[N*T, R, D]``
            initial_state: optional ``[N, H, K, V]`` carried from the previous window
            traj_n: number of parallel trajectories (default 1)
        Returns:
            ``(out [N*T, L, D], final_state [N, H, K, V])``
        """
        if chunk_gated_delta_rule is None:
            raise RuntimeError(
                f"chunk_gated_delta_rule unavailable for traj_chunk_parallel: {_FLA_IMPORT_ERROR}"
            )
        if query_states.ndim != 3 or memory_kv.ndim != 3:
            raise ValueError(
                "traj_chunk_parallel expects query_states [N*T,L,D] and memory_kv [N*T,R,D]; "
                f"got {tuple(query_states.shape)} and {tuple(memory_kv.shape)}"
            )
        BT, L, _ = query_states.shape
        BTm, R, _ = memory_kv.shape
        if BTm != BT:
            raise ValueError(
                f"traj_chunk_parallel T mismatch: query BT={BT} vs memory BT={BTm}"
            )
        if BT == 0:
            raise ValueError("traj_chunk_parallel empty window")
        n = int(traj_n or 1)
        if n <= 0:
            raise ValueError(f"traj_n must be >=1, got {n}")
        if BT % n != 0:
            raise ValueError(
                f"traj_chunk_parallel BT={BT} not divisible by traj_n={n}"
            )
        T = BT // n

        q_act = F.silu(self.q_proj(query_states))  # [N*T, L, D]
        k_mem = F.silu(self.k_proj(memory_kv))  # [N*T, R, D]
        v_mem = F.silu(self.v_proj(memory_kv))
        q_act, k_mem, v_mem = self._reshape_qkv(q_act, k_mem, v_mem)
        # q_act: [N*T, L, Hk, K]; k_mem/v_mem: [N*T, R, Hk/Hv, ...]

        beta_mem = self.b_proj(memory_kv).sigmoid()  # [N*T, R, Hv]
        g_mem = -self.A_log.float().exp() * F.softplus(
            self.a_proj(memory_kv).float() + self.dt_bias
        )
        g_mem = g_mem.to(dtype=q_act.dtype)

        if self.num_v_heads != self.num_heads:
            repeat = self.num_v_heads // self.num_heads
            if self.num_v_heads % self.num_heads != 0:
                raise ValueError("num_v_heads must be multiple of num_heads for cross-GDN")
            k_mem = k_mem.repeat_interleave(repeat, dim=2)
            q_act = q_act.repeat_interleave(repeat, dim=2)

        Hv = beta_mem.shape[-1]
        Hk = k_mem.shape[2]
        Kdim = k_mem.shape[3]
        Vdim = v_mem.shape[3]

        # Read-only action slots: beta=0 → no state update; k/v zeros.
        beta_act = torch.zeros(BT, L, Hv, device=beta_mem.device, dtype=beta_mem.dtype)
        g_act = torch.zeros(BT, L, Hv, device=g_mem.device, dtype=g_mem.dtype)
        k_act = torch.zeros(BT, L, Hk, Kdim, device=k_mem.device, dtype=k_mem.dtype)
        v_act = torch.zeros(BT, L, Hv, Vdim, device=v_mem.device, dtype=v_mem.dtype)

        # Interleave [mem_R | act_L] per timestep → [N, T*(R+L), ...]
        def _interleave_time(mem_x: torch.Tensor, act_x: torch.Tensor) -> torch.Tensor:
            # mem/act [N*T, R|L, ...] → [N, T*(R+L), ...]
            step = torch.cat([mem_x, act_x], dim=1)
            step = step.view(n, T, R + L, *step.shape[2:])
            return step.reshape(n, T * (R + L), *step.shape[3:])

        q_seq = _interleave_time(k_mem, q_act)
        k_seq = _interleave_time(k_mem, k_act)
        v_seq = _interleave_time(v_mem, v_act)
        beta_seq = _interleave_time(beta_mem, beta_act)
        g_seq = _interleave_time(g_mem, g_act)

        init = initial_state
        if init is not None:
            if init.shape[0] == 1 and n > 1:
                init = init.expand(n, *init.shape[1:]).contiguous()
            elif init.shape[0] != n:
                raise ValueError(
                    f"traj_chunk_parallel initial_state N={init.shape[0]} != traj_n={n}"
                )
            if init.dtype != k_seq.dtype:
                init = init.to(dtype=k_seq.dtype)

        o_seq, state = chunk_gated_delta_rule(
            q_seq,
            k_seq,
            v_seq,
            g=g_seq,
            beta=beta_seq,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # Extract action readouts: positions within each block of (R+L).
        # o_seq: [N, T*(R+L), Hv, V] → flat [N*T, L, ...]
        o_seq = o_seq.reshape(n, T, R + L, *o_seq.shape[2:])
        o = o_seq[:, :, R : R + L]  # [N, T, L, Hv, V]
        o = o.reshape(BT, L, self.value_dim).to(dtype=query_states.dtype)
        out = self.o_proj(o)
        return out, state

    def _forward_sse_cross(self, query_states: torch.Tensor, memory_kv: torch.Tensor) -> torch.Tensor:
        """Cross SSEGDN: sparse-partition write on memory KV, sparse read with action Q.

        Follows FLA ``SSEGDN`` mask path (shared stream ∥ N sparse streams), but splits
        write/read across sequences: KV/eta_write from ``memory_kv``, Q/eta_read from
        ``query_states``.
        """
        B, T, _ = query_states.shape
        R = memory_kv.shape[1]
        N = self.num_sparse_partition
        HD = self.head_dim
        HVD = int(self.head_dim * self.expand_v)

        q1 = self.q_proj(query_states)
        q2 = q1 + self.lora_q_proj(query_states)
        k1 = self.k_proj(memory_kv)
        k2 = k1 + self.lora_k_proj(memory_kv)
        v = self.v_proj(memory_kv)

        b = self.b_proj(memory_kv).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(memory_kv).float() + self.dt_bias)
        b1, b2 = torch.chunk(b, 2, dim=-1)
        g1, g2 = torch.chunk(g, 2, dim=-1)
        g1 = g1.to(dtype=query_states.dtype)
        g2 = g2.to(dtype=query_states.dtype)

        eta_w = self.e_proj(memory_kv)  # [B, R, N]
        eta_r = self.e_read_proj(query_states)  # [B, T, N]

        q1 = F.silu(q1.view(B, T, self.num_heads, HD))
        q2 = F.silu(q2.view(B, T, self.num_heads, HD))
        k1 = F.silu(k1.view(B, R, self.num_heads, HD))
        k2 = F.silu(k2.view(B, R, self.num_heads, HD))
        v = F.silu(v.view(B, R, self.num_v_heads, HVD))

        if self.num_v_heads != self.num_heads:
            repeat = self.num_v_heads // self.num_heads
            if self.num_v_heads % self.num_heads != 0:
                raise ValueError("num_v_heads must be multiple of num_heads for cross-SSE")
            q1 = q1.repeat_interleave(repeat, dim=2)
            q2 = q2.repeat_interleave(repeat, dim=2)
            k1 = k1.repeat_interleave(repeat, dim=2)
            k2 = k2.repeat_interleave(repeat, dim=2)

        H = self.num_v_heads

        # --- Write: shared GDN ∥ sparse top-k writers over memory tokens ---
        # softmax_and_mask: k/v gated by writer mask; q gated by reader (unused for state).
        _, k2_w, v2, _, eta_w_s, mask_w, _mask_r_w = softmax_and_mask(
            k2, k2, v, k2, eta_w, self.num_writer, self.num_reader
        )
        mw = mask_w.to(dtype=g2.dtype).unsqueeze(-1)  # [B,R,N,1]
        g2_exp = (g2.unsqueeze(2).expand(-1, -1, N, -1) * mw).reshape(B, R, N * H)
        b2_exp = (b2.unsqueeze(2).expand(-1, -1, N, -1) * mw).reshape(B, R, N * H)

        k_cat = torch.cat([k1, k2_w], dim=2)  # [B,R,H*(1+N),HD]
        v_cat = torch.cat([v, v2], dim=2)
        g_cat = torch.cat([g1, g2_exp], dim=2)
        b_cat = torch.cat([b1, b2_exp], dim=2)

        _, state = chunk_gated_delta_rule(
            k_cat,
            k_cat,
            v_cat,
            g=g_cat,
            beta=b_cat,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # state: [B, H*(1+N), HD, HVD]

        # Writer load-balance aux (same formula as FLA SSEGDN); optional for training.
        if self.training:
            p = torch.mean(eta_w_s.float(), dim=(0, 1))
            f = torch.mean(mask_w.float(), dim=(0, 1))
            self.last_aux_loss = torch.sum(p * f) * N / max(self.num_writer, 1)
        else:
            self.last_aux_loss = None

        # --- Read: shared Q ∥ sparse top-k readers over action tokens ---
        dummy_v = q2.new_zeros(B, T, H, HVD)
        q2_r, _, _, _, _eta_r_s, _mask_w_r, mask_r = softmax_and_mask(
            q2, q2, dummy_v, q2, eta_r, self.num_writer, self.num_reader
        )
        # Zero out non-selected sparse read heads for state readout (reader mask).
        # q2_r already scaled by reader gate*eta; additionally hard-mask unused partitions.
        mr = mask_r.to(dtype=q2_r.dtype).unsqueeze(-1).unsqueeze(-1)  # [B,T,N,1,1]
        q2_r = q2_r.view(B, T, N, H, HD) * mr
        q2_r = q2_r.reshape(B, T, N * H, HD)

        q_cat = torch.cat([q1, q2_r], dim=2)  # [B,T,H*(1+N),HD]
        o = torch.einsum("bthk,bhkv->bthv", q_cat.to(state.dtype), state).to(q1.dtype)
        o = o.view(B, T, N + 1, H, HVD).sum(dim=2)
        o = o.reshape(B, T, self.value_dim)
        return self.o_proj(o)

    def _forward_ttt_linear_cross(
        self, query_states: torch.Tensor, memory_kv: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = query_states.shape
        R = memory_kv.shape[1]
        H, HD = self.num_heads, self.head_dim
        q = _l2norm(self.q_proj(query_states).view(B, T, H, HD))
        k = _l2norm(self.k_proj(memory_kv).view(B, R, H, HD))
        v = self.v_proj(memory_kv).view(B, R, H, HD)
        eta = (torch.sigmoid(self.lr_proj(memory_kv)) * self.base_lr).unsqueeze(-1)  # [B,R,H,1]

        final_state = None
        final_bias = None
        # FLA TTT chunk kernels need seq_len >= 16 for Triton tile sizes.
        if chunk_ttt_linear is not None and R >= 16:
            _, final_state, final_bias = chunk_ttt_linear(
                k,
                k,
                v,
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                eta,
                chunk_size=min(self.mini_batch_size, R),
                initial_state=None,
                initial_state_bias=None,
                output_final_state=True,
                head_first=False,
            )

        if final_state is None:
            # Short-memory / no-kernel fallback: per-sample scan (must not mix batch).
            W = self.W0.to(dtype=torch.float32).unsqueeze(0).expand(B, -1, -1, -1).clone()
            b = self.b0.to(dtype=torch.float32).unsqueeze(0).expand(B, -1, -1, -1).clone()
            kf = k.float().transpose(1, 2)  # [B,H,R,D]
            vf = v.float().transpose(1, 2)
            et = eta.float().transpose(1, 2)  # [B,H,R,1]
            for i in range(R):
                ki = kf[:, :, i : i + 1]
                vi = vf[:, :, i : i + 1]
                pred = torch.einsum("bhmd,bhde->bhme", ki, W) + b
                g = 2.0 * (pred - vi)
                gW = torch.einsum("bhme,bhmd->bhde", g, ki)
                gb = g.sum(dim=2, keepdim=True)  # [B,H,1,D]
                lr = et[:, :, i : i + 1]  # [B,H,1,1]; already includes base_lr
                W = W - lr * gW
                b = b - lr * gb
            o = torch.einsum("bthd,bhde->bthe", q.float(), W) + b.reshape(B, 1, H, HD)
            o = o.to(q.dtype)
        else:
            # final_state [B,H,D,D], final_bias [B,H,1,D]
            o = torch.einsum("bthd,bhde->bthe", q.to(final_state.dtype), final_state)
            if final_bias is not None:
                o = o + final_bias.to(o.dtype).reshape(B, 1, H, HD)
            o = o.to(q.dtype)

        # Match FLA readout: token LN + residual (both kernel and fallback paths).
        mu = o.mean(dim=-1, keepdim=True)
        var = o.var(dim=-1, unbiased=False, keepdim=True)
        o = (o - mu) * torch.rsqrt(var + 1e-6)
        o = o * self.ttt_norm_weight.to(o.dtype).view(1, 1, H, HD) + self.ttt_norm_bias.to(
            o.dtype
        ).view(1, 1, H, HD)
        o = o + q

        return self.o_proj(o.reshape(B, T, self.key_dim))

    def _forward_ttt_mlp_cross(
        self, query_states: torch.Tensor, memory_kv: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch reference TTT-MLP: update on memory KV, apply to action Q."""
        B, T, _ = query_states.shape
        R = memory_kv.shape[1]
        H, HD = self.num_heads, self.head_dim
        inner = HD * self.mlp_expand
        q = self.q_proj(query_states).view(B, T, H, HD).float()
        k = self.k_proj(memory_kv).view(B, R, H, HD).float()
        v = self.v_proj(memory_kv).view(B, R, H, HD).float()
        eta = torch.sigmoid(self.lr_proj(memory_kv)).transpose(1, 2).unsqueeze(-1)  # [B,H,R,1]

        # Per-sample TTT state (do not reduce/mix across batch).
        W1 = self.W1_0.float().unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B,H,D,I]
        b1 = self.b1_0.float().unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B,H,1,I]
        W2 = self.W2_0.float().unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B,H,I,D]
        b2 = self.b2_0.float().unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B,H,1,D]
        M = self.mini_batch_size
        # pad memory to multiple of M
        pad = (M - (R % M)) % M
        if pad:
            k = F.pad(k.transpose(1, 2), (0, 0, 0, pad)).transpose(1, 2)
            v = F.pad(v.transpose(1, 2), (0, 0, 0, pad)).transpose(1, 2)
            eta = F.pad(eta, (0, 0, 0, pad))
        Rp = k.shape[1]
        NT = Rp // M
        k = k.transpose(1, 2)  # [B,H,R,D]
        v = v.transpose(1, 2)
        inv_m = 1.0 / float(M)
        for i in range(NT):
            sl = slice(i * M, (i + 1) * M)
            ki, vi = k[:, :, sl], v[:, :, sl]
            etai = eta[:, :, sl].mean(dim=2, keepdim=True)  # [B,H,1,1]
            pre = torch.einsum("bhmd,bhdi->bhmi", ki, W1) + b1
            h1 = F.gelu(pre)
            pred = torch.einsum("bhmi,bhid->bhmd", h1, W2) + b2
            g_pred = 2.0 * (pred - vi) * inv_m
            g_b2 = g_pred.sum(dim=2, keepdim=True)  # [B,H,1,D]
            g_W2 = torch.einsum("bhmd,bhmi->bhid", g_pred, h1)
            g_h1 = torch.einsum("bhmd,bhid->bhmi", g_pred, W2)
            c = math.sqrt(2.0 / math.pi)
            tanh_out = torch.tanh(c * (pre + 0.044715 * pre**3))
            gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * pre * (1.0 - tanh_out**2) * c * (
                1.0 + 3.0 * 0.044715 * pre**2
            )
            g_pre = g_h1 * gelu_grad
            g_b1 = g_pre.sum(dim=2, keepdim=True)  # [B,H,1,I]
            g_W1 = torch.einsum("bhmi,bhmd->bhdi", g_pre, ki)
            lr = etai * self.base_lr  # [B,H,1,1]
            W1 = W1 - lr * g_W1
            b1 = b1 - lr * g_b1
            W2 = W2 - lr * g_W2
            b2 = b2 - lr * g_b2

        qh = q.transpose(1, 2)  # [B,H,T,D]
        h1 = F.gelu(torch.einsum("bhtd,bhdi->bhti", qh, W1) + b1)
        o = torch.einsum("bhti,bhid->bhtd", h1, W2) + b2
        o = o.transpose(1, 2).reshape(B, T, self.key_dim).to(dtype=query_states.dtype)
        return self.o_proj(o)


def build_cross_memory_branch(dim: int, cfg: Optional[dict] = None) -> Optional[CrossMemoryBranch]:
    """Factory for ablation: cfg keys include memory_type / cross_memory_type."""
    cfg = dict(cfg or {})
    memory_type = cfg.pop("memory_type", cfg.pop("cross_memory_type", "none"))
    memory_type = (memory_type or "none").lower()
    if memory_type in ("", "none", "null", "disabled"):
        return None
    return CrossMemoryBranch(dim=dim, memory_type=memory_type, **cfg)
