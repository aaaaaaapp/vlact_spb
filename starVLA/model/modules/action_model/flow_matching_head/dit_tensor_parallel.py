# Copyright 2026 starVLA
"""DiT Megatron-style Tensor Parallel (weight sharding), not sequence parallel.

Shards Attention Q/K/V (column) + out_proj (row) and FeedForward GEGLU/out
across a TP process group. Compatible with orthogonal DiT/VLM sequence-parallel
groups (do not reuse the SP group as the TP group).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _tp_meta(process_group) -> tuple[int, int]:
    if process_group is None or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(process_group), dist.get_world_size(process_group)


def _replace_param(module: nn.Module, name: str, tensor: torch.Tensor, *, requires_grad: bool) -> None:
    setattr(module, name, nn.Parameter(tensor, requires_grad=requires_grad))


def _column_shard_linear(linear: nn.Linear, tp_rank: int, tp_size: int) -> None:
    """Shard ``Linear`` output dim (weight rows)."""
    out_f, in_f = linear.weight.shape
    if out_f % tp_size != 0:
        raise ValueError(f"column-parallel out_features={out_f} not divisible by tp_size={tp_size}")
    w = linear.weight.data.chunk(tp_size, dim=0)[tp_rank].contiguous().clone()
    _replace_param(linear, "weight", w, requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        b = linear.bias.data.chunk(tp_size, dim=0)[tp_rank].contiguous().clone()
        _replace_param(linear, "bias", b, requires_grad=linear.bias.requires_grad)
    linear.out_features = w.shape[0]
    linear.in_features = in_f


def _column_shard_geglu_proj(linear: nn.Linear, tp_rank: int, tp_size: int) -> None:
    """Shard GEGLU ``proj`` keeping [hidden|gate] layout valid after chunk(2)."""
    out_f, in_f = linear.weight.shape
    if out_f % 2 != 0:
        raise ValueError(f"GEGLU proj out_features={out_f} must be even")
    half = out_f // 2
    if half % tp_size != 0:
        raise ValueError(f"GEGLU half={half} not divisible by tp_size={tp_size}")
    w_h, w_g = linear.weight.data[:half], linear.weight.data[half:]
    w = torch.cat(
        [
            w_h.chunk(tp_size, dim=0)[tp_rank],
            w_g.chunk(tp_size, dim=0)[tp_rank],
        ],
        dim=0,
    ).contiguous().clone()
    _replace_param(linear, "weight", w, requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        b_h, b_g = linear.bias.data[:half], linear.bias.data[half:]
        b = torch.cat(
            [
                b_h.chunk(tp_size, dim=0)[tp_rank],
                b_g.chunk(tp_size, dim=0)[tp_rank],
            ],
            dim=0,
        ).contiguous().clone()
        _replace_param(linear, "bias", b, requires_grad=linear.bias.requires_grad)
    linear.out_features = w.shape[0]
    linear.in_features = in_f


def _row_shard_linear(linear: nn.Linear, tp_rank: int, tp_size: int) -> None:
    """Shard ``Linear`` input dim (weight cols). Bias kept only on rank 0."""
    out_f, in_f = linear.weight.shape
    if in_f % tp_size != 0:
        raise ValueError(f"row-parallel in_features={in_f} not divisible by tp_size={tp_size}")
    w = linear.weight.data.chunk(tp_size, dim=1)[tp_rank].contiguous().clone()
    _replace_param(linear, "weight", w, requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        if tp_rank == 0:
            b = linear.bias.data.contiguous().clone()
            _replace_param(linear, "bias", b, requires_grad=linear.bias.requires_grad)
        else:
            # Keep a zero bias Parameter so optimizer/DDP param lists stay aligned in shape
            # across TP ranks for modules that still iterate named_parameters naively.
            # RowParallelLinear.forward ignores non-zero-rank bias.
            b = torch.zeros_like(linear.bias.data)
            _replace_param(linear, "bias", b, requires_grad=False)
            linear._tp_bias_rank0_only = True  # type: ignore[attr-defined]
    linear.out_features = out_f
    linear.in_features = w.shape[1]


class _RowParallelLinear(nn.Module):
    """Thin wrapper: local matmul + all-reduce over TP group."""

    def __init__(self, linear: nn.Linear, process_group):
        super().__init__()
        self.linear = linear
        self.process_group = process_group
        self._tp_rank, self._tp_size = _tp_meta(process_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.linear.bias if self._tp_rank == 0 else None
        out = F.linear(x, self.linear.weight, bias)
        if self._tp_size > 1 and self.process_group is not None:
            # Must be autograd-aware so row-parallel output gets correct grads.
            if out.requires_grad:
                from torch.distributed.nn.functional import all_reduce as diff_all_reduce

                out = diff_all_reduce(out, group=self.process_group)
            else:
                dist.all_reduce(out, group=self.process_group)
        return out


def _patch_attention_tp(attn: nn.Module, process_group) -> None:
    tp_rank, tp_size = _tp_meta(process_group)
    if tp_size <= 1:
        return
    heads = int(getattr(attn, "heads", 0) or 0)
    if heads <= 0 or heads % tp_size != 0:
        raise ValueError(f"Attention heads={heads} must be >0 and divisible by tp_size={tp_size}")

    _column_shard_linear(attn.to_q, tp_rank, tp_size)
    _column_shard_linear(attn.to_k, tp_rank, tp_size)
    _column_shard_linear(attn.to_v, tp_rank, tp_size)
    _row_shard_linear(attn.to_out[0], tp_rank, tp_size)
    attn.to_out[0] = _RowParallelLinear(attn.to_out[0], process_group)
    attn.heads = heads // tp_size
    if hasattr(attn, "inner_dim"):
        attn.inner_dim = int(attn.to_q.out_features)


def _patch_ff_tp(ff: nn.Module, process_group) -> None:
    tp_rank, tp_size = _tp_meta(process_group)
    if tp_size <= 1:
        return
    # diffusers FeedForward: net[0]=GELU/GEGLU(.proj), net[2]=Linear out
    act = ff.net[0]
    if not hasattr(act, "proj"):
        raise TypeError(f"Expected FF activation with .proj, got {type(act)}")
    # GEGLU packs [hidden|gate]; plain GELU is a normal column-parallel expand.
    is_geglu = act.__class__.__name__.upper().startswith("GEGLU") or (
        act.proj.out_features == 2 * ff.net[2].in_features
    )
    if is_geglu:
        _column_shard_geglu_proj(act.proj, tp_rank, tp_size)
    else:
        _column_shard_linear(act.proj, tp_rank, tp_size)
    _row_shard_linear(ff.net[2], tp_rank, tp_size)
    ff.net[2] = _RowParallelLinear(ff.net[2], process_group)


def install_dit_tensor_parallel(dit: nn.Module, process_group) -> int:
    """Shard DiT block Attention + FeedForward weights for TP. Returns #blocks patched."""
    _, tp_size = _tp_meta(process_group)
    if tp_size <= 1:
        return 0
    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("DiT has no transformer_blocks")
    n = 0
    for block in blocks:
        if hasattr(block, "attn1"):
            _patch_attention_tp(block.attn1, process_group)
        if hasattr(block, "ff"):
            _patch_ff_tp(block.ff, process_group)
        n += 1
        setattr(block, "_dit_tp_size", tp_size)
    setattr(dit, "_dit_tp_size", tp_size)
    setattr(dit, "_dit_tp_group", process_group)
    return n


def resolve_dit_tp_degree(cfg=None, default: int = 1) -> int:
    import os

    env = (os.environ.get("STARVLA_DIT_TP_DEGREE") or "").strip()
    if env:
        return max(1, int(env))
    flag = (os.environ.get("STARVLA_DIT_TP") or "").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return 1
    if cfg is None:
        return max(1, int(default))
    action = getattr(getattr(cfg, "framework", None), "action_model", None)
    if action is None:
        return max(1, int(default))
    if hasattr(action, "get"):
        return max(1, int(action.get("dit_tp_degree", default) or default))
    return max(1, int(getattr(action, "dit_tp_degree", default) or default))
