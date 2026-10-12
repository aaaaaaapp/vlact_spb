# Copyright 2026 starVLA
"""Packing-compatible Ulysses SP for SpikeBrain-2.0-VL.

Eval SP pre-shards sequence to ``L/SP`` before the language model, then attention
all-to-all reconstructs full ``L`` with ``H/SP`` heads. Training packing used to
feed the *full* packed sequence into ``set_sp_group`` attentions, which breaks
``cu_seqlens`` (``L/SP`` vs ``L``).

This module installs a runtime wrap on ``language_model.forward`` that:
  1. Pads seq len to a multiple of SP
  2. Keeps only the local ``L/SP`` shard of embeds / position_ids / masks
  3. Leaves ``cu_seqlens`` as the *full* (padded) layout for post-a2a varlen ops
  4. All-gathers the LM hidden states back to full ``L`` for DiT / lm_head hook
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from starVLA.training.trainer_utils.sequence_parallel import SequenceParallelState


def _pad_seq_to_sp(tensor: torch.Tensor, sp_size: int, seq_dim: int) -> torch.Tensor:
    if seq_dim < 0:
        seq_dim = tensor.ndim + seq_dim
    if seq_dim < 0 or seq_dim >= tensor.ndim:
        raise ValueError(f"seq_dim={seq_dim} invalid for tensor ndim={tensor.ndim}")
    length = tensor.size(seq_dim)
    pad = (sp_size - (length % sp_size)) % sp_size
    if pad == 0:
        return tensor
    pad_shape = [0] * (2 * tensor.ndim)
    # F.pad pads from last dim; index for seq_dim from the end
    idx = 2 * (tensor.ndim - 1 - seq_dim)
    pad_shape[idx + 1] = pad
    return F.pad(tensor, pad_shape)


def _shard_seq(tensor: torch.Tensor, sp: SequenceParallelState, seq_dim: int) -> torch.Tensor:
    if seq_dim < 0:
        seq_dim = tensor.ndim + seq_dim
    length = tensor.size(seq_dim)
    chunk = length // sp.sp_size
    start = sp.sp_rank * chunk
    return tensor.narrow(seq_dim, start, chunk).contiguous()


def _all_gather_seq(local: torch.Tensor, sp: SequenceParallelState, seq_dim: int = 1) -> torch.Tensor:
    if not sp.enabled or sp.sp_group is None:
        return local
    parts = [torch.empty_like(local) for _ in range(sp.sp_size)]
    dist.all_gather(parts, local.contiguous(), group=sp.sp_group)
    return torch.cat(parts, dim=seq_dim)


def install_packing_ulysses_preshard(
    vl_root: torch.nn.Module,
    sp: SequenceParallelState,
) -> bool:
    """Wrap SpikeBrain ``language_model.forward`` for packing+Ulysses.

    Returns True if a patch was installed.
    """
    if vl_root is None or not sp.enabled:
        return False

    # AutoModelForCausalLM → .model (SPB2VLModel) → .language_model
    model = getattr(vl_root, "model", vl_root)
    lm = getattr(model, "language_model", None)
    if lm is None and hasattr(model, "model"):
        lm = getattr(model.model, "language_model", None)
    if lm is None or not hasattr(lm, "forward"):
        return False
    if getattr(lm, "_starvla_packing_ulysses_patched", False):
        lm._starvla_sp_state = sp
        return True

    orig_forward = lm.forward

    def _forward_with_preshard(*args, **kwargs):
        sp_state: SequenceParallelState = getattr(lm, "_starvla_sp_state", sp)
        if not sp_state.enabled:
            return orig_forward(*args, **kwargs)

        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            return orig_forward(*args, **kwargs)

        orig_len = int(embeds.shape[1])
        # Pad then shard sequence for Ulysses (eval-aligned).
        embeds = _pad_seq_to_sp(embeds, sp_state.sp_size, seq_dim=1)
        full_len = int(embeds.shape[1])
        kwargs["inputs_embeds"] = _shard_seq(embeds, sp_state, seq_dim=1)

        pos = kwargs.get("position_ids")
        if pos is not None:
            # (3, B, S) multimodal RoPE or (B, S)
            if pos.ndim == 3:
                pos = _pad_seq_to_sp(pos, sp_state.sp_size, seq_dim=2)
                kwargs["position_ids"] = _shard_seq(pos, sp_state, seq_dim=2)
            elif pos.ndim == 2:
                pos = _pad_seq_to_sp(pos, sp_state.sp_size, seq_dim=1)
                kwargs["position_ids"] = _shard_seq(pos, sp_state, seq_dim=1)

        # SP path requires no dense attention_mask (would rebuild wrong cu_seqlens).
        kwargs["attention_mask"] = None

        vmask = kwargs.get("visual_pos_masks")
        if vmask is not None:
            # Vision tokens already fused into embeds; drop deepstack + mask under SP
            # to avoid absolute-index / pad bugs on sharded sequences.
            kwargs["visual_pos_masks"] = None
            kwargs["deepstack_visual_embeds"] = None

        cu = kwargs.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device=embeds.device)
            if int(cu[-1].item()) != full_len:
                # Extend last boundary to padded full length (pad tokens = extra tail).
                cu = cu.clone()
                cu[-1] = full_len
            kwargs["cu_seqlens"] = cu

        cache_pos = kwargs.get("cache_position")
        if cache_pos is not None and cache_pos.ndim == 1:
            cache_pos = _pad_seq_to_sp(cache_pos.unsqueeze(0), sp_state.sp_size, seq_dim=1).squeeze(0)
            kwargs["cache_position"] = _shard_seq(cache_pos.unsqueeze(0), sp_state, seq_dim=1).squeeze(0)

        out = orig_forward(*args, **kwargs)

        # Gather full sequence then truncate pad so DiT cu_seqlens still matches.
        def _gather_hidden(h: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if h is None or not torch.is_tensor(h):
                return h
            if h.ndim < 2:
                return h
            h = _all_gather_seq(h, sp_state, seq_dim=1)
            if h.shape[1] > orig_len:
                h = h[:, :orig_len].contiguous()
            return h

        def _gather_out(obj):
            if obj is None:
                return None
            # ModelOutput is a tuple subclass — handle by attribute BEFORE iterating.
            if hasattr(obj, "last_hidden_state"):
                hs = getattr(obj, "last_hidden_state", None)
                if torch.is_tensor(hs) and hs.ndim >= 2:
                    try:
                        obj.last_hidden_state = _gather_hidden(hs)
                    except Exception:
                        pass
                return obj
            if isinstance(obj, tuple):
                if not obj:
                    return obj
                return (_gather_out(obj[0]),) + tuple(obj[1:])
            if torch.is_tensor(obj) and obj.ndim >= 2:
                return _gather_hidden(obj)
            return obj

        return _gather_out(out)

    lm.forward = _forward_with_preshard  # type: ignore[method-assign]
    lm._starvla_packing_ulysses_patched = True
    lm._starvla_sp_state = sp
    lm._starvla_packing_ulysses_orig_forward = orig_forward
    return True
