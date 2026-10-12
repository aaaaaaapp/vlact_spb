# Copyright 2026
# Liger-Kernel patches for SpikeBrain-2.0-VL (SPB2), aligned with lmms-engine qwen3_vl path.
# VLA training uses backbone forward without LM CE — fused_linear_cross_entropy is off by default.

from __future__ import annotations

import importlib
from typing import Optional

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from transformers import PreTrainedModel

logger = get_logger(__name__)

_LIGER_PATCHED_MODEL_IDS: set[int] = set()

_LIGER_AVAILABLE = False
_LIGER_IMPORT_ERROR: Optional[Exception] = None

try:
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    _LIGER_AVAILABLE = True
except Exception as e:
    _LIGER_IMPORT_ERROR = e


def _import_sibling_modules(modeling_module_name: str) -> list:
    """Import sse_swa / moba_attn / attn from the same HF remote-code package."""
    if "." not in modeling_module_name:
        return []
    pkg = modeling_module_name.rsplit(".", 1)[0]
    mods = []
    for sub in ("sse_swa", "moba_attn", "attn"):
        try:
            mods.append(importlib.import_module(f"{pkg}.{sub}"))
        except ImportError:
            pass
    return mods


def _resolve_spb2_inner(model: PreTrainedModel):
    """Return (language_model, visual) from SPB2VLForConditionalGeneration or PEFT wrap."""
    target = model
    if hasattr(target, "get_base_model"):
        target = target.get_base_model()
    inner = getattr(target, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model, getattr(inner, "visual", None)
    if hasattr(target, "language_model"):
        return target.language_model, getattr(target, "visual", None)
    return None, None


def _spb2_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """SPB2 uses split-half RoPE, not Liger's generic rotary layout."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _spb2_text_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """SPB2 text RoPE preserving the remote-code unsqueeze_dim contract."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_spb2_rotate_half(q) * sin), (k * cos) + (_spb2_rotate_half(k) * sin)


def _spb2_seq_first_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """SPB2 sibling attention RoPE: q/k are [batch, seq, heads, head_dim]."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (_spb2_rotate_half(q) * sin), (k * cos) + (_spb2_rotate_half(k) * sin)


def apply_liger_kernel_to_spb2_vl(
    model: Optional[PreTrainedModel] = None,
    *,
    rope: bool = True,
    rms_norm: bool = False,
    swiglu: bool = True,
    fused_linear_cross_entropy: bool = False,
) -> bool:
    """
    Apply Liger kernels to SPB2-VL (custom modeling_spb2_vl module).

    Default: only non-sensitive fused ops (RoPE, SwiGLU). Text/attn RMSNorm stays on native
    fp32-stable implementations — do not set ``rms_norm=True`` unless you know the risk.

    For starVLA / QwenGR00T VLA, keep fused_linear_cross_entropy=False (no LM loss on backbone).

    Returns:
        True if patches were applied, False if liger-kernel is missing or model is None.
    """
    if not _LIGER_AVAILABLE:
        logger.warning(
            "[SpikeBrain] liger-kernel not installed; skip Liger patch. "
            f"Install with: pip install liger-kernel  ({_LIGER_IMPORT_ERROR})"
        )
        return False

    if model is None:
        logger.warning("[SpikeBrain] apply_liger_kernel_to_spb2_vl: model is None, skip")
        return False

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    model_key = id(base)
    if model_key in _LIGER_PATCHED_MODEL_IDS:
        logger.info("[SpikeBrain] Liger kernel already applied on this SPB2-VL instance, skip")
        return True

    modeling_mod_name = type(model).__module__
    try:
        modeling_spb2 = importlib.import_module(modeling_mod_name)
    except Exception as e:
        logger.warning(f"[SpikeBrain] cannot import modeling module {modeling_mod_name}: {e}")
        return False

    if rope:
        modeling_spb2.apply_rotary_pos_emb = _spb2_text_rotary_pos_emb
        for sib in _import_sibling_modules(modeling_mod_name):
            if hasattr(sib, "apply_rotary_pos_emb"):
                sib.apply_rotary_pos_emb = _spb2_seq_first_rotary_pos_emb

    # Sensitive: SPB2VLTextRMSNorm / decoder norms / attn q_norm,k_norm — never replace with Liger.
    if rms_norm:
        logger.warning(
            "[SpikeBrain] liger rms_norm=True replaces SPB2VLTextRMSNorm; "
            "not recommended for full VLM fine-tune stability."
        )
        if hasattr(modeling_spb2, "SPB2VLTextRMSNorm"):
            modeling_spb2.SPB2VLTextRMSNorm = LigerRMSNorm

    if swiglu and hasattr(modeling_spb2, "SPB2VLTextMLP"):
        modeling_spb2.SPB2VLTextMLP = LigerSwiGLUMLP

    if fused_linear_cross_entropy:
        logger.warning(
            "[SpikeBrain] fused_linear_cross_entropy=True is not wired for SPB2-VL in starVLA; "
            "ignored (VLA uses backbone features only)."
        )

    text_model, vision_model = _resolve_spb2_inner(model)

    if rms_norm and vision_model is not None and hasattr(vision_model, "blocks"):
        for block in vision_model.blocks:
            if hasattr(block, "norm1"):
                _patch_rms_norm_module(block.norm1)
            if hasattr(block, "norm2"):
                _patch_rms_norm_module(block.norm2)

    if text_model is not None and hasattr(text_model, "layers"):
        for decoder_layer in text_model.layers:
            if swiglu and hasattr(decoder_layer, "mlp"):
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)

    n_layers = len(text_model.layers) if text_model is not None and hasattr(text_model, "layers") else 0
    _LIGER_PATCHED_MODEL_IDS.add(model_key)
    logger.info(
        "[SpikeBrain] Liger kernel ENABLED on SPB2-VL (non-sensitive only: "
        f"rope={rope} (SPB2-native-compatible), rms_norm={rms_norm}, "
        f"swiglu={swiglu}, text_layers={n_layers})"
    )
    return True
