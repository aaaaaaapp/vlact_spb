# Copyright 2026 starVLA community.
"""DeepSpeed-path numerical patches aligned with lmms-engine SPB2-VL SFT (no FSDP)."""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
from accelerate.logging import get_logger

from starVLA.model.modules.vlm.spikebrain import (
    ATTN_RMSNORM_CLASS,
    GDN_FP32_MODULE_NAME,
    SPB2_TEXT_RMSNORM_CLASS,
    _is_fp32_sensitive_module,
)

logger = get_logger(__name__)

_SSE_LAYER_CLASSES = frozenset({"SSEGLAH", "SSEGDNH", "SSEGLA", "SSEGDN"})


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _attn_rmsnorm_fp32_output_forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
    if hasattr(self, "variance_epsilon"):
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return self.weight.float() * x
    orig = getattr(self, "_spb2_orig_rmsnorm_forward", None)
    if orig is None:
        raise RuntimeError("Missing _spb2_orig_rmsnorm_forward on fla RMSNorm")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        out = orig(hidden_states, *args, **kwargs)
    if torch.is_tensor(out):
        return out.float()
    if isinstance(out, tuple):
        return tuple(t.float() if torch.is_tensor(t) and t.is_floating_point() else t for t in out)
    return out


def _generic_fp32_autocast_forward(orig_forward):
    def forward(self, *args, **kwargs):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            out = orig_forward(self, *args, **kwargs)
        if torch.is_tensor(out):
            return out.float()
        if isinstance(out, tuple):
            return tuple(t.float() if torch.is_tensor(t) and t.is_floating_point() else t for t in out)
        return out

    return forward


def patch_att_rmsnorm_fp32_output(model: nn.Module, *, log: bool = True) -> int:
    """
    Keep attn-path RMSNorm outputs in fp32 (lmms FSDP ``output_dtype=fp32`` equivalent).

    Does not touch decoder ``input_layernorm`` / ``post_attention_layernorm`` (SPB2VLTextRMSNorm).
    """
    count = 0
    for name, module in model.named_modules():
        if not _is_fp32_sensitive_module(name, module):
            continue
        cls = module.__class__.__name__
        if cls == ATTN_RMSNORM_CLASS:
            if getattr(module, "_spb2_fp32_output_patched", False):
                continue
            if not hasattr(module, "_spb2_orig_rmsnorm_forward"):
                module._spb2_orig_rmsnorm_forward = module.forward
            module.forward = types.MethodType(_attn_rmsnorm_fp32_output_forward, module)
            module._spb2_fp32_output_patched = True
            count += 1
        elif cls == "FusedRMSNormGated":
            if getattr(module, "_spb2_fp32_output_patched", False):
                continue
            orig = module.forward
            module.forward = types.MethodType(_generic_fp32_autocast_forward(orig), module)
            module._spb2_fp32_output_patched = True
            count += 1
    if log:
        logger.info("[SpikeBrain] attn RMSNorm fp32-output patch applied to %d module(s)", count)
    return count


def patch_sse_fp32_island(model: nn.Module, *, log: bool = True) -> int:
    """
    Optional: run SSE linear-attention kernels under fp32 autocast island.
    SPB2-VL SFT keeps SSE in bf16; enable only for experiments.
    """
    count = 0
    for module in model.modules():
        if module.__class__.__name__ not in _SSE_LAYER_CLASSES:
            continue
        for method_name in ("sse_linear_attention_varlen", "sse_linear_attention_mask"):
            if not hasattr(module, method_name):
                continue
            attr = f"_spb2_fp32_island_{method_name}"
            if getattr(module, attr, False):
                continue
            orig = getattr(module, method_name)

            def _make_wrapped(original):
                def wrapped(*args, **kwargs):
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.autocast(device_type=device_type, enabled=False):
                        f_args = tuple(
                            a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args
                        )
                        f_kwargs = {
                            k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                            for k, v in kwargs.items()
                        }
                        out = original(*f_args, **f_kwargs)
                    if isinstance(out, tuple):
                        return tuple(
                            t.to(torch.bfloat16)
                            if torch.is_tensor(t) and t.is_floating_point()
                            else t
                            for t in out
                        )
                    if torch.is_tensor(out) and out.is_floating_point():
                        return out.to(torch.bfloat16)
                    return out

                return wrapped

            setattr(module, method_name, _make_wrapped(orig))
            setattr(module, attr, True)
            count += 1
    if log:
        logger.info("[SpikeBrain] SSE fp32-island patch applied to %d method(s)", count)
    return count


def apply_spb2_numerical_patches(model: nn.Module, sb_cfg: Any = None, *, log: bool = True) -> dict:
    """Apply all enabled SPB2 numerical stability patches to a loaded VLM."""
    sb_cfg = sb_cfg or {}
    stats = {}
    if _cfg_get(sb_cfg, "rmsnorm_fp32_forward", True):
        stats["rmsnorm_fp32"] = patch_att_rmsnorm_fp32_output(model, log=log)
    if _cfg_get(sb_cfg, "sse_fp32_island", False):
        stats["sse_fp32_island"] = patch_sse_fp32_island(model, log=log)
    return stats


def verify_fp32_sensitive_layers(
    model: nn.Module, *, log: bool = True, require_norm_fp32: bool = True
) -> bool:
    """Re-check GDN / attn RMSNorm dtypes after DeepSpeed/FSDP ``accelerator.prepare()``.

    Under FSDP1, attn RMSNorm weights stay bf16 inside the decoder FlatParam; set
    ``require_norm_fp32=False`` and rely on ``rmsnorm_fp32_forward`` patches.
    """
    gdn_issues: list[str] = []
    norm_issues: list[str] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls == GDN_FP32_MODULE_NAME:
            for pname, param in module.named_parameters(recurse=False):
                if param.dtype != torch.float32:
                    gdn_issues.append(f"{name}.{pname}={param.dtype}")
        elif cls in (ATTN_RMSNORM_CLASS, "FusedRMSNormGated") and _is_fp32_sensitive_module(name, module):
            for pname, param in module.named_parameters(recurse=False):
                if param.dtype != torch.float32:
                    norm_issues.append(f"{name}.{pname}={param.dtype}")
        elif cls == SPB2_TEXT_RMSNORM_CLASS:
            continue

    ok = not gdn_issues and (not require_norm_fp32 or not norm_issues)
    if log:
        if ok and (not norm_issues or not require_norm_fp32):
            if norm_issues and not require_norm_fp32:
                logger.info(
                    "[SpikeBrain] verify_fp32_sensitive_layers: GDN OK; "
                    "attn RMSNorm kept bf16 under FSDP1 (%d modules)",
                    len(norm_issues),
                )
            else:
                logger.info("[SpikeBrain] verify_fp32_sensitive_layers: OK")
        else:
            if gdn_issues:
                logger.warning(
                    "[SpikeBrain] GDN params not fp32 after prepare: %s",
                    ", ".join(gdn_issues[:8]) + (" ..." if len(gdn_issues) > 8 else ""),
                )
            if norm_issues and require_norm_fp32:
                logger.warning(
                    "[SpikeBrain] attn RMSNorm params not fp32 after prepare: %s",
                    ", ".join(norm_issues[:8]) + (" ..." if len(norm_issues) > 8 else ""),
                )
    return ok
