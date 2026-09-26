"""Numerically guarded Liger kernels for SpikeBrain-2.0-VL.

SpikeBrain builds interleaved T/H/W MRoPE frequencies before attention.  The
patches below consume those already-built cos/sin tensors; they do not reuse
Qwen's position-id or frequency construction.
"""

from __future__ import annotations

import inspect
import logging
import sys
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_LIGER_AVAILABLE = False
_LIGER_IMPORT_ERROR: Optional[Exception] = None

try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
    from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    _LIGER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the runtime image
    _LIGER_IMPORT_ERROR = exc


def _cfg_bool(cfg, name: str, default: bool) -> bool:
    if cfg is None:
        return default
    value = cfg.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _bind(module, name, function):
    setattr(module, name, MethodType(function, module))


def spikebrain_rope_head_first(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Liger RoPE for SpikeBrain tensors laid out as [batch, heads, seq, dim]."""
    return liger_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)


def spikebrain_rope_seq_first(q, k, cos, sin):
    """Liger RoPE for custom SpikeBrain attention's [batch, seq, heads, dim]."""
    q_out, k_out = liger_rotary_pos_emb(
        q.transpose(1, 2),
        k.transpose(1, 2),
        cos,
        sin,
    )
    return q_out.transpose(1, 2), k_out.transpose(1, 2)


def _native_rope_head_first(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        left, right = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-right, left), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _interleaved_mrope_cos_sin(device, dtype, batch=2, seq=17, head_dim=128):
    """Build cos/sin with SpikeBrain's [T,H,W] interleaving and split-half layout."""
    torch.manual_seed(17)
    position_ids = torch.randint(0, 512, (3, batch, seq), device=device)
    inv_freq = 1.0 / (
        5_000_000.0
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = (
        inv_freq[None, None, :, None].expand(3, batch, -1, 1)
        @ position_ids[:, :, None, :].float()
    ).transpose(2, 3)
    sections = (24, 20, 20)
    interleaved = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        interleaved[..., slice(offset, sections[dim] * 3, 3)] = freqs[
            dim, ..., slice(offset, sections[dim] * 3, 3)
        ]
    embedding = torch.cat((interleaved, interleaved), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _tensor_error(reference, actual):
    error = (reference.float() - actual.float()).abs()
    return {"max": error.max().item(), "mean": error.mean().item()}


@torch.no_grad()
def _check_rope(device, dtype):
    torch.manual_seed(19)
    batch, seq, q_heads, kv_heads, head_dim = 2, 17, 4, 2, 128
    cos, sin = _interleaved_mrope_cos_sin(device, dtype, batch, seq, head_dim)
    q = torch.randn(batch, q_heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq, head_dim, device=device, dtype=dtype)
    q_ref, k_ref = _native_rope_head_first(q, k, cos, sin)
    q_liger, k_liger = spikebrain_rope_head_first(q.clone(), k.clone(), cos, sin)

    q_seq, k_seq = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
    q_seq_liger, k_seq_liger = spikebrain_rope_seq_first(
        q_seq.clone(), k_seq.clone(), cos, sin
    )
    return {
        "head_q": _tensor_error(q_ref, q_liger),
        "head_k": _tensor_error(k_ref, k_liger),
        "seq_q": _tensor_error(q_ref.transpose(1, 2), q_seq_liger),
        "seq_k": _tensor_error(k_ref.transpose(1, 2), k_seq_liger),
        "interleaved_layout": True,
    }


def _check_rms_norm(device, dtype, hidden_size=2560, eps=1e-6):
    torch.manual_seed(23)
    x_ref = torch.randn(
        2, 17, hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    weight_ref = torch.randn(
        hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    upstream = torch.randn_like(x_ref)
    variance = x_ref.float().pow(2).mean(-1, keepdim=True)
    reference = weight_ref * (x_ref.float() * torch.rsqrt(variance + eps)).to(dtype)
    reference.backward(upstream)

    x_liger = x_ref.detach().clone().requires_grad_(True)
    liger = LigerRMSNorm(
        hidden_size,
        eps=eps,
        offset=0.0,
        casting_mode="llama",
        in_place=True,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        liger.weight.copy_(weight_ref.detach())
    actual = liger(x_liger)
    actual.backward(upstream)
    return {
        "forward": _tensor_error(reference.detach(), actual.detach()),
        "input_grad": _tensor_error(x_ref.grad, x_liger.grad),
        "weight_grad": _tensor_error(weight_ref.grad, liger.weight.grad),
    }


def _check_swiglu(device, dtype):
    torch.manual_seed(29)
    gate_ref = torch.randn(
        2, 17, 1024, device=device, dtype=dtype, requires_grad=True
    )
    up_ref = torch.randn(
        2, 17, 1024, device=device, dtype=dtype, requires_grad=True
    )
    upstream = torch.randn_like(gate_ref)
    reference = F.silu(gate_ref) * up_ref
    reference.backward(upstream)

    gate_liger = gate_ref.detach().clone().requires_grad_(True)
    up_liger = up_ref.detach().clone().requires_grad_(True)
    actual = LigerSiLUMulFunction.apply(gate_liger, up_liger)
    actual.backward(upstream)
    return {
        "forward": _tensor_error(reference.detach(), actual.detach()),
        "gate_grad": _tensor_error(gate_ref.grad, gate_liger.grad),
        "up_grad": _tensor_error(up_ref.grad, up_liger.grad),
    }


@torch.no_grad()
def _check_fused_ce(device, dtype, vocab=4096, hidden_size=256):
    torch.manual_seed(31)
    batch, seq = 2, 31
    hidden = torch.randn(batch, seq, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(vocab, hidden_size, device=device, dtype=dtype)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    labels[:, :5] = -100
    logits = F.linear(hidden, weight)
    # Match SpikeBrain's native path exactly: CrossEntropyLoss receives BF16
    # logits directly. Qwen's HF loss upcasts logits, but SpikeBrain does not.
    reference = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, vocab),
        labels[..., 1:].contiguous().view(-1),
    )
    result = LigerForCausalLMLoss(
        hidden_states=hidden,
        lm_head_weight=weight,
        labels=labels,
        hidden_size=hidden_size,
    )
    actual, _, _ = unpack_cross_entropy_result(result)
    absolute = (reference - actual.float()).abs().item()
    return {
        "reference": reference.item(),
        "liger": actual.float().item(),
        "abs": absolute,
        "rel": absolute / max(reference.abs().item(), 1e-12),
    }


def _patch_rms_norm_module(module):
    module.offset = 0.0
    module.casting_mode = "llama"
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(
        module, "eps", 1e-6
    )
    module.in_place = True
    module.row_mode = None
    _bind(module, "forward", LigerRMSNorm.forward)
    _bind(module, "extra_repr", LigerRMSNorm.extra_repr)


def _text_model(model):
    backbone = getattr(model, "model", None)
    text = getattr(backbone, "language_model", None)
    if text is None:
        raise RuntimeError("SpikeBrain language_model not found")
    return text


def _patch_rms_norms(model):
    text = _text_model(model)
    candidates = [text.norm]
    for layer in text.layers:
        candidates.extend((layer.input_layernorm, layer.post_attention_layernorm))
        for child in layer.self_attn.modules():
            if type(child).__name__ == "RMSNorm":
                candidates.append(child)

    patched = 0
    seen = set()
    for module in candidates:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if getattr(module, "bias", None) is not None:
            logger.warning("[SpikeBrain] skip biased RMSNorm %s", type(module).__name__)
            continue
        _patch_rms_norm_module(module)
        patched += 1
    return patched


def _patch_swiglu(model):
    patched = 0
    for layer in _text_model(model).layers:
        mlp = layer.mlp
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            continue
        _bind(mlp, "forward", LigerSwiGLUMLP.forward)
        patched += 1
    return patched


def _patch_rope(model):
    patched_modules = set()
    modeling_module = inspect.getmodule(type(model))
    if modeling_module is None:
        raise RuntimeError("SpikeBrain modeling module not found")
    modeling_module.apply_rotary_pos_emb = spikebrain_rope_head_first
    patched_modules.add(modeling_module.__name__)

    for layer in _text_model(model).layers:
        module_name = type(layer.self_attn).__module__
        attention_module = sys.modules.get(module_name)
        if attention_module is None or not hasattr(attention_module, "apply_rotary_pos_emb"):
            continue
        attention_module.apply_rotary_pos_emb = spikebrain_rope_seq_first
        patched_modules.add(module_name)
    return sorted(patched_modules)


def _spikebrain_fused_ce_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    cache_position=None,
    logits_to_keep=0,
    **kwargs,
):
    cu_seqlens = kwargs.get("cu_seqlens")
    if labels is None or cu_seqlens is not None or not self.training:
        return self._vlact_native_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs[0]
    result = LigerForCausalLMLoss(
        hidden_states=hidden_states,
        lm_head_weight=self.lm_head.weight,
        labels=labels,
        hidden_size=self.config.text_config.hidden_size,
    )
    loss, _, _ = unpack_cross_entropy_result(result)

    aux_loss = None
    if outputs.aux_losses is not None:
        compute_device = outputs.aux_losses[0].device
        aux_loss = sum(value.to(compute_device) for value in outputs.aux_losses)
        loss = loss + self.aux_loss_coef * aux_loss.to(loss.device)

    output_class = getattr(inspect.getmodule(type(self)), "SPB2VLCausalLMOutputWithPast")
    return output_class(
        loss=loss,
        aux_loss=aux_loss,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=(hidden_states,),
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )


def _patch_fused_ce(model):
    model._vlact_native_forward = model.forward
    _bind(model, "forward", _spikebrain_fused_ce_forward)


def apply_liger_kernel_to_spikebrain(
    model,
    *,
    rope=True,
    fused_linear_cross_entropy=False,
    rms_norm=False,
    swiglu=False,
    skip_checks=False,
):
    requested = {
        "rope": bool(rope),
        "fused_linear_cross_entropy": bool(fused_linear_cross_entropy),
        "rms_norm": bool(rms_norm),
        "swiglu": bool(swiglu),
    }
    status = {
        "available": _LIGER_AVAILABLE,
        "rope": False,
        "fused_linear_cross_entropy": False,
        "rms_norm": False,
        "swiglu": False,
        "rope_check": None,
        "lm_head_check": None,
        "rms_check": None,
        "swiglu_check": None,
        "error": None,
    }
    if not _LIGER_AVAILABLE:
        status["error"] = f"liger-kernel not installed ({_LIGER_IMPORT_ERROR})"
        logger.warning("[SpikeBrain] %s", status["error"])
        return status

    parameter = next(model.parameters())
    dtype = parameter.dtype
    device = parameter.device
    check_device = device if device.type == "cuda" else (
        torch.device("cuda") if torch.cuda.is_available() else device
    )
    if check_device.type != "cuda":
        skip_checks = True
        logger.warning("[SpikeBrain] Liger numerical checks skipped because CUDA is unavailable")

    if not skip_checks:
        if rope:
            try:
                status["rope_check"] = _check_rope(check_device, dtype)
                rope = all(
                    status["rope_check"][key]["max"] == 0.0
                    for key in ("head_q", "head_k", "seq_q", "seq_k")
                )
            except Exception as exc:
                status["rope_check"] = {"error": str(exc)}
                rope = False
        if rms_norm:
            try:
                status["rms_check"] = _check_rms_norm(check_device, dtype)
                rms_norm = all(
                    metrics["max"] == 0.0
                    for metrics in status["rms_check"].values()
                )
            except Exception as exc:
                status["rms_check"] = {"error": str(exc)}
                rms_norm = False
        if swiglu:
            try:
                status["swiglu_check"] = _check_swiglu(check_device, dtype)
                swiglu = all(
                    metrics["max"] == 0.0
                    for metrics in status["swiglu_check"].values()
                )
            except Exception as exc:
                status["swiglu_check"] = {"error": str(exc)}
                swiglu = False
        if fused_linear_cross_entropy:
            try:
                status["lm_head_check"] = _check_fused_ce(check_device, dtype)
                fused_linear_cross_entropy = status["lm_head_check"]["rel"] <= 1e-5
            except Exception as exc:
                status["lm_head_check"] = {"error": str(exc)}
                fused_linear_cross_entropy = False

    if rope:
        status["rope_modules"] = _patch_rope(model)
        status["rope"] = True
    if rms_norm:
        status["rms_norm_count"] = _patch_rms_norms(model)
        status["rms_norm"] = True
    if swiglu:
        status["swiglu_count"] = _patch_swiglu(model)
        status["swiglu"] = True
    if fused_linear_cross_entropy:
        _patch_fused_ce(model)
        status["fused_linear_cross_entropy"] = True

    for name in ("rope", "rms_norm", "swiglu", "fused_linear_cross_entropy"):
        if requested[name] and not status[name]:
            logger.warning("[SpikeBrain] Liger %s disabled after numerical guard", name)
    return status


def apply_spikebrain_liger_from_config(model, config):
    if not _cfg_bool(config, "use_liger_kernel", False):
        logger.info("[SpikeBrain] Liger disabled (framework.qwenvl.use_liger_kernel=false)")
        return {"enabled": False}
    return apply_liger_kernel_to_spikebrain(
        model,
        rope=_cfg_bool(config, "liger_rope", True),
        fused_linear_cross_entropy=_cfg_bool(
            config, "liger_fused_linear_cross_entropy", False
        ),
        rms_norm=_cfg_bool(config, "liger_rms_norm", False),
        swiglu=_cfg_bool(config, "liger_swiglu", False),
    )
