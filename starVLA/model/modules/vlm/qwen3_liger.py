# Liger-Kernel patches for HuggingFace Qwen3-VL.
# Text RoPE is Qwen split-half (cat(freqs, freqs)); vision RoPE is NOT patched
# because q/k are [seq, heads, dim] and Liger RoPE expects [bs, heads, seq, dim].

from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from transformers import PreTrainedModel
from transformers.models.qwen3_vl import modeling_qwen3_vl

logger = get_logger(__name__)

_LIGER_AVAILABLE = False
_LIGER_IMPORT_ERROR: Optional[Exception] = None
_ORIGINAL_TEXT_ROPE = modeling_qwen3_vl.apply_rotary_pos_emb

try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
    from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
    from liger_kernel.transformers.model.output_classes import LigerQwen3VLCausalLMOutputWithPast
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    _LIGER_AVAILABLE = True
except Exception as e:
    _LIGER_IMPORT_ERROR = e


def _bind(module, name, fn):
    setattr(module, name, MethodType(fn, module))


def _patch_rms_norm_module(module, *, offset=0.0, eps=1e-6, casting_mode="llama", in_place=True):
    module.offset = offset
    module.casting_mode = casting_mode
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    module.in_place = in_place
    module.row_mode = None
    _bind(module, "forward", LigerRMSNorm.forward)
    _bind(module, "extra_repr", LigerRMSNorm.extra_repr)


def _patch_text_rms_norm(model: PreTrainedModel) -> int:
    text = getattr(model, "language_model", None) or getattr(getattr(model, "model", None), "language_model", None)
    if text is None:
        raise RuntimeError("Qwen3-VL language_model not found")
    n = 0
    _patch_rms_norm_module(text.norm)
    n += 1
    for layer in text.layers:
        _patch_rms_norm_module(layer.input_layernorm)
        _patch_rms_norm_module(layer.post_attention_layernorm)
        n += 2
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            if getattr(attn, "q_norm", None) is not None:
                _patch_rms_norm_module(attn.q_norm)
                n += 1
            if getattr(attn, "k_norm", None) is not None:
                _patch_rms_norm_module(attn.k_norm)
                n += 1
    return n


def _patch_text_swiglu(model: PreTrainedModel) -> int:
    text = getattr(model, "language_model", None) or getattr(getattr(model, "model", None), "language_model", None)
    if text is None:
        raise RuntimeError("Qwen3-VL language_model not found")
    n = 0
    for layer in text.layers:
        _bind(layer.mlp, "forward", LigerSwiGLUMLP.forward)
        n += 1
    return n


def _cfg_bool(cfg, name: str, default: bool) -> bool:
    if cfg is None:
        return default
    val = cfg.get(name, default)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def qwen_split_half_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Reference Qwen3-VL / GPT-J split-half RoPE (same formula as modeling_qwen3_vl)."""
    return _ORIGINAL_TEXT_ROPE(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)


def _qwen3_vl_fused_lm_head_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep=0,
    skip_logits: Optional[bool] = None,
    **kwargs,
):
    """HF Qwen3-VL forward + Liger fused linear CE when labels are present (no S×vocab logits)."""
    output_attentions = kwargs.pop("output_attentions", None)
    output_hidden_states = kwargs.pop("output_hidden_states", None)
    return_dict = kwargs.pop("return_dict", None)
    use_cache = kwargs.pop("use_cache", None)
    shift_labels = kwargs.pop("shift_labels", None)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None
    token_accuracy = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
        )
        loss, _, token_accuracy = unpack_cross_entropy_result(result)
    else:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = (loss,) + output if loss is not None else output
        return output

    out_cls = LigerQwen3VLCausalLMOutputWithPast if LigerQwen3VLCausalLMOutputWithPast is not None else type(outputs)
    return out_cls(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        rope_deltas=getattr(outputs, "rope_deltas", None),
        **({"token_accuracy": token_accuracy} if LigerQwen3VLCausalLMOutputWithPast is not None else {}),
    )


def _qwen_split_half_closed_form(q, k, cos, sin):
    """Qwen3-VL text RoPE: rotate_half + cat(freqs,freqs) → y=[x1,x2]*[c,c]+[-x2,x1]*[s,s]."""
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_embed, k_embed


@torch.no_grad()
def _check_text_rope(device: torch.device, dtype: torch.dtype) -> dict:
    """Compare Qwen split-half RoPE vs Liger Triton on Qwen3-VL cos/sin layout.

    Formula match is judged in fp32. Training-dtype match uses the same bf16
    path as Qwen text RoPE (no extra fp32 cast). with_cast is not used: it
    diverges from Qwen by 1 ULP.
    """
    torch.manual_seed(0)
    bs, n_q, n_k, sl, hd = 2, 4, 2, 16, 64
    freqs = torch.randn(bs, sl, hd // 2, device=device, dtype=torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_f = emb.cos()
    sin_f = emb.sin()
    q_f = torch.randn(bs, n_q, sl, hd, device=device, dtype=torch.float32)
    k_f = torch.randn(bs, n_k, sl, hd, device=device, dtype=torch.float32)

    q_hf_f, k_hf_f = _qwen_split_half_closed_form(q_f, k_f, cos_f, sin_f)
    q_lg_f, k_lg_f = liger_rotary_pos_emb(q_f.clone(), k_f.clone(), cos_f, sin_f)
    q_err_f = (q_hf_f - q_lg_f).abs()
    k_err_f = (k_hf_f - k_lg_f).abs()

    q_b = q_f.to(dtype)
    k_b = k_f.to(dtype)
    cos_b = cos_f.to(dtype)
    sin_b = sin_f.to(dtype)
    q_hf_b, k_hf_b = qwen_split_half_rope(q_b, k_b, cos_b, sin_b)
    # Match Qwen text RoPE dtype path: stay in bf16. with_cast (fp32 then round)
    # differs from Qwen by 1 ULP and is NOT used.
    q_lg_b, k_lg_b = liger_rotary_pos_emb(q_b.clone(), k_b.clone(), cos_b, sin_b)
    q_err_b = (q_hf_b.float() - q_lg_b.float()).abs()
    k_err_b = (k_hf_b.float() - k_lg_b.float()).abs()

    cos_dup = (cos_f[..., : hd // 2] - cos_f[..., hd // 2 :]).abs().max().item()
    return {
        "fp32_q_max": q_err_f.max().item(),
        "fp32_k_max": k_err_f.max().item(),
        "fp32_q_mean": q_err_f.mean().item(),
        "bf16_q_max": q_err_b.max().item(),
        "bf16_k_max": k_err_b.max().item(),
        "bf16_q_mean": q_err_b.mean().item(),
        "cos_dup_max": cos_dup,
        "half_layout_ok": cos_dup < 1e-6,
    }


def _check_fused_lm_head(device: torch.device, dtype: torch.dtype, vocab: int = 4096, hidden: int = 256) -> dict:
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.manual_seed(1)
    bs, sl = 2, 32
    h = torch.randn(bs, sl, hidden, device=device, dtype=dtype)
    w = torch.randn(vocab, hidden, device=device, dtype=dtype)
    labels = torch.randint(0, vocab, (bs, sl), device=device)
    labels[:, :4] = -100

    logits_f = F.linear(h.float(), w.float())
    oracle = ForCausalLMLoss(logits_f, labels, vocab)
    # Actual HF Qwen path: bf16 GEMM then ForCausalLMLoss upcasts logits to fp32.
    hf = ForCausalLMLoss(F.linear(h, w), labels, vocab)

    fused = LigerForCausalLMLoss(
        hidden_states=h,
        lm_head_weight=w,
        labels=labels,
        hidden_size=hidden,
    )
    if isinstance(fused, tuple):
        fused = fused[0]
    fused_v = fused.float()
    abs_hf = (fused_v - hf).abs().item()
    rel_hf = abs_hf / max(hf.abs().item(), 1e-6)
    abs_or = (fused_v - oracle).abs().item()
    rel_or = abs_or / max(oracle.abs().item(), 1e-6)
    return {
        "oracle": oracle.item(),
        "hf": hf.item(),
        "fused": fused_v.item(),
        "abs_vs_hf": abs_hf,
        "rel_vs_hf": rel_hf,
        "abs_vs_oracle": abs_or,
        "rel_vs_oracle": rel_or,
    }


@torch.no_grad()
def _check_rms_norm(device: torch.device, dtype: torch.dtype) -> dict:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

    out = {}
    for dim, tag in ((2560, "hidden"), (128, "qk")):
        torch.manual_seed(0)
        x = torch.randn(2, 32, dim, device=device, dtype=dtype)
        w = torch.randn(dim, device=device, dtype=dtype)
        hf = Qwen3VLTextRMSNorm(dim, eps=1e-6).to(device=device, dtype=dtype)
        lg = LigerRMSNorm(dim, eps=1e-6, offset=0.0, casting_mode="llama").to(device=device, dtype=dtype)
        hf.weight.copy_(w)
        lg.weight.copy_(w)
        err = (hf(x).float() - lg(x.clone()).float()).abs()
        out[f"{tag}_max"] = err.max().item()
        out[f"{tag}_mean"] = err.mean().item()
    return out


@torch.no_grad()
def _check_swiglu(device: torch.device, dtype: torch.dtype) -> dict:
    torch.manual_seed(0)
    gate = torch.randn(2, 32, 1024, device=device, dtype=dtype)
    up = torch.randn(2, 32, 1024, device=device, dtype=dtype)
    hf = F.silu(gate) * up
    lg = LigerSiLUMulFunction.apply(gate.clone(), up.clone())
    err = (hf.float() - lg.float()).abs()
    return {"max": err.max().item(), "mean": err.mean().item()}


def apply_liger_kernel_to_qwen3_vl_vlact(
    model: PreTrainedModel,
    *,
    rope: bool = True,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    skip_checks: bool = False,
) -> dict:
    """
    Enable Liger on a loaded Qwen3VLForConditionalGeneration.

    Returns a status dict: which ops were enabled and numerical-check errors.
    """
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
        logger.warning(f"[Qwen3-VL] {status['error']}")
        return status
    if model is None:
        status["error"] = "model is None"
        return status

    device = next(model.parameters()).device
    dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else next(model.parameters()).dtype
    check_device = device if device.type == "cuda" else (
        torch.device("cuda") if torch.cuda.is_available() else device
    )
    if check_device.type != "cuda":
        logger.warning("[Qwen3-VL] Liger numerical checks skipped (no CUDA)")
        skip_checks = True
    else:
        device = check_device

    if rope and not skip_checks:
        try:
            status["rope_check"] = _check_text_rope(device, dtype)
            # fp32: formula. bf16: must match Qwen's native bf16 path (no extra fp32 cast).
            ok = (
                status["rope_check"]["half_layout_ok"]
                and status["rope_check"]["fp32_q_max"] < 1e-4
                and status["rope_check"]["fp32_k_max"] < 1e-4
                and status["rope_check"]["bf16_q_max"] == 0.0
                and status["rope_check"]["bf16_k_max"] == 0.0
            )
            if not ok:
                logger.warning(
                    f"[Qwen3-VL] Liger text RoPE check FAILED {status['rope_check']}; keep native RoPE"
                )
                rope = False
            else:
                logger.info(f"[Qwen3-VL] Liger text RoPE check OK {status['rope_check']}")
        except Exception as e:
            logger.warning(f"[Qwen3-VL] Liger text RoPE check error ({e}); keep native RoPE")
            status["rope_check"] = {"error": str(e)}
            rope = False

    if fused_linear_cross_entropy and not skip_checks:
        try:
            status["lm_head_check"] = _check_fused_lm_head(device, dtype)
            if status["lm_head_check"]["rel_vs_hf"] > 1e-5:
                logger.warning(
                    f"[Qwen3-VL] Liger fused LM-head CE check FAILED {status['lm_head_check']}; keep native CE"
                )
                fused_linear_cross_entropy = False
            else:
                logger.info(f"[Qwen3-VL] Liger fused LM-head CE check OK {status['lm_head_check']}")
        except Exception as e:
            logger.warning(f"[Qwen3-VL] Liger fused LM-head CE check error ({e}); keep native CE")
            status["lm_head_check"] = {"error": str(e)}
            fused_linear_cross_entropy = False

    if rope:
        # Text only, same bf16 path as Qwen (no fp32 with_cast).
        # Do not patch apply_rotary_pos_emb_vision (seq-first layout).
        modeling_qwen3_vl.apply_rotary_pos_emb = liger_rotary_pos_emb
        status["rope"] = True
        logger.info(
            "[Qwen3-VL] Liger text RoPE enabled (split-half, bf16 bit-match vs Qwen, cat(freqs,freqs)). "
            "Vision RoPE left native."
        )

    if fused_linear_cross_entropy:
        model.forward = MethodType(_qwen3_vl_fused_lm_head_forward, model)
        status["fused_linear_cross_entropy"] = True
        logger.info("[Qwen3-VL] Liger fused linear CE enabled on lm_head (skip S×vocab logits when labels present)")

    if rms_norm and not skip_checks:
        try:
            status["rms_check"] = _check_rms_norm(device, dtype)
            ok = status["rms_check"]["hidden_max"] == 0.0 and status["rms_check"]["qk_max"] == 0.0
            if not ok:
                logger.warning(f"[Qwen3-VL] Liger RMSNorm check FAILED {status['rms_check']}; keep native RMSNorm")
                rms_norm = False
            else:
                logger.info(f"[Qwen3-VL] Liger RMSNorm check OK {status['rms_check']}")
        except Exception as e:
            logger.warning(f"[Qwen3-VL] Liger RMSNorm check error ({e}); keep native RMSNorm")
            status["rms_check"] = {"error": str(e)}
            rms_norm = False

    if swiglu and not skip_checks:
        try:
            status["swiglu_check"] = _check_swiglu(device, dtype)
            if status["swiglu_check"]["max"] != 0.0:
                logger.warning(f"[Qwen3-VL] Liger SwiGLU check FAILED {status['swiglu_check']}; keep native MLP")
                swiglu = False
            else:
                logger.info(f"[Qwen3-VL] Liger SwiGLU check OK {status['swiglu_check']}")
        except Exception as e:
            logger.warning(f"[Qwen3-VL] Liger SwiGLU check error ({e}); keep native MLP")
            status["swiglu_check"] = {"error": str(e)}
            swiglu = False

    if rms_norm:
        n = _patch_text_rms_norm(model)
        status["rms_norm"] = True
        logger.info(
            f"[Qwen3-VL] Liger RMSNorm enabled on {n} text norms "
            "(final/input/post-attn/q_norm/k_norm, llama fp32-rstd then bf16 weight). Vision LayerNorm left native."
        )

    if swiglu:
        n = _patch_text_swiglu(model)
        status["swiglu"] = True
        logger.info(f"[Qwen3-VL] Liger SwiGLU enabled on {n} text MLPs (silu(gate)*up). Vision GELU MLP left native.")

    return status


def apply_qwen3_liger_from_config(model: PreTrainedModel, qwenvl_cfg) -> dict:
    if not _cfg_bool(qwenvl_cfg, "use_liger_kernel", False):
        logger.info("[Qwen3-VL] Liger disabled (framework.qwenvl.use_liger_kernel=false)")
        return {"enabled": False}
    return apply_liger_kernel_to_qwen3_vl_vlact(
        model,
        rope=_cfg_bool(qwenvl_cfg, "liger_rope", True),
        fused_linear_cross_entropy=_cfg_bool(qwenvl_cfg, "liger_fused_linear_cross_entropy", True),
        rms_norm=_cfg_bool(qwenvl_cfg, "liger_rms_norm", True),
        swiglu=_cfg_bool(qwenvl_cfg, "liger_swiglu", True),
    )
