# Copyright 2026
# SpikeBrain VLM interface for starVLA

import os
from typing import List, Optional

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from starVLA.model.modules.vlm.spb2_liger import apply_liger_kernel_to_spb2_vl
from starVLA.model.modules.vlm.spb2_parallel_gc import (
    _trainer_gc_enabled,
    maybe_enable_gradient_checkpointing_from_config,
    validate_zero3_gc_alignment,
)
from starVLA.training.trainer_utils.vlm_packing import pack_per_sample_inputs, relocate_packed_inputs

logger = get_logger(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 151665
_ACTION_TOKEN_MAX = 153712
GDN_FP32_MODULE_NAME = "GDNFP32Params"
SPB2_TEXT_RMSNORM_CLASS = "SPB2VLTextRMSNorm"
ATTN_RMSNORM_CLASS = "RMSNorm"


def _is_fp32_sensitive_module(name: str, module: nn.Module) -> bool:
    """
    Match lmms-engine ``fp32_layer_cls_to_wrap`` (see fsdp2_utils.apply_fsdp2):

    - GDNFP32Params: always fp32
    - ``RMSNorm`` in text/attn (MoBA/SSE), excluding decoder input/post layernorm names
    - ``SPB2VLTextRMSNorm``: keep bf16 weights — forward already runs variance in fp32
      (do NOT .float() input_layernorm / post_attention_layernorm; that broke DiT at step 2)
    """
    if "visual" in name:
        return False
    cls = module.__class__.__name__
    if cls == GDN_FP32_MODULE_NAME:
        return True
    if cls == ATTN_RMSNORM_CLASS:
        if "input_layernorm" in name or "post_attention_layernorm" in name:
            return False
        return True
    return False


def ensure_gdn_fp32_params(model: nn.Module, *, log: bool = True) -> int:
    """
    Keep SSE-SWA GDN gating params (A_log, dt_bias) in float32 under bf16 / DeepSpeed training.
    Equivalent intent to lmms-engine FSDP fp32_layer_cls_to_wrap: [GDNFP32Params].
    """
    count = 0
    non_fp32: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != GDN_FP32_MODULE_NAME:
            continue
        module.float()
        count += 1
        for pname, param in module.named_parameters(recurse=False):
            if param.dtype != torch.float32:
                non_fp32.append(f"{name}.{pname}={param.dtype}")

    if log:
        if count == 0:
            logger.warning(
                "[SpikeBrain] GDNFP32Params: no modules found (model may not use linear_attn_type=gdn)"
            )
        else:
            logger.info(f"[SpikeBrain] GDNFP32Params: ensured float32 on {count} module(s)")
        if non_fp32:
            logger.warning(
                "[SpikeBrain] GDNFP32Params: parameters still not float32 after cast: "
                + ", ".join(non_fp32[:8])
                + (" ..." if len(non_fp32) > 8 else "")
            )
        elif count > 0:
            sample = next(
                (m for m in model.modules() if m.__class__.__name__ == GDN_FP32_MODULE_NAME),
                None,
            )
            if sample is not None:
                dtypes = {n: p.dtype for n, p in sample.named_parameters(recurse=False)}
                logger.info(f"[SpikeBrain] GDNFP32Params sample dtypes: {dtypes}")

    return count


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


def use_fsdp2_env() -> bool:
    return _env_flag("STARVLA_USE_FSDP2")


def use_fsdp1_env() -> bool:
    """FSDP1 via STARVLA_USE_FSDP / accelerate FSDP launch (not FSDP2)."""
    if use_fsdp2_env():
        return False
    if _env_flag("STARVLA_USE_FSDP") or _env_flag("ACCELERATE_USE_FSDP"):
        return True
    # accelerate launch --config_file fsdp*.yaml may not export ACCELERATE_USE_FSDP
    # until Accelerator init; launch scripts must set STARVLA_USE_FSDP=1.
    return False


def _defer_fp32_patches_for_fsdp() -> bool:
    """FSDP flat-param requires uniform dtype at wrap time; patches run after prepare."""
    return use_fsdp1_env() or use_fsdp2_env()


def ensure_fp32_sensitive_layers(model: nn.Module, *, log: bool = True) -> tuple[int, int]:
    """
    Keep numerically sensitive VLM submodules in fp32 (DeepSpeed path, no FSDP).

    Aligns with lmms-engine ``fp32_layer_cls_to_wrap: [GDNFP32Params]`` plus text/attn
    RMSNorm layers (excluding vision tower norms).
    """
    gdn_count = ensure_gdn_fp32_params(model, log=False)
    norm_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == GDN_FP32_MODULE_NAME:
            continue
        if not _is_fp32_sensitive_module(name, module):
            continue
        module.float()
        norm_count += 1

    if log:
        logger.info(
            "[SpikeBrain] fp32 sensitive layers: GDNFP32Params=%d, RMSNorm-like=%d",
            gdn_count,
            norm_count,
        )
    return gdn_count, norm_count


def prepare_fsdp1_uniform_dtypes(model: nn.Module, *, log: bool = True) -> dict:
    """
    Make FSDP1 FlatParam units dtype-uniform before ``accelerator.prepare``.

    - Nest-wrap ``GDNFP32Params`` (see fsdp1_qwengr00t.yaml) keeps GDN in fp32.
    - Everything else (incl. DiT / action_model, often default fp32) must be bf16,
      otherwise the root FSDP unit mixes bf16 VLM leftovers with fp32 action params.
    """
    gdn_modules = [m for m in model.modules() if m.__class__.__name__ == GDN_FP32_MODULE_NAME]
    gdn_param_ids = {id(p) for m in gdn_modules for p in m.parameters(recurse=False)}

    cast_params = 0
    cast_buffers = 0
    for p in model.parameters():
        if not p.is_floating_point():
            continue
        if id(p) in gdn_param_ids:
            if p.dtype != torch.float32:
                p.data = p.data.float()
                cast_params += 1
            continue
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(dtype=torch.bfloat16)
            cast_params += 1

    for m in gdn_modules:
        m.float()

    for b in model.buffers():
        if not b.is_floating_point():
            continue
        # Keep GDN-adjacent buffers alone; buffers are not flattened into FlatParam
        # the same way, but align non-fp32 compute buffers to bf16 for consistency.
        if b.dtype not in (torch.bfloat16, torch.float32):
            continue
        # Leave fp32 buffers (rare); cast other float buffers that are fp32 only if
        # not owned by GDN modules. Simpler: skip buffers.
        _ = b
        cast_buffers += 0

    stats = {
        "gdn_modules": len(gdn_modules),
        "cast_params": cast_params,
        "cast_buffers": cast_buffers,
    }
    if log:
        logger.info(
            "[SpikeBrain] FSDP1 dtype prep: GDNFP32Params=%d kept/cast fp32; "
            "other floating params cast to bf16 (%d tensors)",
            stats["gdn_modules"],
            stats["cast_params"],
        )
    return stats


def _find_lora_target_modules(model: nn.Module):
    allow = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in allow:
                found.add(leaf)
    return sorted(found)


def _resolve_attn_implementation(sb_cfg) -> str:
    """Pick attention backend at load time (must be passed to from_pretrained)."""
    attn_impl = sb_cfg.get("attn_implementation", "flash_attention_2")
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "[SpikeBrain] flash_attn not installed, falling back to sdpa. "
                "Install flash-attn for lower VRAM / faster attention."
            )
            attn_impl = "sdpa"
    return attn_impl


def _resolve_spb2_backbone(model: nn.Module) -> Optional[nn.Module]:
    """
    Return SPB2VLModel backbone inside SPB2VLForConditionalGeneration (handles PEFT wrap).
    """
    target = model
    if hasattr(target, "get_base_model"):
        target = target.get_base_model()
    inner = getattr(target, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner
    if hasattr(target, "language_model"):
        return target
    return None


def _align_config_hidden_size(model: nn.Module) -> None:
    """Expose text hidden_size on top-level config for QwenGR00T / DiT cross_attention_dim."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        cfg.hidden_size = int(text_cfg.hidden_size)


def _log_model_load_summary(model: nn.Module, model_id: str, attn_impl: str) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[SpikeBrain] loaded from {model_id} | class={model.__class__.__name__} | "
        f"params={n_params / 1e9:.3f}B | attn_implementation={attn_impl}"
    )
    backbone = _resolve_spb2_backbone(model)
    if backbone is None:
        logger.warning(
            "[SpikeBrain] Could not find SPB2VLModel backbone (language_model). "
            "Check that AutoModelForCausalLM loaded SPB2VLForConditionalGeneration."
        )
    else:
        logger.info(f"[SpikeBrain] backbone class={backbone.__class__.__name__}")


class _SpikeBrain_VL_Interface(nn.Module):
    """
    SpikeBrain wrapper with the same external interface as Qwen wrappers:
      - forward(...)
      - generate(...)
      - build_qwenvl_inputs(...)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config

        sb_cfg = config.framework.get("qwenvl", {})
        model_id = sb_cfg.get("base_vlm", "")

        model_id = str(model_id).replace("\r", "").replace("\n", "").strip()
        if os.path.isdir(model_id):
            model_id = os.path.realpath(model_id)

        if not model_id:
            raise ValueError("framework.qwenvl.base_vlm is empty")

        logger.info(f"[SpikeBrain] loading model from: {model_id}")

        if os.path.isdir(model_id):
            shard1 = os.path.join(model_id, "model-00001-of-00003.safetensors")
            if not os.path.exists(shard1):
                raise FileNotFoundError(
                    f"Missing shard: {repr(shard1)}; files={os.listdir(model_id)}"
                )

        local_only = os.path.isdir(model_id)

        hf_cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=local_only
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=local_only
        )
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        from starVLA.model.modules.vlm.SpikeBrain2VL import apply_processor_image_pixel_budget

        apply_processor_image_pixel_budget(
            self.processor, sb_cfg, log_prefix="[SpikeBrain]"
        )

        attn_impl = _resolve_attn_implementation(sb_cfg)

        # Use AutoModelForCausalLM -> SPB2VLForConditionalGeneration so checkpoint keys
        # (model.language_model.*, model.visual.*, lm_head.*) match safetensors.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=hf_cfg,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
            attn_implementation=attn_impl,
        )
        _align_config_hidden_size(self.model)
        _log_model_load_summary(self.model, model_id, attn_impl)

        use_liger = bool(sb_cfg.get("use_liger_kernel", False))
        if os.environ.get("STARVLA_DISABLE_LIGER", "0").lower() in ("1", "true", "yes"):
            use_liger = False
        if use_liger:
            # Liger on non-sensitive ops only; keep text/attn RMSNorm on native fp32 paths.
            apply_liger_kernel_to_spb2_vl(
                self.model,
                rope=bool(sb_cfg.get("liger_rope", True)),
                rms_norm=bool(sb_cfg.get("liger_rms_norm", False)),
                swiglu=bool(sb_cfg.get("liger_swiglu", True)),
            )

        enable_hf_lora = bool(sb_cfg.get("enable_hf_lora", False))
        if enable_hf_lora:
            target_modules = sb_cfg.get("lora_target_modules", [])
            target_modules = list(target_modules) if target_modules is not None else []
            if len(target_modules) == 0:
                target_modules = _find_lora_target_modules(self.model)

            if len(target_modules) == 0:
                raise ValueError(
                    "LoRA target_modules is empty. Please set framework.qwenvl.lora_target_modules manually."
                )

            peft_task_type = (
                TaskType.CAUSAL_LM
                if hasattr(self.model, "prepare_inputs_for_generation")
                else TaskType.FEATURE_EXTRACTION
            )

            lora_cfg = LoraConfig(
                task_type=peft_task_type,
                r=int(sb_cfg.get("lora_r", 16)),
                lora_alpha=int(sb_cfg.get("lora_alpha", 32)),
                lora_dropout=float(sb_cfg.get("lora_dropout", 0.05)),
                bias=str(sb_cfg.get("lora_bias", "none")),
                target_modules=target_modules,
            )

            self.model = get_peft_model(self.model, lora_cfg)
            logger.info(f"[SpikeBrain] HF LoRA enabled, target_modules={target_modules}")
            self.model.print_trainable_parameters()
            _align_config_hidden_size(self.model)

        gc_info = maybe_enable_gradient_checkpointing_from_config(
            self.model,
            config,
            sb_cfg=sb_cfg,
            logger=logger,
        )
        if gc_info is not None:
            validate_zero3_gc_alignment(config, gc_info, logger=logger)
        elif _trainer_gc_enabled(config, sb_cfg):
            ablation = os.environ.get("STARVLA_FSDP2_LM_ABLATION", "0").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if ablation:
                logger.warning(
                    "[SpikeBrain] VLM GC skipped: STARVLA_FSDP2_LM_ABLATION=1 "
                    "(not a layer-class align failure)"
                )
            else:
                logger.warning(
                    "[SpikeBrain] gradient_checkpointing requested but aligned "
                    "enable failed on model"
                )

        if not _defer_fp32_patches_for_fsdp():
            if sb_cfg.get("ensure_fp32_sensitive_layers", True) or sb_cfg.get(
                "ensure_gdn_fp32_params", True
            ):
                ensure_fp32_sensitive_layers(self.model)

            from starVLA.model.modules.vlm.spb2_numerical_stability import apply_spb2_numerical_patches

            apply_spb2_numerical_patches(self.model, sb_cfg)
        else:
            logger.info(
                "[SpikeBrain] FSDP mode: defer fp32/numerical patches until after accelerator.prepare"
            )
            # FSDP1 nest-wraps GDNFP32Params (see fsdp1_qwengr00t.yaml). Cast GDN to fp32
            # before prepare so each nested FlatParam is uniform dtype; leave attn RMSNorm
            # in bf16 (fp32 via rmsnorm_fp32_forward patches after prepare).
            if use_fsdp1_env() and (
                sb_cfg.get("ensure_fp32_sensitive_layers", True)
                or sb_cfg.get("ensure_gdn_fp32_params", True)
            ):
                ensure_gdn_fp32_params(self.model)

        self._vlm_forward_fp32 = bool(sb_cfg.get("vlm_forward_fp32", False))
        if self._vlm_forward_fp32:
            logger.info(
                "[SpikeBrain] VLM forward: bf16 compute (flash-attn) + fp32 hidden output"
            )

        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    @staticmethod
    def _cfg_get(section, key: str, default=""):
        if section is None:
            return default
        if hasattr(section, "get"):
            return section.get(key, default)
        return getattr(section, key, default)

    def _render_cot_prompt(self, instruction: str) -> str:
        vla_cfg = self._cfg_get(getattr(self.config, "datasets", None), "vla_data", None)
        template = self._cfg_get(vla_cfg, "CoT_prompt", "")
        if not template:
            return instruction

        action_cfg = self._cfg_get(getattr(self.config, "framework", None), "action_model", None)
        replacements = {
            "instruction": instruction,
            "action_horizon": self._cfg_get(action_cfg, "action_horizon", ""),
            "future_action_window_size": self._cfg_get(action_cfg, "future_action_window_size", ""),
            "past_action_window_size": self._cfg_get(action_cfg, "past_action_window_size", ""),
        }
        prompt = str(template)
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def _forward_backbone(
        self,
        backbone: nn.Module,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        """
        VLA feature path: run SPB2VLModel only (skip lm_head).
        SPB2VLForConditionalGeneration.forward does not set last_hidden_state on outputs.
        """
        if use_cache is None:
            use_cache = False

        return backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _wrap_backbone_as_causal_output(self, backbone_out, return_dict: bool):
        if not return_dict:
            return backbone_out
        if getattr(backbone_out, "last_hidden_state", None) is not None:
            return backbone_out
        # transformers>=5.8: CausalLMOutputWithPast has no last_hidden_state field
        return BaseModelOutputWithPast(
            last_hidden_state=getattr(backbone_out, "last_hidden_state", None),
            past_key_values=getattr(backbone_out, "past_key_values", None),
            hidden_states=getattr(backbone_out, "hidden_states", None),
            attentions=getattr(backbone_out, "attentions", None),
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = device_type == "cuda"
        dtype = torch.bfloat16 if use_amp else torch.float32

        backbone = _resolve_spb2_backbone(self.model)
        # QwenGR00T: no labels -> extract features via backbone (provides last_hidden_state).
        use_backbone_only = labels is None and backbone is not None

        def _run():
            if use_backbone_only:
                backbone_out = self._forward_backbone(
                    backbone,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
                return self._wrap_backbone_as_causal_output(backbone_out, return_dict)

            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        if use_amp:
            with torch.autocast(device_type=device_type, dtype=dtype):
                outputs = _run()
        else:
            outputs = _run()

        # Full CausalLM path may omit last_hidden_state; fill from hidden_states if needed.
        if return_dict and outputs is not None:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None and getattr(outputs, "hidden_states", None):
                outputs.last_hidden_state = outputs.hidden_states[-1]

        return outputs

    def generate(self, **kwargs):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = device_type == "cuda"
        dtype = torch.float16 if use_amp else torch.float32

        if use_amp:
            with torch.autocast(device_type=device_type, dtype=dtype):
                return self.model.generate(**kwargs)
        return self.model.generate(**kwargs)

    def _build_messages(self, images, instructions, solutions=None):
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        messages = []
        for idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            if imgs is None:
                imgs = []
            elif not isinstance(imgs, (list, tuple)):
                imgs = [imgs]
            elif imgs and isinstance(imgs[0], (list, tuple)):
                # Traj segment layout [T][cams] → current timestep cams.
                imgs = list(imgs[-1])
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = self._render_cot_prompt(instruction)
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[idx]}]})
            messages.append(msg)
        return messages

    def _processor_single_sample(self, message, *, with_labels: bool = False):
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info([message])
        sample = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        if with_labels:
            labels = sample["input_ids"].clone()
            seq = labels[0]
            mask_seq = (seq >= self._ACTION_TOKEN_MIN) & (seq <= self._ACTION_TOKEN_MAX)
            nz = torch.nonzero(mask_seq, as_tuple=False)
            if nz.numel() > 0:
                seq[: nz[0].item()] = IGNORE_INDEX
            else:
                seq[:] = IGNORE_INDEX
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                pad_id = self.processor.tokenizer.pad_token_id
                if pad_id is not None:
                    labels[labels == pad_id] = IGNORE_INDEX
            sample["labels"] = labels
        return sample

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        messages = self._build_messages(images, instructions, solutions=solutions)

        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if solutions is not None:
            labels = batch_input["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= self._ACTION_TOKEN_MIN) & (seq <= self._ACTION_TOKEN_MAX)
                nz = torch.nonzero(mask_seq, as_tuple=False)
                if nz.numel() > 0:
                    first_idx = nz[0].item()
                    seq[:first_idx] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX

            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                pad_id = self.processor.tokenizer.pad_token_id
                if pad_id is not None:
                    labels[labels == pad_id] = IGNORE_INDEX
            batch_input["labels"] = labels

        return batch_input.to(self.model.device)

    def build_qwenvl_inputs_per_sample(self, images, instructions, solutions=None):
        messages = self._build_messages(images, instructions, solutions=solutions)
        return [
            self._processor_single_sample(msg, with_labels=solutions is not None)
            for msg in messages
        ]

    def pack_qwenvl_inputs(self, per_sample, indices, *, max_seqlen: int | None = None):
        subset = [per_sample[i] for i in indices]
        packed, _ = pack_per_sample_inputs(subset, max_seqlen=max_seqlen)
        return relocate_packed_inputs(packed, self.model.device)

    def build_qwenvl_inputs_packed(
        self,
        images,
        instructions,
        *,
        max_seqlen: int | None = None,
        solutions=None,
    ):
        """
        Pack all samples into one varlen forward (caller must ensure total_len <= max_seqlen).
        """
        per_sample = self.build_qwenvl_inputs_per_sample(
            images, instructions, solutions=solutions
        )
        packed, _ = pack_per_sample_inputs(per_sample, max_seqlen=max_seqlen)
        return relocate_packed_inputs(packed, self.model.device)
