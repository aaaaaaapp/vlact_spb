# SpikeBrain-2.0-VL interface via lm_head pre-hook (X5A stable path)
# Aligned with starVLA-starVLA_dev/examples/X5A_Cans/starvla_train_x5a_cans_spikebrain.yaml
# GC: explicit layer alignment via spb2_parallel_gc (see framework.qwenvl.gradient_checkpointing_layer_cls_to_wrap)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import ModelOutput

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


def _resolve_attn_implementation(sb_cfg) -> str:
    attn_impl = sb_cfg.get("attn_implementation", "flash_attention_2")
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "[SpikeBrain2VL] flash_attn not installed, falling back to sdpa"
            )
            attn_impl = "sdpa"
    return attn_impl


def _resolve_optional_int(*candidates) -> Optional[int]:
    for value in candidates:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def apply_processor_image_pixel_budget(processor, sb_cfg=None, *, log_prefix: str = "[SpikeBrain2VL]") -> None:
    """Override image processor min/max pixels (matches lmms-engine image_min/max_pixels).

    Priority: STARVLA_VLM_MIN/MAX_PIXELS env → framework.qwenvl.image_min/max_pixels → unchanged.
    Also syncs size['shortest_edge'/'longest_edge'] used by Qwen3-VL processors.
    """
    sb_cfg = sb_cfg or {}
    min_pixels = _resolve_optional_int(
        os.environ.get("STARVLA_VLM_MIN_PIXELS"),
        sb_cfg.get("image_min_pixels"),
    )
    max_pixels = _resolve_optional_int(
        os.environ.get("STARVLA_VLM_MAX_PIXELS"),
        sb_cfg.get("image_max_pixels"),
    )
    if min_pixels is None and max_pixels is None:
        return

    ip = getattr(processor, "image_processor", None)
    if ip is None:
        logger.warning("%s no image_processor; skip pixel budget override", log_prefix)
        return

    before_min = getattr(ip, "min_pixels", None)
    before_max = getattr(ip, "max_pixels", None)
    if min_pixels is not None:
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = min_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = min_pixels
    if max_pixels is not None:
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["longest_edge"] = max_pixels

    logger.info(
        "%s image pixel budget: min %s→%s, max %s→%s",
        log_prefix,
        before_min,
        getattr(ip, "min_pixels", None),
        before_max,
        getattr(ip, "max_pixels", None),
    )


@dataclass
class SpikeBrain2VLOutput(ModelOutput):
    """Structured VLM forward output for DeepSpeed ZeRO-3 hook compatibility."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class _SpikeBrain2_VL_Interface(nn.Module):
    """
    Minimal SPB2-VL wrapper aligned with starVLA-starVLA_dev / X5A_Cans.
    SpikeBrain top-level forward does not propagate ``output_hidden_states`` reliably;
    we capture the final hidden state via a pre-forward hook on ``lm_head``.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        sb_cfg = config.framework.get("qwenvl", {})
        model_id = str(sb_cfg.get("base_vlm", "")).replace("\r", "").replace("\n", "").strip()
        if os.path.isdir(model_id):
            model_id = os.path.realpath(model_id)

        if not model_id:
            raise ValueError("framework.qwenvl.base_vlm must point to SpikeBrain-2.0-VL")

        if os.path.isdir(model_id):
            shard1 = os.path.join(model_id, "model-00001-of-00003.safetensors")
            if not os.path.exists(shard1):
                raise FileNotFoundError(
                    f"Missing shard: {repr(shard1)}; files={os.listdir(model_id)}"
                )

        local_only = os.path.isdir(model_id)
        attn_impl = _resolve_attn_implementation(sb_cfg)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=local_only
        )
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        apply_processor_image_pixel_budget(self.processor, sb_cfg)

        # Expose text hidden_size on top-level config for QwenGR00T / DiT cross_attention_dim.
        text_cfg = getattr(getattr(self.model, "config", None), "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            self.model.config.hidden_size = int(text_cfg.hidden_size)

        # Keep lm_head pre-hook as fallback; VLA training prefers backbone-only
        # (skip S×vocab logits materialization).
        self._captured_hidden: Optional[torch.Tensor] = None
        lm_head = self._find_lm_head(self.model)
        if lm_head is None:
            raise RuntimeError("[SpikeBrain2VL] could not locate lm_head; hidden capture impossible")
        lm_head.register_forward_pre_hook(self._capture_pre_hook)
        self._lm_head = lm_head
        self._backbone_only = bool(sb_cfg.get("vlm_backbone_only", True))
        self._backbone_only_strict = bool(sb_cfg.get("vlm_backbone_only_strict", True))
        self._backbone_fallback_logged = False
        if self._backbone_only:
            input_embeddings = (
                self.model.get_input_embeddings()
                if hasattr(self.model, "get_input_embeddings")
                else None
            )
            input_weight = getattr(input_embeddings, "weight", None)
            head_weight = getattr(lm_head, "weight", None)
            tied = input_weight is not None and head_weight is input_weight
            if not tied:
                for parameter in lm_head.parameters():
                    parameter.requires_grad_(False)
                logger.info(
                    "[SpikeBrain2VL] froze untied lm_head parameters "
                    "(unused by VLA action loss)"
                )

        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", -1)
        logger.info(
            "[SpikeBrain2VL] loaded from %s | attn=%s | hidden=%d | "
            "backbone_only=%s | lm_head fallback on %s",
            model_id,
            attn_impl,
            hidden_size,
            self._backbone_only,
            type(lm_head).__name__,
        )

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
                    "[SpikeBrain2VL] VLM GC skipped: STARVLA_FSDP2_LM_ABLATION=1 "
                    "(not a layer-class align failure)"
                )
            else:
                logger.warning(
                    "[SpikeBrain2VL] gradient_checkpointing requested but aligned "
                    "enable failed on model"
                )

        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    @staticmethod
    def _find_lm_head(model: nn.Module) -> Optional[nn.Module]:
        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
            return model.lm_head
        for name, module in model.named_modules():
            if name.endswith("lm_head") and isinstance(module, nn.Linear):
                return module
        return None

    def _capture_pre_hook(self, _module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            self._captured_hidden = inputs[0]

    @staticmethod
    def _extract_backbone_hidden(outputs) -> Optional[torch.Tensor]:
        if torch.is_tensor(outputs):
            return outputs
        if hasattr(outputs, "get"):
            hidden = outputs.get("last_hidden_state", None)
            if hidden is not None:
                return hidden
            hidden_states = outputs.get("hidden_states", None)
            if hidden_states:
                return hidden_states[-1]
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden
        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        return None

    def _fsdp2_active(self) -> bool:
        """True only for native FSDP2 (STARVLA_USE_FSDP2).

        Do NOT treat FSDP1-managed modules as FSDP2: FSDP1 also sets
        ``_is_fsdp_managed_module``, and the DTensor root route is wrong there
        (illegal memory access on first step).
        """
        try:
            from starVLA.training.fsdp2_prepare import use_fsdp2

            return bool(use_fsdp2())
        except Exception:
            return False

    def _forward_backbone_only_via_fsdp_root(self, **model_kwargs):
        """Enter the FSDP2-wrapped ConditionalGeneration root.

        Calling the inner ``.model`` backbone directly bypasses FSDP2 all-gather,
        so ``embed_tokens.weight`` stays a DTensor while ``input_ids`` is a plain
        Tensor → ``aten.embedding`` mixed-type RuntimeError (packing & nopack).
        """
        lm_head = getattr(self, "_lm_head", None) or self._find_lm_head(self.model)
        orig_forward = None
        if lm_head is not None:
            orig_forward = lm_head.forward

            def _skip_vocab_logits(hidden_states, *args, **kwargs):
                # Pre-hook already captured hidden; emit a tiny stand-in so the
                # CausalLM forward can finish without allocating S×vocab.
                return hidden_states.new_zeros(
                    hidden_states.shape[0], hidden_states.shape[1], 1
                )

            lm_head.forward = _skip_vocab_logits  # type: ignore[method-assign]

        model_kwargs = dict(model_kwargs)
        # Don't force output_hidden_states=True — the lm_head pre-hook
        # (_capture_pre_hook) already captures the last hidden state.
        # Forcing True makes the model materialise ALL layer hidden states
        # (~3 GiB for 36 layers × 12k tokens), which are never used but stay
        # alive on the autograd graph.
        model_kwargs["output_hidden_states"] = False
        model_kwargs["return_dict"] = True
        model_kwargs.pop("labels", None)
        try:
            outputs = self.model(labels=None, **model_kwargs)
        finally:
            if lm_head is not None and orig_forward is not None:
                lm_head.forward = orig_forward  # type: ignore[method-assign]

        hidden = self._captured_hidden
        if hidden is None:
            hidden = self._extract_backbone_hidden(outputs)
        if hidden is None:
            hs = getattr(outputs, "hidden_states", None)
            if hs:
                hidden = hs[-1]
        if hidden is None:
            keys = list(outputs.keys()) if hasattr(outputs, "keys") else "n/a"
            raise AttributeError(
                f"{type(self.model).__name__} FSDP2 root forward produced no hidden; keys={keys}"
            )
        # CausalLM always attaches `.logits`; our patched lm_head only emits a
        # 1-wide stand-in. Null it so backbone-only invariant still holds.
        if getattr(outputs, "logits", None) is not None:
            outputs.logits = None
        return outputs, hidden

    def _forward_backbone_only(self, **model_kwargs):
        """Run the multimodal backbone without allocating ``S x vocab`` logits."""
        # FSDP2 only: enter ConditionalGeneration root (DTensor embeddings).
        # FSDP1: use inner backbone (production-style); GDN is FSDP-ignored + fp32.
        if self._fsdp2_active():
            if not getattr(self, "_fsdp_backbone_route_logged", False):
                logger.info(
                    "[SpikeBrain2VL] FSDP2: backbone-only routes through "
                    "ConditionalGeneration root"
                )
                self._fsdp_backbone_route_logged = True
            return self._forward_backbone_only_via_fsdp_root(**model_kwargs)

        backbone = getattr(self.model, "model", None)
        if not isinstance(backbone, nn.Module):
            raise AttributeError(
                f"{type(self.model).__name__} has no nn.Module `.model` backbone"
            )
        outputs = backbone(**model_kwargs)
        hidden = self._extract_backbone_hidden(outputs)
        if hidden is None:
            keys = list(outputs.keys()) if hasattr(outputs, "keys") else "n/a"
            raise AttributeError(
                f"{type(backbone).__name__} did not return last hidden state; keys={keys}"
            )
        return outputs, hidden

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> SpikeBrain2VLOutput:
        self._captured_hidden = None
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=use_cache if use_cache is not None else False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        with torch.autocast(device_type, dtype=torch.bfloat16):
            if self._backbone_only and labels is None:
                try:
                    raw_outputs, last_hidden = self._forward_backbone_only(**model_kwargs)
                except (AttributeError, TypeError) as exc:
                    if self._backbone_only_strict:
                        raise RuntimeError(
                            "[SpikeBrain2VL] strict backbone-only forward failed. "
                            "Refusing to fall back to lm_head because that would "
                            f"materialize S*vocab logits: {exc}"
                        ) from exc
                    if not self._backbone_fallback_logged:
                        logger.warning(
                            "[SpikeBrain2VL] backbone-only forward unavailable (%s); "
                            "falling back to the logits-producing path",
                            exc,
                        )
                        self._backbone_fallback_logged = True
                    raw_outputs = self.model(labels=None, **model_kwargs)
                    last_hidden = self._captured_hidden
            else:
                raw_outputs = self.model(labels=labels, **model_kwargs)
                last_hidden = self._captured_hidden

        if last_hidden is None:
            raise RuntimeError(
                "[SpikeBrain2VL] forward produced no last hidden state; "
                "backbone output and lm_head hook both failed"
            )
        logits = getattr(raw_outputs, "logits", None)
        if self._backbone_only and labels is None and logits is not None:
            raise RuntimeError(
                "[SpikeBrain2VL] backbone-only invariant violated: logits were materialized"
            )

        return SpikeBrain2VLOutput(
            last_hidden_state=last_hidden,
            hidden_states=getattr(raw_outputs, "hidden_states", None),
            logits=logits,
            loss=getattr(raw_outputs, "loss", None),
            past_key_values=getattr(raw_outputs, "past_key_values", None),
            attentions=getattr(raw_outputs, "attentions", None),
        )

    def generate(self, **kwargs):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type, dtype=torch.float16):
            return self.model.generate(**kwargs)

    def _render_cot_prompt(self, instruction: str) -> str:
        vla_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        template = getattr(vla_cfg, "CoT_prompt", "") or ""
        if not template:
            return instruction
        action_cfg = getattr(getattr(self.config, "framework", None), "action_model", None)
        replacements = {
            "instruction": instruction,
            "action_horizon": getattr(action_cfg, "action_horizon", ""),
            "future_action_window_size": getattr(action_cfg, "future_action_window_size", ""),
            "past_action_window_size": getattr(action_cfg, "past_action_window_size", ""),
        }
        prompt = str(template)
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def _build_messages(self, images, instructions, solutions=None):
        messages = []
        for idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            # Accept PIL / single image, or List[PIL] cams. Traj layout
            # List[List[PIL]] must be collapsed to the last timestep first.
            if imgs is None:
                imgs = []
            elif not isinstance(imgs, (list, tuple)):
                imgs = [imgs]
            elif imgs and isinstance(imgs[0], (list, tuple)):
                imgs = list(imgs[-1])
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": self._render_cot_prompt(instruction)})
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
                    seq[: nz[0].item()] = IGNORE_INDEX
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

    def pack_qwenvl_inputs(self, per_sample, indices, *, max_seqlen=None):
        subset = [per_sample[i] for i in indices]
        packed, _ = pack_per_sample_inputs(subset, max_seqlen=max_seqlen)
        return relocate_packed_inputs(packed, self.model.device)

    def build_qwenvl_inputs_packed(self, images, instructions, *, max_seqlen=None, solutions=None):
        per_sample = self.build_qwenvl_inputs_per_sample(images, instructions, solutions=solutions)
        packed, _ = pack_per_sample_inputs(per_sample, max_seqlen=max_seqlen)
        return relocate_packed_inputs(packed, self.model.device)
