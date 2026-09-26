"""SpikeBrain-2.0-VL adapter for VLAct.

The upstream SpikeBrain processor ships with a large fixed image pixel budget.
VLAct resizes observations in its data/framework pipeline, so this adapter uses
the actual VLAct batch resolution as the processor budget instead.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import ModelOutput

logger = get_logger(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 151665
_ACTION_TOKEN_MAX = 153712


@dataclass
class SpikeBrain2VLOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def _resolve_attention_implementation(config) -> str:
    implementation = config.get("attn_implementation", "flash_attention_2")
    if implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning("flash_attn is unavailable; SpikeBrain will use SDPA")
            implementation = "sdpa"
    return implementation


def _image_pixel_budget(images) -> Optional[dict[str, int]]:
    """Build a per-batch budget from already-resized VLAct images.

    Qwen/SpikeBrain still rounds dimensions to the vision patch factor, but the
    checkpoint's fixed 512-ish minimum no longer overrides VLAct's resolution.
    """
    areas = []
    for image in images or []:
        size = getattr(image, "size", None)
        if not size or len(size) != 2:
            continue
        width, height = int(size[0]), int(size[1])
        if width > 0 and height > 0:
            areas.append(width * height)
    if not areas:
        return None
    return {"min_pixels": min(areas), "max_pixels": max(areas)}


def _cast_all_floating_state_to_bf16(model: nn.Module) -> None:
    """Force every floating parameter and buffer, including GDN state, to BF16."""
    model.to(dtype=torch.bfloat16)
    remaining = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_floating_point() and value.dtype != torch.bfloat16
    ]
    if remaining:
        raise RuntimeError(
            "SpikeBrain contains non-BF16 floating state after conversion: "
            + ", ".join(remaining[:8])
        )


class _SpikeBrain2_VL_Interface(nn.Module):
    """Expose SpikeBrain through the interface used by VLAct Qwen frameworks."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        spikebrain_config = config.framework.get("qwenvl", {})
        model_id = str(spikebrain_config.get("base_vlm", "")).strip()
        if not model_id:
            raise ValueError("framework.qwenvl.base_vlm must point to SpikeBrain-2.0-VL")

        local_only = os.path.isdir(model_id)
        if local_only:
            model_id = os.path.realpath(model_id)
            index_path = os.path.join(model_id, "model.safetensors.index.json")
            if not os.path.isfile(index_path):
                raise FileNotFoundError(f"SpikeBrain checkpoint index is missing: {index_path}")

        attention_implementation = _resolve_attention_implementation(spikebrain_config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
            attn_implementation=attention_implementation,
        )
        # VLAct SpikeBrain runs uniformly in BF16. This intentionally also
        # casts SpikeBrain's natively-FP32 GDN parameters and attention norms.
        _cast_all_floating_state_to_bf16(self.model)
        from starVLA.model.modules.vlm.spikebrain_liger import (
            apply_spikebrain_liger_from_config,
        )

        liger_status = apply_spikebrain_liger_from_config(
            self.model, spikebrain_config
        )
        logger.info("SpikeBrain Liger status: %s", liger_status)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"

        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            self.model.config.hidden_size = int(text_config.hidden_size)

        self._fail_on_nonfinite = bool(spikebrain_config.get("fail_on_nonfinite", True))
        self._nonfinite_check_interval = max(
            1, int(spikebrain_config.get("nonfinite_check_interval", 1))
        )
        self._forward_calls = 0

        trainer_config = config.get("trainer", {})
        if trainer_config.get("enable_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled for SpikeBrain-2.0-VL")

        self._captured_hidden = None
        lm_head = self._find_lm_head(self.model)
        if lm_head is not None:
            lm_head.register_forward_pre_hook(self._capture_hidden)

        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX
        logger.info(
            "Loaded SpikeBrain-2.0-VL from %s with %s attention; image resolution is controlled by VLAct",
            model_id,
            attention_implementation,
        )

    @staticmethod
    def _find_lm_head(model):
        direct_head = getattr(model, "lm_head", None)
        if isinstance(direct_head, nn.Module):
            return direct_head
        for name, module in model.named_modules():
            if name.endswith("lm_head") and isinstance(module, nn.Linear):
                return module
        return None

    def _capture_hidden(self, _module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            self._captured_hidden = inputs[0]

    @staticmethod
    def _last_hidden(outputs):
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        return None

    @staticmethod
    def _autocast_context():
        if torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        image_grid_thw=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        model_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        self._captured_hidden = None
        self._forward_calls = getattr(self, "_forward_calls", 0) + 1

        with self._autocast_context():
            backbone = getattr(self.model, "model", None)
            if labels is None and isinstance(backbone, nn.Module):
                raw_outputs = backbone(**model_kwargs)
            else:
                raw_outputs = self.model(labels=labels, **model_kwargs)

        last_hidden = self._last_hidden(raw_outputs)
        if last_hidden is None:
            last_hidden = self._captured_hidden
        if last_hidden is None:
            raise RuntimeError("SpikeBrain forward did not expose a final hidden state")

        check_nonfinite = getattr(self, "_fail_on_nonfinite", True) and (
            not self.training
            or self._forward_calls <= 3
            or self._forward_calls % getattr(self, "_nonfinite_check_interval", 1) == 0
        )
        if check_nonfinite and not torch.isfinite(last_hidden).all():
            bad_count = int((~torch.isfinite(last_hidden)).sum().item())
            raise FloatingPointError(
                f"SpikeBrain produced {bad_count} non-finite hidden-state values"
            )

        hidden_states = getattr(raw_outputs, "hidden_states", None)
        if not hidden_states:
            hidden_states = (last_hidden,)
        loss = getattr(raw_outputs, "loss", None)
        if check_nonfinite and loss is not None and not torch.isfinite(loss).all():
            raise FloatingPointError("SpikeBrain produced a non-finite language-model loss")
        return SpikeBrain2VLOutput(
            last_hidden_state=last_hidden,
            hidden_states=tuple(hidden_states),
            logits=getattr(raw_outputs, "logits", None),
            loss=loss,
            past_key_values=getattr(raw_outputs, "past_key_values", None),
            attentions=getattr(raw_outputs, "attentions", None),
        )

    def generate(self, **kwargs):
        with self._autocast_context():
            return self.model.generate(**kwargs)

    def _prompt(self, instruction: str) -> str:
        vla_config = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        template = getattr(vla_config, "CoT_prompt", "") or ""
        return str(template).replace("{instruction}", instruction) if template else instruction

    def _messages(self, images, instructions, solutions=None):
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")
        messages = []
        for index, (sample_images, instruction) in enumerate(zip(images, instructions)):
            if sample_images is None:
                sample_images = []
            elif not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            elif sample_images and isinstance(sample_images[0], (list, tuple)):
                sample_images = sample_images[-1]
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append({"type": "text", "text": self._prompt(instruction)})
            conversation = [{"role": "user", "content": content}]
            if solutions is not None:
                conversation.append(
                    {"role": "assistant", "content": [{"type": "text", "text": solutions[index]}]}
                )
            messages.append(conversation)
        return messages

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        messages = self._messages(images, instructions, solutions)
        texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=solutions is None,
            )
            for message in messages
        ]
        image_processor = getattr(self.processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 16))
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=patch_size,
        )
        processor_kwargs = dict(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        pixel_budget = _image_pixel_budget(image_inputs)
        if pixel_budget is not None:
            processor_kwargs["images_kwargs"] = dict(pixel_budget)
        batch_input = self.processor(**processor_kwargs)

        if solutions is not None:
            labels = batch_input["input_ids"].clone()
            for sequence in labels:
                action_mask = (sequence >= self._ACTION_TOKEN_MIN) & (
                    sequence <= self._ACTION_TOKEN_MAX
                )
                action_positions = torch.nonzero(action_mask, as_tuple=False)
                if action_positions.numel():
                    sequence[: action_positions[0].item()] = IGNORE_INDEX
                else:
                    sequence[:] = IGNORE_INDEX
            pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
            if pad_id is not None:
                labels[labels == pad_id] = IGNORE_INDEX
            batch_input["labels"] = labels

        return batch_input.to(self.model.device)
