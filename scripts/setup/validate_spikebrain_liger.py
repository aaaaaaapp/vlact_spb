#!/usr/bin/env python3
"""Full-model numerical comparison for native vs guarded SpikeBrain Liger ops."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from starVLA.model.modules.vlm.SpikeBrain2VL import _cast_all_floating_state_to_bf16
from starVLA.model.modules.vlm.spikebrain_liger import apply_liger_kernel_to_spikebrain


DEFAULT_MODEL = "/mnt/jfzn/liujn/code/Starvla/Starvla/spikebrain/SpikeBrain-2.0-VL"
PARAMETERS_TO_COMPARE = (
    "lm_head.weight",
    "model.language_model.norm.weight",
    "model.language_model.layers.18.input_layernorm.weight",
    "model.language_model.layers.18.mlp.gate_proj.weight",
    "model.language_model.layers.35.post_attention_layernorm.weight",
    "model.language_model.layers.35.mlp.down_proj.weight",
)


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    difference = (reference - actual).abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "relative_l2": difference.norm().item() / max(reference.norm().item(), 1e-12),
    }


def sample_indices(numel: int, sample_size: int, device: torch.device) -> torch.Tensor:
    if numel <= sample_size:
        return torch.arange(numel, device=device)
    # Avoid float32 linspace rounding beyond numel for very large matrices such
    # as SpikeBrain's tied 151936 x 2560 embedding/lm_head weight.
    positions = torch.arange(sample_size, device=device, dtype=torch.int64)
    return positions.mul(numel - 1).div(sample_size - 1, rounding_mode="floor")


def selected_gradient_state(model: nn.Module, sample_size: int) -> dict:
    # SpikeBrain ties lm_head.weight to the input embedding.  Keep duplicate
    # names so the lm_head alias remains addressable for the comparison.
    named = dict(model.named_parameters(remove_duplicate=False))
    state = {}
    for name in PARAMETERS_TO_COMPARE:
        parameter = named[name]
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        flat = parameter.grad.detach().reshape(-1)
        indices = sample_indices(flat.numel(), sample_size, flat.device)
        state[name] = {
            "sample": flat.index_select(0, indices).float().cpu(),
            "norm": flat.float().norm().item(),
            "finite": bool(torch.isfinite(flat).all().item()),
        }
    return state


def validate_parameter_names(model: nn.Module) -> None:
    named = dict(model.named_parameters(remove_duplicate=False))
    missing = [name for name in PARAMETERS_TO_COMPARE if name not in named]
    if missing:
        raise RuntimeError(f"comparison parameters not found: {missing}")


def run_pass(model, inputs, capture, sample_size):
    model.zero_grad(set_to_none=True)
    capture.clear()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1234)
    started = time.perf_counter()
    output = model(**inputs)
    forward_seconds = time.perf_counter() - started
    if not capture:
        hidden_states = output.hidden_states[-1] if output.hidden_states else None
    else:
        hidden_states = capture[-1]
    if hidden_states is None:
        raise RuntimeError("failed to capture final hidden states")
    hidden_copy = hidden_states.detach().float().cpu()
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"invalid loss: {loss}")
    started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - started
    gradients = selected_gradient_state(model, sample_size)
    return {
        "loss": loss.detach().float().item(),
        "hidden": hidden_copy,
        "gradients": gradients,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def compare_passes(reference, actual):
    loss_abs = abs(reference["loss"] - actual["loss"])
    result = {
        "loss": {
            "reference": reference["loss"],
            "actual": actual["loss"],
            "abs": loss_abs,
            "relative": loss_abs / max(abs(reference["loss"]), 1e-12),
        },
        "hidden": tensor_metrics(reference["hidden"], actual["hidden"]),
        "gradients": {},
    }
    for name in PARAMETERS_TO_COMPARE:
        reference_gradient = reference["gradients"][name]
        actual_gradient = actual["gradients"][name]
        metrics = tensor_metrics(reference_gradient["sample"], actual_gradient["sample"])
        metrics.update(
            {
                "reference_norm": reference_gradient["norm"],
                "actual_norm": actual_gradient["norm"],
                "norm_relative": abs(reference_gradient["norm"] - actual_gradient["norm"])
                / max(reference_gradient["norm"], 1e-12),
                "reference_finite": reference_gradient["finite"],
                "actual_finite": actual_gradient["finite"],
            }
        )
        result["gradients"][name] = metrics
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--gradient-samples", type=int, default=4096)
    parser.add_argument("--rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rms-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--swiglu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fused-linear-cross-entropy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    model_path = os.path.realpath(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    _cast_all_floating_state_to_bf16(model)
    validate_parameter_names(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()

    capture = []

    def capture_hidden(_module, module_inputs):
        capture.append(module_inputs[0].detach())

    model.lm_head.register_forward_pre_hook(capture_hidden)
    vocab_size = int(model.config.text_config.vocab_size)
    torch.manual_seed(101)
    input_ids = torch.randint(
        100, min(vocab_size, 100_000), (1, args.sequence_length), device=device
    )
    labels = input_ids.clone()
    labels[:, : args.sequence_length // 2] = -100
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "use_cache": False,
    }

    native_warmup = run_pass(model, inputs, capture, args.gradient_samples)
    print("[validation] native warmup forward/backward complete", flush=True)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    native = run_pass(model, inputs, capture, args.gradient_samples)
    print("[validation] native reference forward/backward complete", flush=True)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    liger_status = apply_liger_kernel_to_spikebrain(
        model,
        rope=args.rope,
        fused_linear_cross_entropy=args.fused_linear_cross_entropy,
        rms_norm=args.rms_norm,
        swiglu=args.swiglu,
    )
    print(f"[validation] Liger status: {liger_status}", flush=True)
    guarded = run_pass(model, inputs, capture, args.gradient_samples)
    print("[validation] Liger forward/backward complete", flush=True)

    native_repeatability = compare_passes(native_warmup, native)
    liger_comparison = compare_passes(native, guarded)
    result = {
        "model": model_path,
        "dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "requested": {
            "rope": args.rope,
            "rms_norm": args.rms_norm,
            "swiglu": args.swiglu,
            "fused_linear_cross_entropy": args.fused_linear_cross_entropy,
        },
        "liger_status": liger_status,
        "native_repeatability": native_repeatability,
        "liger_comparison": liger_comparison,
        "timing": {
            "warmup_native_forward_seconds": native_warmup["forward_seconds"],
            "warmup_native_backward_seconds": native_warmup["backward_seconds"],
            "native_forward_seconds": native["forward_seconds"],
            "native_backward_seconds": native["backward_seconds"],
            "liger_forward_seconds": guarded["forward_seconds"],
            "liger_backward_seconds": guarded["backward_seconds"],
        },
        "peak_memory_gib": {
            "warmup_native": native_warmup["peak_memory_gib"],
            "native": native["peak_memory_gib"],
            "liger": guarded["peak_memory_gib"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
