# Copyright 2026 starVLA
"""Explicit alignment between SPB2-VL gradient checkpointing and ZeRO-3 parameter gather.

Mirrors lmms-engine FSDP settings:
  fsdp_config.transformer_layer_cls_to_wrap: [SPB2VLTextDecoderLayer]
  (+ SPB2VLVisionBlock for the vision tower)

GC checkpoint unit == ZeRO-3 ``stage3_max_live_parameters`` budget unit (one layer).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Sequence

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Keep in sync with starVLA/config/fsdp/fsdp1_qwengr00t.yaml
DEFAULT_GC_LAYER_CLS_TO_WRAP: tuple[str, ...] = (
    "SPB2VLTextDecoderLayer",
    "SPB2VLVisionBlock",
)

# DiT: checkpoint each BasicTransformerBlock inside DiT.forward (see cross_attention_dit.py)
DEFAULT_DIT_GC_ROOT_CLS: tuple[str, ...] = ("DiT",)
DEFAULT_DIT_GC_BLOCK_CLS: tuple[str, ...] = ("BasicTransformerBlock",)

DEFAULT_ZERO3_MAX_LIVE_PARAMETERS = 100_000_000
DEFAULT_ZERO3_PREFETCH_BUCKET_SIZE = 100_000_000


@dataclass(frozen=True)
class AlignedGCInfo:
    layer_cls_names: tuple[str, ...]
    enabled_layer_count: int
    disabled_module_count: int
    max_layer_param_count: int
    recommended_zero3_max_live_parameters: int
    recommended_zero3_prefetch_bucket_size: int


def _cfg_get(section: Any, key: str, default=None):
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


def _match_layer_cls(module: nn.Module, names: set[str]) -> Optional[str]:
    """Match ``module`` against target class ``names``, returning the matched name.

    FSDP2 ``fully_shard`` rewrites ``module.__class__`` in place to a dynamically
    created subclass named ``FSDP<OriginalName>``. A plain
    ``module.__class__.__name__ in names`` check then fails for every sharded
    layer — and worse, the mismatch arm of enable_aligned_gradient_checkpointing
    actively clears the layer's ``gradient_checkpointing`` flag, silently
    disabling GC for the whole text decoder after FSDP2 wrap. Walk the MRO so
    sharded layers still resolve to their original class name.
    """
    for klass in module.__class__.__mro__:
        cls_name = klass.__name__
        if cls_name in names:
            return cls_name
        if cls_name.startswith("FSDP") and cls_name[len("FSDP"):] in names:
            return cls_name[len("FSDP"):]
    return None


def _trainer_gc_enabled(cfg: Any, section: Any = None) -> bool:
    trainer = _cfg_get(cfg, "trainer", None)
    enable = bool(_cfg_get(section, "enable_gradient_checkpointing", False))
    enable = enable or bool(_cfg_get(trainer, "enable_gradient_checkpointing", False))
    enable = enable or bool(_cfg_get(trainer, "gradient_checkpointing", False))
    return enable


def resolve_gc_layer_cls_names(cfg: Any) -> tuple[str, ...]:
    """Read ``framework.qwenvl.gradient_checkpointing_layer_cls_to_wrap`` or defaults."""
    qwenvl = _cfg_get(_cfg_get(cfg, "framework", None), "qwenvl", None)
    raw = _cfg_get(qwenvl, "gradient_checkpointing_layer_cls_to_wrap", None)
    if raw is None:
        return DEFAULT_GC_LAYER_CLS_TO_WRAP
    if isinstance(raw, str):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return tuple(str(x) for x in raw)


def resolve_dit_gc_block_cls_names(cfg: Any) -> tuple[str, ...]:
    """Per-block GC unit inside DiT (for ZeRO-3 budget estimation)."""
    action_cfg = _cfg_get(_cfg_get(cfg, "framework", None), "action_model", None)
    raw = _cfg_get(action_cfg, "gradient_checkpointing_layer_cls_to_wrap", None)
    if raw is None:
        return DEFAULT_DIT_GC_BLOCK_CLS
    if isinstance(raw, str):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return tuple(str(x) for x in raw)


def merge_aligned_gc_info(*infos: AlignedGCInfo) -> Optional[AlignedGCInfo]:
    valid = [i for i in infos if i is not None and i.enabled_layer_count > 0]
    if not valid:
        return None
    max_params = max(i.max_layer_param_count for i in valid)
    budget = _recommended_zero3_budget(max_params)
    return AlignedGCInfo(
        layer_cls_names=tuple(dict.fromkeys(c for i in valid for c in i.layer_cls_names)),
        enabled_layer_count=sum(i.enabled_layer_count for i in valid),
        disabled_module_count=sum(i.disabled_module_count for i in valid),
        max_layer_param_count=max_params,
        recommended_zero3_max_live_parameters=budget,
        recommended_zero3_prefetch_bucket_size=budget,
    )


def resolve_action_dit(action_head: nn.Module) -> Optional[nn.Module]:
    """Return diffusers ``DiT`` inside ``FlowmatchingActionHead`` (or compatible)."""
    dit = getattr(action_head, "model", None)
    if dit is None:
        return None
    if dit.__class__.__name__ in DEFAULT_DIT_GC_ROOT_CLS or hasattr(dit, "transformer_blocks"):
        return dit
    return None


def enable_dit_gradient_checkpointing(
    action_head: nn.Module,
    cfg: Any,
    *,
    logger: Any = None,
    force: bool = False,
) -> Optional[AlignedGCInfo]:
    """Enable per-block GC on the flow-matching DiT action head.

    Aligns with HF ``GradientCheckpointingLayer`` contract: in addition to the
    DiT root, each ``BasicTransformerBlock`` also gets its own
    ``_gradient_checkpointing_func`` so that the checkpoint path cannot be
    silently disabled by FSDP2 wrapping or by a custom ``__call__`` on a block.
    """
    action_cfg = _cfg_get(_cfg_get(cfg, "framework", None), "action_model", None)
    if not force and not _trainer_gc_enabled(cfg, action_cfg):
        return None

    dit = resolve_action_dit(action_head)
    if dit is None:
        if logger is not None:
            logger.warning("[DiT] gradient_checkpointing requested but no DiT model found on action head")
        return None

    use_reentrant = bool(
        _cfg_get(action_cfg, "gradient_checkpointing_use_reentrant", None)
        or _cfg_get(_cfg_get(cfg, "trainer", None), "gradient_checkpointing_use_reentrant", None)
        or False
    )
    gc_kwargs = {"use_reentrant": use_reentrant}
    if hasattr(dit, "enable_gradient_checkpointing"):
        dit.enable_gradient_checkpointing(gradient_checkpointing_kwargs=gc_kwargs)
    else:
        dit.gradient_checkpointing = True
        dit._gradient_checkpointing_func = partial(checkpoint, **gc_kwargs)

    # Mirror the HF contract used for VLM: each checkpointed block carries its
    # own function. This also makes it easy to verify GC after FSDP2 wrap.
    ckpt_func = partial(checkpoint, **gc_kwargs)
    block_cls = set(resolve_dit_gc_block_cls_names(cfg))
    max_block_params = 0
    block_count = 0
    for module in dit.modules():
        if _match_layer_cls(module, block_cls) is not None:
            block_count += 1
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            module._gradient_checkpointing_func = ckpt_func
            max_block_params = max(
                max_block_params,
                sum(p.numel() for p in module.parameters(recurse=True)),
            )

    budget = _recommended_zero3_budget(max_block_params)
    info = AlignedGCInfo(
        layer_cls_names=tuple(block_cls),
        enabled_layer_count=block_count,
        disabled_module_count=0,
        max_layer_param_count=max_block_params,
        recommended_zero3_max_live_parameters=budget,
        recommended_zero3_prefetch_bucket_size=budget,
    )
    if logger is not None:
        logger.info(
            "[DiT] gradient_checkpointing ALIGNED (use_reentrant=%s) | "
            "blocks=%s count=%d | max_block_params=%.2fM",
            use_reentrant,
            list(block_cls),
            block_count,
            max_block_params / 1e6,
        )
    return info


def verify_dit_gradient_checkpointing(
    action_head: nn.Module,
    cfg: Any,
    *,
    logger: Any = None,
) -> AlignedGCInfo:
    """Return a fresh AlignedGCInfo by inspecting the actual DiT state.

    Logs warnings for blocks that are missing the flag or the checkpoint
    function so that GC failures are visible instead of silent.
    """
    info = enable_dit_gradient_checkpointing(action_head, cfg, logger=logger, force=True)
    if info is not None:
        return info

    dit = resolve_action_dit(action_head)
    if dit is None:
        raise RuntimeError("verify_dit_gradient_checkpointing: no DiT found on action head")

    block_cls = set(resolve_dit_gc_block_cls_names(cfg))
    missing_flag: list[str] = []
    missing_func: list[str] = []
    block_count = 0
    max_block_params = 0
    for name, module in dit.named_modules():
        if _match_layer_cls(module, block_cls) is None:
            continue
        block_count += 1
        max_block_params = max(
            max_block_params,
            sum(p.numel() for p in module.parameters(recurse=True)),
        )
        if not getattr(module, "gradient_checkpointing", False):
            missing_flag.append(name)
        if not callable(getattr(module, "_gradient_checkpointing_func", None)):
            missing_func.append(name)

    if logger is not None and (missing_flag or missing_func):
        logger.warning(
            "[DiT] GC verification found issues: missing_flag=%d blocks, "
            "missing_func=%d blocks (of %d). Re-run enable_dit_gradient_checkpointing.",
            len(missing_flag),
            len(missing_func),
            block_count,
        )

    budget = _recommended_zero3_budget(max_block_params)
    return AlignedGCInfo(
        layer_cls_names=tuple(block_cls),
        enabled_layer_count=block_count - len(missing_flag),
        disabled_module_count=0,
        max_layer_param_count=max_block_params,
        recommended_zero3_max_live_parameters=budget,
        recommended_zero3_prefetch_bucket_size=budget,
    )


def should_align_zero3_with_gc(cfg: Any) -> bool:
    qwenvl = _cfg_get(_cfg_get(cfg, "framework", None), "qwenvl", None)
    if qwenvl is None:
        return True
    explicit = _cfg_get(qwenvl, "zero3_align_gc_with_layers", None)
    if explicit is not None:
        return bool(explicit)
    return bool(_cfg_get(qwenvl, "zero3_align_gc_layers", True))


def maybe_enable_framework_gradient_checkpointing(
    framework: nn.Module,
    cfg: Any,
    *,
    logger: Any = None,
) -> Optional[AlignedGCInfo]:
    """Enable aligned GC on VLM (if interface already built) and DiT action head."""
    vlm_info: Optional[AlignedGCInfo] = None
    if hasattr(framework, "qwen_vl_interface"):
        vlm_info = maybe_enable_gradient_checkpointing_from_config(
            framework.qwen_vl_interface,
            cfg,
            logger=logger,
        )
    dit_info: Optional[AlignedGCInfo] = None
    if hasattr(framework, "action_model"):
        dit_info = enable_dit_gradient_checkpointing(
            framework.action_model,
            cfg,
            logger=logger,
        )
    merged = merge_aligned_gc_info(vlm_info, dit_info)
    if merged is not None:
        validate_zero3_gc_alignment(cfg, merged, logger=logger)
    return merged


def resolve_pretrained_vlm_root(model: nn.Module) -> nn.Module:
    """Unwrap PEFT / QwenGR00T to SPB2VLForConditionalGeneration (or compatible)."""
    target = model
    if hasattr(target, "get_base_model"):
        target = target.get_base_model()
    if hasattr(target, "qwen_vl_interface"):
        target = target.qwen_vl_interface
    if hasattr(target, "model") and hasattr(target.model, "language_model"):
        inner = target.model
        if hasattr(inner, "gradient_checkpointing_enable") or hasattr(inner, "config"):
            return target
    if hasattr(target, "language_model"):
        return target
    return target


def _recommended_zero3_budget(max_layer_params: int, *, headroom: float = 1.05) -> int:
    if max_layer_params <= 0:
        return DEFAULT_ZERO3_MAX_LIVE_PARAMETERS
    return int(max_layer_params * headroom)


def enable_aligned_gradient_checkpointing(
    model: nn.Module,
    layer_cls_names: Sequence[str],
    *,
    use_reentrant: bool = False,
    require_all_layer_classes: bool = False,
    logger: Any = None,
) -> AlignedGCInfo:
    """
    Enable HF gradient checkpointing only on named layer classes; disable elsewhere.

    Must match Transformers ``PreTrainedModel._set_gradient_checkpointing``: every module
    with ``gradient_checkpointing=True`` also needs ``_gradient_checkpointing_func``.
    ``GradientCheckpointingLayer.__call__`` uses that per-layer func; setting it only on
    the root is a no-op / crash for SPB2VL decoder & vision blocks.
    """
    root = resolve_pretrained_vlm_root(model)
    names = set(layer_cls_names)
    ckpt_func = partial(checkpoint, use_reentrant=use_reentrant)
    root._gradient_checkpointing_func = ckpt_func

    enabled = 0
    disabled = 0
    max_layer_params = 0
    missing_func = 0
    enabled_by_class = {name: 0 for name in names}
    seen_by_class = {name: 0 for name in names}

    for name, module in root.named_modules():
        matched_cls = _match_layer_cls(module, names)
        if matched_cls is not None:
            seen_by_class[matched_cls] += 1
        if not hasattr(module, "gradient_checkpointing"):
            continue
        if matched_cls is not None:
            # HF contract: flag + checkpoint fn on the *same* module that overrides __call__.
            module._gradient_checkpointing_func = ckpt_func
            module.gradient_checkpointing = True
            if not callable(getattr(module, "_gradient_checkpointing_func", None)):
                missing_func += 1
            nparams = sum(p.numel() for p in module.parameters(recurse=True))
            max_layer_params = max(max_layer_params, nparams)
            enabled += 1
            enabled_by_class[matched_cls] += 1
        else:
            module.gradient_checkpointing = False
            disabled += 1

    if hasattr(root, "gradient_checkpointing"):
        root.gradient_checkpointing = enabled > 0

    # Strict mode guards against silent GC loss: a configured layer class that
    # is absent from the model (e.g. renamed by a remote-code update) must fail
    # loudly, not silently match zero layers. "Missing" covers both
    # never-seen (seen == 0) and seen-but-not-enabled (enabled == 0).
    missing_classes = [
        cls_name
        for cls_name in layer_cls_names
        if seen_by_class.get(cls_name, 0) == 0 or enabled_by_class.get(cls_name, 0) == 0
    ]
    if require_all_layer_classes and (missing_classes or missing_func):
        raise RuntimeError(
            "[SpikeBrain] strict GC contract failed: "
            f"missing_or_ineligible_classes={missing_classes}, "
            f"seen={seen_by_class}, enabled={enabled_by_class}, "
            f"missing_checkpoint_func={missing_func}"
        )

    cfg = getattr(root, "config", None)
    if cfg is not None and hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    budget = _recommended_zero3_budget(max_layer_params)
    info = AlignedGCInfo(
        layer_cls_names=tuple(layer_cls_names),
        enabled_layer_count=enabled,
        disabled_module_count=disabled,
        max_layer_param_count=max_layer_params,
        recommended_zero3_max_live_parameters=budget,
        recommended_zero3_prefetch_bucket_size=budget,
    )
    if logger is not None:
        logger.info(
            "[SpikeBrain] gradient_checkpointing ALIGNED (use_reentrant=%s) | "
            "layers=%s enabled=%d disabled_elsewhere=%d | "
            "max_layer_params=%.2fM | ZeRO-3 recommend stage3_max_live_parameters>=%d",
            use_reentrant,
            list(layer_cls_names),
            enabled,
            disabled,
            max_layer_params / 1e6,
            budget,
        )
        if missing_func:
            logger.warning(
                "[SpikeBrain] %d GC layers missing _gradient_checkpointing_func after enable",
                missing_func,
            )
        elif enabled == 0:
            logger.warning(
                "[SpikeBrain] gradient_checkpointing requested but matched 0 layers for %s",
                list(layer_cls_names),
            )
    return info


def maybe_enable_gradient_checkpointing_from_config(
    model: nn.Module,
    cfg: Any,
    *,
    sb_cfg: Any = None,
    logger: Any = None,
    ignore_lm_ablation: bool = False,
) -> Optional[AlignedGCInfo]:
    """Enable aligned GC when trainer / qwenvl flags request it.

    Args:
        model: VLM root (or a wrapper that ``resolve_pretrained_vlm_root`` can
            unwrap to the actual SPB2VL model).
        cfg: Full trainer/framework config.
        sb_cfg: Optional framework.qwenvl section; inferred from ``cfg`` if None.
        logger: Optional logger.
        ignore_lm_ablation: When True, ignore ``STARVLA_FSDP2_LM_ABLATION``.
            This is the safe default for full VLA training, where the VLM GC
            must be active even if an LM-ablation shell leaked the env var.
    """
    sb_cfg = sb_cfg or _cfg_get(_cfg_get(cfg, "framework", None), "qwenvl", None)
    ablation = os.environ.get("STARVLA_FSDP2_LM_ABLATION", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if ablation and not ignore_lm_ablation:
        # FSDP2 LM-only probes set this; it must not leak into full VLA training.
        if logger is not None and _trainer_gc_enabled(cfg, sb_cfg):
            logger.warning(
                "[SpikeBrain] skip aligned VLM GC: STARVLA_FSDP2_LM_ABLATION=1 "
                "(unset for full VLA / pack+GC runs; DiT GC is unaffected)"
            )
        return None
    if not _trainer_gc_enabled(cfg, sb_cfg):
        return None

    layer_cls = resolve_gc_layer_cls_names(cfg)
    use_reentrant = bool(
        _cfg_get(sb_cfg, "gradient_checkpointing_use_reentrant", None)
        or _cfg_get(_cfg_get(cfg, "trainer", None), "gradient_checkpointing_use_reentrant", None)
        or False
    )
    strict = bool(_cfg_get(sb_cfg, "gradient_checkpointing_strict", True))
    return enable_aligned_gradient_checkpointing(
        model,
        layer_cls,
        use_reentrant=use_reentrant,
        require_all_layer_classes=strict,
        logger=logger,
    )


def force_enable_vlm_gradient_checkpointing(
    model: nn.Module,
    cfg: Any,
    *,
    sb_cfg: Any = None,
    logger: Any = None,
) -> Optional[AlignedGCInfo]:
    """Same as ``maybe_enable_gradient_checkpointing_from_config`` but ignores
    ``STARVLA_FSDP2_LM_ABLATION`` and always applies when the trainer flags
    request GC. Use this for full VLA training to guarantee VLM GC is on.
    """
    return maybe_enable_gradient_checkpointing_from_config(
        model,
        cfg,
        sb_cfg=sb_cfg,
        logger=logger,
        ignore_lm_ablation=True,
    )


def verify_vlm_gradient_checkpointing(
    model: nn.Module,
    cfg: Any,
    *,
    sb_cfg: Any = None,
    logger: Any = None,
) -> AlignedGCInfo:
    """Inspect the actual VLM state and return a fresh AlignedGCInfo.

    Logs warnings for target layer classes that are missing the flag or the
    per-layer checkpoint function. The trainer loop can call this after
    FSDP2/DeepSpeed prepare to confirm GC was not lost during wrap.
    """
    layer_cls = resolve_gc_layer_cls_names(cfg)
    root = resolve_pretrained_vlm_root(model)
    enabled = 0
    missing_flag: list[str] = []
    missing_func: list[str] = []
    max_layer_params = 0
    for name, module in root.named_modules():
        if _match_layer_cls(module, set(layer_cls)) is None:
            continue
        nparams = sum(p.numel() for p in module.parameters(recurse=True))
        max_layer_params = max(max_layer_params, nparams)
        if getattr(module, "gradient_checkpointing", False):
            enabled += 1
        else:
            missing_flag.append(name)
        if not callable(getattr(module, "_gradient_checkpointing_func", None)):
            missing_func.append(name)

    if logger is not None and (missing_flag or missing_func):
        logger.warning(
            "[SpikeBrain] GC verification found issues: missing_flag=%d layers, "
            "missing_func=%d layers (target classes=%s).",
            len(missing_flag),
            len(missing_func),
            list(layer_cls),
        )

    # HF GradientCheckpointingLayer.__call__ only checkpoints when the module is
    # in train mode. from_pretrained() leaves the model in eval() mode, so if no
    # one calls model.train() afterwards, GC is silently skipped and the forward
    # pass keeps every activation. Surface that here instead of staying silent.
    if logger is not None and enabled > 0 and not root.training:
        logger.warning(
            "[SpikeBrain] GC layers are enabled but the VLM is in eval() mode "
            "(training=False). GradientCheckpointingLayer.__call__ will SKIP "
            "checkpointing and the forward will NOT save memory. "
            "Call model.train() before the training loop."
        )

    budget = _recommended_zero3_budget(max_layer_params)
    return AlignedGCInfo(
        layer_cls_names=tuple(layer_cls),
        enabled_layer_count=enabled,
        disabled_module_count=0,
        max_layer_param_count=max_layer_params,
        recommended_zero3_max_live_parameters=budget,
        recommended_zero3_prefetch_bucket_size=budget,
    )


def _resolve_deepspeed_json_path() -> Optional[Path]:
    def _resolve_ref(owner: Path, raw_ref: Any) -> Optional[Path]:
        if not raw_ref:
            return None
        ref = Path(str(raw_ref))
        candidates = [ref] if ref.is_absolute() else [Path.cwd() / ref, owner.parent / ref]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve() if candidates else None

    ds_yaml = os.environ.get("STARVLA_DS_CONFIG", "").strip()
    if ds_yaml:
        yaml_path = Path(ds_yaml)
        if yaml_path.suffix == ".json" and yaml_path.is_file():
            return yaml_path
        if yaml_path.is_file():
            try:
                import yaml

                data = yaml.safe_load(yaml_path.read_text())
                rel = _cfg_get(_cfg_get(data, "deepspeed_config", None), "deepspeed_config_file", None)
                if rel:
                    return _resolve_ref(yaml_path, rel)
            except Exception:
                pass
    accel = os.environ.get("ACCELERATE_CONFIG_FILE", "").strip()
    if accel:
        accel_path = Path(accel)
        if accel_path.is_file():
            try:
                import yaml

                data = yaml.safe_load(accel_path.read_text())
                rel = _cfg_get(_cfg_get(data, "deepspeed_config", None), "deepspeed_config_file", None)
                if rel:
                    return _resolve_ref(accel_path, rel)
            except Exception:
                pass
    return None


def validate_zero3_gc_alignment(
    cfg: Any,
    gc_info: AlignedGCInfo,
    *,
    logger: Any = None,
) -> None:
    """Log a warning if DeepSpeed JSON ``stage3_max_live_parameters`` is misaligned with GC layers."""
    if not should_align_zero3_with_gc(cfg):
        return

    ds_path = _resolve_deepspeed_json_path()
    if ds_path is None or not ds_path.is_file():
        if logger is not None:
            logger.info(
                "[SpikeBrain] ZeRO-3/GC align: no DeepSpeed JSON found "
                "(set STARVLA_DS_CONFIG or use accelerate --config_file); "
                "recommend stage3_max_live_parameters=%d",
                gc_info.recommended_zero3_max_live_parameters,
            )
        return

    try:
        ds_cfg = json.loads(ds_path.read_text())
    except Exception as exc:
        if logger is not None:
            logger.warning("[SpikeBrain] ZeRO-3/GC align: failed to read %s: %s", ds_path, exc)
        return

    zero = ds_cfg.get("zero_optimization", {})
    live = int(zero.get("stage3_max_live_parameters", 0) or 0)
    prefetch = int(zero.get("stage3_prefetch_bucket_size", 0) or 0)
    need = gc_info.recommended_zero3_max_live_parameters

    if logger is None:
        return

    if live > 0 and live < need:
        logger.warning(
            "[SpikeBrain] ZeRO-3/GC MISALIGNED: %s stage3_max_live_parameters=%d < "
            "largest GC layer %.2fM params (recommend >= %d). "
            "Use zero3_longpack.json or set framework.qwenvl.zero3_max_live_parameters.",
            ds_path.name,
            live,
            gc_info.max_layer_param_count / 1e6,
            need,
        )
    else:
        logger.info(
            "[SpikeBrain] ZeRO-3/GC aligned: %s stage3_max_live_parameters=%s "
            "(GC max layer %.2fM, recommend >= %d) prefetch=%s",
            ds_path.name,
            live if live else "auto",
            gc_info.max_layer_param_count / 1e6,
            need,
            prefetch if prefetch else "auto",
        )


def patch_zero3_json_for_gc_layers(
    json_path: Path,
    gc_info: AlignedGCInfo,
    *,
    logger: Any = None,
) -> Path:
    """
    Write an aligned copy of a DeepSpeed ZeRO-3 JSON next to the original.

    Returns path to the patched file (``*_gc_aligned.json``).
    """
    data = json.loads(json_path.read_text())
    zero = data.setdefault("zero_optimization", {})
    budget = gc_info.recommended_zero3_max_live_parameters
    zero["stage3_max_live_parameters"] = budget
    zero["stage3_prefetch_bucket_size"] = budget
    zero["stage3_max_reuse_distance"] = 0

    out = json_path.with_name(json_path.stem + "_gc_aligned.json")
    out.write_text(json.dumps(data, indent=4) + "\n")
    if logger is not None:
        logger.info(
            "[SpikeBrain] wrote ZeRO-3/GC aligned config: %s (stage3_max_live_parameters=%d)",
            out,
            budget,
        )
    return out
