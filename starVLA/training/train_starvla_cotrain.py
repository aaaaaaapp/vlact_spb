# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader, make_fixed_vla_eval_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    collect_gradient_norms,
    enable_cuda_fast_math,
    normalize_dotlist_args,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def save_run_configuration(cfg, output_dir: Path, accelerator: Accelerator) -> None:
    """Persist the resolved launch configuration before expensive initialization."""
    if not accelerator.is_main_process:
        return

    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    env_keys = (
        "RUN_ID", "VLA_BS", "VLM_BS", "GRAD_ACCUM", "NUM_PROCESSES",
        "MAX_TRAIN_STEPS", "BASE_LR", "QWEN_LR", "ACTION_LR", "WEIGHT_DECAY", "VLM_PACK",
        "CAMERA_BUCKET_SAMPLING", "CAMERA_BUCKET_WINDOW_SIZE", "EVAL_INTERVAL",
        "EVAL_NUM_BATCHES", "EVAL_SEED", "IS_RESUME", "RESUME_FROM_CHECKPOINT",
    )
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    metadata = {
        "argv": sys.argv,
        "environment_overrides": {key: os.environ[key] for key in env_keys if key in os.environ},
        "git_commit": git_commit,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "world_size": accelerator.num_processes,
    }
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def create_accelerator(cfg) -> Accelerator:
    gradient_accumulation_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=gradient_accumulation_steps)
    # rank0 filters ~100k trajectories before the steps broadcast; default 30-min
    # process-group timeout kills waiting ranks, so allow hours for a cold cache.
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=6))],
    )
    accelerator.print(accelerator.state)
    return accelerator


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    """Prepare co-training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)
    vla_eval_dataloader = None
    if accelerator.is_main_process and int(cfg.trainer.eval_interval) > 0:
        vla_eval_dataloader = make_fixed_vla_eval_dataloader(vla_train_dataloader, cfg)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader, vlm_train_dataloader, vla_eval_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and learning rate scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLAMTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        vlm_train_dataloader,
        vla_eval_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._batch_fetch_executor: ThreadPoolExecutor | None = None
        self._prefetched_vla = None
        self._prefetched_vlm = None
        self._micro_steps = 0
        self._did_optimizer_step = False

    def prepare_training(self):
        is_resume = _as_bool(self.config.trainer.get("is_resume", False))
        if (
            not is_resume
            and hasattr(self.config.trainer, "pretrained_checkpoint")
            and self.config.trainer.pretrained_checkpoint
        ):
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        (
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.vlm_train_dataloader,
        ) = (
            self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vlm_train_dataloader,
            )
        )
        # Keep scheduler stepping tied to one global optimizer update. An
        # AcceleratedScheduler advances once per process when batches are
        # sharded, while max_train_steps here already counts global updates.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        self._init_wandb()
        # Rank0 wandb.init can take minutes. Synchronize before restoring RNG so
        # tracker initialization cannot perturb a resumed run's random state.
        self.accelerator.wait_for_everyone()
        self._init_checkpointing()
        self.accelerator.wait_for_everyone()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        self._wandb_enabled = False
        if not self.accelerator.is_main_process:
            return
        placeholders = {
            "",
            "your_wandb_entity",
            "your-wandb-entity",
            "your_name",
            "liujn",
            "jianingliu",
        }
        entity = str(getattr(self.config, "wandb_entity", "") or os.environ.get("WANDB_ENTITY", "")).strip()
        if entity in placeholders:
            entity = "app123"
        os.environ["WANDB_ENTITY"] = entity
        init_kwargs = {
            "name": self.config.run_id,
            "dir": os.path.join(self.config.output_dir, "wandb"),
            "project": self.config.wandb_project,
            "group": "vla-train",
            "settings": wandb.Settings(init_timeout=180),
        }

        def _try_init(**extra):
            return wandb.init(**init_kwargs, **extra)

        try:
            if entity:
                _try_init(entity=entity)
            else:
                _try_init()
            self._wandb_enabled = True
            return
        except Exception as e:
            logger.warning(f"wandb.init(entity={entity or os.environ.get('WANDB_ENTITY')}) failed: {e}")
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

        os.environ.pop("WANDB_ENTITY", None)
        try:
            logger.warning("retry wandb.init without entity (use logged-in user)")
            _try_init()
            self._wandb_enabled = True
            return
        except Exception as e:
            logger.warning(f"wandb online failed: {e}; falling back to offline")
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

        os.environ["WANDB_MODE"] = "offline"
        _try_init(mode="offline")
        self._wandb_enabled = True

    def _init_checkpointing(self):
        """Initialize checkpointing and restore a complete state when requested."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if not _as_bool(self.config.trainer.get("is_resume", False)):
            return

        checkpoint_path = self.config.trainer.get("resume_from_checkpoint", None)
        if checkpoint_path in (None, "", "latest"):
            candidates = []
            for path in Path(self.checkpoint_dir).glob("steps_*_state"):
                # trainer_state.json is written only after Accelerate finishes
                # saving the distributed state, so it also acts as a completion
                # marker when a previous job was interrupted mid-checkpoint.
                if not (path / "trainer_state.json").is_file():
                    continue
                try:
                    step = int(path.name.removeprefix("steps_").removesuffix("_state"))
                except ValueError:
                    continue
                candidates.append((step, path))
            if not candidates:
                raise FileNotFoundError(
                    f"is_resume=True but no full-state checkpoint exists under {self.checkpoint_dir}"
                )
            _, checkpoint_path = max(candidates, key=lambda item: item[0])

        self._load_checkpoint(str(checkpoint_path))

    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, scheduler, scaler, RNG, and trainer counters."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_dir():
            raise ValueError(
                f"Resume checkpoint must be a full-state directory, got {checkpoint_path}. "
                "Use trainer.pretrained_checkpoint for model-only checkpoints."
            )
        self.accelerator.load_state(str(checkpoint_path))
        trainer_state_path = checkpoint_path / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"Missing resume metadata: {trainer_state_path}")
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        self.completed_steps = int(trainer_state["completed_steps"])
        self._micro_steps = int(
            trainer_state.get(
                "micro_steps",
                self.completed_steps * max(1, int(self.accelerator.gradient_accumulation_steps)),
            )
        )
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save resumable state plus a portable model-only checkpoint."""
        checkpoint_name = f"steps_{self.completed_steps}"
        state_path = Path(self.checkpoint_dir) / f"{checkpoint_name}_state"
        self.accelerator.save_state(str(state_path), safe_serialization=True)

        if self.accelerator.is_main_process:
            (state_path / "trainer_state.json").write_text(
                json.dumps(
                    {"completed_steps": self.completed_steps, "micro_steps": self._micro_steps},
                    indent=2,
                ),
                encoding="utf-8",
            )

        # ZeRO state gathering can be collective, so every rank participates.
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "accessed_config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        logging_frequency = int(self.config.trainer.logging_frequency)
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if logging_frequency > 0 and self.completed_steps % logging_frequency == 0 and is_main_process:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _should_collect_gradient_norms(self):
        """Collect norms only for optimizer steps that will actually be logged."""
        logging_frequency = int(self.config.trainer.logging_frequency)
        next_step = self.completed_steps + 1
        return logging_frequency > 0 and next_step % logging_frequency == 0

    def _create_data_iterators(self):
        """Create data iterators and spawn DataLoader workers on the main thread."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.vlm_train_dataloader)
        if self._batch_fetch_executor is None:
            self._batch_fetch_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="cotrain-batch",
            )
        # Prime both iterators here so worker `fork()` happens before parallel next().
        self._prefetched_vla = next(self.vla_iter)
        self._prefetched_vlm = next(self.vlm_iter)

    def _fetch_vla_batch(self):
        return next(self.vla_iter)

    def _fetch_vlm_batch(self):
        return next(self.vlm_iter)

    def _reset_and_next_vla(self):
        if not hasattr(self, "vla_epoch_count"):
            self.vla_epoch_count = 0
        self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
            self.vla_train_dataloader, self.vla_epoch_count
        )
        return next(self.vla_iter)

    def _reset_and_next_vlm(self):
        if not hasattr(self, "vlm_epoch_count"):
            self.vlm_epoch_count = 0
        self.vlm_iter, self.vlm_epoch_count = TrainerUtils._reset_dataloader(
            self.vlm_train_dataloader, self.vlm_epoch_count
        )
        return next(self.vlm_iter)

    def _get_next_batch(self):
        """Fetch VLA and VLM batches in parallel after workers are already running."""
        if self._batch_fetch_executor is None:
            self._create_data_iterators()

        if self._prefetched_vla is not None:
            batch_vla, batch_vlm = self._prefetched_vla, self._prefetched_vlm
            self._prefetched_vla = None
            self._prefetched_vlm = None
            return batch_vla, batch_vlm

        vla_future = self._batch_fetch_executor.submit(self._fetch_vla_batch)
        vlm_future = self._batch_fetch_executor.submit(self._fetch_vlm_batch)
        try:
            batch_vla = vla_future.result()
        except StopIteration:
            batch_vla = self._reset_and_next_vla()
        try:
            batch_vlm = vlm_future.result()
        except StopIteration:
            batch_vlm = self._reset_and_next_vlm()
        return batch_vla, batch_vlm

    def train(self):
        """Execute training loop."""
        # from_pretrained 默认 eval 模式；不转 train 的话 gradient checkpointing 不生效
        self.model.train()
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
            t_end_model = time.perf_counter()

            if self.accelerator.is_main_process:
                data_s = t_end_data - t_start_data
                model_s = t_end_model - t_start_model
                action = step_metrics.get("action_dit_loss")
                vlm = step_metrics.get("vlm_loss")
                logger.info(
                    "hb micro=%s step=%s opt=%s data=%.3fs model=%.3fs action=%s vlm=%s"
                    % (
                        self._micro_steps,
                        self.completed_steps,
                        int(self._did_optimizer_step),
                        data_s,
                        model_s,
                        f"{float(action):.4f}" if action is not None else "nan",
                        f"{float(vlm):.4f}" if vlm is not None else "nan",
                    )
                )
                try:
                    hb = {
                        "ts": time.time(),
                        "micro": int(self._micro_steps),
                        "step": int(self.completed_steps),
                        "opt": bool(self._did_optimizer_step),
                        "data": data_s,
                        "model": model_s,
                        "action": None if action is None else float(action),
                        "vlm": None if vlm is None else float(vlm),
                    }
                    hb_path = Path(self.config.output_dir) / "train_heartbeat.json"
                    hb_path.write_text(json.dumps(hb), encoding="utf-8")
                except Exception:
                    pass

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if not self._did_optimizer_step:
                continue

            progress_bar.update(1)
            self.completed_steps += 1

            eval_interval = int(self.config.trainer.eval_interval)
            if eval_interval > 0 and self.completed_steps % eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            save_interval = int(self.config.trainer.save_interval)
            if save_interval > 0 and self.completed_steps % save_interval == 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Evaluate on a fixed panel without advancing either training iterator."""
        if self.accelerator.is_main_process:
            if self.vla_eval_dataloader is None:
                raise RuntimeError("VLA evaluation requested without a fixed evaluation dataloader")

            eval_seed = int(self.config.trainer.get("eval_seed", self.config.get("seed", 42)))
            squared_error_sum = 0.0
            valid_element_count = 0
            bucket_totals = {}
            was_training = self.model.training
            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            try:
                self.model.eval()
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(eval_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(eval_seed)
                    for examples in self.vla_eval_dataloader:
                        actions = [example["action"] for example in examples]
                        action_masks = [example.get("action_valid_mask") for example in examples]
                        output_dict = self.model.predict_action(examples=examples)
                        pred_actions, gt_actions, valid_mask = self._align_eval_actions(
                            predicted=output_dict["normalized_actions"],
                            ground_truth=actions,
                            ground_truth_masks=action_masks,
                        )
                        squared_error = np.square(pred_actions - gt_actions) * valid_mask
                        squared_error_sum += float(squared_error.sum())
                        valid_element_count += int(valid_mask.sum())

                        for index, example in enumerate(examples):
                            camera_count = len(example["image"])
                            item_error = float(squared_error[index].sum())
                            item_count = int(valid_mask[index].sum())
                            total_error, total_count = bucket_totals.get(camera_count, (0.0, 0))
                            bucket_totals[camera_count] = (
                                total_error + item_error,
                                total_count + item_count,
                            )
            finally:
                self.model.train(was_training)

            mse = squared_error_sum / valid_element_count if valid_element_count else 0.0
            step_metrics["mse_score"] = float(mse)
            step_metrics["eval_num_samples"] = len(self.vla_eval_dataloader.dataset)
            for camera_count, (error_sum, element_count) in bucket_totals.items():
                step_metrics[f"mse_score_{camera_count}img"] = (
                    float(error_sum / element_count) if element_count else 0.0
                )

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    @staticmethod
    def _align_eval_actions(predicted, ground_truth, ground_truth_masks=None):
        """Pad heterogeneous action tensors for masked eval."""
        gt_list = [np.asarray(action, dtype=np.float32) for action in ground_truth]

        if isinstance(predicted, np.ndarray):
            if predicted.ndim == 2:
                pred_list = [np.asarray(predicted, dtype=np.float32)]
            elif predicted.ndim == 3:
                pred_list = [np.asarray(predicted[idx], dtype=np.float32) for idx in range(predicted.shape[0])]
            else:
                raise ValueError(f"Unsupported predicted action shape: {predicted.shape}")
        elif isinstance(predicted, (list, tuple)):
            pred_list = [np.asarray(action, dtype=np.float32) for action in predicted]
        else:
            raise TypeError(f"Unsupported predicted action type: {type(predicted).__name__}")

        if len(pred_list) != len(gt_list):
            raise ValueError(
                f"Prediction/ground-truth batch size mismatch: {len(pred_list)} vs {len(gt_list)}"
            )

        max_horizon = max(max(pred.shape[0], gt.shape[0]) for pred, gt in zip(pred_list, gt_list))
        max_dim = max(max(pred.shape[-1], gt.shape[-1]) for pred, gt in zip(pred_list, gt_list))

        pred_batch = np.zeros((len(pred_list), max_horizon, max_dim), dtype=np.float32)
        gt_batch = np.zeros((len(gt_list), max_horizon, max_dim), dtype=np.float32)
        valid_mask = np.zeros((len(gt_list), max_horizon, max_dim), dtype=np.float32)

        for idx, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            pred_horizon = min(pred.shape[0], max_horizon)
            pred_dim = min(pred.shape[-1], max_dim)
            gt_horizon = min(gt.shape[0], max_horizon)
            gt_dim = min(gt.shape[-1], max_dim)

            pred_batch[idx, :pred_horizon, :pred_dim] = pred[:pred_horizon, :pred_dim]
            gt_batch[idx, :gt_horizon, :gt_dim] = gt[:gt_horizon, :gt_dim]
            valid_mask[idx, :gt_horizon, :gt_dim] = 1.0

            if ground_truth_masks is not None and ground_truth_masks[idx] is not None:
                item_mask = np.asarray(ground_truth_masks[idx], dtype=np.float32)
                if item_mask.ndim == 1:
                    item_mask = item_mask[None, :]
                mask_horizon = min(item_mask.shape[0], gt_horizon)
                mask_dim = min(item_mask.shape[-1], gt_dim)
                if item_mask.shape[0] == 1:
                    valid_mask[idx, :gt_horizon, :mask_dim] *= item_mask[0, :mask_dim]
                else:
                    valid_mask[idx, :mask_horizon, :mask_dim] *= item_mask[:mask_horizon, :mask_dim]

        return pred_batch, gt_batch, valid_mask

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    @staticmethod
    def _memdbg(tag):
        if os.environ.get("MEMDBG") == "1" and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[MEMDBG] {tag}: alloc={torch.cuda.memory_allocated()/2**30:.1f}G "
                f"reserved={torch.cuda.memory_reserved()/2**30:.1f}G "
                f"max={torch.cuda.max_memory_allocated()/2**30:.1f}G",
                flush=True,
            )

    def _deepspeed_zero_stage(self) -> int:
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return 0
        return int(getattr(plugin, "zero_stage", 0) or 0)

    def _deepspeed_engine(self):
        wrapped = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
        if wrapped is not None and getattr(wrapped, "engine", None) is not None:
            return wrapped.engine
        return self.model

    def _train_step(self, batch_vla, batch_vlm):
        """Execute single training step."""
        log_dict = {}
        gas = max(1, int(self.accelerator.gradient_accumulation_steps))
        self._micro_steps += 1
        self._did_optimizer_step = self._micro_steps % gas == 0
        # ZeRO-2 cannot use accelerate.no_sync(); DeepSpeed already owns GAS.
        # accelerator.backward() also calls engine.step(), so two losses would
        # count as two micro-steps — call engine.backward twice and step once.
        use_accelerate_accum = self._deepspeed_zero_stage() < 2

        if use_accelerate_accum:
            with self.accelerator.accumulate(self.model):
                action_loss, vlm_loss, output_dict = self._backward_cotrain_losses(
                    batch_vla, batch_vlm, backward_fn=self.accelerator.backward
                )
                if self._did_optimizer_step and self._should_collect_gradient_norms():
                    log_dict.update(collect_gradient_norms(self.optimizer))
                if self._did_optimizer_step:
                    if self.config.trainer.gradient_clipping is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.trainer.gradient_clipping,
                        )
                self.optimizer.step()
                if self._did_optimizer_step:
                    self.lr_scheduler.step()
                # AcceleratedOptimizer suppresses zero_grad on accumulation
                # micro-steps and clears gradients after the real update.
                self.optimizer.zero_grad()
        else:
            engine = self._deepspeed_engine()
            engine.set_gradient_accumulation_boundary(False)
            action_loss, vlm_loss, output_dict = self._backward_cotrain_losses(
                batch_vla,
                batch_vlm,
                backward_fn=engine.backward,
                between_backwards=lambda: engine.set_gradient_accumulation_boundary(self._did_optimizer_step),
            )
            if self._did_optimizer_step and self._should_collect_gradient_norms():
                log_dict.update(
                    collect_gradient_norms(self.optimizer, deepspeed_engine=engine)
                )
            engine.step()
            engine.set_gradient_accumulation_boundary(None)
            if self._did_optimizer_step:
                self.lr_scheduler.step()

        log_dict.update(
            {
                "action_dit_loss": action_loss.item(),
                "vlm_loss": vlm_loss.item(),
            }
        )
        if "oft_loss" in output_dict:
            log_dict["oft_loss"] = output_dict["oft_loss"].item()
        if "gr00t_loss" in output_dict:
            log_dict["gr00t_loss"] = output_dict["gr00t_loss"].item()
        if "pi_loss" in output_dict:
            log_dict["pi_loss"] = output_dict["pi_loss"].item()

        return log_dict

    def _backward_cotrain_losses(self, batch_vla, batch_vlm, backward_fn, between_backwards=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
        self._memdbg("after VLA fwd")
        backward_fn(action_loss)
        self._memdbg("after VLA bwd")
        if between_backwards is not None:
            between_backwards()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vlm_output = self.model.qwen_vl_interface(**batch_vlm)
            vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
        self._memdbg("after VLM fwd")
        backward_fn(vlm_loss)
        self._memdbg("after VLM bwd")
        return action_loss, vlm_loss, output_dict

    def _finalize_training(self):
        """Training end processing."""
        if self._batch_fetch_executor is not None:
            self._batch_fetch_executor.shutdown(wait=True)
            self._batch_fetch_executor = None

        # ZeRO state gathering can be collective, so every rank participates.
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")
    enable_cuda_fast_math()
    logger.info(
        "CUDA fast math: tf32_matmul=%s cudnn_benchmark=%s",
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.benchmark,
    )

    accelerator = create_accelerator(cfg)
    output_dir = setup_directories(cfg=cfg)
    save_run_configuration(cfg, output_dir, accelerator)
    # Keep model initialization identical across ranks. Accelerate handles
    # rank-specific dataloader sharding after construction.
    set_seed(int(cfg.get("seed", 42)))

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    vla = build_framework(cfg)
    vla_train_dataloader, vlm_train_dataloader, vla_eval_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        vla_eval_dataloader=vla_eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
