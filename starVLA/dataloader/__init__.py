import json
import os
import random
from accelerate.logging import get_logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def _balanced_bucket_window(bucket_ids, bucket_probs, window_size, seed, epoch, window_index):
    """Build one shuffled window with integer quotas closest to bucket probabilities."""
    raw_counts = np.asarray(bucket_probs, dtype=np.float64) * int(window_size)
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(window_size) - int(counts.sum())
    if remainder:
        fractional = raw_counts - counts
        order = np.argsort(-fractional, kind="stable")
        counts[order[:remainder]] += 1

    schedule = np.repeat(np.asarray(bucket_ids, dtype=np.int64), counts)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), int(window_index)])
    )
    rng.shuffle(schedule)
    return schedule


class CameraBucketBatchSampler:
    """Yield batches whose samples come from datasets with the same view count.

    The mixture dataset still chooses the concrete dataset using its configured
    mixture weights; the bucket only constrains the camera-count dimension that
    causes multimodal padding.
    """

    def __init__(self, dataset, batch_size: int, seed: int = 42, window_size: int = 100):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError(f"camera bucket window_size must be positive, got {window_size}")
        self._camera_counts = np.asarray(
            [len(item.modality_keys.get("video", [])) for item in dataset.datasets], dtype=np.int64
        )
        dataset_weights = np.asarray(dataset.dataset_sampling_weights, dtype=np.float64)
        self._bucket_ids = np.unique(self._camera_counts)
        weights = np.asarray(
            [dataset_weights[self._camera_counts == bucket].sum() for bucket in self._bucket_ids],
            dtype=np.float64,
        )
        self._bucket_probs = weights / weights.sum()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if callable(getattr(self.dataset, "set_epoch", None)):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        schedule = None
        current_window = None
        for batch_index in range(len(self)):
            window_index, window_offset = divmod(batch_index, self.window_size)
            if window_index != current_window:
                schedule = _balanced_bucket_window(
                    self._bucket_ids,
                    self._bucket_probs,
                    self.window_size,
                    self.seed,
                    getattr(self, "epoch", 0),
                    window_index,
                )
                current_window = window_index
            bucket = int(schedule[window_offset])
            yield [(batch_index * self.batch_size + offset, bucket) for offset in range(self.batch_size)]


class CameraBucketBatchDataset:
    """Expose a camera-homogeneous batch as one DataLoader sample.

    DeepSpeed requires a concrete DataLoader batch size. Wrapping the logical
    VLA batch this way keeps that outer size at one while preserving the actual
    per-device batch consumed by the model.
    """

    def __init__(self, dataset, batch_size: int, seed: int = 42, window_size: int = 100):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError(f"camera bucket window_size must be positive, got {window_size}")
        self.epoch = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._camera_counts = np.asarray(
            [len(item.modality_keys.get("video", [])) for item in dataset.datasets], dtype=np.int64
        )
        dataset_weights = np.asarray(dataset.dataset_sampling_weights, dtype=np.float64)
        self._bucket_ids = np.unique(self._camera_counts)
        weights = np.asarray(
            [dataset_weights[self._camera_counts == bucket].sum() for bucket in self._bucket_ids],
            dtype=np.float64,
        )
        self._bucket_probs = weights / weights.sum()
        if self.rank == 0:
            preview = _balanced_bucket_window(
                self._bucket_ids,
                self._bucket_probs,
                self.window_size,
                self.seed,
                self.epoch,
                0,
            )
            quotas = {
                int(bucket): int((preview == bucket).sum())
                for bucket in self._bucket_ids
            }
            print(
                "Camera bucket schedule: "
                f"ids={self._bucket_ids.tolist()} "
                f"probs={self._bucket_probs.tolist()} "
                f"window_size={self.window_size} quotas={quotas}"
            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if callable(getattr(self.dataset, "set_epoch", None)):
            self.dataset.set_epoch(epoch)

    def __len__(self):
        # Accelerate/DistributedSampler shards wrapper indices across ranks.
        # Reserve one index per rank for every logical step so all ranks can
        # derive the same bucket while still reading different samples.
        logical_batches = len(self.dataset) // self.batch_size
        return logical_batches * self.world_size

    def __getitem__(self, batch_index):
        batch_index = int(batch_index)
        logical_step = batch_index // self.world_size
        rank_slot = batch_index % self.world_size
        window_index, window_offset = divmod(logical_step, self.window_size)
        schedule = _balanced_bucket_window(
            self._bucket_ids,
            self._bucket_probs,
            self.window_size,
            self.seed,
            self.epoch,
            window_index,
        )
        bucket = int(schedule[window_offset])
        sample_base = (logical_step * self.world_size + rank_slot) * self.batch_size
        return [
            self.dataset[(sample_base + offset, bucket)]
            for offset in range(self.batch_size)
        ]


def _unwrap_camera_bucket_batch(batch):
    if len(batch) != 1:
        raise ValueError(f"Expected one wrapped camera bucket batch, got {len(batch)}")
    return batch[0]


class FixedVLAEvalDataset:
    """In-memory VLA examples materialized once for repeatable evaluation."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def make_fixed_vla_eval_dataloader(train_dataloader, cfg):
    """Materialize a deterministic, camera-balanced evaluation panel on rank 0.

    The panel has its own DataLoader and never advances the training iterator.
    Samples are cached after transforms, so data augmentation is fixed as well.
    """
    batch_size = int(
        cfg.trainer.get(
            "eval_batch_size",
            cfg.datasets.vla_data.per_device_batch_size,
        )
    )
    num_batches = int(cfg.trainer.get("eval_num_batches", 1))
    eval_seed = int(cfg.trainer.get("eval_seed", cfg.get("seed", 42)))
    sample_count = batch_size * num_batches
    if batch_size < 1 or num_batches < 1:
        raise ValueError(
            f"eval_batch_size and eval_num_batches must be positive, got "
            f"{batch_size} and {num_batches}"
        )

    source_dataset = train_dataloader.dataset
    if isinstance(source_dataset, CameraBucketBatchDataset):
        source_dataset = source_dataset.dataset

    rng = np.random.default_rng(eval_seed)
    sample_indices = rng.choice(
        len(source_dataset),
        size=sample_count,
        replace=sample_count > len(source_dataset),
    )

    camera_counts = np.asarray(
        [len(dataset.modality_keys.get("video", [])) for dataset in source_dataset.datasets],
        dtype=np.int64,
    )
    bucket_ids = np.unique(camera_counts)
    dataset_weights = np.asarray(source_dataset.dataset_sampling_weights, dtype=np.float64)
    bucket_weights = np.asarray(
        [dataset_weights[camera_counts == bucket].sum() for bucket in bucket_ids],
        dtype=np.float64,
    )
    bucket_schedule = _balanced_bucket_window(
        bucket_ids,
        bucket_weights / bucket_weights.sum(),
        sample_count,
        eval_seed,
        0,
        0,
    )

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    original_mode = getattr(source_dataset, "mode", None)
    original_sequential = getattr(source_dataset, "_sequential_step_sampling", None)
    transform_training_modes = [
        (dataset.transforms, dataset.transforms.training)
        for dataset in source_dataset.datasets
        if hasattr(dataset, "transforms")
    ]
    examples = []
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(eval_seed)
            if original_mode is not None:
                source_dataset.mode = "eval"
            if original_sequential is not None:
                source_dataset._sequential_step_sampling = False
            for transforms, _ in transform_training_modes:
                transforms.eval()
            for index, camera_count in zip(sample_indices, bucket_schedule):
                examples.append(source_dataset[(int(index), int(camera_count))])
    finally:
        if original_mode is not None:
            source_dataset.mode = original_mode
        if original_sequential is not None:
            source_dataset._sequential_step_sampling = original_sequential
        for transforms, was_training in transform_training_modes:
            transforms.train() if was_training else transforms.eval()
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)

    quotas = {
        int(bucket): int((bucket_schedule == bucket).sum())
        for bucket in bucket_ids
    }
    print(
        "Fixed VLA eval panel: "
        f"samples={sample_count} batch_size={batch_size} "
        f"seed={eval_seed} camera_quotas={quotas}"
    )
    return DataLoader(
        FixedVLAEvalDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        if cfg.datasets.vla_data.get("camera_bucket_sampling", False):
            bucketed_dataset = CameraBucketBatchDataset(
                vla_dataset,
                cfg.datasets.vla_data.per_device_batch_size,
                seed=int(cfg.get("seed", 42)),
                window_size=int(cfg.datasets.vla_data.get("camera_bucket_window_size", 100)),
            )
            vla_train_dataloader = DataLoader(
                bucketed_dataset,
                batch_size=1,
                collate_fn=_unwrap_camera_bucket_batch,
                num_workers=4,
            )
        else:
            vla_train_dataloader = DataLoader(
                vla_dataset,
                batch_size=cfg.datasets.vla_data.per_device_batch_size,
                collate_fn=collate_fn,
                num_workers=4,
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
