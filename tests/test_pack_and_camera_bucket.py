from types import SimpleNamespace

import numpy as np
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader import (
    CameraBucketBatchDataset,
    _balanced_bucket_window,
    make_fixed_vla_eval_dataloader,
)
from starVLA.dataloader.vlm_datasets import (
    EpochRandomSampler,
    IGNORE_INDEX,
    FlattenedDataCollatorForSupervisedDataset,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class _Tokenizer:
    pad_token_id = 0
    padding_side = "left"
    model_max_length = 32


def _instance(input_ids, labels):
    length = len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "position_ids": torch.arange(length).view(1, 1, -1).expand(3, -1, -1),
    }


def test_pack_masks_segment_starts_and_preserves_boundaries():
    instances = [
        _instance([10, 11, 12], [10, 11, 12]),
        _instance([20, 21], [20, 21]),
    ]

    batch = FlattenedDataCollatorForSupervisedDataset(_Tokenizer())(instances)

    assert batch["cu_seq_lens_q"].dtype == torch.int32
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5]
    assert batch["labels"][0].tolist() == [IGNORE_INDEX, 11, 12, IGNORE_INDEX, 21]
    assert batch["position_ids"][0, 0].tolist() == [0, 1, 2, 0, 1]


class _Mixture:
    def __init__(self):
        self.datasets = [
            SimpleNamespace(modality_keys={"video": ["cam0"]}),
            SimpleNamespace(modality_keys={"video": ["cam0", "cam1"]}),
        ]
        self.dataset_sampling_weights = np.asarray([0.25, 0.75])
        self.epoch = None

    def __len__(self):
        return 64

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_camera_bucket_propagates_epoch_and_keeps_rank_samples_distinct():
    mixture = _Mixture()
    dataset = CameraBucketBatchDataset(mixture, batch_size=4, seed=42)
    dataset.world_size = 2
    dataset.set_epoch(3)

    rank_zero_batch = dataset[0]
    rank_one_batch = dataset[1]

    assert mixture.epoch == 3
    assert [index for index, _ in rank_zero_batch] == [0, 1, 2, 3]
    assert [index for index, _ in rank_one_batch] == [4, 5, 6, 7]
    assert {bucket for _, bucket in rank_zero_batch} == {rank_one_batch[0][1]}


def test_camera_bucket_window_has_balanced_quotas_and_is_shuffled():
    bucket_ids = np.asarray([2, 3])
    bucket_probs = np.asarray([0.37, 0.63])

    first = _balanced_bucket_window(bucket_ids, bucket_probs, 100, 42, 0, 0)
    repeated = _balanced_bucket_window(bucket_ids, bucket_probs, 100, 42, 0, 0)
    next_window = _balanced_bucket_window(bucket_ids, bucket_probs, 100, 42, 0, 1)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, next_window)
    assert dict(zip(*np.unique(first, return_counts=True))) == {2: 37, 3: 63}
    assert np.any(first[:-1] != first[1:])


def test_reset_dataloader_prefers_dataloader_epoch_api():
    class _Loader:
        def __init__(self):
            self.epoch = None

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            return iter(())

    loader = _Loader()
    _, epoch = TrainerUtils._reset_dataloader(loader, 0)

    assert epoch == 1
    assert loader.epoch == 1


def test_vlm_sampler_is_repeatable_and_changes_each_epoch():
    sampler = EpochRandomSampler(list(range(32)), seed=42)
    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_one != epoch_zero
    assert sorted(epoch_one) == list(range(32))


class _EvalMixture(_Mixture):
    def __init__(self):
        super().__init__()
        self.mode = "train"
        self._sequential_step_sampling = True
        self.requested = []
        self.transforms = [_ModeTransform(), _ModeTransform(training=False)]
        for dataset, transforms in zip(self.datasets, self.transforms):
            dataset.transforms = transforms

    def __len__(self):
        return 100

    def __getitem__(self, index):
        sample_index, camera_count = index
        self.requested.append((sample_index, camera_count, self.mode, self._sequential_step_sampling))
        return {
            "sample_index": sample_index,
            "image": [object()] * camera_count,
            "action": np.zeros((2, 3), dtype=np.float32),
            "lang": "test",
        }


class _ModeTransform:
    def __init__(self, training=True):
        self.training = training

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


def test_fixed_eval_panel_is_repeatable_balanced_and_does_not_mutate_training_mode():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "datasets": {"vla_data": {"per_device_batch_size": 4}},
            "trainer": {"eval_batch_size": 4, "eval_num_batches": 2, "eval_seed": 17},
        }
    )
    mixture = _EvalMixture()
    train_loader = DataLoader(mixture, batch_size=4)

    first = make_fixed_vla_eval_dataloader(train_loader, cfg)
    first_signature = [
        (sample["sample_index"], len(sample["image"]))
        for batch in first
        for sample in batch
    ]
    second = make_fixed_vla_eval_dataloader(train_loader, cfg)
    second_signature = [
        (sample["sample_index"], len(sample["image"]))
        for batch in second
        for sample in batch
    ]

    assert first_signature == second_signature
    assert len(first_signature) == 8
    assert [camera_count for _, camera_count in first_signature].count(1) == 2
    assert [camera_count for _, camera_count in first_signature].count(2) == 6
    assert all(mode == "eval" and not sequential for _, _, mode, sequential in mixture.requested)
    assert mixture.mode == "train"
    assert mixture._sequential_step_sampling is True
    assert [transform.training for transform in mixture.transforms] == [True, False]
