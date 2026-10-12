"""Dataset registration kept local to the aloha_real_ljn example."""

import random
from typing import Any

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.data_config import AgilexDataConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionDropout,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResizeWithPad,
    VideoToNumpy,
    VideoToTensor,
)


class AlohaGripperContinuous(ModalityTransform):
    """Map raw ALOHA gripper qpos to a continuous [-1, 1] model value."""

    ranges: dict[str, tuple[float, float]]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # 部署侧 normalize_state 只传 state key，缺 action key 时跳过
                continue
            lower, upper = self.ranges[key]
            value = data[key]
            if isinstance(value, torch.Tensor):
                normalized = (value - lower) / (upper - lower)
                data[key] = torch.clamp(2.0 * normalized - 1.0, -1.0, 1.0)
            else:
                normalized = (np.asarray(value) - lower) / (upper - lower)
                data[key] = np.clip(2.0 * normalized - 1.0, -1.0, 1.0)
        return data


class AlohaStateGripperToActionScale(ModalityTransform):
    """Convert measured state gripper qpos into the action command scale."""

    source_ranges: dict[str, tuple[float, float]]
    target_ranges: dict[str, tuple[float, float]]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            source_lower, source_upper = self.source_ranges[key]
            target_lower, target_upper = self.target_ranges[key]
            value = data[key]
            ratio = (value - source_lower) / (source_upper - source_lower)
            data[key] = target_lower + ratio * (target_upper - target_lower)
        return data


class AlohaImageAugmentationGate(ModalityTransform):
    """Apply the complete image augmentation branch to half of train samples."""

    augmentation_transforms: list[ModalityTransform]
    probability: float = 0.5

    def set_metadata(self, dataset_metadata):
        super().set_metadata(dataset_metadata)
        for transform in self.augmentation_transforms:
            transform.set_metadata(dataset_metadata)

    def train(self):
        self.training = True
        for transform in self.augmentation_transforms:
            transform.train()

    def eval(self):
        self.training = False
        for transform in self.augmentation_transforms:
            transform.eval()

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training or random.random() >= self.probability:
            return data
        for transform in self.augmentation_transforms:
            data = transform(data)
        return data


class VideoBranchSkipMissing(ModalityTransform):
    """Only run the video sub-transforms when the video keys are present.

    训练样本始终包含 video key，行为与直接串联完全一致；但部署侧
    PolicyNormProcessor 会复用同一条 ComposedModalityTransform 处理
    纯 state（normalize_state）/ 纯 action（unapply_actions）字典，
    此时 video key 缺失，共享的 VideoToTensor/VideoResizeWithPad 等会在
    check_input 断言失败。该包装层在缺 key 时整体跳过视频分支。
    """

    video_keys: list[str]
    sub_transforms: list[ModalityTransform]

    def set_metadata(self, dataset_metadata):
        super().set_metadata(dataset_metadata)
        if not getattr(dataset_metadata.modalities, "video", None):
            # 部署侧 PolicyNormProcessor 构造的 metadata 不含视频信息（video={}），
            # VideoTransform.set_metadata 会因找不到分辨率而抛错；视频分支在缺 key
            # 时本就不会执行，这里跳过子变换的元数据初始化。训练侧 metadata 含视频
            # 信息，正常初始化。
            return
        for transform in self.sub_transforms:
            transform.set_metadata(dataset_metadata)

    def train(self):
        self.training = True
        for transform in self.sub_transforms:
            transform.train()

    def eval(self):
        self.training = False
        for transform in self.sub_transforms:
            transform.eval()

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if any(key not in data for key in self.video_keys):
            return data
        for transform in self.sub_transforms:
            data = transform(data)
        return data


class AlohaRealLjnDataConfig(AgilexDataConfig):
    """Three-camera, 14D ALOHA follower data in runtime joint order."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    action_indices = list(range(50))

    def transform(self):
        video_keys = list(self.video_keys)
        state_grippers = ["state.left_gripper", "state.right_gripper"]
        action_grippers = ["action.left_gripper", "action.right_gripper"]
        state_joints = ["state.left_joints", "state.right_joints"]
        action_joints = ["action.left_joints", "action.right_joints"]
        state_gripper_source_ranges = {
            "state.left_gripper": (-0.022141, 0.371891),
            "state.right_gripper": (-0.025066, 0.369284),
        }
        gripper_action_ranges = {
            "left": (-0.359344, 4.40000),
            "right": (-0.354767, 4.40000),
        }
        state_gripper_target_ranges = {
            "state.left_gripper": gripper_action_ranges["left"],
            "state.right_gripper": gripper_action_ranges["right"],
        }
        continuous_ranges = {
            "state.left_gripper": gripper_action_ranges["left"],
            "state.right_gripper": gripper_action_ranges["right"],
            "action.left_gripper": gripper_action_ranges["left"],
            "action.right_gripper": gripper_action_ranges["right"],
        }
        return ComposedModalityTransform(
            transforms=[
                VideoBranchSkipMissing(
                    apply_to=video_keys,
                    video_keys=video_keys,
                    sub_transforms=[
                        VideoToTensor(apply_to=video_keys),
                        VideoResizeWithPad(apply_to=video_keys, height=224, width=224),
                        AlohaImageAugmentationGate(
                            apply_to=video_keys,
                            probability=0.5,
                            augmentation_transforms=[
                                VideoCrop(
                                    apply_to=video_keys,
                                    height=224,
                                    width=224,
                                    scale=0.95,
                                ),
                                VideoResizeWithPad(apply_to=video_keys, height=224, width=224),
                                VideoColorJitter(
                                    apply_to=video_keys,
                                    brightness=0.2,
                                    contrast=0.2,
                                    saturation=0.2,
                                    hue=0.04,
                                ),
                            ],
                        ),
                        VideoToNumpy(apply_to=video_keys),
                    ],
                ),
                AlohaStateGripperToActionScale(
                    apply_to=state_grippers,
                    source_ranges=state_gripper_source_ranges,
                    target_ranges=state_gripper_target_ranges,
                ),
                StateActionToTensor(apply_to=self.state_keys + self.action_keys),
                StateActionTransform(
                    apply_to=state_joints,
                    normalization_modes={key: "q99" for key in state_joints},
                ),
                StateActionTransform(
                    apply_to=action_joints,
                    normalization_modes={key: "q99" for key in action_joints},
                ),
                AlohaGripperContinuous(
                    apply_to=state_grippers + action_grippers,
                    ranges=continuous_ranges,
                ),
                StateActionDropout(
                    apply_to=self.state_keys,
                    dropout_prob=0.5,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"aloha_real_ljn": AlohaRealLjnDataConfig()}

DATASET_NAMED_MIXTURES = {
    "aloha_real_ljn": [
        (
            "fold_shoebox_and_insert_shoes_data_",
            1.0,
            "aloha_real_ljn",
        ),
    ],
    "aloha_real_ljn_clean_1e-3": [
        (
            "fold_shoebox_and_insert_shoes_data_clean_1e-3",
            1.0,
            "aloha_real_ljn",
        ),
    ],
    "aloha_real_ljn_clean_1e-3_drop_glitch_train": [
        (
            "fold_shoebox_and_insert_shoes_data_clean_1e-3_drop_glitch_train",
            1.0,
            "aloha_real_ljn",
        ),
    ],
    "aloha_real_ljn_clean_1e-3_drop_glitch_val": [
        (
            "fold_shoebox_and_insert_shoes_data_clean_1e-3_drop_glitch_val",
            1.0,
            "aloha_real_ljn",
        ),
    ],
}
