import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.transform.video import VideoResize, VideoToTensor


VIDEO_KEYS = ["video.exterior", "video.wrist"]


def test_video_to_tensor_accepts_mixed_camera_resolutions():
    transform = VideoToTensor(apply_to=VIDEO_KEYS)
    transform.train_transform = transform.get_transform("train")

    result = transform(
        {
            VIDEO_KEYS[0]: np.zeros((1, 360, 640, 3), dtype=np.uint8),
            VIDEO_KEYS[1]: np.zeros((1, 480, 640, 3), dtype=np.uint8),
        }
    )

    assert result[VIDEO_KEYS[0]].shape == (1, 3, 360, 640)
    assert result[VIDEO_KEYS[1]].shape == (1, 3, 480, 640)


def test_video_resize_normalizes_mixed_camera_resolutions():
    transform = VideoResize(apply_to=VIDEO_KEYS, height=224, width=224)
    transform.train_transform = transform.get_transform("train")

    result = transform(
        {
            VIDEO_KEYS[0]: torch.zeros((1, 3, 360, 640)),
            VIDEO_KEYS[1]: torch.zeros((1, 3, 480, 640)),
        }
    )

    assert result[VIDEO_KEYS[0]].shape == (1, 3, 224, 224)
    assert result[VIDEO_KEYS[1]].shape == (1, 3, 224, 224)
