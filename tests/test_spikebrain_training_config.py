from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
    collect_gradient_norms,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SCRIPT = ROOT / "scripts/run_scripts/Pretrain/pretrain_spikebrain_single_node.sh"


class _MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(2, 2)
        self.oft_action_model = nn.Linear(2, 2)
        self.gr00t_action_model = nn.Linear(2, 2)
        self.pi_action_model = nn.Linear(2, 2)
        self.other = nn.Linear(2, 2)


def test_multi_head_lr_alias_and_weight_decay():
    model = _MultiHeadModel()
    cfg = OmegaConf.create(
        {
            "trainer": {
                "learning_rate": {
                    "base": 1e-5,
                    "qwen_vl_interface": 2e-6,
                    "action_model": 1e-4,
                },
                "optimizer": {"weight_decay": 0.01},
                "freeze_modules": "",
            }
        }
    )

    groups = build_param_lr_groups(model, cfg)
    by_name = {group["name"]: group for group in groups}
    action_ids = {id(parameter) for name in (
        "oft_action_model",
        "gr00t_action_model",
        "pi_action_model",
    ) for parameter in getattr(model, name).parameters()}

    assert by_name["qwen_vl_interface"]["lr"] == pytest.approx(2e-6)
    assert by_name["action_model"]["lr"] == pytest.approx(1e-4)
    assert {id(parameter) for parameter in by_name["action_model"]["params"]} == action_ids
    assert all(group["weight_decay"] == pytest.approx(0.01) for group in groups)


def test_collect_gradient_norms_reports_every_group_and_global():
    first = nn.Parameter(torch.tensor([0.0]))
    second = nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "name": "qwen_vl_interface"},
            {"params": [second], "name": "action_model"},
        ],
        lr=1e-3,
    )

    metrics = collect_gradient_norms(optimizer)

    assert metrics == {
        "grad_norm/qwen_vl_interface": pytest.approx(3.0),
        "grad_norm/action_model": pytest.approx(4.0),
        "grad_norm/global": pytest.approx(5.0),
    }


def test_collect_gradient_norms_uses_zero2_partition_norms_and_loss_scale():
    optimizer = SimpleNamespace(
        param_groups=[
            {"params": [], "name": "qwen_vl_interface"},
            {"params": [], "name": "action_model"},
        ]
    )

    class _ZeroOptimizer:
        averaged_gradients = {0: [torch.tensor([6.0])], 1: [torch.tensor([8.0])]}
        bit16_groups = [[], []]
        params_in_partition = [[], []]
        loss_scale = 2.0

        @staticmethod
        def get_grad_norm_direct(gradients, parameters):
            del parameters
            return gradients[0].abs().sum()

    engine = SimpleNamespace(optimizer=_ZeroOptimizer())

    metrics = collect_gradient_norms(optimizer, deepspeed_engine=engine)

    assert metrics == {
        "grad_norm/qwen_vl_interface": pytest.approx(3.0),
        "grad_norm/action_model": pytest.approx(4.0),
        "grad_norm/global": pytest.approx(5.0),
    }


def test_spikebrain_launch_disables_pack_and_bucket_and_sets_hyperparameters():
    script = LAUNCH_SCRIPT.read_text()

    assert "qwen_lr=${QWEN_LR:-1e-05}" in script
    assert "action_lr=${ACTION_LR:-1e-04}" in script
    assert "weight_decay=${WEIGHT_DECAY:-0.01}" in script
    assert "vlm_pack=false" in script
    assert "camera_bucket_sampling=false" in script
    assert "image_size_buckets" not in script
    assert "ensure_fp32_sensitive_layers" not in script
    assert "rmsnorm_fp32_forward" not in script
    assert "setup_spikebrain_hf_modules_cache" in script
    assert 'test -f "${package_dir}/ulysses.py"' in script
    assert "/mnt/jfzn/liujn/code/VLAct-main/playground/Datasets" in script
    assert 'export STARVLA_DATA_ROOT="${data_root_dir}"' in script
    assert "/mnt/jfzn/liujn/code/VLAct-main/playground/Pretrained_models/Qwen3-0.6B" in script
    assert "--framework.action_model.expert_model_path ${action_expert_model}" in script
    assert 'local_dependency_dir="${STARVLA_ROOT}/.acp_deps"' in script
    assert 'enable_gradient_checkpointing=${ENABLE_GC:-true}' in script
    assert "--trainer.enable_gradient_checkpointing ${enable_gradient_checkpointing}" in script
    assert 'export TRITON_CACHE_DIR=' in script
    assert 'nonfinite_check_interval=${NONFINITE_CHECK_INTERVAL:-100}' in script
    assert 'use_liger_kernel=${USE_LIGER_KERNEL:-true}' in script
    assert 'liger_rope=${LIGER_ROPE:-true}' in script
    assert 'liger_rms_norm=${LIGER_RMS_NORM:-false}' in script
    assert 'liger_swiglu=${LIGER_SWIGLU:-false}' in script
    assert 'liger_fused_linear_cross_entropy=${LIGER_FUSED_LINEAR_CROSS_ENTROPY:-false}' in script
    assert "--framework.qwenvl.use_liger_kernel ${use_liger_kernel}" in script
    assert "--trainer.weight_decay ${weight_decay}" in script
    assert "--trainer.optimizer.weight_decay ${weight_decay}" in script
