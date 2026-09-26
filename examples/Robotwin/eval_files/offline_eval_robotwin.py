"""Open-loop offline metrics on RoboTwin LeRobot demos (press_stapler etc.)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
EVAL_FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EVAL_FILES))

from offline_eval_metrics import compute_metrics, joint_deg_report, pick_episode_indices  # noqa: E402


def _extract_steps(ckpt: Path) -> int | None:
    m = re.search(r"steps_(\d+)", ckpt.name)
    return int(m.group(1)) if m else None


def _summary_row(results: dict) -> dict:
    n = results["normalized"]
    jd = results["joint_deg"]
    thr = {int(r["threshold_deg"]): r for r in jd["thresholds"]}
    return {
        "steps": results.get("steps"),
        "normalized_mse": n["mse"],
        "joint_median_deg": jd["element"]["median"],
        "window_peak_median_deg": jd["window_peak"]["median"],
        "window_ge_20deg_frac": thr.get(20, {}).get("window_frac", None),
        "window_ge_50deg_frac": thr.get(50, {}).get("window_frac", None),
        "max_joint_deg": jd["element"]["max"],
        "max_window_peak_deg": jd["window_peak"]["max"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--data_root",
        default="/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin",
    )
    ap.add_argument("--data_name", default="Clean/press_stapler")
    ap.add_argument("--robot_type", default="robotwin_wrap32")
    ap.add_argument("--frame_cache_dir", default="")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_episodes", type=int, default=0, help="0 = all episodes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model_tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    from accelerate import PartialState
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.framework.share_tools import read_mode_config

    PartialState()
    os.chdir(REPO)

    ckpt = Path(args.ckpt)
    if ckpt.is_dir():
        ckpt = ckpt / "steps_100000_pytorch_model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    print(f"[offline] loading {ckpt}", flush=True)
    t0 = time.time()
    model = baseframework.from_pretrained(str(ckpt))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"[offline] model on {device} in {time.time() - t0:.0f}s", flush=True)

    vla_cfg, norm_stats = read_mode_config(str(ckpt))
    framework_name = str((vla_cfg.get("framework") or {}).get("name", ""))
    action_cfg = (vla_cfg.get("framework") or {}).get("action_model") or {}
    n_infer = int(action_cfg.get("num_inference_timesteps", 4) or 4)
    action_horizon = int(action_cfg.get("action_horizon", action_cfg.get("future_action_window_size", 31) + 1))

    vla_data_cfg = (vla_cfg.get("datasets") or {}).get("vla_data") or {}
    image_size_buckets = vla_data_cfg.get("image_size_buckets") or None
    obs_image_size = vla_data_cfg.get("image_size") or [224, 224]

    unnorm_key = next(iter(norm_stats.keys()))
    action_stats = norm_stats[unnorm_key]["action"]
    lo = np.asarray(action_stats["min"], dtype=np.float32)
    hi = np.asarray(action_stats["max"], dtype=np.float32)
    print(
        f"[offline] framework={framework_name} unnorm_key={unnorm_key} "
        f"n_infer={n_infer} h={action_horizon} buckets={image_size_buckets}",
        flush=True,
    )

    data_cfg = {
        "video_backend": "pyav",
        "lerobot_version": "v2.0",
        "obs_image_size": obs_image_size,
        "include_state": True,
        "frame_cache_dir": args.frame_cache_dir.strip() or None,
    }
    if image_size_buckets:
        data_cfg["image_size_buckets"] = image_size_buckets

    ds = make_LeRobotSingleDataset(
        Path(args.data_root),
        args.data_name,
        args.robot_type,
        data_cfg=data_cfg,
    )
    if hasattr(ds, "transforms") and hasattr(ds.transforms, "eval"):
        ds.transforms.eval()

    episode_ids = [int(x) for x in ds.trajectory_ids]
    if args.num_episodes > 0 and args.num_episodes < len(episode_ids):
        rng = np.random.default_rng(int(args.seed))
        episode_ids = sorted(int(x) for x in rng.choice(episode_ids, size=int(args.num_episodes), replace=False))
        print(f"[offline] sampled episodes={episode_ids}", flush=True)

    indices, meta = pick_episode_indices(ds, episode_ids, args.stride)
    print(
        f"[offline] data={args.data_name} episodes={len(episode_ids)} "
        f"windows={len(indices)} stride={args.stride} batch={args.batch_size}",
        flush=True,
    )

    predict_kwargs = {}
    if "qwenpi" in framework_name.lower():
        predict_kwargs["num_inference_steps"] = n_infer

    preds, gts = [], []
    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start : start + args.batch_size]
        examples = [ds[i] for i in batch_idx]
        with torch.inference_mode():
            out = model.predict_action(examples=examples, **predict_kwargs)
        pred = np.asarray(out["normalized_actions"], dtype=np.float32)
        gt = np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in examples])
        if pred.shape != gt.shape:
            raise RuntimeError(f"shape mismatch pred={pred.shape} gt={gt.shape}")
        preds.append(pred)
        gts.append(gt)
        done = start + len(batch_idx)
        if done == len(indices) or done % 128 == 0 or start == 0:
            print(f"  {done}/{len(indices)}", flush=True)

    pred_all = np.concatenate(preds, axis=0)
    gt_all = np.concatenate(gts, axis=0)

    def to_physical(arr: np.ndarray) -> np.ndarray:
        x = np.clip(arr, -1.0, 1.0)
        return ((x + 1.0) / 2.0) * (hi - lo) + lo

    pred_phys = to_physical(pred_all)
    gt_phys = to_physical(gt_all)
    joint_deg = joint_deg_report(pred_phys, gt_phys, meta)

    model_tag = args.model_tag or framework_name
    results = {
        "model_tag": model_tag,
        "steps": _extract_steps(ckpt),
        "ckpt": str(ckpt),
        "data_root": args.data_root,
        "data_name": args.data_name,
        "split": args.split,
        "framework": framework_name,
        "episodes_evaluated": episode_ids,
        "num_episodes": len(episode_ids),
        "stride": args.stride,
        "action_horizon": action_horizon,
        "num_inference_steps": n_infer if predict_kwargs else None,
        "normalized": compute_metrics(pred_all, gt_all, "normalized"),
        "physical": compute_metrics(pred_phys, gt_phys, "physical"),
        "joint_deg": joint_deg,
        "summary": _summary_row(
            {
                "normalized": compute_metrics(pred_all, gt_all, "normalized"),
                "joint_deg": joint_deg,
                "steps": _extract_steps(ckpt),
            }
        ),
    }

    out_path = Path(args.out) if args.out else (
        REPO
        / "logs/robotwin_offline"
        / f"offline_{model_tag}_{args.data_name.replace('/', '_')}_{ckpt.parent.name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    s = results["summary"]
    print(
        f"[offline] steps={s['steps']} norm_mse={s['normalized_mse']:.6f} "
        f"joint_med={s['joint_median_deg']:.2f} win_peak_med={s['window_peak_median_deg']:.2f} "
        f">=20win={s['window_ge_20deg_frac']:.4f} >=50win={s['window_ge_50deg_frac']:.4f} "
        f"max={s['max_joint_deg']:.2f}",
        flush=True,
    )
    print(f"[offline] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
