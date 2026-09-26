"""Shared offline action-prediction metrics (from ALOHA offline eval)."""

from __future__ import annotations

import numpy as np

ACTION_DIM_NAMES = [
    "left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5",
    "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5",
    "left_gripper", "right_gripper",
]


def pick_episode_indices(ds, episode_ids, stride: int):
    id_to_len = {int(tid): int(L) for tid, L in zip(ds.trajectory_ids, ds.trajectory_lengths)}
    ep_set = {int(e) for e in episode_ids}
    stride = max(1, int(stride))
    indices, meta = [], []
    for i, (tid, base) in enumerate(ds.all_steps):
        tid, base = int(tid), int(base)
        if tid not in ep_set or (base % stride) != 0:
            continue
        indices.append(i)
        meta.append({"episode": tid, "base_index": base, "length": id_to_len[tid]})
    return indices, meta


def compute_metrics(pred: np.ndarray, gt: np.ndarray, label: str) -> dict:
    diff = pred - gt
    abs_diff = np.abs(diff)
    per_dim_mse = np.mean(diff ** 2, axis=(0, 1))
    per_dim_mae = np.mean(abs_diff, axis=(0, 1))
    per_dim_max = np.max(abs_diff, axis=(0, 1))
    names = ACTION_DIM_NAMES[: pred.shape[2]]
    return {
        "label": label,
        "n_windows": int(pred.shape[0]),
        "horizon": int(pred.shape[1]),
        "action_dim": int(pred.shape[2]),
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_diff": float(abs_diff.max()),
        "max_abs_diff_t0": float(abs_diff[:, 0].max()),
        "pred_abs_max": float(np.abs(pred).max()),
        "gt_abs_max": float(np.abs(gt).max()),
        "per_dim_mse": {names[i]: float(per_dim_mse[i]) for i in range(pred.shape[2])},
        "per_dim_mae": {names[i]: float(per_dim_mae[i]) for i in range(pred.shape[2])},
        "per_dim_max_abs_diff": {names[i]: float(per_dim_max[i]) for i in range(pred.shape[2])},
    }


def _summarize_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "p99_9": float(np.percentile(x, 99.9)),
        "max": float(x.max()),
    }


def _hist_frac(x: np.ndarray, edges: list[float]) -> list[dict]:
    x = np.asarray(x, dtype=np.float64).ravel()
    counts, _ = np.histogram(x, bins=np.asarray(edges, dtype=np.float64))
    n = max(int(x.size), 1)
    return [
        {"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(c), "frac": float(c) / n}
        for i, c in enumerate(counts)
    ]


def _threshold_table(abs_deg: np.ndarray, thresholds: list[float], episode_ids: np.ndarray) -> list[dict]:
    n_win = int(abs_deg.shape[0])
    n_elem = int(abs_deg.size)
    win_peak = abs_deg.max(axis=(1, 2))
    t0_peak = abs_deg[:, 0, :].max(axis=1)
    h25_peak = abs_deg[:, : min(25, abs_deg.shape[1]), :].max(axis=(1, 2))
    rows = []
    for thr in thresholds:
        elem_mask = abs_deg >= thr
        win_mask = win_peak >= thr
        ep_hit = {int(e) for e, hit in zip(episode_ids, win_mask) if hit}
        rows.append(
            {
                "threshold_deg": float(thr),
                "element_count": int(elem_mask.sum()),
                "element_frac": float(elem_mask.mean()) if n_elem else 0.0,
                "window_count": int(win_mask.sum()),
                "window_frac": float(win_mask.mean()) if n_win else 0.0,
                "episode_count": int(len(ep_hit)),
                "t0_window_count": int((t0_peak >= thr).sum()),
                "t0_window_frac": float((t0_peak >= thr).mean()) if n_win else 0.0,
                "h25_window_count": int((h25_peak >= thr).sum()),
                "h25_window_frac": float((h25_peak >= thr).mean()) if n_win else 0.0,
            }
        )
    return rows


def joint_deg_report(pred_phys: np.ndarray, gt_phys: np.ndarray, meta: list[dict]) -> dict:
    rad2deg = 180.0 / np.pi
    abs_deg = np.abs(pred_phys[..., :12] - gt_phys[..., :12]) * rad2deg
    names = ACTION_DIM_NAMES[:12]
    episode_ids = np.asarray([int(m["episode"]) for m in meta], dtype=np.int32)
    base_index = np.asarray([int(m["base_index"]) for m in meta], dtype=np.int32)
    n, t, j = abs_deg.shape
    flat_idx = int(np.argmax(abs_deg))
    wi, ti, ji = np.unravel_index(flat_idx, abs_deg.shape)
    t0_idx = int(np.argmax(abs_deg[:, 0, :]))
    w0, j0 = np.unravel_index(t0_idx, abs_deg[:, 0, :].shape)

    win_peak = abs_deg.max(axis=(1, 2))
    t0_peak = abs_deg[:, 0, :].max(axis=1)
    h20_peak = abs_deg[:, : min(20, t), :].max(axis=(1, 2))
    h25_peak = abs_deg[:, : min(25, t), :].max(axis=(1, 2))

    return {
        "unit": "degree",
        "n_windows": int(n),
        "horizon": int(t),
        "n_joints": int(j),
        "element": _summarize_1d(abs_deg),
        "window_peak": _summarize_1d(win_peak),
        "t0_peak": _summarize_1d(t0_peak),
        "h20_peak": _summarize_1d(h20_peak),
        "h25_peak": _summarize_1d(h25_peak),
        "element_hist_deg": _hist_frac(abs_deg, [0, 1, 2, 5, 10, 15, 20, 30, 50, 90, 180]),
        "window_peak_hist_deg": _hist_frac(win_peak, [0, 2, 5, 10, 15, 20, 30, 50, 90, 180]),
        "t0_peak_hist_deg": _hist_frac(t0_peak, [0, 2, 5, 10, 15, 20, 30, 50, 90, 180]),
        "thresholds": _threshold_table(abs_deg, [5, 10, 15, 20, 30, 50, 90], episode_ids),
        "global_argmax": {
            "deg": float(abs_deg[wi, ti, ji]),
            "episode": int(episode_ids[wi]),
            "base_index": int(base_index[wi]),
            "t": int(ti),
            "joint": names[int(ji)],
        },
        "t0_argmax": {
            "deg": float(abs_deg[w0, 0, j0]),
            "episode": int(episode_ids[w0]),
            "base_index": int(base_index[w0]),
            "joint": names[int(j0)],
        },
    }
