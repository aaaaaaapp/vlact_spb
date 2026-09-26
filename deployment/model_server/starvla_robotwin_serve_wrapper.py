"""Serve VLAct checkpoints with the Starvla RobotWin websocket client protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from starVLA.model.framework.share_tools import read_mode_config


class StarvlaRobotwinServeWrapper:
    """Normalize state / unnormalize actions for Starvla model2robotwin_interface."""

    def __init__(
        self,
        framework,
        ckpt_path: str,
        default_unnorm_key: Optional[str] = None,
    ) -> None:
        self._framework = framework
        self._ckpt_path = str(ckpt_path)
        model_cfg, norm_stats = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        self._norm_stats = norm_stats

        if default_unnorm_key is None:
            if len(norm_stats) == 1:
                default_unnorm_key = next(iter(norm_stats.keys()))
        elif default_unnorm_key not in norm_stats:
            raise KeyError(
                f"unnorm_key={default_unnorm_key!r} not in {list(norm_stats.keys())}"
            )
        if default_unnorm_key is None:
            raise ValueError(
                f"Multiple unnorm_keys in dataset_statistics.json: {list(norm_stats.keys())}"
            )
        self._default_unnorm_key = default_unnorm_key
        self._use_wrap = "wrap" in str(
            model_cfg.get("datasets", {}).get("vla_data", {}).get("data_mix", "")
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        action_model_cfg = self._model_cfg["framework"]["action_model"]
        if "action_horizon" in action_model_cfg:
            action_chunk_size = int(action_model_cfg["action_horizon"])
        else:
            action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1

        vla_cfg = self._model_cfg.get("datasets", {}).get("vla_data", {})
        return {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": action_chunk_size,
            "available_unnorm_keys": list(self._norm_stats.keys()),
            "default_unnorm_key": self._default_unnorm_key,
            "training_action_mode": vla_cfg.get("action_mode", "abs"),
            "training_normalization_mode": vla_cfg.get(
                "action_normalization_mode",
                vla_cfg.get("normalization_mode", "min_max"),
            ),
        }

    def reset_memory(self) -> None:
        return None

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        action_stats = self._norm_stats[effective_key]["action"]
        state_stats = self._norm_stats[effective_key]["state"]

        kwargs.pop("type", None)
        kwargs.pop("request_id", None)
        kwargs.pop("payload", None)
        kwargs.pop("memory_states", None)
        kwargs.pop("return_memory_states", None)

        normalized_examples = []
        for example in examples:
            example_copy = dict(example)
            state = example_copy.get("state", None)
            if state is not None:
                example_copy["state"] = self._normalize_state(
                    np.asarray(state, dtype=np.float32), state_stats
                )
            normalized_examples.append(example_copy)

        out = self._framework.predict_action(examples=normalized_examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"], dtype=np.float32)
        unnorm = np.stack(
            [
                self._unnormalize_actions(normalized[b], action_stats)
                for b in range(normalized.shape[0])
            ],
            axis=0,
        )
        return {"actions": unnorm}

    @staticmethod
    def _normalize_state(state: np.ndarray, state_stats: Dict[str, Any]) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        mask = np.array([True] * 12 + [False, False], dtype=bool)
        state_high = np.asarray(state_stats["max"], dtype=np.float32)
        state_low = np.asarray(state_stats["min"], dtype=np.float32)
        normalized = np.where(
            mask,
            (state - state_low) / np.maximum(state_high - state_low, 1e-8) * 2 - 1,
            state,
        )
        normalized = np.where(
            ~mask,
            (normalized > 0.5).astype(normalized.dtype),
            normalized,
        )
        return normalized[0] if squeeze else normalized

    def _unnormalize_actions(
        self,
        normalized_actions: np.ndarray,
        action_stats: Dict[str, Any],
    ) -> np.ndarray:
        mask = np.asarray(
            action_stats.get(
                "mask", np.ones_like(action_stats["min"], dtype=bool)
            ),
            dtype=bool,
        )
        raw_actions = np.array(normalized_actions, copy=True)
        clipped = np.clip(raw_actions, -1, 1)
        actions = np.array(clipped, copy=True)

        if not np.any(mask):
            return actions

        if self._use_wrap:
            actions[..., mask] = (raw_actions[..., mask] + np.pi) % (2 * np.pi) - np.pi
            return actions

        action_high = np.asarray(action_stats["max"], dtype=np.float32)
        action_low = np.asarray(action_stats["min"], dtype=np.float32)
        actions[..., mask] = (
            0.5 * (clipped[..., mask] + 1) * (action_high[mask] - action_low[mask])
            + action_low[mask]
        )
        return actions
