# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboracerDataset — single-camera AV action dataset for Cosmos3-Nano policy fine-tuning.

Reads a LeRobot v3.0 dataset converted from ROS2 bags recorded on the
UT Austin RoboRacer 1/10th-scale autonomous vehicle platform.

Action space: AV 9D = [pos_x, pos_y, pos_z, rot_0..rot_5]
  - pos_xyz: frame-to-frame position delta in body frame (from odom)
  - rot_0..5: rot6d yaw delta (from odom quaternion)
  Domain: "av" (domain_id=1), matching Cosmos3 AV pretraining priors.

Single camera: observation.image.camera_0 (front-facing RGB)

Place this file at:
    cosmos_framework/data/vfm/action/datasets/roboracer_dataset.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset

# Video key matching the converter output
_VIDEO_KEY = "observation.image.camera_0"

# Action column names matching the converter parquet output
_ACTION_COLS = [
    "action.pos_x", "action.pos_y", "action.pos_z",
    "action.rot_0", "action.rot_1", "action.rot_2",
    "action.rot_3", "action.rot_4", "action.rot_5",
]

# Path to normalization stats (computed after conversion, see compute_stats.py)
# For initial training, set action_normalization=None to skip normalization.
_STATS_PATH = Path(__file__).parent / "stats" / "roboracer_stats.json"


class RoboracerDataset(ActionBaseDataset):
    """LeRobot v3.0 dataset for the RoboRacer AV platform.

    Reads single-camera video + 9D AV ego-pose actions from a LeRobot
    dataset converted from ROS2 bags via convert_roboracer_to_lerobot.py.

    Args:
        root: Path to the LeRobot v3.0 dataset root directory.
        fps: Dataset frame rate (default 15, matching conversion).
        chunk_length: Number of future action steps per sample (default 32,
            matching DROID recipe for consistency).
        mode: "policy" | "forward_dynamics" | "inverse_dynamics" | "joint".
        action_normalization: "minmax" (default) maps the train-split min/max
            per channel to exactly [-1, 1] — recomputed via compute_roboracer_stats.py
            from the train split only. Avoid "quantile": with turning frames a small
            minority of the dataset, the q01/q99 band sits inside genuine turn
            magnitudes, so real turns get clipped outside the network's normal
            input range (see check_roboracer_predictions.py session notes).
        use_image_augmentation: Apply random crop+rescale and color jitter.
        oversample_turns: Duplicate windows containing a turn in ``_valid_windows``
            so any downstream sampler (shuffled or sequential) sees them more often
            per epoch. Straight-driving frames otherwise dominate by sheer count —
            this only weights which windows get sampled more, it does not touch
            normalization. Should be True for training, False for eval/test (an
            oversampled eval set would no longer be a faithful measure of real-world
            performance). Tiers (by max |rot_1| over the window's chunk):
            <=0.03 (13377 raw windows, ~41%): 1x; 0.03-0.08 (13651, ~41%): 5x;
            >0.08 (5915, ~18%): 20x — yields ~199932 oversampled windows total,
            ~6.3% straight-by-weighted-count (v2's 1x/2x/5x tiers, ~2.13x total,
            left straight-driving still majority and the model never learned to
            turn; see roboracer_dataset.py _OVERSAMPLE_TIERS comment).
    """

    # Curvature tiers for oversample_turns, keyed by max |rot_1| over a window's
    # chunk. See compute_roboracer_stats.py / session notes for how these thresholds
    # were chosen from the train split's actual curvature distribution.
    #
    # v2 tiers (1x/2x/5x, ~2.13x total oversampled size) were not strong enough:
    # the model still converged to predicting a near-input-independent default
    # action ("always slightly-straight") that minimizes loss on the still-
    # majority straight frames, even with 32-step inference sampling ruling out
    # an inference-time mode-averaging artifact. Raised substantially so turning
    # windows dominate the per-epoch sample count (~199932 vs 32943 raw windows,
    # ~6.3% straight-by-weighted-count) rather than merely being a large minority.
    _OVERSAMPLE_TIERS = ((0.03, 1), (0.08, 5), (float("inf"), 20))

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        # "policy" is the only mode that matches what's actually available at
        # live-deployment time: just the current/past frame, no future frames.
        # It used to fail outright (v2-v6) because it ALSO noises/generates
        # almost the entire video simultaneously with the action sequence — a
        # large competing video-reconstruction objective sharing the same
        # fine-tuned backbone (moe_gen) as action prediction. Switching to
        # "inverse_dynamics" (v7, all video frames clean conditioning) proved
        # that diagnosis correct and dramatically improved predictions — but
        # inverse_dynamics conditions on the REAL future video frames, which
        # don't exist yet on a live car, so it isn't deployable. The
        # action_policy_roboracer_nano.py experiment config now explicitly
        # passes mode="joint" for training (matching the reference DROID
        # recipe's own default — mixes forward_dynamics/inverse_dynamics/
        # policy per sample, base_dataset.py _MODE_CHOICES) and mode="policy"
        # for validation (so the val loss/early-stopping decision reflects the
        # real deployable task, not an inflated inverse_dynamics number).
        # "policy" remains the class default here so any future caller that
        # forgets to pass mode= gets the safe, deployable behavior rather than
        # silently training/testing against future frames it won't have.
        mode: str = "policy",
        action_normalization: str = "minmax",
        use_image_augmentation: bool = False,
        oversample_turns: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            domain_name="av",           # domain_id=1, dim=9 from domain_utils.py
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",
            tolerance_s=2e-4,
            viewpoint="ego_view",
            action_normalization=action_normalization,
        )
        self._use_image_augmentation = use_image_augmentation

        # Build compact per-episode frame index from the rows loaded by base class.
        # Groups rows by episode, keeps only episodes with enough frames for a chunk.
        ep_to_rows: dict[int, list[int]] = {}
        for i, row in enumerate(self._rows):
            ep = int(row["episode_index"])
            ep_to_rows.setdefault(ep, []).append(i)

        self._valid_windows: list[tuple[int, int]] = []  # (row_start_idx, episode_index)
        for ep, row_indices in sorted(ep_to_rows.items()):
            n = len(row_indices)
            for start in range(0, n - chunk_length - 1):
                self._valid_windows.append((row_indices[start], ep))

        if oversample_turns:
            self._valid_windows = self._oversample_turning_windows(self._valid_windows)

    def _oversample_turning_windows(
        self, windows: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Replicate windows by curvature tier (see ``_OVERSAMPLE_TIERS``)."""
        oversampled: list[tuple[int, int]] = []
        for row_start, ep in windows:
            action_rows = self._rows[row_start + 1: row_start + 1 + self._chunk_length]
            curvature = max(abs(float(row.get("action.rot_1", 0.0))) for row in action_rows)
            weight = next(w for threshold, w in self._OVERSAMPLE_TIERS if curvature <= threshold)
            oversampled.extend([(row_start, ep)] * weight)
        return oversampled

    # ------------------------------------------------------------------
    # ActionBaseDataset abstract interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 9

    def _action_spec(self) -> ActionSpec:
        # AV 9D: Pos() (3D) + Rot("rot6d") (6D) — matches domain_utils.py
        return build_action_spec(Pos(), Rot("rot6d"))

    @classmethod
    def _stats_path(cls) -> Path:
        return _STATS_PATH

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        row_start, episode_index = self._valid_windows[idx]

        # Observation rows: chunk_length + 1 frames (like DROID joint_pos recipe)
        observation_rows = self._rows[row_start: row_start + self._chunk_length + 1]
        episode = self._episodes[episode_index]

        # Load video frames
        video = self._load_video(episode, observation_rows)

        # Build 9D action tensor: shape (chunk_length, 9)
        action = self._build_action(observation_rows)

        # Task caption
        task_idx = int(observation_rows[0].get("task_index", 0))
        ai_caption = self._tasks.get(task_idx, "Drive the roboracer vehicle.")

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            additional_view_description=(
                "A single front-facing camera mounted on a 1/10th-scale autonomous vehicle."
            ),
        )

    def _video_path(self, episode, video_key):
        """Override base class: derive chunk/file index from episode_index directly."""
        ep_idx = int(episode["episode_index"])
        chunks_size = self._info.get("chunks_size", 1000)
        chunk_idx = ep_idx // chunks_size
        file_idx = ep_idx % chunks_size
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        return self._root / rel
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Load and return video frames as float32 tensor [T, C, H, W] in [0,1]."""
        video_path = self._video_path(episode, _VIDEO_KEY)
        timestamps = [float(row["timestamp"]) for row in observation_rows]

        # Episode video starts at t=0; timestamps are already relative
        from_ts = float(episode.get(f"videos/{_VIDEO_KEY}/from_timestamp", 0.0))
        abs_timestamps = [from_ts + ts for ts in timestamps]

        frames = decode_video_frames(
            video_path,
            abs_timestamps,
            self._tolerance_s,
        )
        return frames  # [T, C, H, W], float32 in [0, 1]

    def _build_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """Build 9D action tensor from parquet action columns.

        Uses rows[1:] as the chunk_length commanded actions (matching
        the DROID joint_pos convention where row[0] is initial state).
        Shape: (chunk_length, 9)
        """
        action_rows = observation_rows[1:]  # skip initial observation frame
        actions = np.array(
            [[float(row.get(col, 0.0)) for col in _ACTION_COLS] for row in action_rows],
            dtype=np.float32,
        )  # (chunk_length, 9)
        return torch.from_numpy(actions).float()


# ---------------------------------------------------------------------------
# Factory function (mirrors get_action_droid_sft_dataset pattern)
# ---------------------------------------------------------------------------

def get_action_roboracer_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    mode: str = "policy",  # safe/deployable default — see RoboracerDataset.__init__
    action_normalization: str | None = None,
    use_image_augmentation: bool = False,
    oversample_turns: bool = False,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
):
    """Build the RoboRacer action SFT dataset wrapped with ActionTransformPipeline.

    Mirrors get_action_droid_sft_dataset from action_sft_dataset.py.
    """
    from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

    dataset = RoboracerDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
        oversample_turns=oversample_turns,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)

   
