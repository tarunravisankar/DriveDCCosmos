#!/usr/bin/env python3
"""
compute_roboracer_stats.py

Recomputes cosmos_framework/data/vfm/action/datasets/stats/roboracer_stats.json
from the TRAIN split only (never test/eval — that would leak held-out episode
statistics into the model's normalization). Two fixes vs. the original stats:

1. Computed from train bags only (no leakage from test/eval bags).
2. Drops the synthetic all-zero last-frame-per-episode row (compute_actions in
   convert_roboracer_to_lerobot.py has no "next frame" to diff against there,
   so it's a placeholder, not a real action) and any frame whose delta is
   physically implausible (>1m position or >45deg yaw in 1/15s) — defensive,
   since the train split already scanned at 0% glitches in
   scan_roboracer_bag_quality.py, but kept in case new bags are added later.

mean/std/min/max are still recorded for reference, but min/max are what get
used by action_normalization.py's "minmax" method — every real action in the
train split maps to exactly [-1, 1] with no clipping (unlike "quantile", which
clipped genuine turns to the q01/q99 band — see check_roboracer_predictions.py
session notes).

Usage:
    python compute_roboracer_stats.py --train-root /scratch/tarunrav/roboracer_lerobot_train
    # or, combining multiple datasets (e.g. when training on more than one
    # source via PackingDataLoader's multi-dataset config):
    python compute_roboracer_stats.py \\
        --train-root /scratch/tarunrav/roboracer_lerobot_train /scratch/tarunrav/roboracer_lerobot_train_orin13
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ACTION_COLS = [
    "action.pos_x", "action.pos_y", "action.pos_z",
    "action.rot_0", "action.rot_1", "action.rot_2",
    "action.rot_3", "action.rot_4", "action.rot_5",
]

MAX_POS_DELTA_M = 1.0
MAX_YAW_DELTA_DEG = 45.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root",
        type=Path,
        nargs="+",
        default=[Path("/scratch/tarunrav/roboracer_lerobot_train")],
        help="One or more LeRobot train-split roots; stats are computed over their combined rows "
             "(e.g. when training on multiple datasets via PackingDataLoader's multi-dataset config).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cosmos_framework/data/vfm/action/datasets/stats/roboracer_stats.json"),
    )
    args = parser.parse_args()

    # episode_index resets to 0 within each separately-converted root, so a plain
    # concat would collide episode indices across roots. Tag rows with their
    # source root before concatenating, and group on (root, episode_index) below.
    dfs = []
    for root_idx, root in enumerate(args.train_root):
        table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
        root_df = table.to_pandas()
        root_df["_root_idx"] = root_idx
        dfs.append(root_df)
        print(f"Loaded {len(root_df)} rows from {root}")
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

    # Drop the synthetic zero placeholder at the last frame of every episode.
    last_frame_mask = (
        df.groupby(["_root_idx", "episode_index"])["frame_index"].transform("max") == df["frame_index"]
    )
    kept = df[~last_frame_mask].copy()
    n_episodes = len(df[["_root_idx", "episode_index"]].drop_duplicates())
    print(f"Loaded {len(df)} rows, dropped {last_frame_mask.sum()} synthetic last-frame zero rows "
          f"({n_episodes} episodes across {len(args.train_root)} root(s)), {len(kept)} remain")

    actions = kept[_ACTION_COLS].to_numpy(dtype=np.float32)  # [N, 9]
    yaw_delta_deg = np.degrees(np.arctan2(actions[:, 4], actions[:, 3]))
    glitch_mask = (
        (np.abs(actions[:, 0]) > MAX_POS_DELTA_M)
        | (np.abs(actions[:, 1]) > MAX_POS_DELTA_M)
        | (np.abs(yaw_delta_deg) > MAX_YAW_DELTA_DEG)
    )
    if glitch_mask.any():
        print(f"WARNING: dropping {int(glitch_mask.sum())} implausible-delta rows from train split "
              f"(expected 0 — scan_roboracer_bag_quality.py found the train bags 100% clean)")
    actions = actions[~glitch_mask]

    stats = {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
    }

    names = ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "rot_5"]
    print(f"\nComputed from {len(actions)} clean training-split frames:")
    for k in ["mean", "std", "min", "max", "q01", "q99"]:
        print(f"  {k:5s}", {n: round(v, 5) for n, v in zip(names, stats[k])})

    if args.out.exists():
        backup = args.out.with_suffix(".json.bak")
        backup.write_text(args.out.read_text())
        print(f"\nBacked up old stats to {backup}")

    args.out.write_text(json.dumps(stats, indent=4))
    print(f"Wrote new train-only stats to {args.out}")


if __name__ == "__main__":
    main()
