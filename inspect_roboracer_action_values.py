#!/usr/bin/env python3
"""
inspect_roboracer_action_values.py

Answers the mentor's question directly: "what values are being input to the
network during fine-tuning?" Pulls real samples through the exact same
dataset + ActionTransformPipeline used by action_policy_roboracer_nano.py's
training dataloader (get_action_roboracer_sft_dataset), and prints the
"action" tensor per channel — both raw (real units: meters/frame, rot6d) and
normalized (what the network actually sees) — side by side.

No model/GPU/VAE needed; this only exercises the dataset + ActionProcessor.

Usage:
    cd ~/cosmos-framework
    python inspect_roboracer_action_values.py --num-samples 10
"""

import argparse
from pathlib import Path

import torch

from cosmos_framework.data.vfm.action.action_normalization import denormalize_action, load_action_stats
from cosmos_framework.data.vfm.action.datasets.roboracer_dataset import get_action_roboracer_sft_dataset

_ACTION_NAMES = ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "rot_5"]
_STATS_PATH = Path(__file__).parent / "cosmos_framework/data/vfm/action/datasets/stats/roboracer_stats.json"


def summarize_columns(tag: str, values: torch.Tensor) -> None:
    for c in range(values.shape[-1]):
        col = values[:, c]
        print(
            f"      {_ACTION_NAMES[c]:8s} min={col.min():+.5f}  mean={col.mean():+.5f}  "
            f"max={col.max():+.5f}  std={col.std():+.5f}"
        )


def summarize(ds, idx: int, stats: dict) -> None:
    sample = ds[idx]
    action = sample["action"]  # [T, max_action_dim] — exactly what the network's "action" input looks like
    raw_action_dim = int(sample["raw_action_dim"])
    network_input = action[:, :raw_action_dim]  # normalized, real channels only
    real_units = denormalize_action(network_input, "minmax", stats)
    pad = action[:, raw_action_dim:]

    print(f"  action shape={tuple(action.shape)} raw_action_dim={raw_action_dim}")
    print("  [NORMALIZED — what the network actually receives as input]")
    summarize_columns("network_input", network_input)
    print("  [RAW, real units (meters/frame, rot6d) — same values, denormalized for reference]")
    summarize_columns("real_units", real_units)
    print(f"  padding channels (should be exactly 0): max abs = {pad.abs().max():.8f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="/scratch/tarunrav/roboracer_lerobot")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    # Exact same dataset/transform pipeline the training dataloader uses
    # (action_policy_roboracer_nano.py's dataloader_train -> get_action_roboracer_sft_dataset).
    ds = get_action_roboracer_sft_dataset(
        root=args.dataset_root,
        fps=15.0,
        chunk_length=32,
        action_normalization="minmax",
        use_image_augmentation=True,
        resolution="256",
        max_action_dim=64,
        cfg_dropout_rate=0.1,
    )

    stats_raw = load_action_stats(str(_STATS_PATH))
    stats = {k: torch.from_numpy(v).float() for k, v in stats_raw.items()}

    print(f"Dataset has {len(ds)} windows\n")
    step = max(1, len(ds) // args.num_samples)
    shown = 0
    for idx in range(0, len(ds), step):
        if shown >= args.num_samples:
            break
        print(f"=== sample {idx} ===")
        summarize(ds, idx, stats)
        print()
        shown += 1


if __name__ == "__main__":
    main()
