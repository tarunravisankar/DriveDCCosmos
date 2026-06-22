#!/usr/bin/env python3
"""
scan_roboracer_bag_quality.py

Scans every roboracer ROS2 bag for odometry quality before deciding the
train/test/eval split, so "bad data" is identified from evidence (glitch
frame counts, episode length) rather than guessed.

A frame-to-frame odom delta is flagged as a glitch if it implies >1 m of
travel or >45 deg of yaw change in a single ~1/30s bag frame (i.e. >30 m/s or
>1350 deg/s) — physically impossible for this 1/10-scale vehicle. Reuses the
exact odom-decoding/delta logic from convert_roboracer_to_lerobot.py so the
glitch definition is consistent with how the dataset itself is built.

Usage:
    python scan_roboracer_bag_quality.py \
        --bags-dir /robodata/fri/spring26/imitation_learning/rosbags/orin10
"""

import argparse
import math
from pathlib import Path

from convert_roboracer_to_lerobot import compute_pose_delta_9d, read_bag, sync_odom_to_frames

MAX_POS_DELTA_M = 1.0
MAX_YAW_DELTA_DEG = 45.0


def scan_bag(bag_path: Path) -> dict:
    frames, odom = read_bag(bag_path)
    poses = sync_odom_to_frames(frames, odom)
    n_valid_odom = sum(1 for p in poses if p is not None)

    n_glitch = 0
    first_glitch_idx = None
    for i in range(len(poses) - 1):
        if poses[i] is None or poses[i + 1] is None:
            continue
        action = compute_pose_delta_9d(poses[i], poses[i + 1])
        dx, dy = float(action[0]), float(action[1])
        yaw_delta_deg = math.degrees(math.atan2(float(action[4]), float(action[3])))
        if abs(dx) > MAX_POS_DELTA_M or abs(dy) > MAX_POS_DELTA_M or abs(yaw_delta_deg) > MAX_YAW_DELTA_DEG:
            n_glitch += 1
            if first_glitch_idx is None:
                first_glitch_idx = i

    return {
        "name": bag_path.parent.name,
        "n_frames": len(frames),
        "n_odom": len(odom),
        "n_valid_odom": n_valid_odom,
        "n_glitch": n_glitch,
        "first_glitch_idx": first_glitch_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bags-dir", type=Path, default=Path("/robodata/fri/spring26/imitation_learning/rosbags/orin10")
    )
    args = parser.parse_args()

    bag_files = sorted(args.bags_dir.rglob("*.db3"))
    print(f"Found {len(bag_files)} bags\n")

    results = []
    for bag_path in bag_files:
        r = scan_bag(bag_path)
        results.append(r)
        glitch_pct = 100.0 * r["n_glitch"] / max(1, r["n_frames"])
        flag = "  <-- BAD" if glitch_pct > 1.0 or r["n_valid_odom"] < 0.5 * r["n_frames"] else ""
        print(
            f"{r['name']}: frames={r['n_frames']:5d} valid_odom={r['n_valid_odom']:5d} "
            f"glitches={r['n_glitch']:4d} ({glitch_pct:5.1f}%) first_glitch_idx={r['first_glitch_idx']}{flag}"
        )

    total_frames = sum(r["n_frames"] for r in results)
    total_glitch = sum(r["n_glitch"] for r in results)
    print(f"\nTotal: {len(results)} bags, {total_frames} frames, {total_glitch} glitch frames "
          f"({100.0*total_glitch/max(1,total_frames):.2f}%)")


if __name__ == "__main__":
    main()
