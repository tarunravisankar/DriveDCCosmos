"""Velocity-stratified prediction evaluation, at a sample size that supports a claim.

The existing check_roboracer_predictions.py ranks candidate windows by |rot_1|
(turn magnitude), which is right for testing steering but gives almost no
coverage of the velocity axis -- the 8-sample runs landed on only TWO distinct
ground-truth speeds (~0.05 and ~0.10), which is far too thin to say anything
about whether the model tracks velocity.

This script instead stratifies across the observed pos_x (forward velocity)
range, so ground truth spans the full distribution including near-stopped
frames, and evaluates enough samples for the statistics to mean something.

Reports Pearson r, Spearman rho (rank-based, robust to the clustering that made
Pearson unstable before), MAE, and a slow/fast group comparison.

Usage:
  velocity_eval.py --checkpoint-path <dcp>/model --dataset-root <root> \
                   [--num-samples 48] [--experiment action_policy_roboracer_edge]
"""

import argparse
import math
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cosmos_framework.data.generator.action.action_normalization import (  # noqa: E402
    denormalize_action,
    load_action_stats,
)
from cosmos_framework.data.generator.action.datasets.roboracer_dataset import (  # noqa: E402
    _STATS_PATH,
    RoboracerDataset,
    get_action_roboracer_sft_dataset,
)
from cosmos_framework.data.generator.action.transforms import (  # noqa: E402
    build_sequence_plan_from_mode,
)
from cosmos_framework.inference.args import OmniSetupOverrides  # noqa: E402
from cosmos_framework.inference.common.args import CheckpointType  # noqa: E402
from cosmos_framework.inference.inference import OmniInference  # noqa: E402

MAX_POS_DELTA_M = 1.0
MAX_YAW_DELTA_DEG = 45.0


def stratified_by_velocity(ds: RoboracerDataset, n: int) -> list[int]:
    """Pick n windows spread evenly across the observed pos_x range."""
    cands = []
    step = max(1, len(ds) // 4000)
    for i in range(0, len(ds), step):
        row = ds._rows[ds._valid_windows[i][0] + 1]
        px = float(row.get("action.pos_x", 0.0))
        py = float(row.get("action.pos_y", 0.0))
        r0 = float(row.get("action.rot_0", 1.0))
        r1 = float(row.get("action.rot_1", 0.0))
        if abs(px) > MAX_POS_DELTA_M or abs(py) > MAX_POS_DELTA_M:
            continue
        if abs(math.degrees(math.atan2(r1, r0))) > MAX_YAW_DELTA_DEG:
            continue
        cands.append((i, px))
    if not cands:
        return []
    cands.sort(key=lambda t: t[1])
    # even quantile spread across the sorted-by-velocity list
    picks, seen = [], set()
    for k in range(n):
        j = min(len(cands) - 1, round(k * (len(cands) - 1) / max(1, n - 1)))
        if cands[j][0] not in seen:
            seen.add(cands[j][0])
            picks.append(cands[j][0])
    return picks


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--num-samples", type=int, default=48)
    p.add_argument("--experiment", default="action_policy_roboracer_edge")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--label", default="model")
    args = p.parse_args()

    ds = RoboracerDataset(root=args.dataset_root, fps=15.0, chunk_length=32,
                          action_normalization="minmax", mode="wam")
    sft_ds = get_action_roboracer_sft_dataset(
        root=args.dataset_root, fps=15.0, chunk_length=32, mode="wam",
        action_normalization="minmax", use_image_augmentation=False,
        oversample_turns=False, resolution="256", max_action_dim=64,
        cfg_dropout_rate=0.0,
    )
    idxs = stratified_by_velocity(ds, args.num_samples)
    print(f"[{args.label}] {args.dataset_root.split('/')[-1]}: {len(idxs)} velocity-stratified samples "
          f"from {len(ds)} windows", flush=True)

    setup = OmniSetupOverrides.model_validate({
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_type": CheckpointType.DCP,
        "experiment": args.experiment,
        "experiment_overrides": [
            "model.config.tokenizer.vae_path=/scratch/tarunrav/cosmos-framework/"
            "examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
        ],
        "output_dir": "/tmp/velocity_eval_out",
        "guardrails": False,
        "use_ema_weights": False,
    })
    pipe = OmniInference.create(setup.build_setup())
    model = pipe.model
    model.eval()

    stats_raw = load_action_stats(str(_STATS_PATH))
    stats = {k: torch.from_numpy(v).float() for k, v in stats_raw.items()}

    gts, preds, gt_yaw, pr_yaw = [], [], [], []
    for n, idx in enumerate(idxs):
        gt_raw = denormalize_action(ds[idx]["action"], "minmax", stats)
        ms = sft_ds[idx]
        video = ms["video"]
        az = torch.zeros_like(ms["action"])
        plan = build_sequence_plan_from_mode(mode="wam", video_length=video.shape[1],
                                             action_length=az.shape[0])
        batch = {
            "video": [[video]], "action": [[az]],
            "ai_caption": [ms["ai_caption"]], "conditioning_fps": [ms["conditioning_fps"]],
            "domain_id": [ms["domain_id"]], "raw_action_dim": [ms["raw_action_dim"]],
            "action_processing_record": [ms["action_processing_record"]],
            "sequence_plan": [plan],
        }
        with torch.inference_mode():
            out = model.generate_samples_from_batch(
                batch, guidance=args.guidance, seed=[0],
                num_steps=args.num_steps, shift=args.shift)
        pr_raw = denormalize_action(out["action"][0][:, :9].detach().cpu(), "minmax", stats)
        gts.append(float(gt_raw[0, 0])); preds.append(float(pr_raw[0, 0]))
        gt_yaw.append(math.degrees(math.atan2(float(gt_raw[0, 4]), float(gt_raw[0, 3]))))
        pr_yaw.append(math.degrees(math.atan2(float(pr_raw[0, 4]), float(pr_raw[0, 3]))))
        if (n + 1) % 8 == 0:
            print(f"    ...{n+1}/{len(idxs)}", flush=True)

    mae = sum(abs(a - b) for a, b in zip(gts, preds)) / len(gts)
    lo, hi = min(gts), max(gts)
    mid = (lo + hi) / 2
    slow = [preds[i] for i, g in enumerate(gts) if g <= mid]
    fast = [preds[i] for i, g in enumerate(gts) if g > mid]

    print()
    print(f"=== [{args.label}] VELOCITY (pos_x), n={len(gts)} ===")
    print(f"  GT   range [{lo:.4f}, {hi:.4f}]  spread {hi-lo:.4f}")
    print(f"  PRED range [{min(preds):.4f}, {max(preds):.4f}]  spread {max(preds)-min(preds):.4f}")
    print(f"  Pearson r  = {pearson(gts,preds):+.3f}")
    print(f"  Spearman r = {spearman(gts,preds):+.3f}   <- rank-based, robust")
    print(f"  MAE        = {mae:.4f}")
    if slow and fast:
        print(f"  slow-half pred mean = {st.mean(slow):.4f}  (n={len(slow)})")
        print(f"  fast-half pred mean = {st.mean(fast):.4f}  (n={len(fast)})")
        print(f"  gap = {st.mean(fast)-st.mean(slow):+.4f}   (GT gap = "
              f"{st.mean([g for g in gts if g>mid])-st.mean([g for g in gts if g<=mid]):+.4f})")
    print(f"  YAW Spearman r = {spearman(gt_yaw,pr_yaw):+.3f}")


if __name__ == "__main__":
    main()
