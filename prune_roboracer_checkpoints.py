#!/usr/bin/env python3
"""
prune_roboracer_checkpoints.py

Runs alongside the roboracer retrain (each checkpoint is ~83GB and /scratch is
a shared 18T filesystem that already hit 0 bytes free once this session).
With no built-in "keep last N checkpoints" option in cosmos_framework's
checkpoint code, this script keeps disk usage bounded by deleting every
checkpoint directory except:
  - the current best: the single checkpoint with the lowest validation loss
    WITHIN the best EPOCH so far, where "best epoch" is computed the same way
    early_stopping_watchdog.py decides it (mean of that epoch's validation
    checks) — not just whichever single raw check happens to be lowest. This
    matters because the watchdog's patience/stopping decision is epoch-level;
    the checkpoint kept as "best" should be the one the watchdog would
    actually consider the rollback point if it stopped right now, i.e. the
    model from the epoch where the patience window last reset, not an
    unrelated single lucky check from a worse epoch.
  - the single most recently *completed* checkpoint (needed to resume if the
    job dies; "completed" = has all 4 expected DCP subdirectories, so we never
    touch a checkpoint that's still being written)

Polls every --interval-s seconds until the training log shows the run has
exited (or --max-iter is reached).

Usage:
    python prune_roboracer_checkpoints.py \
        --log-file /tmp/roboracer_retrain_v5.log \
        --checkpoint-dir /scratch/.../action_policy_roboracer_repro_v5/checkpoints \
        --iters-per-epoch 3124
"""

import argparse
import re
import shutil
import time
from pathlib import Path

_VAL_LOSS_RE = re.compile(r"Validation loss \(iteration (\d+)\): ([\d.]+)")
_EXPECTED_SUBDIR_COUNT = 4  # model/optim/scheduler/trainer — a fully-written checkpoint


def parse_validation_losses(log_path: Path) -> dict[int, float]:
    losses: dict[int, float] = {}
    text = log_path.read_text(errors="ignore")
    for it_str, loss_str in _VAL_LOSS_RE.findall(text):
        losses[int(it_str)] = float(loss_str)
    return losses


def list_checkpoint_iters(checkpoint_dir: Path) -> list[int]:
    iters = []
    for p in checkpoint_dir.iterdir():
        if p.is_dir() and p.name.startswith("iter_"):
            iters.append(int(p.name.removeprefix("iter_")))
    return sorted(iters)


def is_complete(checkpoint_dir: Path, it: int) -> bool:
    ckpt_path = checkpoint_dir / f"iter_{it:09d}"
    return ckpt_path.is_dir() and sum(1 for _ in ckpt_path.iterdir()) >= _EXPECTED_SUBDIR_COUNT


def best_iter_by_epoch(losses: dict[int, float], iters_per_epoch: float) -> int | None:
    """Mirror early_stopping_watchdog.py's epoch-aggregation: bucket every
    validation check by epoch_idx = iteration // iters_per_epoch, find the
    epoch with the lowest MEAN loss, then return the single checkpoint
    iteration with the lowest individual loss within that epoch (the
    rollback point the watchdog would actually pick)."""
    if not losses:
        return None
    by_epoch: dict[int, list[tuple[int, float]]] = {}
    for it, loss in losses.items():
        epoch_idx = int(it // iters_per_epoch)
        by_epoch.setdefault(epoch_idx, []).append((it, loss))

    best_epoch = min(by_epoch, key=lambda e: sum(l for _, l in by_epoch[e]) / len(by_epoch[e]))
    return min(by_epoch[best_epoch], key=lambda x: x[1])[0]


def prune(checkpoint_dir: Path, log_path: Path, iters_per_epoch: float) -> None:
    losses = parse_validation_losses(log_path)
    iters = [it for it in list_checkpoint_iters(checkpoint_dir) if is_complete(checkpoint_dir, it)]
    if not iters:
        return

    latest = max(iters)
    # Best-epoch checkpoint among iterations we actually have both a logged
    # validation loss AND a checkpoint on disk for; fall back to latest if
    # none logged yet (e.g. iteration 0 hasn't been parsed).
    scored_losses = {it: losses[it] for it in iters if it in losses}
    best = best_iter_by_epoch(scored_losses, iters_per_epoch) if scored_losses else latest

    keep = {best, latest}
    for it in iters:
        if it in keep:
            continue
        path = checkpoint_dir / f"iter_{it:09d}"
        print(f"[prune] removing superseded checkpoint {path} (keeping best={best}, latest={latest})")
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--iters-per-epoch", type=float, required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    print(
        f"Pruning {args.checkpoint_dir} every {args.interval_s}s, keeping only "
        f"best-epoch-checkpoint+latest (iters_per_epoch={args.iters_per_epoch})..."
    )
    while True:
        try:
            prune(args.checkpoint_dir, args.log_file, args.iters_per_epoch)
        except Exception as e:  # noqa: BLE001 - janitor must never crash the loop
            print(f"[prune] warning: {e}")

        losses = parse_validation_losses(args.log_file)
        if losses and max(losses) >= args.max_iter:
            print("[prune] reached max_iter in log, exiting.")
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
