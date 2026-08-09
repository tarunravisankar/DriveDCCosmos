#!/usr/bin/env python3
"""
roboracer_epoch_summary.py

Clean, av_imitation-style per-epoch progress display ("Epoch X/1000 Train
Loss: A Val Loss: B") for the roboracer retrain, instead of digging through
cosmos_framework's verbose per-iteration/per-rank log lines.

Read-only / non-destructive: this script only watches the training log and
the watchdog log and prints/reports — it never sends signals to the training
process. (The earlier checkpoint-deletion mistake this session came from a
script with write/delete side effects; keeping this one observation-only
avoids that risk entirely.)

Epoch aggregation mirrors early_stopping_watchdog.py exactly (epoch_idx =
iteration // iters_per_epoch, mean loss per epoch) so the numbers shown here
are consistent with what actually drives the early-stopping decision — this
script does not implement its own, possibly-divergent notion of "epoch".

Also marks the run's status in wandb via the API (not just local files) once
the outcome is known, so it's checkable from the dashboard itself, not just
from local logs:
  - "early_stopped"   — watchdog's SIGTERM-fired line appears in its log
  - "completed_max_iter" — training log shows "Done with training"
  - "crashed"         — training process is gone but neither of the above
                         appeared (unexpected exit)

Usage:
    python roboracer_epoch_summary.py \
        --log-file /tmp/roboracer_retrain_v5.log \
        --watchdog-log-file /tmp/roboracer_watchdog_v5.log \
        --pid 2602986 \
        --iters-per-epoch 3124 --max-epochs 1000 \
        --min-delta 0.001 \
        --wandb-entity tarun-vidyut-university-of-texas-at-austin \
        --wandb-project cosmos3_action --wandb-run-id zea3622p
"""

import argparse
import os
import re
import time
from pathlib import Path

_TRAIN_LOSS_RE = re.compile(r"RANK 0\] (\d+) : iter_speed [\d.]+ seconds per iteration \| Loss: ([\d.]+)")
_VAL_LOSS_RE = re.compile(r"Validation loss \(iteration (\d+)\):\s*([\d.]+)\s*$", re.MULTILINE)
_DONE_RE = re.compile(r"Done with training")
_WATCHDOG_FIRED_RE = re.compile(r"Sending SIGTERM to PID")
_WATCHDOG_BEST_RE = re.compile(r"Best epoch: (\d+) \(loss=([\d.]+)\)")


def parse_pairs(log_path: Path, pattern: re.Pattern) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="ignore")
    return [(int(it), float(loss)) for it, loss in pattern.findall(text)]


def bucket_by_epoch(pairs: list[tuple[int, float]], iters_per_epoch: float) -> dict[int, list[float]]:
    by_epoch: dict[int, list[float]] = {}
    for it, loss in pairs:
        epoch_idx = int(it // iters_per_epoch)
        by_epoch.setdefault(epoch_idx, []).append(loss)
    return by_epoch


def mark_wandb_status(entity: str, project: str, run_id: str, status: str, **extra) -> None:
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        run.summary["roboracer_status"] = status
        for k, v in extra.items():
            run.summary[f"roboracer_{k}"] = v
        existing_tags = [t for t in run.tags if not t.startswith("roboracer-status:")]
        run.tags = existing_tags + [f"roboracer-status:{status}"]
        run.update()
        print(f"[summary] marked wandb run {entity}/{project}/{run_id} as '{status}' ({extra})")
    except Exception as e:  # noqa: BLE001 - never let a wandb API hiccup crash the monitor
        print(f"[summary] warning: failed to mark wandb status: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--watchdog-log-file", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True, help="Training PID, watched read-only (never signaled).")
    parser.add_argument("--iters-per-epoch", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--interval-s", type=float, default=15.0)
    args = parser.parse_args()

    def wandb_mark(status: str, **extra) -> None:
        if args.wandb_entity and args.wandb_project and args.wandb_run_id:
            mark_wandb_status(args.wandb_entity, args.wandb_project, args.wandb_run_id, status, **extra)

    printed_epochs: set[int] = set()
    best_val = float("inf")
    best_epoch = None

    while True:
        train_pairs = parse_pairs(args.log_file, _TRAIN_LOSS_RE)
        val_pairs = parse_pairs(args.log_file, _VAL_LOSS_RE)
        train_by_epoch = bucket_by_epoch(train_pairs, args.iters_per_epoch)
        val_by_epoch = bucket_by_epoch(val_pairs, args.iters_per_epoch)

        # Only print an epoch once its train-loss bucket has a full
        # iters_per_epoch worth of iterations (i.e. it's actually complete),
        # to avoid printing a partial, misleadingly-low/high in-progress mean.
        complete_epochs = sorted(
            e for e, losses in train_by_epoch.items()
            if len(losses) >= args.iters_per_epoch and e not in printed_epochs
        )
        for epoch in complete_epochs:
            train_mean = sum(train_by_epoch[epoch]) / len(train_by_epoch[epoch])
            val_losses = val_by_epoch.get(epoch)
            if val_losses:
                val_mean = sum(val_losses) / len(val_losses)
                if val_mean < best_val - args.min_delta:
                    best_val = val_mean
                    best_epoch = epoch
                val_str = f"{val_mean:.4f}"
                best_str = f" (best={best_val:.4f} @ epoch {best_epoch})"
            else:
                val_str = "N/A"
                best_str = ""
            print(f"Epoch [{epoch}/{args.max_epochs}] Train Loss: {train_mean:.4f} Val Loss: {val_str}{best_str}")
            printed_epochs.add(epoch)

        # Terminal-state detection (read-only — just observes and reports).
        watchdog_text = args.watchdog_log_file.read_text(errors="ignore") if args.watchdog_log_file.exists() else ""
        train_text = args.log_file.read_text(errors="ignore") if args.log_file.exists() else ""
        proc_alive = os.path.exists(f"/proc/{args.pid}")

        if _WATCHDOG_FIRED_RE.search(watchdog_text):
            m = _WATCHDOG_BEST_RE.search(watchdog_text)
            best_iter_info = {"best_epoch": int(m.group(1)), "best_loss": float(m.group(2))} if m else {}
            print(f"[summary] early stopping fired. {best_iter_info}")
            wandb_mark("early_stopped", **best_iter_info)
            return
        if _DONE_RE.search(train_text):
            print("[summary] training completed naturally (reached max_iter).")
            wandb_mark("completed_max_iter")
            return
        if not proc_alive:
            print(f"[summary] training process {args.pid} is gone, but no early-stop or completion message seen — crash.")
            wandb_mark("crashed")
            return

        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
