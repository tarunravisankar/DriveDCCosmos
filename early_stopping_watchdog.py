#!/usr/bin/env python3
"""
early_stopping_watchdog.py

Epsilon-tolerant, epochs-based patience early stopping, applied automatically
rather than relying on a human watching the log live — the v2 retrain ran
unattended all the way to its max_iter hard cap because the real
early-stopping point came and went while nobody was watching.

v2 of this script (single derived "checks_per_epoch = iters_per_epoch /
validation_iter" conversion, then comparing every raw 200-iteration validation
check against patience_checks) conflated two different granularities into one
loop. Per the professor's explicit guidance, this version uses two separate
loops instead:
  - an INNER loop (over individual validation checks as they appear in the
    log, every --validation-iter iterations) that just buckets each check's
    loss into the epoch it falls in;
  - an OUTER loop (over completed epochs) that fires exactly once per epoch
    boundary, aggregates that epoch's checks into one epoch-level loss (mean),
    and runs the actual best/patience comparison against OTHER epochs — not
    against raw sub-epoch checks.

Algorithm (per professor's guidance):
    best_epoch_loss = inf, patience_counter = 0  # counted in EPOCHS, not checks
    for each completed epoch (outer loop):
        epoch_loss = mean of that epoch's validation checks (inner loop output)
        if epoch_loss < best_epoch_loss - min_delta:   # real improvement, beyond noise
            best_epoch_loss = epoch_loss; best_epoch = epoch; patience_counter = 0
        else:                                          # flat within min_delta, or worse
            patience_counter += 1
        if patience_counter >= patience_epochs: SIGTERM and exit

min_delta (epsilon, default 0.001) means an epoch loss that stays within a
small band of the best epoch's loss for a long time counts as "non-decreasing"
and contributes to stopping, not just literal increases. patience
(--patience-epochs) and the ceiling are both expressed directly in epochs.

Usage:
    python early_stopping_watchdog.py \
        --log-file /tmp/roboracer_retrain_v5.log \
        --pid 123456 \
        --iters-per-epoch 3124 --validation-iter 200 \
        --patience-epochs 20 --min-delta 0.001
"""

import argparse
import os
import re
import signal
import time
from pathlib import Path

_VAL_LOSS_RE = re.compile(r"Validation loss \(iteration (\d+)\):\s*([\d.]+)\s*$", re.MULTILINE)
_DONE_RE = re.compile(r"Done with training")


def parse_validation_losses(log_path: Path) -> list[tuple[int, float]]:
    """Return (iteration, loss) pairs in log order, skipping NaN/unparseable entries."""
    text = log_path.read_text(errors="ignore")
    return [(int(it), float(loss)) for it, loss in _VAL_LOSS_RE.findall(text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True, help="PID of the torchrun parent process to SIGTERM")
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=20,
        help="Epochs of non-decreasing (within --min-delta) EPOCH-LEVEL validation loss to tolerate before stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help=(
            "Minimum improvement over best_epoch_loss required to count as a real decrease "
            "and reset the patience counter. Improvements smaller than this (including exact "
            "ties or slight increases) count as 'non-decreasing'."
        ),
    )
    parser.add_argument("--iters-per-epoch", type=float, required=True)
    parser.add_argument("--validation-iter", type=float, required=True, help="Iterations between validation checks.")
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=0,
        help=(
            "Don't allow the watchdog to fire before this EPOCH, even if patience is "
            "exhausted — validation loss is noisy enough early on (large swings before "
            "the model has real exposure to the data) that a non-decreasing streak can "
            "happen from noise alone. best_epoch_loss is still tracked from epoch 0; "
            "only the kill action is gated."
        ),
    )
    parser.add_argument("--interval-s", type=float, default=15.0)
    args = parser.parse_args()

    print(
        f"Early stopping watchdog (two-loop, epoch-aggregated): patience={args.patience_epochs} epochs, "
        f"min_delta={args.min_delta}, min_epoch={args.min_epoch}, iters_per_epoch={args.iters_per_epoch}, "
        f"watching {args.log_file}"
    )

    seen = 0
    best_epoch_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    current_epoch = 0
    current_epoch_checks: list[float] = []

    def finalize_epoch(epoch_idx: int, checks: list[float]) -> bool:
        """OUTER loop body: runs once per completed epoch. Returns True if the
        watchdog should fire (and has already sent SIGTERM)."""
        nonlocal best_epoch_loss, best_epoch, patience_counter
        if not checks:
            return False
        epoch_loss = sum(checks) / len(checks)
        if epoch_loss < best_epoch_loss - args.min_delta:
            best_epoch_loss = epoch_loss
            best_epoch = epoch_idx
            patience_counter = 0
            print(f"[watchdog] epoch {epoch_idx}: new best epoch loss={epoch_loss:.6f} (patience reset)")
        else:
            patience_counter += 1
            print(
                f"[watchdog] epoch {epoch_idx}: epoch loss={epoch_loss:.6f}, non-decreasing vs. "
                f"best={best_epoch_loss:.6f} @ epoch {best_epoch} (within min_delta={args.min_delta}) "
                f"({patience_counter}/{args.patience_epochs} epochs)"
            )
        if patience_counter >= args.patience_epochs:
            if epoch_idx < args.min_epoch:
                print(
                    f"[watchdog] epoch {epoch_idx}: patience exhausted but epoch < "
                    f"min_epoch={args.min_epoch} — not stopping yet, just resetting the counter."
                )
                patience_counter = 0
            else:
                print(
                    f"[watchdog] patience exhausted ({args.patience_epochs} epochs without a "
                    f">{args.min_delta} improvement). Best epoch: {best_epoch} (loss={best_epoch_loss:.6f}). "
                    f"Sending SIGTERM to PID {args.pid}."
                )
                try:
                    os.kill(args.pid, signal.SIGTERM)
                except ProcessLookupError:
                    print(f"[watchdog] PID {args.pid} already gone.")
                return True
        return False

    while True:
        time.sleep(args.interval_s)

        if args.log_file.exists() and _DONE_RE.search(args.log_file.read_text(errors="ignore")):
            print("[watchdog] training finished on its own (reached max_iter) — exiting.")
            if finalize_epoch(current_epoch, current_epoch_checks):
                pass
            break

        if not args.log_file.exists():
            continue
        losses = parse_validation_losses(args.log_file)
        if len(losses) <= seen:
            continue

        # INNER loop: bucket each new raw validation check into its epoch.
        stop = False
        for iteration, loss in losses[seen:]:
            epoch_idx = int(iteration // args.iters_per_epoch)
            if epoch_idx != current_epoch:
                # Epoch boundary crossed — OUTER loop fires for the epoch that just completed.
                if finalize_epoch(current_epoch, current_epoch_checks):
                    stop = True
                    break
                current_epoch = epoch_idx
                current_epoch_checks = []
            current_epoch_checks.append(loss)
        seen = len(losses)
        if stop:
            return


if __name__ == "__main__":
    main()
