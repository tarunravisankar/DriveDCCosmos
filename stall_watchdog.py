#!/usr/bin/env python3
"""
stall_watchdog.py

Detects a hung-but-still-alive training process — e.g. a silent NCCL
collective deadlock (CPU pegged in a busy-poll wait, GPU idle, full memory
still held, no exception ever raised despite TORCH_NCCL_ASYNC_ERROR_HANDLING
already being set by cosmos_framework/utils/distributed.py) — and kills it so
it can be resumed from the latest checkpoint, instead of silently burning
GPU-hours doing nothing. This happened once on the v8 robolang run: training
froze for ~8 hours after an anomalously fast iteration (2.47s vs. the usual
~16-18s), with no error ever logged.

Unlike early_stopping_watchdog.py (which watches validation LOSS values),
this only watches for forward PROGRESS — whether the log file is growing at
all, regardless of content. A true stall produces zero new log output of any
kind (the stuck ranks are spinning in C++ NCCL wait code, not producing any
Python-level logging), so "log file size unchanged for N minutes while the
process is still alive" is a clean, reliable signal — no legitimate pause
(even a slow validation+checkpoint-save cycle) should come close to the
default 15-minute threshold.

This script is read-only with respect to training EXCEPT for the kill action
itself — it never modifies checkpoints, config, or any other state. It does
NOT auto-relaunch training after killing a stalled process; resuming is a
separate manual step (same as after the existing early-stopping watchdog
fires), to avoid building a crash-loop into something used unattended.

Usage:
    python stall_watchdog.py \
        --log-file /tmp/roboracer_retrain_v8_robolang.log \
        --pid 3479983 \
        --stall-timeout-s 900 --interval-s 30
"""

import argparse
import os
import re
import signal
import time
from pathlib import Path

_DONE_RE = re.compile(r"Done with training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True, help="Training PID, killed (SIGTERM) if stalled.")
    parser.add_argument(
        "--stall-timeout-s",
        type=float,
        default=900.0,
        help=(
            "If the log file hasn't grown in this many seconds while the process is still "
            "alive, treat it as a stall and kill it. Default 900s (15 min) is comfortably "
            "longer than any legitimate validation+checkpoint-save pause observed so far "
            "(~170-190s), but far shorter than letting a real hang run for hours."
        ),
    )
    parser.add_argument("--interval-s", type=float, default=30.0)
    args = parser.parse_args()

    print(
        f"Stall watchdog: pid={args.pid}, stall_timeout={args.stall_timeout_s}s, "
        f"watching {args.log_file} for log growth"
    )

    last_size = -1
    last_growth_time = time.time()

    while True:
        time.sleep(args.interval_s)

        if not os.path.exists(f"/proc/{args.pid}"):
            print(f"[stall-watchdog] pid {args.pid} is gone — exiting.")
            return

        if not args.log_file.exists():
            continue

        text_for_done_check = args.log_file.read_text(errors="ignore")
        if _DONE_RE.search(text_for_done_check):
            print("[stall-watchdog] training finished on its own — exiting.")
            return

        size = args.log_file.stat().st_size
        now = time.time()
        if size != last_size:
            last_size = size
            last_growth_time = now
            continue

        stalled_for = now - last_growth_time
        if stalled_for >= args.stall_timeout_s:
            print(
                f"[stall-watchdog] no log growth for {stalled_for:.0f}s "
                f"(>= {args.stall_timeout_s}s) while pid {args.pid} is still alive — "
                f"likely a silent hang (e.g. NCCL collective deadlock). Sending SIGTERM."
            )
            try:
                os.kill(args.pid, signal.SIGTERM)
            except ProcessLookupError:
                print(f"[stall-watchdog] pid {args.pid} already gone.")
            return
        else:
            print(
                f"[stall-watchdog] no log growth for {stalled_for:.0f}s "
                f"(threshold {args.stall_timeout_s}s) — watching."
            )


if __name__ == "__main__":
    main()
