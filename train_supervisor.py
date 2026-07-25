#!/usr/bin/env python3
"""Auto-relaunch supervisor for the roboracer action-policy SFT training job.

Wraps the torchrun launch + stall_watchdog in an outer retry loop so that
the known DataLoader-worker D-state silent hang (occurs at every
checkpoint_boundary+144 iteration) is handled without manual intervention:
  1. Launches training (torchrun) + watchdog (stall_watchdog.py) together.
  2. Waits for training to exit.
  3. If training finished naturally → exits cleanly.
     If training was killed by the watchdog (stall) → relaunches both and
     continues from the latest checkpoint automatically (same job name, so
     the trainer finds latest_checkpoint.txt on its own).
  4. Caps at MAX_RELAUNCHES to prevent infinite loops on real errors.

checkpoint.save_iter is intentionally overridden to a small value (50) here
so that at most ~50 iterations of progress are lost per hang, vs the toml's
default 200. Overhead: each checkpoint write takes ~190s and ~83GB; at 50-iter
cadence and ~17s/iter that's ~190/(50*17) ≈ 22% wall-clock overhead — an
acceptable trade-off given each hang otherwise wastes 143+ iterations.

Usage (run as a nohup background process, keeps training alive across SSH
sessions and auto-recovers from stalls without any manual intervention):
    cd /scratch/tarunrav/cosmos-framework
    nohup .venv/bin/python3 train_supervisor.py > /tmp/train_supervisor.log 2>&1 &
    disown
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_RELAUNCHES = 100
STALL_TIMEOUT_S = 600
LOG_FILE = Path("/tmp/roboracer_retrain_v10_robolang.log")
WATCHDOG_LOG = Path("/tmp/roboracer_stall_watchdog_v10.log")

# Early stopping configuration.
#
# One "epoch" = iters for the largest dataset to exhaust once.
# orin13 is the largest: 38858 windows / 96 max_per_batch = 405 iters/epoch.
# Using this as the reference unit keeps "patience in epochs" meaningful even
# though each rank cycles at a different rate.
EPOCH_ITERS = 405

# Stop if val loss hasn't improved (by more than EPSILON) for this many
# epoch-equivalent iterations.  20 epochs = 8100 iters ≈ 38 h at 17 s/iter.
EARLY_STOP_PATIENCE_EPOCHS = 12
PATIENCE_ITERS = EARLY_STOP_PATIENCE_EPOCHS * EPOCH_ITERS  # 4860

# Don't start tracking best val or counting patience until this many epoch-
# equivalent iterations have elapsed.  Prevents an early fluky low-loss eval
# from anchoring the best and making everything after look like no improvement.
EARLY_STOP_WARMUP_EPOCHS = 5
WARMUP_ITERS = EARLY_STOP_WARMUP_EPOCHS * EPOCH_ITERS  # 2025

# Minimum drop in val loss to count as a real improvement (noise filter).
EARLY_STOP_EPSILON = 0.001

# Marker written to log so is_training_done() recognises early-stop as "done".
EARLY_STOP_MARKER = "[early_stop] Training stopped"

# Checkpoint directory — derived from IMAGINAIRE_OUTPUT_ROOT + job.name in TRAIN_CMD.
CHECKPOINT_DIR = Path(
    "/scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft"
    "/action_policy_roboracer_repro_v10/checkpoints"
)

VENV_NV = "/scratch/tarunrav/cosmos-framework/.venv/lib/python3.13/site-packages/nvidia"
CLEAN_LD = ":".join([
    f"{VENV_NV}/cu13/lib", f"{VENV_NV}/cudnn/lib", f"{VENV_NV}/nccl/lib",
    f"{VENV_NV}/nvshmem/lib", f"{VENV_NV}/nvjpeg2k/lib", f"{VENV_NV}/nvjpeg/lib",
    f"{VENV_NV}/nvtiff/lib", f"{VENV_NV}/cusparselt/lib",
])

TRAIN_ENV = {
    **os.environ,
    # Dataset roots are now hardcoded in action_policy_roboracer_nano.py —
    # only BASE_CHECKPOINT_PATH, WAN_VAE_PATH, and IMAGINAIRE_OUTPUT_ROOT
    # still need to be set here for the TOML's oc.env references.
    "BASE_CHECKPOINT_PATH": "/scratch/tarunrav/cosmos-framework/examples/checkpoints/Cosmos3-Nano",
    "WAN_VAE_PATH": "/home/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
    "IMAGINAIRE_OUTPUT_ROOT": "/scratch/tarunrav/cosmos-framework/outputs",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/roboracer_nccl_trace_",
    "LD_LIBRARY_PATH": CLEAN_LD,
}

TRAIN_CMD = [
    "/scratch/tarunrav/cosmos-framework/.venv/bin/torchrun",
    "--nproc_per_node=8",
    "-m", "cosmos_framework.scripts.train",
    "--sft-toml", "examples/toml/sft_config/action_policy_roboracer_repro.toml",
    "--",
    "job.name=action_policy_roboracer_repro_v10",
    # Let early stopping decide — 100k is a safety ceiling only (~470h at 17s/iter)
    "trainer.max_iter=100000",
    # Targeting ~90% of 80GB VRAM — start at 256 and tune from observed memory
    "dataloader_train.max_samples_per_batch=96",
    # Save every 10 iterations so we have a checkpoint before the ~iter-18 hang.
    "checkpoint.save_iter=50",
    "trainer.run_validation_on_start=False",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[supervisor {ts}] {msg}"
    print(line, flush=True)


_VAL_RE = re.compile(r"Validation loss \(iteration (\d+)\): ([0-9.]+)")
_CKPT_SAVE_RE = re.compile(r"Saved checkpoint to .+?(iter_(\d+))\s*$")


def _prune_checkpoints(best_iter: int) -> None:
    """Delete all checkpoints except the best-val one and the most recent one.

    Safety rules (never skip these):
    1. If best_iter is known, the best checkpoint MUST exist on disk — if it
       doesn't, abort the entire prune rather than silently deleting data.
    2. Never delete if ≤ 2 checkpoints on disk (nothing to prune).
    3. Never delete a checkpoint that is in the keep set.
    """
    if not CHECKPOINT_DIR.exists():
        return

    ckpt_dirs = sorted(
        [d for d in CHECKPOINT_DIR.iterdir()
         if d.is_dir() and re.fullmatch(r"iter_\d+", d.name)],
        key=lambda d: int(d.name[5:]),
    )

    if len(ckpt_dirs) <= 2:
        return  # nothing to prune

    keep = set()
    keep.add(ckpt_dirs[-1].name)  # always keep the latest on disk

    if best_iter >= 0:
        best_name = f"iter_{best_iter:09d}"
        best_path = CHECKPOINT_DIR / best_name
        if not best_path.exists():
            # Best checkpoint is already missing — abort rather than make it worse.
            log(f"[ckpt_prune] ABORT — best checkpoint {best_name} not found on disk! "
                f"Will not delete anything. Manual inspection needed.")
            return
        keep.add(best_name)
    else:
        # No val result yet — keep the two most recent
        keep.add(ckpt_dirs[-2].name)

    for d in ckpt_dirs:
        if d.name not in keep:
            try:
                shutil.rmtree(d)
                log(f"[ckpt_prune] Deleted {d.name}  "
                    f"(keeping best=iter_{best_iter:09d}, latest={ckpt_dirs[-1].name})")
            except OSError as e:
                log(f"[ckpt_prune] Failed to delete {d.name}: {e}")


def _monitor(train_proc: subprocess.Popen) -> None:
    """Daemon thread: combined early-stopping + checkpoint pruning.

    Early stopping — patience measured in iteration distance from best val
    checkpoint, not a count of consecutive bad evaluations.  Only a drop of
    more than EARLY_STOP_EPSILON counts as a real improvement (noise filter).

    Checkpoint pruning — after every new checkpoint save detected in the log,
    keep only the best-val and the most-recent checkpoint on disk.
    """
    best_loss = float("inf")
    best_iter = -1       # iter of best val loss seen so far (-1 = none yet)
    latest_ckpt_iter = -1  # iter of most recent saved checkpoint
    seen_lines = 0

    while train_proc.poll() is None:
        time.sleep(30)  # val runs ~every 3.5 min; 30s poll won't miss anything

        if not LOG_FILE.exists():
            continue

        try:
            lines = LOG_FILE.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        new_checkpoint_saved = False

        for line in lines[seen_lines:]:
            # ── checkpoint save ──────────────────────────────────────────────
            cm = _CKPT_SAVE_RE.search(line)
            if cm:
                latest_ckpt_iter = int(cm.group(2))
                new_checkpoint_saved = True

            # ── validation result ────────────────────────────────────────────
            vm = _VAL_RE.search(line)
            if not vm:
                continue
            iteration, val_loss = int(vm.group(1)), float(vm.group(2))

            if iteration < WARMUP_ITERS:
                log(
                    f"[monitor] val {val_loss:.4f} at iter {iteration} "
                    f"(warmup — early stop inactive until iter {WARMUP_ITERS} "
                    f"/ {EARLY_STOP_WARMUP_EPOCHS} epochs)"
                )
                continue

            if val_loss < best_loss - EARLY_STOP_EPSILON:
                best_loss = val_loss
                best_iter = iteration
                log(
                    f"[monitor] New best val {val_loss:.4f} at iter {iteration} "
                    f"(improved by >{EARLY_STOP_EPSILON})"
                )
            else:
                iters_since = iteration - best_iter if best_iter >= 0 else 0
                epochs_since = iters_since / EPOCH_ITERS
                log(
                    f"[monitor] val {val_loss:.4f} — no real improvement; "
                    f"{iters_since} iters ({epochs_since:.1f}/{EARLY_STOP_PATIENCE_EPOCHS} epochs) "
                    f"since best {best_loss:.4f} at iter {best_iter}"
                )
                if best_iter >= 0 and iters_since >= PATIENCE_ITERS:
                    log(
                        f"[monitor] Early stopping — patience exhausted after "
                        f"{epochs_since:.1f} epochs without >{EARLY_STOP_EPSILON} improvement. "
                        f"Best checkpoint: iter_{best_iter:09d} (val {best_loss:.4f})"
                    )
                    try:
                        with LOG_FILE.open("a") as fh:
                            fh.write(
                                f"\n{EARLY_STOP_MARKER} — {epochs_since:.1f} epochs "
                                f"({iters_since} iters) without >{EARLY_STOP_EPSILON} improvement. "
                                f"Best val {best_loss:.4f} at iter {best_iter}.\n"
                            )
                    except OSError:
                        pass
                    # Prune before killing so the best checkpoint is preserved
                    _prune_checkpoints(best_iter)
                    try:
                        train_proc.terminate()
                    except ProcessLookupError:
                        pass
                    return

        seen_lines = len(lines)

        # Prune after every new checkpoint save (outside the per-line loop so
        # we only call it once per poll cycle even if multiple saves were logged)
        if new_checkpoint_saved:
            _prune_checkpoints(best_iter)


def is_training_done() -> bool:
    """Return True if training finished naturally or was stopped by early stopping."""
    if not LOG_FILE.exists():
        return False
    text = LOG_FILE.read_text(errors="ignore")
    return any(k in text for k in [
        "Training finished", "Reached max_iter", "training done", "max_iter reached",
        EARLY_STOP_MARKER,
    ])


def launch_one_attempt(attempt: int) -> int:
    """Launch training + watchdog, wait for training to exit, return exit code."""
    log(f"=== Attempt {attempt}/{MAX_RELAUNCHES} ===")

    with LOG_FILE.open("a") as log_fh:
        log_fh.write(f"\n\n{'='*60}\n[supervisor] Attempt {attempt} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n\n")

    # Launch training, appending to the shared log file so full history is preserved.
    with LOG_FILE.open("ab") as log_fh:
        train_proc = subprocess.Popen(
            TRAIN_CMD,
            env=TRAIN_ENV,
            cwd="/scratch/tarunrav/cosmos-framework",
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,  # equivalent to setsid — survives supervisor death
        )
    log(f"training launched pid={train_proc.pid}")

    # Give training a few seconds to start writing to the log before starting watchdog.
    time.sleep(15)

    watchdog_proc = None
    with WATCHDOG_LOG.open("a") as wlog_fh:
        wlog_fh.write(f"\n[supervisor] Starting watchdog for attempt {attempt}, train_pid={train_proc.pid}\n")
    with WATCHDOG_LOG.open("ab") as wlog_fh:
        watchdog_proc = subprocess.Popen(
            [
                "/scratch/tarunrav/cosmos-framework/.venv/bin/python3",
                "stall_watchdog.py",
                "--log-file", str(LOG_FILE),
                "--pid", str(train_proc.pid),
                "--stall-timeout-s", str(STALL_TIMEOUT_S),
                "--warmup-timeout-s", "2400",   # 40 min — covers cold validation-on-start
                "--warmup-duration-s", "2700",  # switch to 600s after 45 min from launch
            ],
            cwd="/scratch/tarunrav/cosmos-framework",
            stdout=wlog_fh,
            stderr=wlog_fh,
            start_new_session=True,  # survives even a supervisor crash, not just SSH disconnect
        )
    log(f"watchdog launched pid={watchdog_proc.pid}")

    # Combined early-stopping + checkpoint-pruning monitor (daemon thread).
    monitor_thread = threading.Thread(
        target=_monitor,
        args=(train_proc,),
        name="monitor",
        daemon=True,
    )
    monitor_thread.start()
    log(f"monitor started — early stop: warmup={EARLY_STOP_WARMUP_EPOCHS} epochs/{WARMUP_ITERS} iters, patience={EARLY_STOP_PATIENCE_EPOCHS} epochs/{PATIENCE_ITERS} iters, ε={EARLY_STOP_EPSILON}; checkpoint pruning: keep best+latest")

    # Wait for training to finish (naturally, via watchdog SIGTERM, or early stop).
    exit_code = train_proc.wait()
    log(f"training exited with code={exit_code}")

    # Clean up watchdog.
    if watchdog_proc.poll() is None:
        try:
            watchdog_proc.terminate()
            watchdog_proc.wait(timeout=10)
        except Exception:
            pass

    return exit_code


def main() -> None:
    log("supervisor started")
    log(f"max_relaunches={MAX_RELAUNCHES}, stall_timeout={STALL_TIMEOUT_S}s, save_iter=50")

    if is_training_done():
        log("training already done, exiting")
        return

    for attempt in range(1, MAX_RELAUNCHES + 1):
        exit_code = launch_one_attempt(attempt)

        if is_training_done():
            log("training finished naturally — supervisor done")
            return

        if exit_code == 0:
            log("training exited cleanly (code 0) — supervisor done")
            return

        if attempt < MAX_RELAUNCHES:
            log(f"stall/error detected (exit={exit_code}), relaunching in 10s...")
            time.sleep(10)
        else:
            log(f"hit MAX_RELAUNCHES={MAX_RELAUNCHES} — giving up, manual intervention needed")
            sys.exit(1)


if __name__ == "__main__":
    main()
