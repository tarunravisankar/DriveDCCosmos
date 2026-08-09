#!/usr/bin/env python3
"""Auto-relaunch supervisor for the Cosmos3-Edge RoboRacer action-policy SFT.

Same design as the Nano-line train_supervisor.py (auto-relaunch on silent NCCL
hangs + epoch-aggregated early stopping + checkpoint pruning), retargeted at
the Edge run in /scratch/tarunrav/cosmos-edge.

Differences from the Nano supervisor, and why:
  * EPOCH_ITERS = 605.  One "epoch" = iters for the LARGEST dataset to exhaust
    once.  Measured train-window counts (chunk_length=32):
        orin10 32943 | orin13 38660 | orin02  5562 | orin06 18492
        orin14 34755 | pass_right 1639 | pass_left 1769 | wait 2123
        TOTAL 135943, LARGEST 38660
    38660 / 64 global batch (8 per rank x 8 ranks) = 605.
  * EARLY_STOP_PATIENCE_EPOCHS = 20 (the literal default, per prior guidance).
  * NVLink P2P is left ENABLED. The `Cuda failure 711 'peer mapping resources
    exhausted'` seen previously was robolidar-specific -- 10 GPUs shared with
    other users' processes also holding peer mappings, driven on a
    non-contiguous device set. This run uses all 8 dedicated H100s (0-7) on an
    otherwise idle robolang, which is the standard configuration; disabling P2P
    would needlessly give up NVLink bandwidth.
  * Checkpoint pruning keeps {best, latest} and additionally hard-links the
    current best into ../protected/ (see _snapshot_best), so the best survives
    even if the training log -- from which validation history is re-derived --
    is ever lost.

Usage:
    cd /scratch/tarunrav/cosmos-edge
    setsid nohup /scratch/tarunrav/cosmos-framework/.venv/bin/python3 \
        train_supervisor_edge.py > /tmp/edge_supervisor.log 2>&1 &
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

REPO = "/scratch/tarunrav/cosmos-edge"
VENV = "/scratch/tarunrav/cosmos-framework/.venv"

# Give up after this many relaunches. Deliberately low: a healthy run should be
# ended by EARLY STOPPING, not by this guard. Burning 20 relaunches means the
# boundary hang is firing often enough that a human should look, rather than
# the supervisor silently grinding for days.
MAX_RELAUNCHES = 20

# Watchdog timeouts are sized to ITERATION SPEED, not wall-clock intuition.
# On robolang's H100s this run does ~0.8 s/iter, so the 200-iteration window
# between checkpoint/validation boundaries is only ~2.6 minutes. The Nano-era
# values (stall 600 s, warmup 2400/2700 s) were sized for 5.16 s/iter on
# A6000s, where that same window was ~17 min. Left unchanged here they let a
# boundary hang burn ~25 min before detection -- i.e. ~90% of wall time lost.
# Normal inter-log gaps are sub-second; the longest legitimate quiet period is
# validation + checkpoint save at ~70 s. 180 s therefore has ~2.5x headroom
# over the slowest legitimate gap while catching hangs ~8x sooner.
STALL_TIMEOUT_S = 180
# Startup (model load + validation-on-start) measured at ~2 min on H100.
WARMUP_TIMEOUT_S = 900
WARMUP_DURATION_S = 1200
LOG_FILE = Path("/tmp/roboracer_edge_v3.log")
WATCHDOG_LOG = Path("/tmp/roboracer_edge_v3_stall_watchdog.log")

# One epoch = largest dataset (orin13, 38660 windows) / 64 global batch.
EPOCH_ITERS = 605

# Stop if val loss hasn't improved by > EPSILON for this many epoch-equivalents.
EARLY_STOP_PATIENCE_EPOCHS = 20
PATIENCE_ITERS = EARLY_STOP_PATIENCE_EPOCHS * EPOCH_ITERS  # 12100

# Don't track best / count patience until this many epoch-equivalents elapse —
# prevents an early fluky low-loss eval from anchoring "best" forever.
EARLY_STOP_WARMUP_EPOCHS = 5
WARMUP_ITERS = EARLY_STOP_WARMUP_EPOCHS * EPOCH_ITERS  # 3025

EARLY_STOP_EPSILON = 0.001
EARLY_STOP_MARKER = "[early_stop] Training stopped"

JOB_NAME = "action_policy_roboracer_edge_v3"
CHECKPOINT_DIR = Path(f"{REPO}/outputs/cosmos3_action/action_sft/{JOB_NAME}/checkpoints")

NV = f"{VENV}/lib/python3.13/site-packages/nvidia"
CLEAN_LD = ":".join([
    f"{NV}/cu13/lib", f"{NV}/cudnn/lib", f"{NV}/nccl/lib",
    f"{NV}/nvshmem/lib", f"{NV}/nvjpeg2k/lib", f"{NV}/nvjpeg/lib",
    f"{NV}/nvtiff/lib", f"{NV}/cusparselt/lib",
])

# GPUs to use. robolidar is shared — set explicitly rather than grabbing all 10.
CUDA_DEVICES = os.environ.get("EDGE_CUDA_DEVICES", "0,1,2,3,4,5,6,7")

TRAIN_ENV = {
    **os.environ,
    "PATH": "/home/tarunrav/.local/bin:" + os.environ.get("PATH", ""),
    "PYTHONPATH": REPO,
    "HF_HOME": "/scratch/tarunrav/.cache/huggingface",
    "UV_CACHE_DIR": "/scratch/tarunrav/.cache/uv",
    "WAN_VAE_PATH": "/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
    "IMAGINAIRE_OUTPUT_ROOT": f"{REPO}/outputs",
    "BASE_CHECKPOINT_PATH": f"{REPO}/examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp",
    "CUDA_VISIBLE_DEVICES": CUDA_DEVICES,
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/roboracer_edge_nccl_trace_",
    "LD_LIBRARY_PATH": CLEAN_LD,
}

TRAIN_CMD = [
    f"{VENV}/bin/torchrun",
    "--nproc_per_node=8",
    "-m", "cosmos_framework.scripts.train",
    "--sft-toml", "examples/toml/sft_config/action_policy_roboracer_edge.toml",
    "--",
    f"job.name={JOB_NAME}",
    # Safety ceiling only — early stopping is what should actually end the run.
    "trainer.max_iter=515000",
    "dataloader_train.max_samples_per_batch=8",
    # Matches validation_iter=200 in the experiment py, so every validated
    # iteration has a checkpoint on disk to roll back to.
    "checkpoint.save_iter=200",
]


def log(msg: str) -> None:
    print(f"[edge-supervisor {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


_VAL_RE = re.compile(r"Validation loss \(iteration (\d+)\): ([0-9.]+)")
_CKPT_SAVE_RE = re.compile(r"Saved checkpoint to .+?(iter_(\d+))\s*$")


PROTECTED_DIR = CHECKPOINT_DIR.parent / "protected"


def _snapshot_best(src: Path, best_iter: int, best_val: float) -> None:
    """Hard-link the current best checkpoint into a directory the pruner never touches.

    Belt-and-braces on top of the pruner's own {best, latest} rule. That rule
    depends on the training log surviving, since the validation history is
    re-derived from it; if the log were lost or rotated, the pruner would fall
    back to "two most recent" and could delete the best.

    Hard links cost essentially no disk (same inodes) and, because rmtree only
    unlinks, the data survives even if the original checkpoint directory is
    deleted. Superseded snapshots are removed so at most one is kept.
    """
    dst = PROTECTED_DIR / f"iter_{best_iter:09d}_val{best_val:.4f}"
    if dst.exists():
        return
    try:
        PROTECTED_DIR.mkdir(parents=True, exist_ok=True)
        # `cp -al` = recursive hard-link copy. Reuses the shell tool rather than
        # reimplementing link-walking; it is the same operation used by hand for v1.
        r = subprocess.run(["cp", "-al", str(src), str(dst)], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"[snapshot] FAILED to hard-link {src.name}: {r.stderr.strip()[:160]}")
            return
        log(f"[snapshot] Protected {dst.name} (hard links, ~0 disk)")
        for old in PROTECTED_DIR.iterdir():
            if old.is_dir() and old.name != dst.name:
                shutil.rmtree(old, ignore_errors=True)
                log(f"[snapshot] Removed superseded {old.name}")
    except OSError as e:
        log(f"[snapshot] error: {e}")


def _prune_checkpoints(val_by_iter: dict[int, float]) -> None:
    """Keep exactly {best-val checkpoint, latest checkpoint}; delete the rest.

    Stateless and self-healing: the "best" is re-derived on every call from the
    validation results seen so far, restricted to checkpoints that still exist
    on disk. That means a best which was deleted by an earlier (buggy) prune
    simply stops being considered, instead of wedging the pruner forever.

    Safety rules:
      1. Never prune when <= 2 checkpoints exist (nothing to do).
      2. If no validation result matches any on-disk checkpoint yet, keep the
         two most recent rather than guessing.
    """
    if not CHECKPOINT_DIR.exists():
        return

    ckpt_dirs = sorted(
        [d for d in CHECKPOINT_DIR.iterdir() if d.is_dir() and re.fullmatch(r"iter_\d+", d.name)],
        key=lambda d: int(d.name[5:]),
    )
    if len(ckpt_dirs) <= 2:
        return

    on_disk = {int(d.name[5:]): d for d in ckpt_dirs}
    latest_iter = max(on_disk)

    # Best among checkpoints that actually exist AND have a validation score.
    scored = {i: val_by_iter[i] for i in on_disk if i in val_by_iter}
    if scored:
        best_iter = min(scored, key=lambda i: scored[i])
        keep = {best_iter, latest_iter}
        why = f"best=iter_{best_iter:09d} (val {scored[best_iter]:.4f}), latest=iter_{latest_iter:09d}"
        _snapshot_best(on_disk[best_iter], best_iter, scored[best_iter])
    else:
        second_latest = sorted(on_disk)[-2]
        keep = {latest_iter, second_latest}
        why = f"no val scores yet — keeping two most recent ({second_latest}, {latest_iter})"

    for i, d in sorted(on_disk.items()):
        if i in keep:
            continue
        try:
            shutil.rmtree(d)
            log(f"[ckpt_prune] Deleted {d.name}  ({why})")
        except OSError as e:
            log(f"[ckpt_prune] Failed to delete {d.name}: {e}")


def _monitor(train_proc: subprocess.Popen) -> None:
    """Epoch-aggregated early stopping + checkpoint pruning.

    Patience is measured as iteration distance from the best-val checkpoint,
    not a count of consecutive worse evaluations, and only an improvement
    greater than EARLY_STOP_EPSILON counts (noise filter).
    """
    best_loss = float("inf")
    best_iter = -1
    val_by_iter: dict[int, float] = {}
    seen_lines = 0

    while train_proc.poll() is None:
        time.sleep(30)
        if not LOG_FILE.exists():
            continue
        try:
            lines = LOG_FILE.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        new_checkpoint_saved = False
        for line in lines[seen_lines:]:
            if _CKPT_SAVE_RE.search(line):
                new_checkpoint_saved = True

            vm = _VAL_RE.search(line)
            if not vm:
                continue
            iteration, val_loss = int(vm.group(1)), float(vm.group(2))

            # Record EVERY validation result. Pruning needs these from
            # iteration 0 so it can always keep the true best checkpoint --
            # previously best-tracking started only after warmup, so during
            # warmup the pruner fell back to "two most recent" and deleted
            # genuinely-best checkpoints (iter_600 / val 2.131 was lost).
            val_by_iter[iteration] = val_loss
            improved = val_loss < best_loss - EARLY_STOP_EPSILON
            if improved:
                best_loss, best_iter = val_loss, iteration
                log(f"[monitor] New best val {val_loss:.4f} @ iter {iteration} "
                    f"(improved by >{EARLY_STOP_EPSILON})")

            # Warmup gates EARLY STOPPING only, never best-tracking/pruning.
            if iteration < WARMUP_ITERS:
                log(f"[monitor] val {val_loss:.4f} @ iter {iteration} (warmup — early stop "
                    f"inactive until iter {WARMUP_ITERS} / {EARLY_STOP_WARMUP_EPOCHS} epochs)")
                continue

            if not improved:
                iters_since = iteration - best_iter if best_iter >= 0 else 0
                epochs_since = iters_since / EPOCH_ITERS
                log(f"[monitor] val {val_loss:.4f} — no real improvement; {iters_since} iters "
                    f"({epochs_since:.1f}/{EARLY_STOP_PATIENCE_EPOCHS} epochs) since best "
                    f"{best_loss:.4f} @ iter {best_iter}")
                if best_iter >= 0 and iters_since >= PATIENCE_ITERS:
                    log(f"[monitor] Early stopping — patience exhausted after {epochs_since:.1f} "
                        f"epochs without >{EARLY_STOP_EPSILON} improvement. "
                        f"Best: iter_{best_iter:09d} (val {best_loss:.4f})")
                    try:
                        with LOG_FILE.open("a") as fh:
                            fh.write(f"\n{EARLY_STOP_MARKER} — {epochs_since:.1f} epochs "
                                     f"({iters_since} iters) without >{EARLY_STOP_EPSILON} "
                                     f"improvement. Best val {best_loss:.4f} at iter {best_iter}.\n")
                    except OSError:
                        pass
                    _prune_checkpoints(val_by_iter)  # prune BEFORE killing
                    try:
                        train_proc.terminate()
                    except ProcessLookupError:
                        pass
                    return

        seen_lines = len(lines)
        if new_checkpoint_saved:
            _prune_checkpoints(val_by_iter)


def is_training_done() -> bool:
    if not LOG_FILE.exists():
        return False
    text = LOG_FILE.read_text(errors="ignore")
    return any(k in text for k in [
        "Training finished", "Reached max_iter", "training done", "max_iter reached",
        EARLY_STOP_MARKER,
    ])


def launch_one_attempt(attempt: int) -> int:
    log(f"=== Attempt {attempt}/{MAX_RELAUNCHES} ===")
    with LOG_FILE.open("a") as fh:
        fh.write(f"\n\n{'='*60}\n[edge-supervisor] Attempt {attempt} started at "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n\n")

    with LOG_FILE.open("ab") as log_fh:
        train_proc = subprocess.Popen(
            TRAIN_CMD, env=TRAIN_ENV, cwd=REPO,
            stdout=log_fh, stderr=log_fh,
            start_new_session=True,  # survives supervisor death / SSH disconnect
        )
    log(f"training launched pid={train_proc.pid} on GPUs {CUDA_DEVICES}")

    time.sleep(15)  # let it write to the log before the watchdog starts reading

    with WATCHDOG_LOG.open("ab") as wlog_fh:
        watchdog_proc = subprocess.Popen(
            [
                f"{VENV}/bin/python3", "stall_watchdog.py",
                "--log-file", str(LOG_FILE),
                "--pid", str(train_proc.pid),
                "--stall-timeout-s", str(STALL_TIMEOUT_S),
                "--warmup-timeout-s", str(WARMUP_TIMEOUT_S),
                "--warmup-duration-s", str(WARMUP_DURATION_S),
            ],
            env=TRAIN_ENV, cwd=REPO,
            stdout=wlog_fh, stderr=wlog_fh,
            start_new_session=True,
        )
    log(f"stall watchdog launched pid={watchdog_proc.pid}")

    threading.Thread(target=_monitor, args=(train_proc,), name="monitor", daemon=True).start()
    log(f"monitor started — early stop: warmup={EARLY_STOP_WARMUP_EPOCHS} epochs/{WARMUP_ITERS} "
        f"iters, patience={EARLY_STOP_PATIENCE_EPOCHS} epochs/{PATIENCE_ITERS} iters, "
        f"eps={EARLY_STOP_EPSILON}; pruning: keep best+latest")

    exit_code = train_proc.wait()
    log(f"training exited with code={exit_code}")

    if watchdog_proc.poll() is None:
        try:
            watchdog_proc.terminate()
            watchdog_proc.wait(timeout=10)
        except Exception:
            pass
    return exit_code


def main() -> None:
    log("edge supervisor started")
    log(f"job={JOB_NAME}  epoch_iters={EPOCH_ITERS}  patience={EARLY_STOP_PATIENCE_EPOCHS} epochs")
    if is_training_done():
        log("training already done, exiting")
        return

    for attempt in range(1, MAX_RELAUNCHES + 1):
        exit_code = launch_one_attempt(attempt)
        if is_training_done():
            log("training finished — supervisor done")
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
