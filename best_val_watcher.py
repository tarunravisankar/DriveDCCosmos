#!/usr/bin/env python3
"""best_val_watcher.py

Watches the training log for validation loss lines. Whenever a new
best validation loss is seen, copies the corresponding checkpoint to
a stable "best_val" directory before the trainer's keep-latest-only
policy deletes it.

The trainer already preserves the most recent checkpoint; this script
preserves the best-val checkpoint alongside it.

Usage (run detached, alongside training):
    setsid .venv/bin/python3 best_val_watcher.py \
        < /dev/null >> /tmp/best_val_watcher.log 2>&1 &
"""

import re
import shutil
import subprocess
import time
from pathlib import Path

LOG_FILE = Path("/tmp/roboracer_retrain_v8_combined_robolang.log")
CHECKPOINT_DIR = Path(
    "/scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft"
    "/action_policy_roboracer_repro_v8/checkpoints"
)
BEST_VAL_DIR = CHECKPOINT_DIR.parent / "checkpoints_best_val"

VAL_RE = re.compile(r"Validation loss \(iteration (\d+)\): ([0-9.]+)")


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[best-val-watcher {ts()}] {msg}", flush=True)


def copy_checkpoint(src: Path, dst: Path) -> bool:
    """Copy checkpoint dir using cp -r (faster than shutil for large dirs)."""
    tmp = dst.parent / (dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    result = subprocess.run(["cp", "-r", str(src), str(tmp)])
    if result.returncode != 0:
        log(f"ERROR: cp failed (exit {result.returncode})")
        return False
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return True


def main():
    BEST_VAL_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_iter = None
    seen_iters: set[int] = set()
    pos = 0

    # Seed best_loss from any prior best already saved
    best_ptr = BEST_VAL_DIR / "best_checkpoint.txt"
    if best_ptr.exists():
        info = best_ptr.read_text().strip().splitlines()
        if len(info) >= 2:
            try:
                best_iter = int(info[0])
                best_loss = float(info[1])
                log(f"resuming: prior best iter={best_iter} loss={best_loss:.6f}")
            except ValueError:
                pass

    log(f"watching {LOG_FILE} | checkpoints at {CHECKPOINT_DIR} | best-val at {BEST_VAL_DIR}")

    while True:
        try:
            with LOG_FILE.open("r", errors="ignore") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()

            for m in VAL_RE.finditer(chunk):
                iter_num = int(m.group(1))
                val_loss = float(m.group(2))

                if iter_num in seen_iters:
                    continue
                seen_iters.add(iter_num)

                marker = " ← new best!" if val_loss < best_loss else ""
                log(f"iter {iter_num}: val_loss={val_loss:.6f} (best={best_loss:.6f}){marker}")

                if val_loss < best_loss:
                    # Always update the tracked best, even if we can't copy.
                    # This seeds the baseline from history so we only copy
                    # future checkpoints that beat the true historical best.
                    best_loss = val_loss
                    best_iter = iter_num

                    src = CHECKPOINT_DIR / f"iter_{iter_num:09d}"
                    if not src.exists():
                        log(f"historical best iter={iter_num} loss={val_loss:.6f} — checkpoint already gone, "
                            f"tracking as baseline only (will copy next time we beat this)")
                        best_ptr.write_text(f"{best_iter}\n{best_loss}\n")
                        continue

                    dst = BEST_VAL_DIR / f"iter_{iter_num:09d}"
                    log(f"copying {src.name} -> {BEST_VAL_DIR.name}/{dst.name} ...")
                    t0 = time.time()
                    ok = copy_checkpoint(src, dst)
                    if ok:
                        best_ptr.write_text(f"{best_iter}\n{best_loss}\n")
                        log(f"copy done in {time.time()-t0:.0f}s. NEW BEST: iter={best_iter} loss={best_loss:.6f}")
                    # Remove stale best dirs (keep only current best)
                    for old in BEST_VAL_DIR.iterdir():
                        if old.is_dir() and old != dst:
                            log(f"removing old best checkpoint {old.name}")
                            shutil.rmtree(old)

        except Exception as exc:
            log(f"error: {exc}")

        time.sleep(15)


if __name__ == "__main__":
    main()
