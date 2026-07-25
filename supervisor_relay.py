#!/usr/bin/env python3
"""supervisor_relay.py

Waits for the currently-running train_supervisor.py (PID given on the
command line) to exit, then checks if training finished. If not, it
relaunches a fresh train_supervisor.py so training continues from the
latest checkpoint automatically.

Usage:
    setsid .venv/bin/python3 supervisor_relay.py <old_supervisor_pid> \
        < /dev/null >> /tmp/train_supervisor.log 2>&1 &
"""

import os
import subprocess
import sys
import time
from pathlib import Path

DONE_KEYWORDS = ["Training finished", "Reached max_iter", "training done", "max_iter reached"]
LOG_FILE = Path("/tmp/roboracer_retrain_v8_combined_robolang.log")
SUPERVISOR_LOG = Path("/tmp/train_supervisor.log")
CWD = "/scratch/tarunrav/cosmos-framework"
VENV_PYTHON = f"{CWD}/.venv/bin/python3"


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    if len(sys.argv) < 2:
        print(f"[relay {ts()}] usage: supervisor_relay.py <old_supervisor_pid>", flush=True)
        sys.exit(1)

    old_pid = int(sys.argv[1])
    print(f"[relay {ts()}] watching supervisor PID {old_pid} ...", flush=True)

    while True:
        try:
            os.kill(old_pid, 0)
            time.sleep(30)
        except ProcessLookupError:
            break

    print(f"[relay {ts()}] supervisor PID {old_pid} has exited", flush=True)

    if LOG_FILE.exists():
        text = LOG_FILE.read_text(errors="ignore")
        if any(k in text for k in DONE_KEYWORDS):
            print(f"[relay {ts()}] training finished naturally — relay done", flush=True)
            return

    print(f"[relay {ts()}] training not finished; launching new supervisor ...", flush=True)
    with SUPERVISOR_LOG.open("ab") as fh:
        proc = subprocess.Popen(
            [VENV_PYTHON, "train_supervisor.py"],
            cwd=CWD,
            stdin=open("/dev/null"),
            stdout=fh,
            stderr=fh,
            start_new_session=True,
        )
    print(f"[relay {ts()}] new supervisor launched PID={proc.pid}", flush=True)


if __name__ == "__main__":
    main()
