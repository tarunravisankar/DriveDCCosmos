# Deploying the quantized roboracer policy — step by step

Goal: get the INT4-quantized, distilled 4B roboracer driving policy serving
on robolidar, confirm it produces sane output, then drive the car with it
and fold the new data back into training.

**You need nothing except your own robolidar login.** No git clone, no venv,
no `uv sync`, no bitsandbytes install, no re-running the export/quantization
pipeline. The code, the Python environment, and the already-quantized model
are all sitting on robolidar's shared `/scratch` under Tarun's account with
world-readable permissions (verified `644`/`755` throughout every path
involved) — any account on the machine can read and execute all of it
directly. Step 2's launch command is the entire setup.

## 0. Prerequisites

- Your own login on `robolidar.csres.utexas.edu`.
- The car reachable over the lab's WireGuard VPN (see step 4).

That's it. (A separate git checkout is only useful if you want to *edit*
code — see the very end of this doc — not to run what's already there.)

## 1. Model files — no transfer needed

Both already have world-readable permissions on robolidar's shared
`/scratch` (verified `644`/`755` throughout the path, any account on the
machine can read them):

```
/scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_distill_v1/model_best_4200_int4/
/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth
```

Just reference these paths directly in the launch command below. If you'd
rather have your own copy (optional — it's a local copy on the same
`/scratch` volume, not a network transfer, so it's fast regardless of size):

```bash
mkdir -p ~/roboracer_deploy
cp -r /scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_distill_v1/model_best_4200_int4 ~/roboracer_deploy/
cp /scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth ~/roboracer_deploy/
sed -i "s|/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth|$HOME/roboracer_deploy/Wan2.2_VAE_fp16.pth|" \
  ~/roboracer_deploy/model_best_4200_int4/config.json
```

## 2. Launch the inference server (on robolidar)

Check GPUs are free first:
```bash
nvidia-smi
```

Then, using Tarun's existing venv directly — no setup, no `uv sync`, nothing
to install:

```bash
cd /scratch/tarunrav/cosmos-framework

CUDA_VISIBLE_DEVICES=<free GPU index> \
nohup setsid /scratch/tarunrav/cosmos-framework/.venv/bin/python3 \
  -m cosmos_framework.scripts.action_policy_server_roboracer \
  --checkpoint-path /scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_distill_v1/model_best_4200_int4 \
  --vae-path /scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth \
  --eval-root /scratch/tarunrav/roboracer_lerobot_eval \
  --port 18766 \
  < /dev/null > /tmp/roboracer_int4_server.log 2>&1 &
disown

tail -f /tmp/roboracer_int4_server.log   # watch for "ready" + "listening on ws://0.0.0.0:18766"
```

This is simpler than the old teacher-model launch command — no
`BASE_CHECKPOINT_PATH`, `ROBORACER_TRAIN_ROOT`, or `LD_LIBRARY_PATH`
overrides needed. Those were specific to the old DCP-checkpoint loading
path; the quantized checkpoint is HF-format and auto-detected, bypassing
that path entirely (fixed this session — see `CLAUDE.md`).

If you see a Python traceback instead of "ready", stop — do not proceed to
the car with an unconfirmed server. Capture the full traceback rather than
guessing; several bugs found getting this checkpoint to serve had
misleading top-level error messages (see `CLAUDE.md`'s bug list).

## 3. Smoke-test with a fake request (no car needed)

From any machine that can reach robolidar's port 18766 (or the same
session, connecting to `localhost`):

```python
import asyncio, msgpack, numpy as np, websockets

async def main():
    async with websockets.connect("ws://localhost:18766", max_size=None) as ws:
        print("handshake:", msgpack.unpackb(await ws.recv(), raw=False))
        image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        req = {"image": image.tobytes(), "shape": list(image.shape), "direction": "loop_ccw"}
        await ws.send(msgpack.packb(req))
        result = msgpack.unpackb(await ws.recv(), raw=False)
        if "error" in result:
            print("SERVER ERROR:", result["error"])
        else:
            print("curvature[:5]:", result["curvature"][:5])
            print("velocity[:5]:", result["velocity"][:5])
            print("SUCCESS")

asyncio.run(main())
```

Expected: `curvature` roughly in `[-1, 1]` (1/m), `velocity` roughly in
`[0, 2]` (m/s), no NaN, no `SERVER ERROR`. Random noise input won't produce
a meaningful driving decision, but the numbers should look plausible, not
garbage.

## 4. Get on the car

```bash
ssh into car
ping 10.0.0.211     # confirm the car can reach robolidar over the VPN
docker ps           # confirm orin_roboracer is up; if not, bring up at least the camera node
```

`10.0.0.211` is robolidar's WireGuard VPN address (same mesh as robolang's
`10.0.0.212` used previously — just a different host on it).

## 5. Make sure the client + deps are ready in the container

**Don't assume these files are already on the car** — if you're using a
different physical car than the one this was developed on, `/home/orin/`
won't have them yet. Their verified source location is on **robolang**
(`/scratch/tarunrav/roboracer_chunk_buffered_client.py` and
`/scratch/tarunrav/social_baseline.py` — confirmed present there; notably
**not** present on robolidar). Copy them onto the car first, then into the
container:

```bash
# from robolang (or anywhere that can reach both robolang and the car)
scp tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/roboracer_chunk_buffered_client.py \
    tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/social_baseline.py \
    orin@<car-ip-or-vpn-address>:/home/orin/

# on the car
docker cp /home/orin/roboracer_chunk_buffered_client.py orin_roboracer:/tmp/roboracer_chunk_buffered_client.py
docker cp /home/orin/social_baseline.py orin_roboracer:/tmp/social_baseline.py
```

(If `/home/orin/` already has current copies on the car you're using, this
step is a harmless no-op — just skip straight to the `docker cp` lines.)

## 5.5 Start navstack
```bash
cd ~/roboracer_ws/tmux/navstack/
tmuxinator
```
```bash
pip install websockets # if needed
```
## 6. Dry-run sanity check (inside the container shell)

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.211 --server-port 18766
```

Watch the logged curvature/velocity for a bit — confirm it looks sane (no
garbage, no constant clamping). Ctrl+C to stop.

## 7. Choosing a navigation or social behavior

The server conditions on a text caption, selected via the client's
`--direction` flag (a short token, mapped server-side to the exact trained
caption string — typo-safe, since a free-form caption that doesn't
character-for-character match a training caption falls outside the trained
distribution). Default is `loop_ccw` if you omit the flag.

| Token | Behavior |
|---|---|
| `loop_ccw` | Counter-clockwise around a large indoor loop |
| `loop_cw` | Clockwise around a large indoor loop |
| `oval_ccw` | Counter-clockwise around a large oval track |
| `circle_ccw` | Counter-clockwise around a circular track |
| `rect_small_ccw` | Counter-clockwise around a small rectangular track |
| `rect_small_cw` | Clockwise around a small rectangular track |
| `rect_med_ccw` | Counter-clockwise around a medium rectangular track |
| `square_ccw` | Counter-clockwise around a square track |
| `pass_right` | Pass a person on the right |
| `pass_left` | Pass a person on the left |
| `wait` | Wait for a person to pass |

Pass it directly on any client invocation, e.g. to test the `pass_right`
social behavior in dry-run:

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.211 --server-port 18766 --direction pass_right
```

or live, recording the session, driving a clockwise loop:

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.211 --server-port 18766 \
  --live --direction loop_cw --record-bag roboracer_$(date +%Y%m%d_%H%M%S)_dagger
```

**The script that randomly cycles between the three social behaviors** (what
you're likely thinking of) is `social_baseline.py` — it holds each of
`pass_right` / `pass_left` / `wait` for ~5 seconds, cycling automatically, so
you can observe all three behaviors in one run without manually restarting
the client for each token:

```bash
python3 /tmp/social_baseline.py --server-ip 10.0.0.211 --server-port 18766 --live
```
(drop `--live` to dry-run it first, same as the regular client.)

There's also a `--caption` flag on the regular client for free-form text
outside the trained-token vocabulary above (e.g. `--caption "Drive the
roboracer vehicle to the elevator at the end of the hall."`) — this is for
zero-shot generalization testing only, not a normal deployment path, since
it isn't guaranteed to match anything the model was actually trained on.

## 8. Drive live, recording the session

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.211 --server-port 18766 \
  --live --record-bag roboracer_$(date +%Y%m%d_%H%M%S)_dagger
```

Hold R1 to enable autonomous mode; nudge the joystick whenever you want to
correct it (instantly overrides per-axis). When done, Ctrl+C — this stops
driving and finalizes the bag cleanly.

## 9. Get the bag off the car and into the conversion pipeline

The bag lands in the current working directory of step 8, named
`roboracer_<timestamp>_dagger/`. Copy it wherever the other training bags
live (same convention as the existing 31 train bags, so
`convert_roboracer_to_lerobot.py --bags-dir` finds it alongside them):

```bash
scp -r roboracer_20260628_120000_dagger <destination-matching-existing-bags-location>
```

## 10. Convert and add to the train split

```bash
cd /scratch/tarunrav/cosmos-framework
python convert_roboracer_to_lerobot.py \
  --bags-dir <dir containing roboracer_20260628_120000_dagger> \
  --output-dir /scratch/tarunrav/roboracer_lerobot_train \
  --bag-split-json roboracer_bag_split.json \
  --split train
```

Then add `"roboracer_20260628_120000_dagger"` to the `"train"` list in
`roboracer_bag_split.json`.

## 11. Resume fine-tuning on the updated dataset

Same launch command as any other resume — `ROBORACER_TRAIN_ROOT` already
points at the directory the new episode was just added to, so no extra flag
is needed. Use the usual env vars + torchrun command for whichever
experiment (nano SFT or distillation) you're continuing.

## If something breaks

1. **GPU out of memory** — check `nvidia-smi`, confirm nothing else is using
   the GPU. ~5.7GB minimum for the model+VAE, more headroom is safer.
2. **Car can't ping robolidar** — VPN routing issue, not a code issue; check
   the WireGuard config before touching anything server-side.
3. **Traceback on server launch** — capture the full traceback, don't guess.
   Several bugs fixed this session had misleading top-level errors that
   pointed to the wrong root cause (see `CLAUDE.md`).
4. **Dry-run predictions look like garbage/constant clamping** — stop before
   going live; this means the model or preprocessing has a real problem,
   not something to push through.

## If you want to edit code (not just run it)

Everything above runs directly off Tarun's existing checkout and venv on
robolidar — you're not editing anything there. If you want your own
checkout to make changes:

```bash
git clone git@github.com:tarunravisankar/DriveDCCosmos.git
cd DriveDCCosmos && git checkout roboracer-action-policy
uv venv --python 3.13
UV_CACHE_DIR=/tmp/uv_cache uv sync --group cu130-train --extra train
UV_CACHE_DIR=/tmp/uv_cache VIRTUAL_ENV=.venv .venv/bin/uv pip install bitsandbytes
```

Then point `--checkpoint-path`/`--vae-path` at Tarun's existing model files
(no need to copy those either, per step 1) while using your own venv to run
the server.
