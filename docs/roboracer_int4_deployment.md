# Deploying the quantized roboracer policy — step by step

Goal: get the INT4-quantized, distilled 4B roboracer driving policy serving
on robolidar, confirm it produces sane output, then drive the car with it
and fold the new data back into training.

You (the professor, or anyone with your own robolidar login) do **not** need
to copy any multi-gigabyte model files anywhere — everything below runs
directly off Tarun's already-readable files on robolidar's shared `/scratch`.

## 0. Prerequisites

- Your own login on `robolidar.csres.utexas.edu`.
- The car reachable over the lab's WireGuard VPN (see step 6).
- Repo access to `git@github.com:tarunravisankar/DriveDCCosmos.git`
  (`roboracer-action-policy` branch) if you need the client script or want
  your own checkout — not required just to launch the server, since the
  code already lives on robolidar under Tarun's account.

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

Then, using Tarun's existing venv (or your own, set up per
`git checkout roboracer-action-policy` + `uv sync --group cu130-train --extra train`):

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

```bash
docker cp /home/orin/roboracer_chunk_buffered_client.py orin_roboracer:/tmp/roboracer_chunk_buffered_client.py
docker exec orin_roboracer bash -lc 'python3 -c "import websockets" 2>/dev/null || python3 -m pip install --user --quiet websockets'
docker exec -it orin_roboracer bash -l
```

## 6. Dry-run sanity check (inside the container shell)

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.211 --server-port 18766
```

Watch the logged curvature/velocity for a bit — confirm it looks sane (no
garbage, no constant clamping). Ctrl+C to stop.

## 7. Drive live, recording the session

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.211 --server-port 18766 \
  --live --record-bag roboracer_$(date +%Y%m%d_%H%M%S)_dagger
```

Hold R1 to enable autonomous mode; nudge the joystick whenever you want to
correct it (instantly overrides per-axis). When done, Ctrl+C — this stops
driving and finalizes the bag cleanly.

## 8. Get the bag off the car and into the conversion pipeline

The bag lands in the current working directory of step 7, named
`roboracer_<timestamp>_dagger/`. Copy it wherever the other training bags
live (same convention as the existing 31 train bags, so
`convert_roboracer_to_lerobot.py --bags-dir` finds it alongside them):

```bash
scp -r roboracer_20260628_120000_dagger <destination-matching-existing-bags-location>
```

## 9. Convert and add to the train split

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

## 10. Resume fine-tuning on the updated dataset

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
