# Deploying the quantized roboracer policy — step by step

Goal: get the INT4-quantized, distilled 4B roboracer driving policy running as a
websocket inference server on your own machine, and confirm it produces sane
output before pointing it at the actual car.

## 0. Prerequisites

- An NVIDIA GPU with at least ~10GB free VRAM (model + VAE is ~5.7GB; leave
  headroom for activations).
- Linux, x86_64 (standard install path below — this is NOT the Jetson/ARM64
  case, so none of the `sm_87` bitsandbytes complications apply).
- GitHub access to `tarunravisankar/DriveDCCosmos` (ask Tarun to add you as a
  collaborator if the repo is private).

## 1. Get the code

```bash
git clone git@github.com:tarunravisankar/DriveDCCosmos.git cosmos-framework
cd cosmos-framework
git checkout roboracer-action-policy
```

All six deployment fixes below are already in this branch as of commit
`70ea524` ("Distill Cosmos3-Nano to 4B, quantize to INT4, and fix deployment
pipeline"). If something looks off, check `git log -1` shows this commit (or
later) is present:
- `cosmos_framework/scripts/export_model.py`
- `cosmos_framework/scripts/action_policy_server_roboracer.py`
- `cosmos_framework/utils/distributed.py`
- `cosmos_framework/model/vfm/omni_mot_model.py`
- `cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py`
- `cosmos_framework/model/vfm/vlm/qwen3_vl/qwen3_vl.py`

## 2. Set up the Python environment

```bash
uv venv --python 3.13
UV_CACHE_DIR=/tmp/uv_cache uv sync --group cu130-train --extra train
```

(If `uv` isn't installed: `pip install uv` first, or ask Tarun — this mirrors
the exact setup used on `robolidar`/`robolang` this session.)

Install bitsandbytes (standard x86_64 — no special build needed):
```bash
UV_CACHE_DIR=/tmp/uv_cache VIRTUAL_ENV=.venv .venv/bin/uv pip install bitsandbytes
```

## 3. Copy the model files from robolidar

Two files, ~5.7GB total:

```bash
mkdir -p ~/roboracer_deploy
scp -r robolidar.csres.utexas.edu:/scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_distill_v1/model_best_4200_int4 ~/roboracer_deploy/
scp robolidar.csres.utexas.edu:/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth ~/roboracer_deploy/
```

Then point the checkpoint's own config at wherever you put the VAE (it's
baked into `config.json` at export time):

```bash
sed -i "s|/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth|$HOME/roboracer_deploy/Wan2.2_VAE_fp16.pth|" \
  ~/roboracer_deploy/model_best_4200_int4/config.json
```

## 4. Sanity-check your GPU

```bash
.venv/bin/python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True` and your GPU's name. If `False`, stop here — nothing
below will work without a working CUDA install.

## 5. Launch the server

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -m cosmos_framework.scripts.action_policy_server_roboracer \
  --checkpoint-path ~/roboracer_deploy/model_best_4200_int4 \
  --port 18766 \
  --eval-root /path/to/roboracer_lerobot_eval \
  --vae-path ~/roboracer_deploy/Wan2.2_VAE_fp16.pth
```

Wait for these two lines in the output (model load takes ~30-60s):
```
[roboracer-policy-server] ready
[roboracer-policy-server] listening on ws://0.0.0.0:18766
```

If you see a Python traceback instead, stop and send it back — do not
proceed to the car with an unconfirmed server.

## 6. Test it with a fake request (no car needed)

In a second terminal, on the same machine:

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

Expected: `curvature` values roughly in `[-1, 1]` (1/m), `velocity` values
roughly in `[0, 2]` (m/s) — no NaN, no wildly huge numbers, no `SERVER ERROR`.
Random noise input won't produce a meaningful driving decision, but it should
produce *plausible-looking numbers*, not garbage or a crash.

## 7. Only after step 6 succeeds: point it at the car

Change `--host`/firewall as needed so the car can reach this machine's IP on
port 18766, then on the car side use the existing
`roboracer_chunk_buffered_client.py` in `--dry-run` mode first (logs
predictions, never drives), pointed at `ws://<this-machine-ip>:18766`. Only
move to `--live` after confirming dry-run output looks sane, with a human on
the joystick the entire time.

## If something breaks

The most likely failure points, in order of likelihood:
1. **GPU out of memory** — check `nvidia-smi`, make sure nothing else is
   using the GPU. ~5.7GB minimum, more headroom is safer.
2. **`ModuleNotFoundError`** — dependency install didn't fully complete;
   rerun step 2.
3. **CUDA/driver mismatch** — `torch.cuda.is_available()` returning `False`
   despite having a GPU usually means a driver/CUDA version mismatch; check
   `nvidia-smi` runs cleanly first.
4. Anything else — capture the full traceback and send it back rather than
   trying random fixes; several of the bugs fixed this session had
   misleading top-level error messages that pointed to the wrong root cause.
