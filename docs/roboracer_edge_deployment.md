# Deploying the Cosmos3-Edge roboracer policy — step by step

Goal: serve the Cosmos3-Edge driving policy, confirm it produces sane output,
then drive the car with it.

This is the Edge counterpart to `roboracer_int4_deployment.md` (the Nano
pipeline). **It is simpler**: for the normal server-side deployment there is no
export and no quantization — the training checkpoint is served directly.

Everything lives in **`/scratch/tarunrav/cosmos-edge`** on **robolang**, which is
a separate checkout from `/scratch/tarunrav/cosmos-framework`. The Nano tree is
untouched and still serves the old model.

---

## 0. Which checkpoint, and why

| artifact | path | use |
|---|---|---|
| training checkpoint (DCP) | `outputs/.../action_policy_roboracer_edge_v3/protected/iter_000002400_val0.3159/` | **server** — serve this directly |
| HF export (bf16, 7.3 GB) | `outputs/edge_v3_iter2400_hf/` | intermediate for quantization |
| **INT4 (3.0 GB)** | `outputs/edge_v3_iter2400_int4/` | **on-car Jetson** |
| Wan VAE fp16 (1.4 GB) | `/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth` | both |

`iter_2400` is the best-validation checkpoint (val 0.3159). Training ran to 6805
but never beat it.

**Do not quantize for the server.** bf16 is the training precision, so it is the
accuracy ceiling, and NF4 adds dequantization work on every forward pass. INT4
exists only because the car has 8 GB.

Measured, not estimated:

| | weights | peak allocated | fits 8 GB Jetson? |
|---|---|---|---|
| Edge bf16 | 8.15 GB | **8.34 GB** | ✗ (just over) |
| **Edge INT4** | 3.56 GB | **3.73 GB** | ✓ (~4.3 GB spare) |
| *(Nano INT4, for contrast)* | 6.03 GB | 9.20 GB | ✗ |

---

## 1. Serve on robolang (normal path — no export, no quantization)

```bash
cd /scratch/tarunrav/cosmos-edge
V=/scratch/tarunrav/cosmos-framework/.venv
NV=$V/lib/python3.13/site-packages/nvidia

export PYTHONPATH=/scratch/tarunrav/cosmos-edge
export PATH=/home/tarunrav/.local/bin:$PATH          # uv, for the VAE fetch
export HF_HOME=/scratch/tarunrav/.cache/huggingface
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/nvshmem/lib:$NV/nvjpeg/lib:$NV/nvtiff/lib:$NV/cusparselt/lib
export CUDA_VISIBLE_DEVICES=0                        # pick a free GPU

CKPT=/scratch/tarunrav/cosmos-edge/outputs/cosmos3_action/action_sft/action_policy_roboracer_edge_v3/protected/iter_000002400_val0.3159/model

setsid nohup $V/bin/python3 -m cosmos_framework.scripts.action_policy_server_roboracer \
  --checkpoint-path $CKPT \
  --port 18767 \
  --eval-root /scratch/tarunrav/roboracer_lerobot_eval \
  --vae-path /scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth \
  < /dev/null > /tmp/edge_server.log 2>&1 &
disown

tail -f /tmp/edge_server.log     # wait for "ready" + "listening on ws://0.0.0.0:18767"
```

**Point `--checkpoint-path` at the `model/` subdirectory, not its parent.**
`CheckpointType.from_path()` looks for `*.distcp` at the top level; a training
checkpoint nests them under `model/`, so passing the parent fails with
"Unknown checkpoint type".

On startup the server logs the real memory numbers:

```
CUDA after load: allocated=8.15 GB peak_allocated=8.15 GB reserved=8.17 GB
```

`allocated` is what live tensors hold — the number that matters. `reserved` is
PyTorch's pool, which is what `nvidia-smi` shows. They are now nearly equal
because the server sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
calls `empty_cache()` after load. Without those, `reserved` on the Nano server
sat at ~37 GB against 9.20 GB actually in use.

---

## 2. Smoke-test (no car needed)

```python
import asyncio, msgpack, numpy as np, websockets

async def main():
    async with websockets.connect("ws://localhost:18767", max_size=None) as ws:
        await ws.recv()                                   # handshake
        img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        await ws.send(msgpack.packb({"image": img.tobytes(),
                                     "shape": list(img.shape),
                                     "direction": "loop_ccw"}))
        r = msgpack.unpackb(await ws.recv(), raw=False)
        print(r.get("error") or (r["curvature"][:5], r["velocity"][:5]))

asyncio.run(main())
```

Expect curvature roughly in ±4 (1/m), velocity roughly 0–1.5 (m/s), no NaN.
Measured latency is **~0.23 s** — well inside the 2.13 s chunk-buffer budget, and
~6x faster than the Nano INT4 server. Random noise will not produce a meaningful
driving decision; this only proves the pipeline serves.

---

## 3. Drive the car

The car side is **unchanged from the Nano runbook** — same client, same direction
tokens, same msgpack/websocket protocol. Follow
`roboracer_int4_deployment.md` sections 4–9, changing only the port to `18767`
and the server IP to robolang's (`10.0.0.212`).

Dry-run first (logs predictions, never publishes to the drive topic):

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.212 --server-port 18767
```

Direction tokens are identical (`loop_ccw`, `loop_cw`, `oval_ccw`, `circle_ccw`,
`rect_small_ccw`, `rect_small_cw`, `rect_med_ccw`, `square_ccw`, `pass_right`,
`pass_left`, `wait`).

---

## 4. On-car INT4 (only if running on the Jetson itself)

Already built at `outputs/edge_v3_iter2400_int4/`. To reproduce:

```bash
# DCP -> HF bf16
$V/bin/python3 -m cosmos_framework.scripts.export_model \
  --checkpoint-path <protected>/iter_000002400_val0.3159/model \
  --experiment action_policy_roboracer_edge --no-use-ema-weights \
  -o outputs/edge_v3_iter2400_hf

# HF bf16 -> INT4 (NF4, double-quant, bf16 compute)
$V/bin/python3 quantize_int4.py \
  --src outputs/edge_v3_iter2400_hf \
  --dst outputs/edge_v3_iter2400_int4
```

Both steps normalize two things that otherwise break single-process serving:

1. **`data_parallel_shard_degree`** is baked in as `8` from the 8-GPU training
   run; a single-process server cannot construct an 8-way-sharded model. Reset to `1`.
2. **`vae_path`** is written as a registry URI. Repoint it at the local
   **fp16** VAE — fp32 is 2.8 GB and eats the Jetson margin.

Serve the INT4 dir exactly like section 1, but pass the directory itself (it is
HF-format, so no `model/` suffix).

### Open risk on the Jetson

**bitsandbytes has not been validated on ARM64 / `sm_87`.** Stock PyPI aarch64
wheels target sm75/80/90. Either build from source on-device (~6 min,
community-validated) or use the Jetson AI Lab index
(`pypi.jetson-ai-lab.io/jp6/cu126/bitsandbytes/`) if the JetPack version matches.
Check `cat /etc/nv_tegra_release` first. This gates the whole on-car path and is
worth testing before anything else.

---

## 5. What this model does and doesn't do

Measured on held-out eval splits, 48 velocity-stratified samples, Spearman
correlation between predicted and ground-truth forward velocity:

| dataset | Edge (iter 2400) | Nano (iter 7200) |
|---|---|---|
| `wait` (social) | **+0.577** | +0.703 |
| orin10 (nav loop) | **−0.046** | +0.503 |

**Social behaviour works.** On `wait` the model tracks ground-truth velocity
closely and outputs near-zero when the car should be stopped — at roughly 80% of
Nano's fidelity with a 4x smaller text backbone.

**Nav-loop velocity does not.** On orin10 the model outputs a roughly constant
speed regardless of the frame, where Nano tracks it. Steering is comparable
(yaw Spearman +0.271 Edge vs +0.287 Nano). Practically: expect the car to hold a
roughly constant speed around the loop, and to slow/stop appropriately for people.

Training was stopped by the relaunch guard rather than early stopping, but
validation had not improved for 4,400 iterations, so more training of this recipe
is unlikely to help. The remaining hypothesis is capacity (Edge runs a 2B-active
text backbone vs Nano's 8B).

---

## If something breaks

1. **"Unknown checkpoint type"** — you passed the training checkpoint's parent
   directory. Append `/model`.
2. **GPU OOM** — check `nvidia-smi`. bf16 needs ~8.4 GB, INT4 ~3.8 GB.
3. **`ModuleNotFoundError: bitsandbytes`** — only needed for INT4:
   `UV_CACHE_DIR=/scratch/tarunrav/.cache/uv VIRTUAL_ENV=$V uv pip install bitsandbytes`
   (keep the cache off `$HOME`, which has a ~50 GB quota).
4. **`FileNotFoundError: 'uv'`** — put `/home/tarunrav/.local/bin` on `PATH`; the
   VAE fetch shells out to it.
5. **Traceback on launch** — capture the whole thing rather than guessing.
   Several bugs in this pipeline had misleading top-level messages.
