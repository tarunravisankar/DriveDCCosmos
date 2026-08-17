# Deploying the Cosmos3-Edge roboracer policy — step by step

Goal: serve the Cosmos3-Edge driving policy, confirm it produces sane output,
then drive the car with it.

This is the Edge counterpart to `roboracer_int4_deployment.md` (the Nano
pipeline). **It is simpler**: for the normal server-side deployment there is no
export and no quantization — the training checkpoint is served directly.

Everything lives in **`/scratch/tarunrav/cosmos-edge`** on **robolang**, which is
a separate checkout from `/scratch/tarunrav/cosmos-framework`. The Nano tree is
untouched and still serves the old model.

> **Deploying to a car?** See **[`car_deploy/README.md`](../car_deploy/README.md)**
> — a patch file plus the client, with the four fixes that stopped the car
> driving into a wall. **orin10 already has all of them applied.**
>
> **On-car inference now works** — two bugs (uninitialised meta-init buffers, and
> a 1-frame video window) are fixed; see §7. The tethered path (§4) is still the
> better *control* loop: ~0.23 s per call versus ~1.55 s on-board.

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

## 3. The websocket interface

The server speaks **msgpack over websockets** — no ROS2, no `openpi`, no other
dependency. Anything that can open a websocket can drive it, which is why the
smoke test in section 2 works from plain Python.

### Protocol

**On connect** the server sends an empty msgpack dict as a metadata handshake.
A client must read and discard it before sending anything.

**Request** — one msgpack dict per inference:

| key | type | notes |
|---|---|---|
| `image` | raw bytes | `H x W x 3`, uint8, **RGB**, row-major |
| `shape` | `[H, W, 3]` | needed to reshape the flat buffer |
| `direction` | string | optional, defaults to `loop_ccw` |
| `caption` | string | optional; raw text, **overrides** `direction` |

**Response** — one msgpack dict:

| key | type | notes |
|---|---|---|
| `curvature` | 32 floats | 1/m, **left-positive** |
| `velocity` | 32 floats | m/s, forward-positive |
| `raw_action` | 32 x 9 floats | the un-decoded action, for debugging |

Units match the car's `/ackermann_curvature_drive` message exactly, so a
response can be forwarded to that topic as-is.

On failure the server returns `{"error": "<message>"}` instead — clients should
check for that key before reading `curvature`.

### Why 32 steps

Each response is a **chunk** of 32 future timesteps at `conditioning_fps=15`,
i.e. **2.13 s** of driving. That is the latency budget: the client publishes one
step every 1/15 s and must receive a fresh chunk before the buffer drains.

Measured Edge latency is **~0.23 s** per inference, roughly 9x inside that
budget (the Nano INT4 server was ~1.5 s, which is what made the receding-horizon
buffer necessary in the first place).

### Direction tokens

`direction` is a short token mapped **server-side** to the exact caption string
seen in training. This is deliberately typo-safe: a free-form `caption` that
doesn't match character-for-character falls outside the trained distribution and
the model's behaviour is undefined.

| Token | Behaviour |
|---|---|
| `loop_ccw` | Counter-clockwise, large indoor loop *(default)* |
| `loop_cw` | Clockwise, large indoor loop |
| `oval_ccw` | Counter-clockwise, large oval track |
| `circle_ccw` | Counter-clockwise, circular track |
| `rect_small_ccw` / `rect_small_cw` | Small rectangular track |
| `rect_med_ccw` | Medium rectangular track |
| `square_ccw` | Square track |
| `pass_right` | Pass a person on the right |
| `pass_left` | Pass a person on the left |
| `wait` | Wait for a person to pass |

Use `--caption` only for deliberate zero-shot generalization tests, never for
normal driving.

### Checking a running server

```bash
ss -tlnp | grep 18767                  # is it listening?
grep -E "ready|listening|CUDA" /tmp/edge_server.log
```

---

## 4. Driving the car (server on robolang, car as client)

This is the normal deployment. The model runs on robolang's GPU; the car is a
thin websocket client over the lab WireGuard VPN. **Nothing runs on the Jetson.**

### 4.1 Reach the car

```bash
ssh orin@<car-ip>
ping 10.0.0.212          # robolang's WireGuard address — must reply
docker ps                # orin_roboracer must be up (camera + VESC stack)
```

`10.0.0.212` is robolang. (The Nano runbook used `10.0.0.211`, robolidar — same
VPN mesh, different host. Use the one your server is actually on.)

### 4.2 Get the client into the container

**orin10 is already up to date** — it has the current client with all fixes, at
`/home/orin/roboracer_ws/roboracer_chunk_buffered_client.py`. Skip to 4.3.

For a *different* car, don't assume the scripts are there. Source of truth is
robolang (or `car_deploy/` on this branch, which also carries the framework
patch):

```bash
# from anywhere that can reach both robolang and the car
scp tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/roboracer_chunk_buffered_client.py \
    tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/social_baseline.py \
    orin@<car-ip>:/home/orin/

# on the car
docker cp /home/orin/roboracer_chunk_buffered_client.py orin_roboracer:/tmp/
docker cp /home/orin/social_baseline.py               orin_roboracer:/tmp/
docker exec orin_roboracer bash -lc 'python3 -c "import websockets, msgpack" || pip install websockets msgpack'
```

### 4.3 Start the navstack

```bash
cd ~/roboracer_ws/tmux/navstack/ && tmuxinator
```

This brings up the camera, VESC, joystick and `/odom`. Starting `/odom` is only
safe because `--subgoal-lookahead-s` now defaults to **0**: the training data has
no goal dots (`convert_all_datasets.sh` sets `LOOKAHEAD=0`), but the client used
to default to 5.0 and would begin drawing dots the model has never seen the
moment odometry appeared.

> **Do not use the `cosmos` tmuxinator profile for tethered runs.** It starts an
> *on-car* server, points the client at `127.0.0.1`, and — unlike the standalone
> client — **defaults to LIVE**. `LIVE=0` is the opt-out there. Run the client by
> hand as in 4.4 instead.

### 4.4 Dry-run first — always

`--dry-run` is the default. The client subscribes to the camera, calls the
server at 15 Hz, and **logs** predictions without ever constructing an
`AckermannCurvatureDriveMsg`.

```bash
cd ~/roboracer_ws && ./container shell          # or: docker exec -it orin_roboracer bash -l
python3 /home/orin/roboracer_ws/roboracer_chunk_buffered_client.py \
  --server-ip 10.0.0.212 --server-port 18767 --direction pass_right
```

Watch the logged curvature/velocity against what the car is actually looking at.
Do not proceed if the numbers look constant, clamped, or NaN.

**Walk in front of the camera and confirm the numbers move.** A stationary car
looking at a static scene produces near-constant output even from a healthy
model, so "constant" is only meaningful once the scene changes.

Expect behaviour to differ by token, and only on the axis that token cares about
(numbers in §7): `pass_right` steers strongly, `wait` slows but **under-brakes**
(~29% of the demonstrated slowdown — use a box, not a person), `pass_left` is
weak, and the navigation loops hold a roughly constant speed.

### 4.5 Go live

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.212 --server-port 18767 \
  --live --direction loop_ccw
```

Three independent safety layers, all already in place:

1. `--live` must be passed explicitly; dry-run is the default.
2. The car's `vesc_driver` **ignores the topic entirely** unless a human has
   enabled autonomous mode on the joystick, and any joystick input past a
   deadzone instantly overrides per-axis.
3. `--max-velocity` (default 1.0 m/s) and `--max-curvature` (default **1.3** 1/m)
   hard-clamp every published command. 1.3 is the car's *physical* steering
   limit: `car.lua` sets `max_steering_angle=0.4030`, so
   `tan(0.4030)/wheelbase(0.32) = 1.33` 1/m. The previous default of 3.0 was
   above anything the servo could execute, so the clamp bounded nothing — it
   just handed `vesc_driver` a value that saturated the steering at full lock.

There is also a **fail-safe STOP**: if the chunk buffer empties or goes older
than `--max-buffer-age-s` (default 3 s) — server unreachable, one inference
stalls — the client publishes velocity 0 / curvature 0 rather than continuing on
stale predictions.

### 4.6 Recording a DAgger episode

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 10.0.0.212 --server-port 18767 \
  --live --direction loop_cw --record-bag roboracer_$(date +%Y%m%d_%H%M%S)_dagger
```

Records exactly the two topics `convert_roboracer_to_lerobot.py` needs
(`/camera_0/image_raw/compressed`, `/odom`). Ctrl+C stops driving and finalizes
the bag cleanly via SIGINT.

**Note:** `social_baseline.py` (auto-cycles `pass_right`/`pass_left`/`wait` every
~5 s) parses `--record-bag` but never acts on it. For recording, use the
chunk-buffered client with a fixed `--direction`.

---

## 5. Running the model ON the car (Jetson, INT4)

This is the research goal — no server, no VPN, no network in the loop. It is
also **the least-tested path in this document**: the INT4 checkpoint is built and
verified on x86 (3.73 GB peak, 0.23 s), but it has **never been run on Jetson
hardware**. Treat this section as a plan, not a proven recipe.

### 5.1 Clear the blocker first

**Check bitsandbytes before anything else** — it gates everything here.

```bash
cat /etc/nv_tegra_release          # which JetPack?
python3 -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

Stock PyPI aarch64 wheels target sm75/80/90 and **do not** support Jetson Orin's
`sm_87`. Two options:

- build from source on-device targeting `sm_87` (~6 min, community-validated), or
- use the Jetson AI Lab index if the CUDA version matches:
  `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/ bitsandbytes`

If neither works, on-car INT4 is blocked and the section-4 client mode is the
fallback.

### 5.2 Memory budget

| component | size |
|---|---|
| INT4 weights | ~2.2 GB |
| Wan VAE **fp16** | 1.4 GB |
| activations | ~0.2 GB |
| **peak (measured on x86)** | **3.73 GB** |

Against 8 GB that leaves ~4.3 GB for the OS, camera stack, and ROS2. **Use the
fp16 VAE** — fp32 is 2.8 GB and eats most of the margin.

### 5.3 Copy the artifacts

```bash
# ~3 GB model + 1.4 GB VAE
scp -r tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/cosmos-edge/outputs/edge_v3_iter2400_int4 \
       orin@<car-ip>:/home/orin/
scp tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth \
    orin@<car-ip>:/home/orin/

# repoint the checkpoint's baked-in VAE path at wherever it landed
python3 - <<'PY'
import json; p="/home/orin/edge_v3_iter2400_int4/config.json"; c=json.load(open(p))
def fix(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=="vae_path": o[k]="/home/orin/Wan2.2_VAE_fp16.pth"
            else: fix(v)
    elif isinstance(o,list):
        for v in o: fix(v)
fix(c); json.dump(c,open(p,"w"),indent=2)
PY
```

### 5.4 The framework also has to be on the car

The server imports `cosmos_framework`, so the Jetson needs the package and its
dependencies (torch, transformers, safetensors, bitsandbytes, msgpack,
websockets). This is the substantial part — an aarch64 torch build matching the
JetPack CUDA version, not a `pip install` of the x86 wheel set.

```bash
scp -r tarunrav@robolang.csres.utexas.edu:/scratch/tarunrav/cosmos-edge/cosmos_framework \
       orin@<car-ip>:/home/orin/cosmos-edge/
```

Budget real time for this. If it proves impractical, section 4 client mode is a
legitimate deployment — the car drives identically, just with the GPU elsewhere.

### 5.5 Run it

On the car:

```bash
export PYTHONPATH=/home/orin/cosmos-edge
python3 -m cosmos_framework.scripts.action_policy_server_roboracer \
  --checkpoint-path /home/orin/edge_v3_iter2400_int4 \
  --port 18767 --host 127.0.0.1 \
  --eval-root /home/orin/roboracer_lerobot_eval \
  --vae-path /home/orin/Wan2.2_VAE_fp16.pth
```

Bind to `127.0.0.1`, not `0.0.0.0` — nothing off-board needs to reach it.

Confirm the memory line matches expectations before going further:

```
CUDA after load: allocated=~3.6 GB ...
```

Then point the client at localhost — no VPN, no network hop:

```bash
python3 /tmp/roboracer_chunk_buffered_client.py --server-ip 127.0.0.1 --server-port 18767
```

Dry-run first, exactly as in 4.4.

**`--eval-root` still needs a LeRobot dataset on the car.** The server builds one
`RoboracerDataset` purely to read fixed config (fps, domain_id, viewpoint,
action normalization) consistently with training. Copy any one eval split across
— `roboracer_lerobot_eval` is the smallest that works.

### 5.6 Expect slower inference

0.23 s was measured on an H100. An Orin Nano is far smaller, and NF4
dequantization runs on every forward pass. If latency exceeds **2.13 s** the
chunk buffer drains and the fail-safe STOP fires — you would see the car stutter
and halt rather than drive badly. Measure it in dry-run before going live; if it
is marginal, reducing `--num-steps` on the sampler is the first lever.

---

## 6. Rebuilding the INT4 checkpoint

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

### Open risk on the Jetson (see also 5.1)

**bitsandbytes has not been validated on ARM64 / `sm_87`.** Stock PyPI aarch64
wheels target sm75/80/90. Either build from source on-device (~6 min,
community-validated) or use the Jetson AI Lab index
(`pypi.jetson-ai-lab.io/jp6/cu126/bitsandbytes/`) if the JetPack version matches.
Check `cat /etc/nv_tegra_release` first. This gates the whole on-car path and is
worth testing before anything else.

---

---

## 7. What this model does and doesn't do

All four held-out eval splits, 48 velocity-stratified samples each, Spearman
correlation against ground truth. Run with
`velocity_eval.py --dataset-root <a,b,c> --label <bf16|int4>`. At n=48 the
p=0.05 threshold is **ρ ≈ 0.285**, so anything below that is not distinguishable
from zero.

| split | velocity ρ (bf16) | velocity ρ (INT4) | yaw ρ (bf16) | yaw ρ (INT4) |
|---|---|---|---|---|
| `wait` | **+0.553** | **+0.502** | +0.166 | +0.107 |
| `pass_right` | +0.056 | +0.152 | **+0.715** | **+0.723** |
| `pass_left` | +0.097 | +0.155 | +0.322 | +0.282 |
| orin10 (nav) | −0.046 | −0.056 | +0.271 | +0.287 |

**Each social task is tracked on the axis that matters for it.** `wait` is a
velocity task — stop for the person — and velocity tracks (+0.55). `pass_right`
is a steering task — go around them — and steering tracks strongly (+0.72).
Holding a roughly constant speed while passing is reasonable behaviour, not a
failure. `pass_left` is markedly weaker than `pass_right` (+0.32 vs +0.72) and
there is no explanation for the asymmetry yet.

**It under-brakes.** On `wait` the slow-half predicted mean is 0.053 against a
fast-half 0.069 — a gap of +0.015 where ground truth is +0.054, i.e. about 29%
of the correct magnitude. It does reach near-zero (predicted min −0.006), but it
slows less than the demonstrations do. Do not rely on it to stop hard.

**Nav-loop velocity does not track** (−0.046). Expect a roughly constant speed
around the loop. Steering is weakly positive (+0.27, borderline).

**INT4 costs nothing measurable.** Differences from bf16 are ≤0.10 and run in
*both* directions across splits, and MAE is identical to three decimals
(e.g. `wait` 0.0253 vs 0.0250). Serve INT4 without hesitation.

Training was stopped by the relaunch guard rather than early stopping, but
validation had not improved for 4,400 iterations, so more training of this recipe
is unlikely to help. The remaining hypothesis is capacity (Edge runs a 2B-active
text backbone vs Nano's 8B).

### On-Jetson inference: fixed (was ignoring the camera)

On-car inference previously emitted a near-constant action regardless of what the
camera saw. **Two bugs, both fixed** — see
[`car_deploy/README.md`](../car_deploy/README.md) for the patches:

1. **Uninitialised non-persistent buffers after meta-device init.** `COSMOS_KEEP_META_INIT=1`
   builds the net on meta so INT4 weights fit unified memory; materialisation
   allocates uninitialised storage, and `persistent=False` buffers are absent from
   the checkpoint so nothing restores them. `time_embedder._timestep_frequencies`
   held `sum=+3.26e24` instead of `+14.401979`, so the model could not tell where
   it was on the denoising trajectory. **This was the primary cause.**
2. **The live video window was truncated to 1 frame** instead of 33, which cut the
   generation sequence from 752 tokens to 112.

After both fixes, on-car output matches x86 in structure:

| | before | after | x86 |
|---|---|---|---|
| velocity across 4 frames | +4.19 flat | **0.27 → 1.00 → 1.09 → 1.12** | 0.12 → 1.01 → 1.09 |
| curvature spread, frames | 0.005 | **0.446** | — |
| curvature spread, captions | 0.008 | **0.297** | — |

The caption ordering matches x86 exactly
(`loop_cw < wait < loop_ccw < pass_left < pass_right`), and peak device memory is
**2.47 GB** on the Orin Nano.

Ruled out along the way, each by measurement rather than argument: INT4
quantization, bitsandbytes on `sm_87` (dequantized weights bit-identical), the
SDPA attention fallback (forced on x86 for all 56 calls — correct output), CUDA
RNG (bit-identical), tokenization (identical IDs), and the VAE (identical latents
that track the input). `named_parameters()` looked healthy the whole time — only
the buffers were wrong.

**Still open:** on-car inference runs at ~1.55 s per call versus ~0.23 s tethered,
so the tether gives roughly 6x tighter closed-loop control. Measure again with
`sudo jetson_clocks` pinned before choosing.

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
