# Getting today's fixes onto a car

Everything here is self-contained. The car's `cosmos-edge` checkout is a
different repo lineage (`nathantsoi/cosmos-edge`, one "initial commit") from
this branch, so `git pull` would be messy — apply the patch instead.

**orin10 has already had all of this applied.** These steps are for a *second*
car, a re-image, or reproducing the state from scratch.

## What the fixes are

| fix | file | why it matters |
|---|---|---|
| Curvature blow-up | `action_policy_server_roboracer.py` | `to_curvature_velocity` divided yaw by an arc length floored at `1e-3`. A near-stationary step (in-distribution — `q01` of `dx` is exactly 0.0) became a ~3.0 1/m curvature, saturating the client clamp and **pinning the steering servo to full lock**. This is what drove the car into a wall. |
| UniPC CPU solve | `fm_solvers_unipc.py` | UniPC's corrector calls `torch.linalg.solve` at order ≥2. Jetson JP6's torch ships a `libtorch_cuda_linalg.so` linked against a mismatched cuSOLVER, so **any `num_steps > 1` crashed** at sampling step 3. The matrices are 2×2; solving on CPU is numerically identical (verified against the cuSOLVER path on x86). |
| Goal-dot default | `roboracer_chunk_buffered_client.py` | `--subgoal-lookahead-s` defaulted to **5.0**, but `convert_all_datasets.sh` sets `LOOKAHEAD=0`, so the checkpoints were trained on frames with **no goal dot**. The client would have started drawing dots the model has never seen the moment `/odom` came alive. Now defaults to 0, with a guard that makes 0 actually mean "no dot" (it previously still drew a degenerate bottom-centre dot). |
| Curvature clamp | `roboracer_chunk_buffered_client.py` | `--max-curvature` was 3.0. `car.lua` has `max_steering_angle=0.4030`, so `tan(0.4030)/0.32 = 1.33 1/m` is the physical limit — the old clamp bounded nothing. Now 1.3. |

## Critical: rebuild non-persistent buffers after a meta-device load

**This was the bug that made on-car inference ignore the camera entirely.**

`run_roboracer_server.sh` sets `COSMOS_KEEP_META_INIT=1` so the network is built
on the meta device and the INT4 weights fit unified memory. Materialisation then
allocates *uninitialised* storage — and buffers registered `persistent=False` are
not in the checkpoint, so `load_state_dict` never fills them. They also get cast
to the model precision instead of staying float32.

Measured on orin10 before the fix:

| buffer | expected | actual |
|---|---|---|
| `time_embedder._timestep_frequencies` | float32, sum `+14.401979`, all in (0,1] | **bfloat16, sum `+3.26e24`, min `-1.07e8`** |
| `rotary_emb.original_inv_freq` | float32 | **bfloat16, sum `+4.6e35`** |

`_timestep_frequencies` is the sinusoidal frequency table for the diffusion
timestep embedding. With garbage frequencies the model cannot tell where it is on
the denoising trajectory, and its output decouples from both the image and the
caption — the "same action no matter what the camera sees" symptom.

Insert immediately after `self.model = hf_model.model`:

```python
n_fixed = 0
for _mod in self.model.modules():
    if type(_mod).__name__ == "TimestepEmbedder" and hasattr(_mod, "_timestep_frequencies"):
        _dev = _mod._timestep_frequencies.device
        _f = _mod._build_timestep_frequencies(
            _mod.frequency_embedding_size, max_period=10000, device=_dev)
        _mod.register_buffer("_timestep_frequencies", _f, persistent=False)
        n_fixed += 1
    _inv = getattr(_mod, "inv_freq", None)
    _orig_inv = getattr(_mod, "original_inv_freq", None)
    if _inv is not None and _orig_inv is not None and (
        _orig_inv.dtype != _inv.dtype or _orig_inv.shape != _inv.shape
        or not torch.isfinite(_orig_inv).all()
        or bool((_orig_inv.float() - _inv.float()).abs().max() > 1e-3)
    ):
        _mod.original_inv_freq = _inv.detach().clone()
        n_fixed += 1
if n_fixed:
    log.info(f"[roboracer-policy-server] rebuilt {n_fixed} non-persistent buffer(s) after meta init")
```

Verify on startup: the log must show `rebuilt 2 non-persistent buffer(s)`, and
`_timestep_frequencies` must read `sum=+14.401979`, first values
`[1.0, 0.930572, 0.865964, …]`.

**Any meta-device init path needs this audit** — `named_parameters()` looked
perfectly healthy throughout (549 params, 0 on meta, 0 NaN). Only the buffers
were wrong.

## Also check: the WAM video window must be 33 frames

orin10's server had `_build_live_sample` truncated to a single frame:

```python
# Live WAM: one conditioning frame only. Repeating chunk_length+1 identical
# frames blew Jetson activation memory (attn seq~99) with no extra signal.
video = frame.unsqueeze(0)                                    # WRONG
video = frame.unsqueeze(0).repeat(_CHUNK_LENGTH + 1, 1, 1, 1) # correct
```

"No extra signal" is wrong. In `wam` mode only index 0 is *conditioning*; frames
1..T-1 are the denoising targets the 32 action steps are jointly denoised with,
and `build_sequence_plan_from_mode` derives the plan from `video.shape[1]`. With
T=1 the model runs in a regime it never saw during training, and the caption's
embedded duration degrades to "The video is 0.0 seconds long".

The memory rationale was also unfounded: restoring 33 frames raised peak device
memory from 2.47 GB to **2.67 GB**. This branch's copy is already correct — check
any car whose tree came from elsewhere.

## Apply

```bash
# 1. framework fixes
scp car_deploy/roboracer_car_fixes.patch orin@<car-ip>:/tmp/
ssh orin@<car-ip>
cd ~/roboracer_ws/src/cosmos-edge
git apply --check /tmp/roboracer_car_fixes.patch   # dry run; must print nothing
git apply /tmp/roboracer_car_fixes.patch

# 2. client (not part of the framework package)
scp car_deploy/roboracer_chunk_buffered_client.py orin@<car-ip>:/home/orin/roboracer_ws/
scp car_deploy/social_baseline.py                 orin@<car-ip>:/home/orin/roboracer_ws/

# 3. verify
python3 -c "import ast;ast.parse(open('cosmos_framework/scripts/action_policy_server_roboracer.py').read());print('ok')"
grep -n "_MIN_ARC_M" cosmos_framework/scripts/action_policy_server_roboracer.py   # expect 2 hits
grep -n "default=1.3" ~/roboracer_ws/roboracer_chunk_buffered_client.py          # expect 1 hit
```

If `git apply --check` complains, the car's copy has diverged. Apply by hand
using the table above — each change is a few lines.

## Which deployment path to use

**Use the tethered path.** Model on robolang, car as a thin websocket client.
Measured on x86 at n=48 per dataset (see `docs/roboracer_edge_deployment.md` §5):
`wait` tracks velocity at ρ = +0.55, `pass_right` tracks steering at ρ = +0.72,
and INT4 costs nothing measurable versus bf16.

**On-car standalone inference is NOT ready.** The model loads and runs on the
Jetson (2.47 GB, `num_steps=4` works after the UniPC fix), but its output is
near-constant and independent of both the camera image and the direction token
— caption spread 0.008 on-Jetson versus 1.614 for the identical checkpoint on
x86. Quantization is ruled out (bf16 and INT4 agree closely on x86); the cause
is somewhere in the Jetson numerical stack and is not yet localised. Do not
drive on on-car inference until that is resolved.
