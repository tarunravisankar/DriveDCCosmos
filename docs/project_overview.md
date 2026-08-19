# RoboRacer × Cosmos3 — project brief

Written as a self-contained handoff: what the project is trying to do, how the
system is put together, and specifically **what had to be built on top of the
stock Cosmos3 framework** to make a 1/10-scale car drive with it. No debugging
history — just the design and the inventory.

---

## 1. Goal

Turn a pretrained **world foundation model** (NVIDIA Cosmos3) into a
**vision-language-conditioned driving policy** for a 1/10-scale autonomous
vehicle, and compress it enough to run **onboard an 8 GB Jetson Orin Nano**.

The task is **mapless, socially-aware navigation**: drive an indoor environment
shared with pedestrians using **no HD map, no metric localization, and no LiDAR
in the control loop**. The policy sees one forward-facing RGB frame plus a
natural-language instruction, and outputs future vehicle motion.

Two research questions, of unequal evidential strength:

- **Deployment (well supported).** Can a world foundation model be compressed to
  run on a resource-constrained embedded platform without losing policy quality?
- **Generalization (partially supported).** Does the pretrained model adapt to
  driving from limited demonstrations? Evaluated on held-out *recording sessions*
  of seen routes — **not** unseen rooms or layouts.

Language matters structurally here, not decoratively: the same physical loop was
demonstrated in both directions, several tracks were recorded in one room, and
the social episodes present one scene under three mutually incompatible correct
behaviours. Visually near-identical observations therefore carry conflicting
action targets, so **the instruction is the entire goal specification**.

---

## 2. Hardware and machines

| machine | role | detail |
|---|---|---|
| **robolang** | Edge training + tethered inference | 8× H100 80 GB |
| **robolidar** | Nano training + distillation | 10× RTX A6000 48 GB, shared, separate `/scratch` |
| **orin10** | the car | Jetson Orin Nano Super 8 GB, L4T 36.4.7 (JetPack 6.2), CUDA 12.6, `sm_87`, 6-core Cortex-A78AE |

Car sensing/actuation: ArduCam IMX219 at 320×240, 66° HFOV, 15 fps; VESC for
odometry and drive; `ut_automata` stack; ROS 2 in a Docker container
(`orin_roboracer`); WireGuard VPN links car and servers.

---

## 3. Policy specification

**Input**
- One forward-facing RGB frame `I_t` (state approximated as Markovian)
- One natural-language instruction `l` from a closed vocabulary of **11 captions**
  — 8 route-following + 3 social (`pass_left`, `pass_right`, `wait`)

In `wam` mode the sequence plan sets `condition_frame_indexes_vision=[0]`, so
**only frame 0 conditions the model**. The tensor passed in is a 33-frame window
at `conditioning_fps=15`; frames 1–32 are the *denoising targets* the action
chunk is jointly denoised with, and the sequence plan is derived from
`video.shape[1]` — so the window length is load-bearing, not padding.

**Output**
- **32 future actions**, each **9-D**: `[pos_x, pos_y, pos_z, rot_0..rot_5]`
  — a body-frame translation delta plus a 6-D rotation representation (rot6d) of
  the yaw delta. Both derived from odometry at conversion time.
- Decoded to **curvature** (1/m, left-positive) and **velocity** (m/s) and
  published as `AckermannCurvatureDriveMsg`. The car's `vesc_driver` converts
  curvature to a steering angle itself via `car.lua`.
- 32 steps at 15 Hz = a **2.13 s** planning horizon.

**Caption design.** Captions are deliberate *minimal pairs* — one sentence frame
differing in a single discriminative token:

```
"Drive the roboracer vehicle counter-clockwise around a large indoor loop."
"Drive the roboracer vehicle clockwise         around a large indoor loop."
"Drive the roboracer vehicle and pass the person on the right."
"Drive the roboracer vehicle and wait for the person to pass."
```

This forces the discriminating information into a semantically meaningful word
and makes conditioning directly testable: hold the frame fixed, vary one token,
the action must change.

**Semantics worth knowing:** `pass_right` means *keep the person on your right* —
the car swerves **left**. Verified in 24/24 training episodes (peak yaw +2.55°/step
left). `pass_left` swerves right in 20/24.

---

## 4. Data

Recorded as ROS 2 bags (`/camera_0/image_raw/compressed`, `/odom`), converted to
LeRobot format.

| | datasets | episodes | frames | hours @15 fps |
|---|---|---|---|---|
| train | 12 | 250 | 355,716 | 6.59 |
| eval | 12 | 35 | 85,851 | 1.59 |
| test | 12 | 32 | 51,290 | 0.95 |
| **total** | **12** | **317** | **492,857** | **9.13** |

12 datasets = **9 navigation tracks + 3 social**. Splits are **by recording
session** (bag-level), so evaluation never sees a training run — but does see the
same rooms and routes.

---

## 5. Training recipe

Base checkpoints: `nvidia/Cosmos3-Nano` and `nvidia/Cosmos3-Edge-Policy-DROID`
(the only Edge variant shipping a trained generation expert and action bridges),
converted to DCP.

**The fine-tune is partial — the vision-language backbone stays frozen.** The
optimizer sees only:

```python
keys_to_select = ["moe_gen", "time_embedder",
                  "vae2llm", "llm2vae",
                  "action2llm", "llm2action", "action_modality_embed"]
```

So the *understanding expert* is fixed and only the **generation expert** plus the
modality bridges adapt. That is the concrete form of the bet that the foundation
model already contains the needed visual/spatiotemporal representations.

The pretrained action bridges are **reinitialized**, because Edge-Policy-DROID's
heads encode a 7-DoF manipulator action space unrelated to a 2-DoF car; they get a
5× learning-rate multiplier to compensate.

| | |
|---|---|
| objective | rectified flow / flow matching, on video **and** action |
| loss | `fm_loss_vision × 1.0 + fm_loss_action × 10.0` |
| optimizer | FusedAdam, lr 2e-4, wd 0.05, β=(0.9, 0.99), LambdaLinear |
| parallelism | FSDP, 8 GPUs |
| train mode | `joint` — mixes forward-dynamics, inverse-dynamics and policy per sample |
| val mode | **`wam` only** — so early stopping reflects deployable performance |
| data resolution | 256p; `config.resolution="720"` selects the flow shift (=10) |

**Why `joint` train / `wam` validate.** A pure policy objective forces one
backbone to reconstruct video *and* predict actions, which collapses to an
input-independent straight line. A pure inverse-dynamics variant scores well but
conditions on *future* frames and cannot be deployed. Training on the mixture and
validating strictly in the deployable mode resolves both.

**Runs:** 13 total — 11 on robolidar (`repro`, `repro_v2`–`v10`, `distill_v1`),
2 on robolang (`edge_v2`, `edge_v3`).

| run | hardware | wall time | best |
|---|---|---|---|
| Edge v3 | 8× H100 | 18.3 h | iter 2400, val 0.3159 |
| Distill v1 | A6000 | ~22 h | iter 4200, val 0.378 |
| Nano teacher v10 | A6000 | — | iter 7200 |

---

## 6. Compression

Two paths were built.

**Distillation (Nano).** `DistillOmniMoTModel`: frozen 16B/8B-active teacher plus a
trainable 4B student (Qwen3-VL-4B-Instruct backbone), MSE between student and
teacher action predictions added to the student's own flow-matching loss.
`distill_loss_action` converged to ~0.002.

**Quantization.** bitsandbytes **NF4**, double quantization, bfloat16 compute
dtype. Not TensorRT, not AWQ/GPTQ. Weights are 4-bit; activations dequantize to
bf16 on each forward pass, so it shrinks *weights* only.

| model | total / active | bf16 | INT4 | peak on Orin |
|---|---|---|---|---|
| Nano teacher | 16B / 8B | 81 GB DCP | — | — |
| Distilled student | 4B / 4B | 16 GB | 4.3 GB | 9.20 GB — **did not fit** |
| **Cosmos3-Edge** | 4B / **2B** | 8.15 GB | 3.0 GB | **2.47 GB — fits** |

Distillation + INT4 on the Nano lineage still needed 9.20 GB against an 8 GB
budget, which no further quantization of that architecture could close. The
budget was therefore re-derived from the hardware backward and the model family
changed to edge-scale, reaching 2.47 GB with ~5 GB headroom.

---

## 7. Deployment

Two modes, sharing one protocol.

**Tethered** — model on robolang, car as a thin websocket client over WireGuard.
~0.23 s per inference.

**Onboard** — INT4 server on the Jetson, client on localhost. ~1.55 s per
inference with the server alone; **2.96 s under full navstack load**, which
exceeds the 2.13 s chunk horizon and starves the buffer.

**Protocol** — msgpack over websockets, no ROS 2 dependency on the server side:

```
request : {"image": <HxWx3 uint8 RGB bytes>, "shape": [H,W,3],
           "direction": <token>            # or "caption": <raw text>}
response: {"curvature": [32], "velocity": [32], "raw_action": [32x9]}
```

**Receding-horizon client.** Actuation must be 15 Hz regardless of inference time,
so the client consumes one step per 15 Hz tick and swaps in each new chunk as it
arrives — executing ~4 of 32 steps per chunk when tethered. Safety layers:
`--live` must be explicit (dry-run is default), `vesc_driver` ignores the topic
unless autonomous mode is toggled on the joystick, `--max-velocity 1.0` and
`--max-curvature 1.3` clamp every command, and a **fail-safe STOP** publishes zero
when the buffer empties or goes stale.

`--max-curvature 1.3` is the car's *physical* limit: `car.lua` sets
`max_steering_angle=0.4030`, so `tan(0.4030)/wheelbase(0.32) = 1.33` 1/m.

---

## 8. Evaluation

**Open-loop only.** 48 velocity-stratified samples per split × 4 splits; predicted
action vs. recorded demonstration, one frame at a time. Metrics: Spearman,
Pearson, MAE, slow/fast group gap. Significance threshold at n=48 is **ρ ≈ 0.285**.

Velocity Spearman:

| split | Nano teacher (8B act.) | Distilled (4B act.) | Edge (2B act.) |
|---|---|---|---|
| `wait` | +0.703 | +0.619 | +0.553 |
| **orin10 nav** | **+0.503** | **+0.434** | **−0.046** |
| `pass_left` | +0.150 | −0.075 | +0.097 |
| `pass_right` | +0.236 | −0.064 | +0.056 |

Steering (yaw) Spearman:

| split | teacher | distilled | Edge |
|---|---|---|---|
| `pass_left` | +0.759 | +0.612 | +0.322 |
| `pass_right` | +0.627 | +0.662 | **+0.715** |
| orin10 | +0.287 | +0.229 | +0.271 |

**Reading.** Each social task is tracked on the axis that matters for it — `wait`
is a velocity task and velocity tracks; `pass_right` is a steering task and
steering tracks. Holding roughly constant speed while passing is correct
behaviour, not failure.

**The capacity result.** The distilled 4B retains nav-loop velocity tracking
(+0.434) while Edge loses it entirely (−0.046). Both are ~4B *total*; the
distilled student is 4B **active** and Edge is 2B active. **Active** parameters,
not total, predict the gap.

**Quantization is free.** Edge INT4 vs bf16 differ by ≤0.10 Spearman in *both*
directions with MAE identical to three decimals. Re-run on the Orin itself through
the deployed websocket path: `wait` +0.484, `pass_right` steering +0.739, MAE
within 0.0006 of x86 on every split.

**Not measured — do not claim:** closed-loop success rate, route completion, laps,
collisions, interventions, unseen-instruction handling, unseen-scenario transfer.
The free-text caption channel exists but has never been exercised. There is no
from-scratch behavioral-cloning baseline.

---

## 9. What we had to build

Stock Cosmos3 ships manipulation recipes (DROID, LIBERO). Everything below is
project-authored.

### Data pipeline
| file | purpose |
|---|---|
| `convert_roboracer_to_lerobot.py` | ROS 2 bag → LeRobot dataset: decode compressed images, resample odom to 15 fps, build 9-D rot6d actions, write captions |
| `convert_all_datasets.sh` | orchestrates 12 datasets × 3 splits = 36 conversions with per-dataset captions |
| `roboracer_bag_split_*.json` | 11 manifests pinning which recording sessions are train/eval/test |
| `compute_roboracer_stats.py` | corpus-level action normalization statistics |
| `roboracer_stats.json` | the stats themselves, versioned alongside the checkpoint |
| `project_goal.py` | optional goal-dot projection (**disabled**: `LOOKAHEAD=0`, so training frames are unannotated and conditioning is text-only) |
| `roboracer_dataset.py` | `RoboracerDataset` + `get_action_roboracer_sft_dataset`; declares `EMBODIMENT_TYPE="av"`, 9-D action space, turn-oversampling tiers |

### Training configs
| file | purpose |
|---|---|
| `action_policy_roboracer_nano.py` | Nano SFT recipe |
| `action_policy_roboracer_4b.py` | 4B student, standalone |
| `action_policy_roboracer_distill.py` | teacher+student distillation |
| `action_policy_roboracer_edge.py` | Edge recipe (398 lines) |
| `distill_omni_mot_model.py` | `DistillOmniMoTModel` — frozen teacher, trainable student, action-MSE distillation loss |

### Training infrastructure
| file | purpose |
|---|---|
| `stall_watchdog.py` | detects silent NCCL hangs by log-progress stall; kills and relaunches |
| `train_supervisor.py` / `train_supervisor_edge.py` | auto-relaunch wrapper, bounded retry budget, epoch-aggregated early stopping (patience 20 epochs, min 5, δ=0.001), checkpoint retention |
| `early_stopping_watchdog.py` | two-loop design — inner buckets raw validation checks into epochs, outer compares epoch means (raw checks are too noisy to stop on) |
| `prune_roboracer_checkpoints.py` | keeps best + latest only; mandatory on the shared filesystem. Best is additionally hard-linked into a `protected/` dir so it survives deletion at zero storage cost |
| `roboracer_epoch_summary.py` | epoch-level training summaries |

### Compression / export
| file | purpose |
|---|---|
| `quantize_int4.py` | HF bf16 → NF4 INT4; also normalizes `data_parallel_shard_degree` → 1 and repoints `vae_path` at the local fp16 VAE |

### Deployment
| file | purpose |
|---|---|
| `action_policy_server_roboracer.py` | websocket policy server; auto-detects DCP vs HF checkpoints; maps direction tokens → exact training captions; decodes 9-D actions → curvature/velocity |
| `roboracer_chunk_buffered_client.py` | ROS 2 client: camera subscribe → inference → receding-horizon chunk buffer → 15 Hz publish, with clamps and fail-safe STOP |
| `social_baseline.py` | auto-cycles `pass_right`/`pass_left`/`wait` every ~5 s |
| `generalization_test_client.py` | free-text caption testing |
| `test_server_from_orin.py` | connectivity smoke test |
| `run_roboracer_server.sh` | Jetson launcher: sets `COSMOS_KEEP_META_INIT`, `COSMOS_VAE_DEVICE=cpu`, `COSMOS_VAE_ENCODE_ONLY`, allocator config, and reclaims page cache before load |
| `tmux/cosmos/.tmuxinator.yaml` | one-command bring-up of server + client + camera + VESC + joystick |

### Evaluation
| file | purpose |
|---|---|
| `check_roboracer_predictions.py` | per-sample prediction inspection and plots (ranks by turn magnitude — good for steering, poor velocity coverage) |
| `velocity_eval.py` | velocity-stratified eval at n=48; Spearman/Pearson/MAE/group-gap; handles DCP and HF checkpoints; multiple dataset roots per model load |
| `ws_eval.py` | same metrics over the websocket against a running server — measures the deployed path, and is the only way to run the eval on the Jetson |

### Framework modifications required
Upstream Cosmos3 assumes manipulation data, multi-GPU training, and x86. These
changes were needed for a car, single-process serving, and ARM64:

- `validation_step` returned `pass` — made it run `training_step`
- validation dataloader iterators were never constructed
- eval logging asserted a single dataset name; packed batches carry one per sample
- `sync_model_states` attempted collectives at `world_size=1`
- Wan VAE `load_state_dict(assign=True)` did not materialize meta-device params —
  switched to `to_empty()` then `assign=False`
- `_init_weights` lacked a default for `buffer_device`, breaking stock `from_pretrained`
- `export_model` matched `OmniMoTModel` by substring (also matching
  `DistillOmniMoTModel`), and did not strip distill-only top-level kwargs
- `export_model` metadata builder only understood the DROID dataloader shape;
  added a `PackingDataLoader` fallback
- UniPC's corrector solves an order×order system with `torch.linalg.solve`;
  moved to CPU so `num_steps > 1` works on Jetson (matrices are 2×2)
- `to_curvature_velocity` divided by a floored arc length; near-stationary steps
  are in-distribution, so it now reports zero curvature below 1 cm/step
- added a torch-SDPA attention backend for devices without flash/cuDNN
- **after meta-device init, non-persistent buffers must be rebuilt** — they are
  absent from the checkpoint, so nothing restores them
  (`time_embedder._timestep_frequencies`, `rotary_emb.original_inv_freq`)

Exported checkpoints also need `data_parallel_shard_degree` reset from 8 to 1
before single-process serving.

---

## 10. Current state

- Tethered deployment works and is the better control loop (~0.23 s vs ~1.55 s).
- Onboard INT4 works: 2.47 GB on the Orin, validated at n=48 per split against x86.
- Onboard latency under full navstack load (2.96 s) exceeds the 2.13 s horizon;
  `jetson_clocks` is unpinned (GPU observed at 918 MHz) and `num_steps` could drop
  from 4 to 2.
- One qualitative closed-loop run exists: drove a straight hallway, swerved for a
  person under `pass_right`, then did not re-straighten. That is behavior-cloning
  covariate shift — the policy is memoryless and the corpus contains **zero
  recovery demonstrations**. DAgger recording is wired into the client
  (`--record-bag`) and is the intended remedy.
