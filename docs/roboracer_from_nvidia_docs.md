# From NVIDIA's docs to a driving policy — what to read, what to run, and why

A navigation guide. Cosmos 3 ships a post-training stack built for **robot arms**
(DROID, LIBERO). This traces the path from NVIDIA's own documentation to the
RoboRacer policy, marking at each step **what upstream gives you** and **what had
to be written**.

Start here if you are adapting Cosmos 3 to a new robot.

---

## The path at a glance

| # | Step | Upstream provides | You provide |
|---|---|---|---|
| 0 | Environment | `docs/setup.md` | — |
| 1 | **Dataset** | *nothing* — explicitly out of scope | **`convert_roboracer_to_lerobot.py`** |
| 2 | Base checkpoint | `convert_model_to_dcp` | — |
| 3 | Recipe | `action_policy_droid_nano` as template | **`action_policy_roboracer_nano.py`** |
| 4 | Launch | `launch_sft_action_policy_droid.sh` | **`train_supervisor.py`** |
| 5 | Export | `export_model.py` | patches (see §5) |
| 6 | Compress | *nothing* | **`quantize_int4.py`** |
| 7 | Serve | *nothing* | **server + ROS 2 client** |

**Steps 1, 6 and 7 have no upstream equivalent at all.** Step 3 has a template
you rewrite. Only 2 and 5 are used more or less as shipped.

---

## 0 — Environment

**Read:** [`docs/setup.md`](./setup.md), then
[`docs/environment_variables.md`](./environment_variables.md).

Keep caches off `$HOME` — it has a ~50 GB quota and `uv`/HF caches will fill it:

```bash
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export HF_HOME=/scratch/$USER/.cache/huggingface
```

---

## 1 — Dataset: the gap upstream leaves you

**Read first:** [`docs/action_policy_droid_posttrain.md`](./action_policy_droid_posttrain.md).

Note what it says under *Inputs you provide*:

> "the LeRobot v2.0→v3.0 conversion + success filtering is **run out-of-band (not
> yet in this repo)**"

and under *Dataset*: **"To be released."**

So the framework expects a finished LeRobot v3.0 dataset and does not tell you how
to build one. For a robot that isn't DROID, this is the bulk of the work.

**Use: `convert_roboracer_to_lerobot.py`**

Reads ROS 2 bags (SQLite) and emits LeRobot v3.0. The parts worth copying if your
robot differs:

| what | why it matters |
|---|---|
| `_resolve_topic_id()` | ROS assigns topic IDs per-bag by recording order — hardcoding them silently reads the wrong stream |
| `sync_odom_to_frames()` | camera and odometry publish independently; the camera is the clock and odometry snaps to it |
| `compute_actions()` | absolute pose → **relative** pose delta. Absolute position is useless as a target; it encodes where the track sits in the room |
| 9-D `[pos_xyz, rot6d]` | matches Cosmos 3's `av` domain (`domain_id=1`). **Use the domain format the pretrained action pathway already knows** or transfer gains nothing |
| H.264 video encoding | required by Cosmos's video tokenizer |

**Why rot6d and not a yaw angle:** angles wrap at ±π, and a model regressing
across that discontinuity learns badly near the wrap. rot6d is continuous.

**Also use:**
- `convert_all_datasets.sh` — one caption per campaign, all splits
- `compute_roboracer_stats.py` — normalization statistics over the **full** corpus
- `roboracer_bag_split_*.json` — pins train/eval/test **by recording session**

> **Split by recording, never by frame.** At 15 fps consecutive frames are 67 ms
> apart and nearly identical. A frame-level split puts near-duplicates in train and
> test and the score becomes meaningless.

**Reference:** [`docs/custom_dataset.md`](./custom_dataset.md) §3 for how datasets
plug into the dataloader, including ratio mixing across multiple datasets.

---

## 2 — Base checkpoint

Straight from the upstream doc, unchanged:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o examples/checkpoints/Cosmos3-Nano
```

This is what `BASE_CHECKPOINT_PATH` points at for the roboracer run. Verify it
worked — the output directory must contain `model/*.distcp`:

```bash
ls examples/checkpoints/Cosmos3-Nano/model/ | head
# __0_0.distcp  __0_1.distcp  __0_2.distcp ...
```

**Pick the right starting checkpoint.** A `*-Policy-*` variant ships a trained
generation expert and action bridges; the plain base model does not, and a
cold-started action pathway is much harder to train.

---

## 3 — The recipe

**Read:** `docs/action_policy_droid_posttrain.md` §Recipe, then
[`docs/sft_config.md`](./sft_config.md) for the TOML schema.

**Template:**
`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py`

**Yours:** `…/action_policy_roboracer_nano.py`

What changes from the DROID recipe and why:

| knob | DROID | RoboRacer | why |
|---|---|---|---|
| action space | `joint_pos` 8-D | **`av` 9-D** | car, not arm |
| `use_state` | true | **false** | no proprioception |
| viewpoint | `concat_view` | **single front camera** | one camera |
| resolution | `480` | **`256` data** | smaller images |
| train mode | policy | **`joint`** | see below |
| val mode | — | **`wam` only** | see below |
| eval | disabled | **enabled** | early stopping needs it |

**The two decisions that matter most:**

**Train `joint`, validate `wam`.** Training only in deployable mode collapsed to a
near-constant straight-line action — the shared backbone must reconstruct video
*and* predict actions from one frame, and a constant minimises loss on the
straight-driving majority. Mixing forward/inverse/policy dynamics per sample
teaches the same physics through easier problems. Validation stays strictly in
`wam`, because `inverse_dynamics` sees real future frames and would inflate the
stopping signal with performance the car can never have.

**Check `loss_scale` when you change model tier.** It weights **only** the video
term:

```
total = fm_loss_vision * loss_scale + fm_loss_action * action_loss_weight
```

The tier's base YAML may set it to 10.0 (tuned for manipulation), which makes the
effective action:video ratio 1:1 instead of 10:1. **Training converges perfectly
normally on the wrong objective.** The only symptom is a validation loss ~5× higher
than expected.

**Also non-obvious:** `model.config.resolution` is *not* the data resolution — it
is a lookup key into `shift = {"256": 3, "480": 5, "720": 10}` for the
flow-matching schedule.

---

## 4 — Launch

**Upstream reference:** `examples/launch_sft_action_policy_droid.sh` +
`examples/toml/sft_config/action_policy_droid_repro.toml`. Read
[`docs/training.md`](./training.md) for what the launcher does.

**What the roboracer run actually used.** Not the shell launcher — a supervisor
that invokes `cosmos_framework.scripts.train` directly, so it can restart the job
on a stall. Set three environment variables, then start it detached:

```bash
cd /scratch/$USER/cosmos-framework

export BASE_CHECKPOINT_PATH=$PWD/examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$PWD/outputs

setsid nohup python3 train_supervisor.py < /dev/null > /tmp/supervisor.log 2>&1 &
```

Underneath, that runs:

```bash
.venv/bin/torchrun --nproc_per_node=8 \
  -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_policy_roboracer_repro.toml \
  -- \
  job.name=action_policy_roboracer_repro_v10 \
  trainer.max_iter=100000 \
  dataloader_train.max_samples_per_batch=96 \
  checkpoint.save_iter=50 \
  trainer.run_validation_on_start=False
```

Those trailing overrides are the **effective** values — they beat both the TOML
and the experiment `.py`. Two worth understanding before you copy them:

- `max_samples_per_batch=96` was tuned for 80 GB cards. On 48 GB, start far lower
  and raise it from observed memory rather than guessing.
- `max_iter=100000` is a safety ceiling only. Early stopping is what should end
  the run.

The supervisor adds three things a bare launch doesn't have:

| component | why you need it |
|---|---|
| `stall_watchdog.py` | NCCL hangs **silently** — every GPU pins at 100% with no logs and no timeout. Only a log-progress timeout catches it |
| `early_stopping_watchdog.py` | epoch-aggregated patience; raw per-check validation is too noisy to stop on |
| `prune_roboracer_checkpoints.py` | keeps best + latest. **Mandatory on shared storage** — checkpoints are tens of GB each |

Run detached with `setsid nohup …` — plain `nohup` still dies when the SSH pipe
breaks.

> **Validation sizing:** `max_val_iter × max_samples_per_batch` must not exceed
> your **smallest** eval split, or some ranks exhaust their data and leave the
> collective early while others wait. Presents as 100% GPU utilisation, no logs,
> no timeout.

---

## 5 — Export

```bash
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path <ckpt>/model \
  --experiment <your_experiment> --no-use-ema-weights \
  -o <out_dir>
```

Note `--experiment`, not `--checkpoint.experiment`. And point at the **`model/`
subdirectory** — `CheckpointType.from_path()` looks for `*.distcp` at the top
level, and training checkpoints nest them one level down.

**After export, reset the sharding degree.** The config carries
`data_parallel_shard_degree=8` from training; a single-process server cannot build
an 8-way-sharded model:

```bash
sed -i 's/"data_parallel_shard_degree": 8/"data_parallel_shard_degree": 1/' config.json
```

---

## 6 — Compression (no upstream equivalent)

**Use `quantize_int4.py`.** 4-bit NF4 with double quantization, bfloat16 compute.

It also normalises the two things that otherwise break single-process serving —
the sharding degree above, and `vae_path`, which is written as a registry URI and
must be repointed at a local **fp16** VAE (the fp32 file is 2.8 GB and eats the
Jetson margin).

**Quantize for the edge device only.** bfloat16 is the training precision and
therefore the accuracy ceiling, and NF4 adds dequantization work to every forward
pass — so on a server it costs you and buys nothing.

---

## 7 — Serving (no upstream equivalent)

| file | role |
|---|---|
| `cosmos_framework/scripts/action_policy_server_roboracer.py` | websocket policy server |
| `roboracer_chunk_buffered_client.py` | ROS 2 client with receding-horizon buffer |

**The server has no ROS 2 dependency** — msgpack over websockets, deliberately. It
can be smoke-tested from a 10-line Python script with no robot present, and the
robot-specific code stays on the vehicle.

**The client exists because inference is slower than control.** Predictions arrive
at ~4.3 Hz; steering needs 15 Hz. Each prediction covers 32 steps = 2.13 s, so the
client plays out one step per tick and swaps in each new chunk on arrival —
receding-horizon control. It publishes zero velocity if the buffer empties or goes
stale rather than coasting on an old plan.

Deployment details: [`docs/roboracer_int4_deployment.md`](./roboracer_int4_deployment.md).

---

## Reading order

**Adapting Cosmos 3 to a new robot:**
1. `docs/action_policy_droid_posttrain.md` — the canonical recipe shape
2. `docs/custom_dataset.md` — how data reaches the model
3. `convert_roboracer_to_lerobot.py` — a worked example of the step upstream skips
4. `action_policy_roboracer_nano.py` — a worked example of a non-arm recipe

**Just reproducing this policy:** §2 → §4 → §5 → §7.

**Deploying to an edge device:** §5 → §6 → §7, then
`docs/roboracer_int4_deployment.md`.

---

## The five things that cost the most time

1. **The dataset step is entirely yours.** Upstream says so explicitly; budget accordingly.
2. **Use the pretrained action space** (`av` 9-D here). Inventing your own throws away the transfer.
3. **Reinitialize inherited action heads** if the source checkpoint's action space differs — DROID's are 7-DoF manipulator heads and are actively wrong on a car.
4. **Re-check `loss_scale`** on every model-tier change. It fails silently.
5. **Split by recording session, not by frame.** This one invalidates all your numbers if you get it wrong.
