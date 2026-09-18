# Adapting Cosmos 3 to a new robot

Cosmos 3 ships a post-training stack built for **robot arms** (DROID, LIBERO).
This walks through what it takes to point it at something else — in this case a
1/10-scale car — marking at each step **what the framework gives you** and **what
you have to write yourself**.

Assumes no prior experience with Cosmos 3. Covers training only; serving and
on-vehicle deployment are in
[`roboracer_int4_deployment.md`](./roboracer_int4_deployment.md).

---

## First: five Cosmos 3 concepts

You need these to read anything else in the repo.

**Experiment.** A Python file registering a named configuration — datasets, what
trains, losses, schedules. Lives under
`cosmos_framework/configs/base/experiment/…`. You refer to it by name
(`action_policy_roboracer_nano`), never by path. This is where the real recipe
lives.

**TOML.** A small run-level file under `examples/toml/sft_config/`. Picks which
experiment to run and sets a handful of scalars — precision, how many GPUs to
shard across, how often to checkpoint. It is validated against a strict schema,
so a typo fails immediately rather than silently.

**Three config layers, and which wins.** This trips up everyone:

```
CLI overrides  >  TOML  >  experiment .py
```

Trailing arguments after `--` on the launch command beat everything. If a number
in the `.py` doesn't match what you observe at runtime, check the launch command
before assuming the code is wrong.

**DCP.** *Distributed checkpoint* — PyTorch's sharded format, a directory of
`*.distcp` files rather than one big file. Training reads and writes DCP. The
separate HF (safetensors) format is what you export to for serving.

**The two towers.** Cosmos 3 runs a *reasoner* (vision-language: reads the image
and the instruction) and a *generator* (produces future video and actions). They
are joined by cross-attention. Which parts of which tower get trained is the
single most important thing the experiment file decides.

---

## The path at a glance

| # | Step | Framework provides | You provide |
|---|---|---|---|
| 1 | Environment | `docs/setup.md` | — |
| 2 | **Dataset** | *nothing — explicitly out of scope* | **`convert_roboracer_to_lerobot.py`** |
| 3 | Base checkpoint | `convert_model_to_dcp` | — |
| 4 | Recipe | DROID experiment as a template | **your experiment file** |
| 5 | Launch | `cosmos_framework.scripts.train` | **`train_supervisor.py`** |
| 6 | Export | `export_model.py` | — |

**Step 2 has no framework equivalent at all.** That is where most of the work is.

---

## 1 — Environment

Read [`docs/setup.md`](./setup.md), then
[`docs/environment_variables.md`](./environment_variables.md).

Keep package and model caches off your home directory — it usually has a quota
and these caches will fill it:

```bash
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export HF_HOME=/scratch/$USER/.cache/huggingface
```

---

## 2 — Dataset: the step the framework leaves to you

**Read first:** [`docs/action_policy_droid_posttrain.md`](./action_policy_droid_posttrain.md),
the canonical action-policy recipe. Note what it says under *Inputs you provide*:

> "the LeRobot v2.0→v3.0 conversion + success filtering is **run out-of-band (not
> yet in this repo)**"

and under *Dataset*: **"To be released."**

So the framework expects a finished dataset in **LeRobot v3.0** format and does
not tell you how to build one. For a robot that isn't DROID, this is the bulk of
the work.

### What a LeRobot dataset looks like

```
my_dataset/
├── meta/info.json              episode count, frame count, fps
├── data/chunk-000/*.parquet    per-frame actions + indices
└── videos/<camera>/chunk-000/*.mp4   H.264 (required by Cosmos)
```

Each row pairs one video frame with the action taken at that moment.

### The worked example

`convert_roboracer_to_lerobot.py` turns ROS 2 bags into exactly that. The parts
worth copying:

| what | why it matters |
|---|---|
| `_resolve_topic_id()` | ROS assigns topic IDs per-bag by recording order. Hardcoding them reads the wrong stream, silently. |
| `sync_odom_to_frames()` | Camera and odometry publish independently and never at the same instant. The camera is the clock; odometry snaps to the nearest one. |
| `compute_actions()` | Converts absolute pose into **relative** pose deltas. Absolute position is useless as a target — it encodes where the track happens to sit in the room. |
| 9-D `[pos_xyz, rot6d]` | Matches Cosmos 3's built-in `av` domain (`domain_id=1`). |

**Two decisions to copy rather than reinvent:**

**Use a domain the framework already knows.** Cosmos 3 has registered action
spaces (`av` for vehicles, `joint_pos` for arms). Using one means the pretrained
action pathway transfers. Inventing your own throws that away.

**Represent rotation as rot6d, not an angle.** Angles wrap at ±π, so a model
regressing across that boundary learns badly near it. rot6d is continuous.

### Also needed

- `convert_all_datasets.sh` — one caption per recording campaign, all splits
- `compute_roboracer_stats.py` — normalization statistics over the **full** corpus
- `roboracer_bag_split_*.json` — pins which recordings go to train/eval/test

> **Split by recording session, never by frame.** At 15 fps, consecutive frames
> are 67 ms apart and nearly identical. A frame-level split puts near-duplicates
> in both train and test, and your evaluation becomes meaningless.

**Reference:** [`docs/custom_dataset.md`](./custom_dataset.md) §3 for how a
dataset plugs into the dataloader, including mixing several by ratio.

---

## 3 — Base checkpoint

Convert the published model to DCP:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o examples/checkpoints/Cosmos3-Nano
```

Verify it worked — the output must contain `model/*.distcp`:

```bash
ls examples/checkpoints/Cosmos3-Nano/model/ | head
# __0_0.distcp  __0_1.distcp  __0_2.distcp ...
```

**Prefer a `*-Policy-*` variant if one exists for your tier.** Those ship a
trained generation expert and action bridges; the plain base model does not, and
a cold-started action pathway is much harder to train.

---

## 4 — The recipe

**Read:** `docs/action_policy_droid_posttrain.md` §Recipe for the knob table, then
[`docs/sft_config.md`](./sft_config.md) for what the TOML is allowed to set.

**Copy this template:**
`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py`

**Compare against:** `…/action_policy_roboracer_nano.py`

### Keep the framework's fine-tuning strategy

The DROID experiment sets `keys_to_select`, which restricts the optimizer to a
named list of parameter groups — the generation expert, the timestep embedder,
and the modality bridges. **Everything else, including the entire vision-language
reasoner, stays frozen.**

The roboracer recipe uses this list unchanged, and you probably should too. The
reasoning: a few hours of robot data cannot teach a model to see, and letting it
try would overwrite pretrained perception. Freezing spends your data entirely on
learning control.

Same for `keys_to_skip_loading` and the 5× `lr_multipliers` on the action
bridges — the pretrained action heads encode the *source* robot's action space,
so they get discarded and retrained faster than everything else.

### What you do change

| knob | DROID | RoboRacer | why |
|---|---|---|---|
| action space | `joint_pos` 8-D | `av` 9-D | car, not arm |
| `use_state` | true | false | no proprioception |
| viewpoint | `concat_view` | single camera | one camera |
| resolution | 480 | 256 | smaller images |
| fps | 0 | 15 | matches the data |
| normalization | `None` | `minmax` | see below |
| **validation** | **disabled** | **enabled, deployable mode only** | see below |

### The two additions that matter most

**Validation.** The DROID recipe sets `run_validation=False` and `mode="disabled"`
— and the doc says so explicitly under *Non-goals*: *"Closed-loop / action
evaluation is out of scope."* It trains for a fixed number of steps and stops.

If you want to know when to stop, or which checkpoint to deploy, you have to add
a validation signal yourself.

**Train on a mixture, validate on what you can deploy.** Three modes exist:

| mode | given | predicts | deployable? |
|---|---|---|---|
| `forward_dynamics` | frame + actions | future video | no |
| `inverse_dynamics` | all video | the actions taken | **no** — needs the future |
| `policy` / `wam` | one frame + instruction | video and actions | **yes** |

Training only in the deployable mode collapsed to a near-constant straight-line
action: the shared backbone must reconstruct video *and* predict actions from a
single frame, and a constant minimizes loss on the straight-driving majority.
Training on `mode="joint"` — a random mix of all three per sample — teaches the
same physics through easier problems.

But validation must stay in the deployable mode only. `inverse_dynamics` sees
real future frames, so validating there reports performance the robot can never
achieve, and early stopping would fire on a fiction.

### Two settings that fail silently

**`loss_scale` weights only the video term.**

```
total = fm_loss_vision * loss_scale + fm_loss_action * action_loss_weight
```

Your model tier's base YAML may set `loss_scale=10.0`, tuned for manipulation.
That makes the effective action:video ratio 1:1 instead of 10:1. **Training
converges perfectly normally on the wrong objective.** The only symptom is a
validation loss several times higher than expected. Re-check this on any tier
change.

**`model.config.resolution` is not the data resolution.** It is a lookup key into
`shift = {"256": 3, "480": 5, "720": 10}` for the flow-matching schedule. The data
resolution is set separately on the dataset. Changing this to match your images
silently changes the sampler.

---

## 5 — Launch

**The framework's way** — see [`docs/training.md`](./training.md) — is a shell
launcher, `examples/launch_sft_action_policy_droid.sh`. Fine for one clean run.

**What the roboracer runs used** is a supervisor that calls the trainer directly,
so it can restart the job when it hangs:

```bash
cd /scratch/$USER/cosmos-framework

export BASE_CHECKPOINT_PATH=$PWD/examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$PWD/outputs

setsid nohup python3 train_supervisor.py < /dev/null > /tmp/supervisor.log 2>&1 &
```

Underneath:

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

Those trailing values are the **effective** ones — they beat the TOML and the
experiment file. Two to understand before copying:

- `max_samples_per_batch=96` was tuned for 80 GB cards. On 48 GB start much
  lower and raise it from observed memory, not from this number.
- `max_iter=100000` is a safety ceiling. Early stopping should end the run.

`setsid` matters — plain `nohup` still dies when your SSH session drops.

### Why a supervisor rather than the launcher

| component | what it handles |
|---|---|
| `stall_watchdog.py` | NCCL hangs are **silent** — every GPU sits at 100% with no logs and no timeout. Only a log-progress timeout catches them. |
| `early_stopping_watchdog.py` | Epoch-aggregated patience. Raw per-check validation is too noisy to stop on directly. |
| `prune_roboracer_checkpoints.py` | Keeps best + latest. **Required on shared storage** — checkpoints run tens of GB each. |

> **Size validation against your smallest dataset.**
> `max_val_iter × max_samples_per_batch` must not exceed the number of windows in
> your **smallest** eval split. One dataset is assigned per rank, so a rank
> holding the small one runs out of data and leaves the collective while the
> others wait. It presents as 100% GPU utilization, no logs, no timeout — it
> looks like broken hardware, not a config error.

---

## 6 — Export

Training produces DCP. Serving wants HF safetensors:

```bash
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path <checkpoint_dir>/model \
  --experiment <your_experiment_name> \
  --no-use-ema-weights \
  -o <output_dir>
```

Two things that catch people:

- The flag is `--experiment`, not `--checkpoint.experiment`.
- Point at the **`model/` subdirectory**, not its parent. The format detector
  looks for `*.distcp` at the top level, and training checkpoints nest them one
  level down.

**Then reset the sharding degree.** The exported config carries
`data_parallel_shard_degree=8` from training, and a single-process server cannot
construct an 8-way-sharded model:

```bash
sed -i 's/"data_parallel_shard_degree": 8/"data_parallel_shard_degree": 1/' \
  <output_dir>/config.json
```

---

## Next

Serving, quantization for embedded hardware, and driving the vehicle:
[`roboracer_int4_deployment.md`](./roboracer_int4_deployment.md).

---

## Reading order

**Adapting Cosmos 3 to a new robot:**
1. `docs/action_policy_droid_posttrain.md` — the canonical recipe shape
2. `docs/custom_dataset.md` — how data reaches the model
3. `convert_roboracer_to_lerobot.py` — a worked example of the step the framework skips
4. `action_policy_roboracer_nano.py` — a worked example of a non-arm recipe

**Reproducing this policy:** §3 → §5 → §6.

---

## The five things that cost the most time

1. **The dataset step is entirely yours.** The framework says so; budget for it.
2. **Use a registered action space.** Inventing your own discards the transfer.
3. **Add validation.** The upstream recipe has none, and without it you cannot
   tell when to stop or which checkpoint to keep.
4. **Re-check `loss_scale` on any tier change.** It fails silently.
5. **Split by recording session, not by frame.** Getting this wrong invalidates
   every number you report.
