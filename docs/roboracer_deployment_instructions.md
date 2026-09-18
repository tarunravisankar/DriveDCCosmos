# Cosmos3 and RoboRacer training

Cosmos 3 ships post-training recipes for robot arms (DROID, LIBERO). This
describes what it took to point it at a 1/10-scale car, and marks which steps the
framework handles and which you write yourself.

Written for someone who hasn't used Cosmos 3. Covers training only. For serving
and on-vehicle deployment see
[`roboracer_int4_deployment.md`](./roboracer_int4_deployment.md).

---

## Concepts

**Experiment** — a Python file under `cosmos_framework/configs/base/experiment/`
that registers a named configuration: datasets, which parameters train, losses,
schedules. You refer to it by name (`action_policy_roboracer_nano`). The recipe
lives here.

**TOML** — a file under `examples/toml/sft_config/` that picks an experiment and
sets run-level scalars: precision, GPU count, checkpoint frequency. A strict
schema rejects unknown keys before training starts.

**Config precedence** — `CLI overrides > TOML > experiment .py`. Arguments after
`--` on the launch command win. If a value in the `.py` doesn't match what you
see at runtime, read the launch command.

**DCP** — PyTorch's distributed checkpoint format, a directory of `*.distcp`
shards. Training reads and writes DCP. Serving uses HF safetensors, which you
produce by exporting.

**Two towers** — a reasoner (vision-language: reads the image and instruction)
and a generator (produces future video and actions), joined by cross-attention.
The experiment file decides which parts of which tower train.

---

## Steps

| # | Step | Framework provides | You provide |
|---|---|---|---|
| 1 | Environment | `docs/setup.md` | — |
| 2 | Dataset | nothing | `convert_roboracer_to_lerobot.py` |
| 3 | Base checkpoint | `convert_model_to_dcp` | — |
| 4 | Recipe | DROID experiment as template | your experiment file |
| 5 | Launch | `cosmos_framework.scripts.train` | `train_supervisor.py` |
| 6 | Export | `export_model.py` | — |

Step 2 is most of the work.

---

## 1. Environment

Read [`docs/setup.md`](./setup.md) and
[`docs/environment_variables.md`](./environment_variables.md).

Point the caches at scratch space. They default to your home directory and will
fill its quota.

```bash
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export HF_HOME=/scratch/$USER/.cache/huggingface
```

---

## 2. Dataset

Start with [`docs/action_policy_droid_posttrain.md`](./action_policy_droid_posttrain.md).
Under *Inputs you provide* it says the LeRobot conversion is "run out-of-band
(not yet in this repo)", and the Dataset section says "To be released."

The framework expects a finished LeRobot v3.0 dataset and doesn't tell you how to
build one.

### Format

```
my_dataset/
├── meta/info.json                     episode count, frame count, fps
├── data/chunk-000/*.parquet           per-frame actions and indices
└── videos/<camera>/chunk-000/*.mp4    H.264, required by Cosmos
```

Each parquet row pairs one video frame with the action taken at that moment.

### Converting

`convert_roboracer_to_lerobot.py` turns ROS 2 bags into this layout. Four parts
generalize:

`_resolve_topic_id()` looks up topic IDs by name per bag. ROS assigns them in
recording order, so hardcoding reads the wrong stream without error.

`sync_odom_to_frames()` matches each camera frame to the nearest odometry
reading. The two publish independently and never align.

`compute_actions()` converts absolute pose to relative pose deltas. Absolute
position encodes where the track sits in the room, which the model can't use.

The 9-D `[pos_xyz, rot6d]` action matches Cosmos 3's `av` domain
(`domain_id=1`).

Two choices to copy:

Use a registered action space (`av` for vehicles, `joint_pos` for arms). The
pretrained action pathway only transfers if the format matches.

Represent rotation as rot6d rather than an angle. Angles wrap at ±π and
regression breaks across the discontinuity.

### Supporting scripts

- `convert_all_datasets.sh` — one caption per recording campaign, all splits
- `compute_roboracer_stats.py` — normalization statistics over the full corpus
- `roboracer_bag_split_*.json` — assigns recordings to train, eval, test

Split by recording session, not by frame. At 15 fps consecutive frames are 67 ms
apart and nearly identical, so a frame-level split puts near-duplicates in both
train and test.

See [`docs/custom_dataset.md`](./custom_dataset.md) §3 for how datasets reach the
dataloader, including mixing several by ratio.

---

## 3. Base checkpoint

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o examples/checkpoints/Cosmos3-Nano
```

Check the output contains shards:

```bash
ls examples/checkpoints/Cosmos3-Nano/model/ | head
# __0_0.distcp  __0_1.distcp  __0_2.distcp ...
```

Use a `*-Policy-*` variant if one exists for your tier. Those ship a trained
generation expert and action bridges. The plain base model doesn't, and a
cold-started action pathway is harder to train.

---

## 4. Recipe

Read `docs/action_policy_droid_posttrain.md` §Recipe for the knob table and
[`docs/sft_config.md`](./sft_config.md) for the TOML schema.

Template:
`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py`

Compare against `action_policy_roboracer_nano.py`.

### Keep the framework's fine-tuning strategy

The DROID experiment sets `keys_to_select`, restricting the optimizer to the
generation expert, the timestep embedder, and the modality bridges. Everything
else stays frozen, including the whole vision-language reasoner.

The roboracer recipe uses this list unchanged. A few hours of robot data can't
teach a model to see, and training the reasoner would overwrite pretrained
perception. Freezing it spends your data on control.

`keys_to_skip_loading` and the 5× `lr_multipliers` on the action bridges come
from the same recipe. The pretrained action heads encode the source robot's
action space, so they're discarded and retrained at a higher learning rate.

### What changes

| knob | DROID | RoboRacer | reason |
|---|---|---|---|
| action space | `joint_pos` 8-D | `av` 9-D | car, not arm |
| `use_state` | true | false | no proprioception |
| viewpoint | `concat_view` | single camera | one camera |
| resolution | 480 | 256 | smaller images |
| fps | 0 | 15 | matches the data |
| normalization | `None` | `minmax` | — |
| validation | disabled | enabled, deployable mode only | below |

### Validation

DROID sets `run_validation=False` and `mode="disabled"`. Its doc lists
closed-loop evaluation under *Non-goals*. The recipe trains for a fixed number of
steps and stops.

To know when to stop, or which checkpoint to deploy, add validation yourself.

### Training mode

Three modes exist:

| mode | input | output | deployable |
|---|---|---|---|
| `forward_dynamics` | frame + actions | future video | no |
| `inverse_dynamics` | all video | actions taken | no, needs the future |
| `policy` / `wam` | one frame + instruction | video and actions | yes |

Training only in the deployable mode collapsed to a near-constant straight-line
action. The shared backbone has to reconstruct video and predict actions from one
frame, and a constant minimizes loss on the straight-driving majority.
`mode="joint"` mixes all three per sample and teaches the same dynamics through
easier problems.

Validate in the deployable mode only. `inverse_dynamics` sees real future frames,
so validating there reports performance the robot can't reach and early stopping
fires on it.

### Two settings that fail silently

`loss_scale` weights only the video term:

```
total = fm_loss_vision * loss_scale + fm_loss_action * action_loss_weight
```

Your tier's base YAML may set `loss_scale=10.0`, tuned for manipulation, making
the effective action:video ratio 1:1 instead of 10:1. Training converges normally
on the wrong objective. The only symptom is a validation loss several times
higher than expected. Re-check this whenever you change model tier.

`model.config.resolution` is not the data resolution. It's a key into
`shift = {"256": 3, "480": 5, "720": 10}` for the flow-matching schedule. Data
resolution is set on the dataset. Changing this to match your images changes the
sampler instead.

---

## 5. Launch

[`docs/training.md`](./training.md) documents the shell launcher,
`examples/launch_sft_action_policy_droid.sh`. It works for a single run.

The roboracer runs used a supervisor that calls the trainer directly so it can
restart after a hang:

```bash
cd /scratch/$USER/cosmos-framework

export BASE_CHECKPOINT_PATH=$PWD/examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$PWD/outputs

setsid nohup python3 train_supervisor.py < /dev/null > /tmp/supervisor.log 2>&1 &
```

It runs:

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

Those trailing values override the TOML and the experiment file.

`max_samples_per_batch=96` was tuned for 80 GB cards. On 48 GB start lower and
raise it from observed memory.

`max_iter=100000` is a ceiling. Early stopping ends the run.

Use `setsid`. Plain `nohup` still dies when your SSH session drops.

### Why a supervisor

`stall_watchdog.py` catches silent NCCL hangs. Every GPU sits at 100% with no
logs and no timeout, so only a log-progress timeout detects them.

`early_stopping_watchdog.py` aggregates validation into epochs before comparing.
Raw per-check values are too noisy to stop on.

`prune_roboracer_checkpoints.py` keeps the best and latest checkpoints.
Checkpoints run tens of GB each, so this is required on shared storage.

### Validation sizing

`max_val_iter × max_samples_per_batch` must not exceed the window count of your
smallest eval split. Each rank gets one dataset, so the rank holding the smallest
split runs out of data and leaves the collective while the others wait. It
presents as 100% GPU utilization with no logs and no timeout, which looks like
broken hardware rather than a config error.

---

## 6. Export

Training writes DCP. Serving needs HF safetensors.

```bash
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path <checkpoint_dir>/model \
  --experiment <your_experiment_name> \
  --no-use-ema-weights \
  -o <output_dir>
```

The flag is `--experiment`, not `--checkpoint.experiment`.

Point at the `model/` subdirectory. The format detector looks for `*.distcp` at
the top level and training checkpoints nest them one level down.

Then reset the sharding degree. The exported config carries
`data_parallel_shard_degree=8` from training, and a single-process server can't
build an 8-way-sharded model.

```bash
sed -i 's/"data_parallel_shard_degree": 8/"data_parallel_shard_degree": 1/' \
  <output_dir>/config.json
```

---

## Next

[`roboracer_int4_deployment.md`](./roboracer_int4_deployment.md) covers serving,
compression for embedded hardware, and driving the vehicle.

---

## Reading order

Adapting to a new robot:

1. `docs/action_policy_droid_posttrain.md` — recipe shape
2. `docs/custom_dataset.md` — how data reaches the model
3. `convert_roboracer_to_lerobot.py` — the step the framework skips
4. `action_policy_roboracer_nano.py` — a non-arm recipe

Reproducing this policy: §3, §5, §6.

---

## What costs the most time

1. Building the dataset. The framework doesn't do it.
2. Using a registered action space. Inventing one discards the transfer.
3. Adding validation. The upstream recipe has none.
4. Re-checking `loss_scale` after a tier change. It fails silently.
5. Splitting by recording session. Getting it wrong invalidates your results.
