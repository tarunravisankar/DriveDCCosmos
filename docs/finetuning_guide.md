# Fine-tuning Cosmos 3 into a RoboRacer driving policy

How the `action_policy_roboracer_edge` run works, with pointers to every file
that defines it. 
Stock Cosmos 3 ships manipulation recipes (DROID, LIBERO). Everything here is
what had to be added for a ground vehicle.

---

## What it produces

A policy that maps **one forward-facing RGB frame + one natural-language
instruction** to **32 future motion steps** (2.13 s at 15 Hz), decoded at
deployment into steering curvature and forward velocity.

---

## Prerequisites

| | path / source |
|---|---|
| Base checkpoint | `nvidia/Cosmos3-Edge-Policy-DROID` → DCP via `convert_model_to_dcp` |
| Wan VAE | `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` |
| Datasets | built by `convert_all_datasets.sh` (see `docs/` for the data pipeline) |
| Hardware | 8 GPUs. Run used 8× H100 80 GB; the recipe was sized for 8× A6000 48 GB |

**Why Policy-DROID and not the base Edge model:** Policy-DROID ships a *trained*
generation expert (281 `*_moe_gen` tensors) plus trained action bridges. Starting
from base Edge would require the teacher/student distillation detour that a
cold-started backbone needed. Verified 549 keys on conversion.

Required environment variables:

```bash
export WAN_VAE_PATH=/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/scratch/tarunrav/cosmos-edge/outputs
export BASE_CHECKPOINT_PATH=/scratch/tarunrav/cosmos-edge/examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp
export HF_HOME=/scratch/tarunrav/.cache/huggingface     # keep caches off $HOME
export UV_CACHE_DIR=/scratch/tarunrav/.cache/uv         # ~50 GB quota there
```

---

## The three files that define the run

| file | role |
|---|---|
| `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_roboracer_edge.py` | the recipe — datasets, what trains, losses, schedules |
| `examples/toml/sft_config/action_policy_roboracer_edge.toml` | run-level knobs — parallelism, precision, checkpoint cadence |
| `train_supervisor_edge.py` | launches it, restarts on stalls, prunes checkpoints, early-stops |

Anything in the experiment `.py` is **not** TOML-overridable unless it appears in
the schema whitelist — `validation_iter`, `max_val_iter` and `run_validation` in
particular must be edited in the `.py`.

---

## Reading the experiment config

### Datasets — `_NAV_ROOTS` / `_PEOPLE_ROOTS`

```python
_NAV_ROOTS = {
    "roboracer":        _roots(""),           # orin10 — loop CCW   (31 ep)
    "roboracer_orin13": _roots("_orin13"),    # loop CW             (6 ep)
    "roboracer_orin02": _roots("_orin02"),    # oval CCW            (42 ep)
    "roboracer_orin06": _roots("_orin06"),    # small rectangle CCW (68 ep)
    "roboracer_orin14": _roots("_orin14"),    # square CCW          (9 ep)
}
_PEOPLE_ROOTS = {
    "roboracer_pass_right": _roots("_pass_right"),   # (24 ep)
    "roboracer_pass_left":  _roots("_pass_left"),    # (24 ep)
    "roboracer_wait":       _roots("_wait"),         # (24 ep)
}
```

**Eight datasets**, five navigation plus three social — 228 training episodes,
143,466 frames, ~2.66 hours, **135,942 stride-1 training windows**. Four other
converted nav datasets (orin03/04/05/08) exist but are not in this mix.

`_roots(suffix)` returns `(train_root, eval_root)`. Training takes `roots[0]`,
validation `roots[1]` — so a dataset never appears in both.

### What trains — the whitelist

```python
keys_to_select=[
    "moe_gen",                 # the generation expert (281 tensors)
    "time_embedder",           # diffusion timestep encoder
    "vae2llm", "llm2vae",      # image bridges
    "action2llm", "llm2action",# action bridges
    "action_modality_embed",   # marks action tokens
],
lr=2.0e-04,
lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
```

This is a **whitelist matched as substrings against parameter names**. Anything
not listed is frozen — the entire vision-language understanding expert, the Wan
VAE, and `k_norm_und_for_gen` (deliberately excluded to preserve Edge's
calibrated cross-attention scaling between the two towers).

The rationale: ~2.7 hours of data cannot teach a model to see. Freezing the
understanding expert prevents catastrophic forgetting and spends the data
entirely on learning control.

### What gets discarded

```python
keys_to_skip_loading=[
    "net_ema.", "action2llm", "llm2action",
    "action_modality_embed", "action_pos_embed",
],
```

Policy-DROID's action bridges encode a **7-DoF manipulator** action space. The
car has 2. Those weights are actively wrong, not merely suboptimal — so they are
reinitialized and given the 5× learning-rate multiplier above to catch up.
(`action_pos_embed` does not exist in this architecture; harmless no-op.)

### Training mode vs validation mode

```python
def _train_datasets():
    return {name: dict(ratio=1, dataset=_ds(roots[0], mode="joint",  augment=True))  ...}

def _val_datasets():
    return {name: dict(ratio=1, dataset=_ds(roots[1], mode="wam", augment=False)) ...}
```

**Train `joint`, validate `wam`.** Three modes exist:

| mode | given | predicts |
|---|---|---|
| `forward_dynamics` | frame + actions | future video |
| `inverse_dynamics` | all video | the actions that happened |
| `wam` | **one frame + instruction** | future video and actions |

Only `wam` is deployable — the others need the future. Training on `wam` alone
collapsed to a near-constant straight-line action, because the shared backbone
had to reconstruct video *and* predict actions from one frame, and a constant
minimizes loss on the straight-driving majority. Mixing modes teaches the same
dynamics through easier problems.

Validation runs strictly in `wam`, so early stopping reflects what the car can
actually do.

### Loss weighting — the one that bites

```python
[...]["rectified_flow_training_config"]["loss_scale"]       = 1.0   # video
[...]["rectified_flow_training_config"]["image_loss_scale"] = 1.0
# action_loss_weight = 10.0 inherited from EDGE_MODEL_CONFIG
```

`total = fm_loss_vision * loss_scale + fm_loss_action * action_loss_weight`

`loss_scale` weights **only** the video term. `EDGE_MODEL_CONFIG` inherits
`loss_scale=10.0` from `Cosmos3-Edge.yaml` (tuned for DROID/LIBERO), which makes
the effective action:video ratio **1:1** instead of **10:1**. Training converges
perfectly normally on the wrong objective; the only symptom is validation loss
around 2.13 where <0.4 is expected. Correcting it recovers ~0.318.

**If you port this recipe to another model tier, re-check this value.**

### Resolution — not what it looks like

```python
[...]["model"]["config"]["resolution"] = "720"
```

This is **not** the data resolution (datasets independently produce 256p via
`_ds(resolution="256")`). It is a lookup key into
`shift = {"256": 3, "480": 5, "720": 10}` for the flow-matching schedule.
Setting it to `"256"` silently changes the sampler shift from 10 to 3.

### Validation sizing — deadlock risk

```python
max_val_iter=20,               # in trainer config
max_samples_per_batch=8,       # in dataloader_val
```

`max_val_iter × max_samples_per_batch` must not exceed the **smallest** eval
split. With 20 × 32 = 640 against `pass_right`'s 205 windows, some ranks exhaust
their data and exit the collective early while others wait — every GPU pins at
100% with no log output and no timeout. 20 × 8 = 160 fits.

---

## Launching

The supervisor is the entry point — it wraps `torchrun`, restarts on stalls,
prunes checkpoints, and early-stops.

```bash
cd /scratch/tarunrav/cosmos-edge
setsid nohup python3 train_supervisor_edge.py < /dev/null > /tmp/edge_supervisor.log 2>&1 &
```

`setsid` matters: plain `nohup` still dies when the SSH pipe breaks.

What it runs underneath:

```python
TRAIN_CMD = [
    f"{VENV}/bin/torchrun", "--nproc_per_node=8",
    "-m", "cosmos_framework.scripts.train",
    "--sft-toml", "examples/toml/sft_config/action_policy_roboracer_edge.toml",
    "--",
    f"job.name={JOB_NAME}",
    "trainer.max_iter=515000",            # ceiling only; early stopping decides
    "dataloader_train.max_samples_per_batch=8",
    "checkpoint.save_iter=200",           # matches validation_iter
]
```

`save_iter == validation_iter` so every validated iteration has a checkpoint on
disk to roll back to.

### Supervisor settings

```python
MAX_RELAUNCHES              = 20     # then stop and require a human
STALL_TIMEOUT_S             = 180    # no log progress -> assume wedged
WARMUP_TIMEOUT_S            = 900    # longer allowance during startup
EPOCH_ITERS                 = 605
EARLY_STOP_PATIENCE_EPOCHS  = 20
EARLY_STOP_WARMUP_EPOCHS    = 5      # never stop before this
EARLY_STOP_EPSILON          = 0.001
```

Early stopping is **epoch-aggregated**: an inner loop buckets raw validation
checks into epochs, an outer loop compares epoch means. Raw per-check values are
too noisy to stop on directly.

Checkpoint retention keeps **best + latest only** — mandatory on a shared
filesystem. The best is additionally hard-linked into `protected/`, so it
survives even if the pruner falls back to "two most recent" after a lost log.

---

## Watching a run

```bash
grep -E "Validation loss" /tmp/edge_supervisor.log | tail -20
grep -E "relaunch|early_stop|ckpt_prune" /tmp/edge_supervisor.log | tail
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Healthy: validation loss falling, checkpoints every 200 iterations, no relaunches.

If the launching SSH session dies and stdout is lost, read validation history
directly from wandb's offline binary log via
`wandb.sdk.internal.datastore.DataStore` — poll only the newest `offline-run-*`
directory, since merging across restarts by raw iteration number produces stale
/ fresh collisions.

---

## Outputs

```
outputs/cosmos3_action/action_sft/<job.name>/
├── checkpoints/iter_NNNNNNNNN/     model/ optim/ scheduler/ trainer/
├── protected/iter_..._valX.XXXX/   hard-linked best
└── wandb/offline-run-*/
```

The reference run: **18.3 hours** on 8× H100, best at **iteration 2400,
validation 0.3159**. Training continued to 6800 and never improved, so 2400 is
the deployed checkpoint.

To serve it, point `--checkpoint-path` at the **`model/` subdirectory**, not its
parent — `CheckpointType.from_path()` looks for `*.distcp` at the top level and a
training checkpoint nests them one level down.

---

## Pitfalls

| symptom | cause |
|---|---|
| Validation ~2.1 instead of <0.4 | `loss_scale` inherited as 10.0 — see above |
| All GPUs 100%, no logs, no timeout | `max_val_iter × batch` exceeds smallest eval split |
| Model predicts near-constant actions | training in `wam` only instead of `joint` |
| Great validation, useless on the car | validating in `inverse_dynamics` — it sees future frames |
| Disk fills mid-run | checkpoint pruning not running alongside |
| Job dies when SSH closes | used `nohup` without `setsid` |
| `FileNotFoundError: 'uv'` | `/home/<user>/.local/bin` not on `PATH` |
| Export won't load single-process | `data_parallel_shard_degree` still 8 in `config.json`; reset to 1 |

---

## Adapting to another vehicle

1. **Action space.** `roboracer_dataset.py` declares `EMBODIMENT_TYPE="av"`
   (`domain_id=1`, 9-D). A different embodiment needs its own domain mapping.
2. **Recompute normalization stats** over the full corpus with
   `compute_roboracer_stats.py` and version them with the checkpoint. Expanding
   the corpus invalidates old stats silently — mean forward velocity shifted 39%
   when stop-and-wait episodes entered the mixture.
3. **Reinitialize the action bridges** unless the source checkpoint's action
   space matches yours.
4. **Re-check `loss_scale`** against the new base config.
5. **Design captions as minimal pairs**, and make sure the visual observation is
   genuinely insufficient — otherwise the model learns to ignore language.
