# RoboRacer Action-Policy Fine-Tuning — Handoff Summary

## Goal
Fine-tune Cosmos3-Nano (an OmniMoT VLA model) into an action policy for a 1/10-scale
RC car ("RoboRacer", UT Austin), using a single front camera + 9D ego-pose-delta
actions. Original symptom: the fine-tuned model predicted near-constant
straight-driving regardless of input — it never learned to turn.

**Code**: https://github.com/tarunravisankar/DriveDCCosmos/tree/roboracer-action-policy
(branch on a personal fork, pushed from `~/cosmos-framework` on `robolidar`)

## Current state (as of this handoff)
Training is running on **`robolang.csres.utexas.edu`** (8x H100 80GB), NOT the
original machine (`robolidar`, 10x A6000 48GB, shared with other users —
abandoned due to resource contention crashes). SSH: `ssh tarunrav@robolang.csres.utexas.edu`.

- Job name: `action_policy_roboracer_repro_v8`, resuming from checkpoint `iter_000002800`.
- Config: `mode="joint"` for training, `mode="policy"` for validation (see "Key
  finding #3" below for why these differ).
- `max_samples_per_batch=96` (global batch 768), tuned for ~94% GPU memory
  utilization on the H100s. `iters_per_epoch≈43` at this batch size.
- `max_iter=10000` (~232 epochs) — a safety-net ceiling; early stopping
  (20-epoch patience, see below) should fire before this in practice.
- Output dir: `/scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_repro_v8/`
- Datasets: `ROBORACER_TRAIN_ROOT=/scratch/tarunrav/roboracer_lerobot_train`,
  `ROBORACER_EVAL_ROOT=/scratch/tarunrav/roboracer_lerobot_eval`, test split at
  `/scratch/tarunrav/roboracer_lerobot_test`.
- 5 background processes running (all via `setsid nohup ... &`, with
  `loginctl enable-linger tarunrav` enabled on robolang — see "Infra gotcha" below):
  training, `early_stopping_watchdog.py`, `prune_roboracer_checkpoints.py`,
  a wandb-sync loop, `roboracer_epoch_summary.py`.
- wandb run: https://wandb.ai/tarun-vidyut-university-of-texas-at-austin/cosmos3_action/runs/4qjy09yw
  (offline mode locally, autosynced every 60s).

## Chronology of root causes found (each verified, not assumed)

1. **Action normalization was using quantile clipping** — real turns got clipped
   outside `[-1,1]` since turning frames are a minority by raw count. Fixed:
   train-split-only **minmax** normalization (`RoboracerDataset`,
   `action_normalization="minmax"`).
2. **Framework bugs** (now fixed, general to any OmniMoT action recipe, not
   roboracer-specific):
   - `OmniMoTModel.validation_step` was an unimplemented stub (`pass`) —
     implemented to call `training_step`.
   - `JointDataLoader`-family validation dataloaders build their iterators once
     in `__init__` and never reset — silently exhausted (NaN loss) after one
     pass. Fixed in `trainer.validate()` to reset per call.
   - `wandb_log_eval.py` crashed on packed multi-sample batches (`dataset_name`
     comes back as a list, not a scalar).
3. **THE major finding**: `mode="policy"` (the dataset's old default) forces the
   model to *simultaneously* reconstruct almost the entire video (only frame 0 is
   clean conditioning) **and** predict the action sequence — a large competing
   video-reconstruction objective sharing the same fine-tuned backbone
   (`moe_gen`) as action prediction. Confirmed via `docs/inference.md`'s official
   mode table (`policy` outputs "predicted action sequence **+ future visual
   rollout**" — by design, not a bug) and via empirical correlation testing that
   ruled out other hypotheses (inference-sampling artifact, sign-inversion bug —
   see below).
   - Switching to `mode="inverse_dynamics"` (conditions on the REAL future video
     frames, zero video-reconstruction loss) dramatically improved predictions
     (val loss 0.295 vs ~0.45 for policy mode; turn predictions visually tracked
     ground truth for the first time). **But this is NOT deployable** —
     inverse_dynamics requires future frames that don't exist yet on a live car.
     This was a real "aha" that took a while to recognize: the model was doing
     visual-odometry-style "read the motion off the footage," not policy
     prediction.
   - Settled on **`mode="joint"`** for training (uniformly mixes
     forward_dynamics/inverse_dynamics/policy per sample,
     `base_dataset.py::_choose_mode`, `_MODE_CHOICES`) — matches the reference
     DROID recipe's own dataset default (`droid_lerobot_dataset.py`), which
     never uses "policy" alone either. Validation is deliberately kept at
     **`mode="policy"` only** (not "joint"), so the val loss / early-stopping
     decision honestly reflects the real deployable task.
   - Researched externally (NVIDIA's "Cosmos Policy" paper/cookbook) to confirm
     this general strategy (mixing policy training with dynamics-prediction
     auxiliary tasks) has real published precedent — but no source, including
     DROID's own docs, claims this guarantees real closed-loop deployment
     success; that still has to be tested on hardware.
4. **Two external hypotheses checked and one ruled out**:
   - Sign-convention claim ("domain=av pretraining expects positive yaw=left,
     our data might be flipped") — investigated DROID's actual domain_id (8,
     not 1/"av", so it couldn't have established an av-domain convention
     either way), then **verified empirically against real footage**: at the
     single sharpest turn in the entire train split, the user confirmed by eye
     the car is turning LEFT, and `rot_1` there is positive. No sign flip
     needed. (My own earlier visual read of a blurry thumbnail had wrongly
     suggested the opposite — corrected by checking a clearer, stronger
     example.)
   - Per-channel loss dilution: confirmed real (3 of 9 raw action channels are
     always exactly 0 for this planar vehicle, diluting the aggregate
     flow-matching loss) but didn't pursue a fix (would require modifying
     shared `flow_matching.py` loss code) since the mode fix was higher-impact.
5. **Oversampling experiments (v3-v6) — ultimately set aside.** Tried discrete
   curvature-based tiers (1x/2x/5x, then 1x/5x/20x reaching 93% turning by
   weighted count) — neither fixed turning under `mode="policy"`. Also ported
   av_imitation's continuous Gaussian-importance-weighting scheme but found it
   empirically much milder than expected (only 1.22x growth, ~50/50 balance) —
   the real issue turned out to be the mode/competing-objective problem above,
   not class imbalance. Per the professor's "isolate variables" guidance,
   oversampling is currently OFF (`oversample_turns=False`) so the mode fix's
   effect isn't confounded with it. Could be revisited later.

## Key files
- `cosmos_framework/data/vfm/action/datasets/roboracer_dataset.py` — the
  dataset class. `mode` default is `"policy"` (the safe/deployable default);
  the experiment config explicitly overrides train/val modes separately.
- `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_roboracer_nano.py`
  — the registered experiment (dataloader_train mode="joint",
  dataloader_val mode="policy", action_loss_weight, optimizer keys_to_select, etc).
- `examples/toml/sft_config/action_policy_roboracer_repro.toml` — the
  TOML that sets `max_iter`, `max_samples_per_batch`, job name, scheduler
  cycle_lengths. **Read the header comments — they document the full v2→v8
  history inline.**
- `convert_roboracer_to_lerobot.py` — ROS2 bag → LeRobot v3 conversion
  (quaternion→rot6d math, `compute_pose_delta_9d`).
- `check_roboracer_predictions.py` — the qualitative test script. Loads a
  checkpoint, runs inference on held-out test windows, denormalizes, and
  produces: trajectory plots, curvature/velocity time-series plots,
  rotation/angular-velocity time-series plots, and a frame+arrow visualization
  (ground truth vs. predicted heading/speed drawn on the real camera frame).
  **Always test in `mode="policy"`** (already set as default in this script) —
  testing in inverse_dynamics gives misleadingly good results, as learned above.
- `early_stopping_watchdog.py` — two-loop (epoch-aggregated) early stopping:
  inner loop buckets raw validation checks into epochs, outer loop runs the
  actual best/patience comparison once per *completed* epoch (epoch-mean
  loss), not on raw sub-epoch checks. `--min-delta` (epsilon), `--patience-epochs`,
  `--min-epoch` (floor before it's allowed to fire) are all configurable.
  Per the professor: epsilon-tolerant (flat-within-epsilon counts as
  non-improvement, not just literal increases), epochs-based patience (not
  raw iteration counts).
- `prune_roboracer_checkpoints.py` — checkpoint janitor; keeps only the
  best-epoch checkpoint (matching the watchdog's own epoch-aggregation logic,
  NOT just the single lowest raw check — these were initially inconsistent,
  causing a real best-checkpoint to get deleted once; now fixed to match) plus
  the latest checkpoint. Each checkpoint is ~80GB+ for this model size — this
  script is essential on disk-constrained shared filesystems.
- `roboracer_epoch_summary.py` — clean "Epoch X Train Loss Y Val Loss Z" log
  lines (av_imitation-style), plus marks the wandb run's status
  (`roboracer-status:early_stopped`/`completed_max_iter`/`crashed`) via the
  wandb API once the run reaches a terminal state. Read-only with respect to
  training (never sends signals) — deliberately kept side-effect-free after an
  earlier mistake where a janitor process variant deleted the wrong checkpoint.
- `roboracer_bag_split.json` — the 31 train / 4 test / 5 eval bag split
  (1 bad bag excluded — odometry glitches found via `scan_roboracer_bag_quality.py`).

## Infra gotchas hit this session (useful to know)
- **Disk-full crashes**: each DCP checkpoint is ~80GB+; a shared filesystem can
  fill up fast. The janitor script (above) exists specifically to prevent this.
  Already happened once (corrupted a checkpoint mid-write) before the janitor
  existed.
- **Stale-process resource contention**: `robolidar` is a shared multi-tenant
  GPU box; another user's job once consumed enough memory to trigger a
  SIGKILL on our training via the kernel OOM killer. This is why we migrated
  to `robolang` (mostly dedicated to this group, though another user's vLLM
  job briefly occupied it too — coordinate before assuming GPUs are free).
- **systemd session-kill**: on `robolang`, processes launched with plain
  `nohup ... & disown` got killed when the launching SSH/Claude-Code session
  ended, despite disown — likely systemd's `KillUserProcesses` behavior. Fixed
  by running `loginctl enable-linger tarunrav` on robolang (now enabled) and
  using `setsid nohup ... < /dev/null &` for new background launches (fully
  detaches from any controlling session). If background jobs mysteriously die
  again after an SSH session ends, check `loginctl show-user <user> -p Linger`.
- **PID confusion**: `nohup CMD & echo $!` gives the PID of the actual `CMD`
  process when invoked directly over SSH, but gives a *wrapper* PID when
  invoked through some local tool-harness layers (seen on `robolidar`) — always
  verify with `ps aux | grep torchrun` before trusting a PID for the watchdog.
- **rsync `-z` (compression) on model checkpoints is counterproductive** — the
  data is already-dense binary floats, so gzip just burns CPU without shrinking
  size. Use plain `rsync -a` (no `-z`) for checkpoint transfers, and add
  `--partial` so an interrupted transfer can resume instead of restarting from
  zero.
- **`uv sync` can hit a home-directory disk quota** (separate from `/scratch`
  capacity) on shared university machines — set `UV_CACHE_DIR=/scratch/...`
  and put the whole repo (incl. `.venv`) under `/scratch` with a `~` symlink if
  the home quota is small (50GB hit this on `robolang`).

## What's NOT yet done / open questions for the next session
1. **Has not yet been re-verified with `check_roboracer_predictions.py`
   (mode="policy") against a `joint`-mode-trained checkpoint.** This is the
   actual test that matters — loss numbers alone have repeatedly been
   misleading this session. Do this as soon as there's a meaningful checkpoint
   (a few more epochs in).
2. **Live deployment to the actual car has NOT been attempted yet.** Groundwork
   done: confirmed `/ackermann_curvature_drive`
   (`amrl_msgs/msg/AckermannCurvatureDriveMsg`, fields `velocity` m/s
   forward-positive + `curvature` 1/m left-positive — matches our existing
   `to_curvature_velocity()` output convention exactly, no conversion needed),
   confirmed network path (`robolang`/`robolidar` can reach the car's Orin
   directly, no SSH tunnel needed), confirmed the car's own safety mechanisms
   (`/autonomy_enabler`, `/override_active` Bool topics in
   `ut_automata/vesc_driver.cpp`; instant joystick override past a small
   deadzone; a 0.5s message-timeout watchdog that auto-stops if commands stop
   arriving). Still needed: build the inference server (there's a template at
   `cosmos_framework/scripts/action_policy_server_robolab.py`, built for a
   different robot/arm — would need a roboracer-specific version), a
   chunk-buffering client design (model produces 32-step/~2.1s chunks but the
   car's 0.5s timeout means the client must continuously republish at ~15Hz
   from a buffer, not request-response per chunk), and a dry-run mode (print
   predictions, don't actually publish) before ever letting it actually drive.
3. **Per-channel loss dilution** (3 of 9 action channels always 0) — diagnosed,
   not fixed. Would need to modify the shared `flow_matching.py` loss function.
4. **Oversampling** — currently off. Once the mode fix's effect is understood
   in isolation, could revisit with the continuous (Gaussian-importance-weighted)
   scheme rather than discrete tiers.
