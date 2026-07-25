# CLAUDE.md — RoboRacer Action Policy Project

This file is project-specific context for the **roboracer action-policy** effort within
cosmos-framework (branch `roboracer-action-policy`). See [AGENTS.md](./AGENTS.md) first for
general framework navigation — this file covers only roboracer-specific state, gotchas, and
history that isn't discoverable by reading code alone.

## Research goal

Investigate whether a foundation-model-based driving policy (NVIDIA Cosmos3-Nano) can
generalize to **mapless, socially-aware navigation** — including scenarios/instructions never
seen in training — while being compressed via **knowledge distillation** and **INT4
quantization** for deployment on an **8GB Jetson Orin Nano** (the RoboRacer 1/10-scale AV
platform's onboard compute). No HD maps, no metric localization — vision + text conditioning
only.

## Machines (read this before running anything)

| Machine | Role | Notes |
|---|---|---|
| **robolang** | Where the Bash tool actually executes | `hostname` to confirm. Has its own `/scratch/tarunrav/`. |
| **robolidar** | 8-10× RTX A6000 48GB, shared multi-user | Separate `/scratch` from robolang — nothing is shared automatically. All roboracer training/serving happens here via `ssh robolidar.csres.utexas.edu`. |
| **TACC Lonestar6** | H100s, used for earlier distillation attempts | Apptainer container (`cosmos_tacc.sif`), not currently the active path. |
| **The car** | Jetson Orin Nano 8GB | Final deployment target. JetPack version unknown as of writing — check `cat /etc/nv_tegra_release` on-device before assuming a bitsandbytes build matches. |

**SSH lands in `$HOME`, not the repo root.** Every remote command needs an explicit `cd` or
absolute paths — this has caused repeated silent "command not found" failures. Always use
`/scratch/tarunrav/cosmos-framework/.venv/bin/python3` (full path), not `.venv/bin/python3`.

## Disk quota gotchas (robolidar)

- `$HOME` has a **~50GB quota**. `uv`/`pip`/`HF` caches default there and *will* blow it.
  Always set `UV_CACHE_DIR=/scratch/tarunrav/.cache/uv` and `HF_HOME=/scratch/tarunrav/.cache/huggingface`.
- `/scratch` is an **18TB filesystem shared by every user on the machine** — it has filled to
  100% before (once from unpruned checkpoints, once from unrelated causes). Any long training
  run **must** run `prune_roboracer_checkpoints.py` alongside it (keeps best + latest only) or
  it will eventually crash the run and potentially affect other users.
- `~/.local/share/containers` (rootless Podman storage) has silently eaten tens of GB of `$HOME`
  quota from stale, unrelated containers. If `$HOME` quota errors show up unexpectedly, check
  this first. To remove: `podman unshare umount <path>/overlay; podman unshare rm -rf <path>/storage`
  (must be done inside `podman unshare` — a normal `rm` hits UID-remap permission errors).

## Current model artifacts

All under `/scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/`:

| Artifact | Path | Notes |
|---|---|---|
| Teacher (7B, fine-tuned) | `action_policy_roboracer_repro_v10/checkpoints/iter_000007200` | DCP format. Full bf16 SFT of Cosmos3-Nano on roboracer data. |
| Distilled student (4B), best | `action_policy_roboracer_distill_v1/checkpoints/iter_000004200` | DCP format. Val loss 0.378 (epoch 16), selected by early-stopping watchdog. **Use this one, not "latest".** |
| HF export, bf16 | `action_policy_roboracer_distill_v1/model_best_4200/` | 16GB. Exported via `export_model.py`. |
| **HF export, INT4 quantized** | `action_policy_roboracer_distill_v1/model_best_4200_int4/` | **4.3GB — the deployable artifact.** bitsandbytes NF4. |
| WAN VAE, fp16 | `/scratch/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE_fp16.pth` | 1.4GB (halved from fp32 2.8GB original). Combined footprint with the int4 model: ~5.7GB, fits the 8GB Jetson budget with margin. |

**Validated end-to-end**: the int4 checkpoint + fp16 VAE load and serve correctly via
`action_policy_server_roboracer.py` on an x86_64 A6000, producing sane curvature/velocity
output. **Not yet validated**: on actual Jetson Orin Nano hardware (bitsandbytes ARM64/`sm_87`
compatibility is an open risk — see below).

## Key training discovery: joint-dynamics fixes action collapse

Versions v2–v6 of the SFT recipe collapsed to predicting near-constant straight-line actions.
Root cause: `mode="policy"` forces the shared `moe_gen` backbone to simultaneously reconstruct
almost the entire video AND predict actions — a competing objective that action-loss alone
couldn't overcome. `mode="inverse_dynamics"` (v7) looked great in validation metrics but is
**non-deployable** — it conditions on real future frames that don't exist yet on a live car.
Fix (v8+, still in use): train with `mode="joint"` (mixes forward/inverse/policy dynamics per
sample), but validate/early-stop strictly in `mode="policy"` so the stopping signal reflects
actually-deployable performance.

## Distillation architecture

`DistillOmniMoTModel` (`cosmos_framework/model/vfm/distill_omni_mot_model.py`): frozen 7B
teacher + trainable 4B student (Qwen3-VL-4B-Instruct backbone), MSE loss between student and
teacher action predictions added to the student's own flow-matching loss. Confirmed genuinely
working: `distill_loss_action` converges to ~0.0018–0.0022 (student closely tracking teacher).

Two bugs already fixed here — don't reintroduce them:
- `__init__` must set `self._teacher_checkpoint_path` etc. **before** calling
  `super().__init__(config)`, because the parent's `__init__` calls `self.set_up_model()`
  internally (dynamic dispatch resolves to the override), which needs those attributes already
  set.
- `_load_teacher()` must explicitly pass `experiment_opts=[f"model.config.tokenizer.vae_path={self.config.tokenizer.vae_path}"]`
  to `load_model_from_checkpoint()` — that function composes a **fresh** Hydra config for the
  teacher's experiment name, which does NOT inherit the current run's env-var/TOML overrides.

## Deployment pipeline bugs (all fixed this session — don't reintroduce)

Six real bugs were found getting the quantized checkpoint to actually serve correctly. If any
of these files get reverted or merged from an older branch, re-check:

1. **`export_model.py`** `_coerce_to_base_model`: was using `"OmniMoTModel" in target`
   (substring match) — `"DistillOmniMoTModel"` contains that substring too, so it incorrectly
   skipped the distillation→base-model rewrite. Fixed to `target.endswith(".OmniMoTModel")`.
2. **`export_model.py`** same function: wasn't stripping top-level distill-only kwargs
   (`teacher_checkpoint_path`, `distill_alpha`, `teacher_experiment_name`) from `model_dict`
   itself (only from the nested `config`), causing `OmniMoTModel.__init__()` to choke on
   unexpected kwargs. Fixed by deleting all top-level keys except `_target_`/`config`.
3. **`qwen3_vl.py`** `Qwen3VLPreTrainedModel._init_weights`: missing a default value for
   `buffer_device`, which broke standard HF `from_pretrained()`'s internal weight-init call
   (that calls `_init_weights(module)` with one arg, not the framework's own
   `functools.partial(..., buffer_device=...)` convention). Fixed by adding `= None` default.
4. **`distributed.py`** `sync_model_states`: attempted real collective ops (allgather/broadcast)
   even for a single-process (world_size=1) inference server, which crashed with "no meta
   kernel" since the model was still mid-construction on the meta device. Fixed with an early
   return when `world_size <= 1` (syncing rank 0 to itself is a no-op by definition).
5. **Exported checkpoint `config.json`**: still carries `data_parallel_shard_degree=8` baked in
   from the original 8-GPU training run. A single-process server trying to construct an
   8-way-sharded model crashes. **Any newly-exported checkpoint needs this manually reset to 1**
   before single-process serving (`sed -i 's/"data_parallel_shard_degree": 8/"data_parallel_shard_degree": 1/'`).
6. **`wan2pt2_vae_4x16x16.py`** VAE loading: `model.load_state_dict(ckpt, assign=True)` (the
   documented pattern for materializing meta-device models) empirically does **not** materialize
   parameters correctly for this nested module structure — verified with 100% key coverage,
   zero missing/unexpected keys reported, yet all params remained on `meta` after the call
   returned. Fixed by forcing `model.to_empty(device=device)` first, then loading with
   `assign=False` (standard copy-based semantics) instead of relying on `assign=True`.

Also: **`action_policy_server_roboracer.py`** previously hardcoded `CheckpointType.DCP`. It now
auto-detects via `CheckpointType.from_path()` and, for HF-format checkpoints, loads via plain
`Cosmos3OmniModel.from_pretrained()` directly rather than `OmniInference.create()` — the latter's
HF-checkpoint loading path (`torch.distributed.checkpoint.hf_storage` bridge) raises a spurious
"missing key: lm_head.weight" even though the key demonstrably exists in the checkpoint's own
safetensors index. Root cause of *that* one was never isolated — the direct-`from_pretrained`
path was faster to get working and is what `quantize_int4.py`-style scripts already used
successfully.

## Jetson deployment: open risk

Standard PyPI bitsandbytes aarch64 wheels do **not** support Jetson Orin's `sm_87` compute
capability (they target sm75/80/90). Options: build from source on-device targeting `sm_87`
(~6 min, community-validated), or use Jetson AI Lab's prebuilt index
(`pypi.jetson-ai-lab.io/jp6/cu126/bitsandbytes/`) if the JetPack/CUDA version matches. Check the
car's actual JetPack version before assuming either path works.

## Training infrastructure (reusable patterns)

- `stall_watchdog.py` — kills+relaunches on silent NCCL hangs.
- `train_supervisor.py` — auto-relaunch wrapper around the above.
- `early_stopping_watchdog.py` — epoch-aggregated patience-based stopping. Parses
  `"Validation loss (iteration N): X.XXX"` lines from a log file. Two-loop design (inner buckets
  raw checks into epochs, outer compares epoch-means) per explicit prior guidance — don't
  simplify back to single-loop raw-check comparison.
- `prune_roboracer_checkpoints.py` — keeps best-epoch + latest checkpoint only. **Must run
  alongside any long training job** given the shared-disk constraint above.
- **If a training job's own stdout pipe dies** (e.g. the launching SSH session ends), don't try
  to resurrect it — read validation-loss history directly from wandb's offline binary log via
  `wandb.sdk.internal.datastore.DataStore` (works even while the log is being actively written).
  Poll only the *latest* `offline-run-*` directory if the job may have restarted with a reset
  iteration counter — merging history across restarts by raw iteration number produces
  misleading stale/fresh collisions.
- All background/detached processes on robolidar should use `setsid nohup ... &` (not just
  `nohup`) — plain `nohup` still dies when the SSH session's pipe breaks; `setsid` fully detaches
  from the controlling terminal.

## Full paper/report material

An IEEE-style paper draft describing this project (`DriveDC`/`[NAME]` platform) went through
extensive fact-checking this session — an earlier AI-authored draft had fabricated latency
numbers, fabricated success-rate percentages, and mischaracterized the goal-conditioning system
as multimodal image-based rather than text/direction-token-based. If resuming paper work, verify
every quantitative claim against actual measured logs before trusting a prior draft.
