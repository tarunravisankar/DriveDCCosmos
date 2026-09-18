# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_roboracer_nano`` — Cosmos3-Nano RoboRacer action policy SFT.

Fine-tunes Cosmos3-Nano on the UT Austin RoboRacer 1/10th-scale AV platform
using single-camera video + 9D AV ego-pose actions (domain_id=1).

12 datasets combined via PackingDataLoader (ratio=1 each = equal sampling):
  Navigation (9):
    orin10  — counter-clockwise large indoor loop       (31 train bags)
    orin13  — clockwise large indoor loop               ( 6 train bags)
    orin02  — counter-clockwise large oval              (42 train bags)
    orin03  — counter-clockwise circular                ( 4 train bags)
    orin04  — counter-clockwise small rectangle         ( 9 train bags)
    orin05  — counter-clockwise medium rectangle        ( 2 train bags)
    orin06  — counter-clockwise small rectangle         (68 train bags, same caption as orin04)
    orin08  — clockwise small rectangle                 ( 7 train bags)
    orin14  — counter-clockwise square                  ( 9 train bags)
  Social navigation (3):
    pass_right — pass the person on the right           (24 train bags)
    pass_left  — pass the person on the left            (24 train bags)
    wait       — wait for the person to pass            (24 train bags)

All frames have a 5s subgoal dot baked in (convert_all_datasets.sh). The dot
projects the robot's odom position 5 seconds ahead onto the camera frame so
the model learns to steer toward the dot — enabling mapless goal-conditioned
navigation at intersections (dot on right → turn right, etc.).

Each dataset's caption is distinct per track shape+direction so the model can
condition on which track type it's navigating. orin04/orin06 share a caption
(same shape, same direction, different physical room — correct behavior).

roboracer_stats.json must be regenerated from ALL train roots before training:
    python compute_roboracer_stats.py --train-root \\
        /scratch/tarunrav/roboracer_lerobot_train \\
        /scratch/tarunrav/roboracer_lerobot_train_orin13 \\
        /scratch/tarunrav/roboracer_lerobot_train_orin02 \\
        ... (all 12 train roots)

Usage (1 node, 8 GPU)::

    BASE_CHECKPOINT_PATH=/scratch/tarunrav/cosmos-framework/examples/checkpoints/Cosmos3-Nano \\
    WAN_VAE_PATH=$HOME/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth \\
    IMAGINAIRE_OUTPUT_ROOT=/scratch/tarunrav/cosmos-framework/outputs \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_roboracer_repro.toml

Place this file at:
    cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_roboracer_nano.py
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.vfm.action.datasets.roboracer_dataset import (
    get_action_roboracer_sft_dataset,
    roboracer_worker_init_fn,
)

cs = ConfigStore.instance()

# ── Dataset root paths (all produced by convert_all_datasets.sh) ─────────────
_S = "/scratch/tarunrav"

# Returns a (train_root, eval_root) tuple for a given suffix
def _roots(suffix): return f"{_S}/roboracer_lerobot_train{suffix}", f"{_S}/roboracer_lerobot_eval{suffix}"

_NAV_ROOTS = {
    "roboracer":        _roots(""),           # orin10 — large indoor loop CCW (31 ep)
    "roboracer_orin13": _roots("_orin13"),    # orin13 — large indoor loop CW  (6 ep)
    "roboracer_orin02": _roots("_orin02"),    # large oval CCW                  (42 ep)
    "roboracer_orin06": _roots("_orin06"),    # small rectangle CCW             (68 ep)
    "roboracer_orin14": _roots("_orin14"),    # square CCW                      (9 ep)
    # orin03/orin04/orin08 dropped to make room for people datasets (8-GPU limit)
}
_PEOPLE_ROOTS = {
    "roboracer_pass_right": _roots("_pass_right"),  # pass person on right (24 ep) — re-encoded with -g 30
    "roboracer_pass_left":  _roots("_pass_left"),   # pass person on left  (24 ep) — re-encoded with -g 30
    "roboracer_wait":       _roots("_wait"),         # wait for person      (24 ep) — re-encoded with -g 30
}


def _ds(root, mode, augment):
    """Build a single get_action_roboracer_sft_dataset LazyCall entry."""
    return L(get_action_roboracer_sft_dataset)(
        root=root,
        fps=15.0,
        chunk_length=32,
        mode=mode,
        action_normalization="minmax",
        use_image_augmentation=augment,
        oversample_turns=False,
        resolution="256",
        max_action_dim="${model.config.max_action_dim}",
        cfg_dropout_rate=0.1 if augment else 0.0,
        tokenizer_config="${model.config.vlm_config.tokenizer}",
    )


def _train_datasets():
    return {
        name: dict(ratio=1, dataset=_ds(roots[0], mode="joint", augment=True))
        for name, roots in {**_NAV_ROOTS, **_PEOPLE_ROOTS}.items()
    }


def _val_datasets():
    return {
        name: dict(ratio=1, dataset=_ds(roots[1], mode="policy", augment=False))
        for name, roots in {**_NAV_ROOTS, **_PEOPLE_ROOTS}.items()
    }


action_policy_roboracer_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # Match DROID recipe optimizer (FusedAdam fp32 master weights)
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_roboracer_nano",
            # Offline so curves are visualizable (wandb local dashboard / synced
            # later) without needing network/login during the training run.
            wandb_mode="offline",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Train generation + action heads only (same as DROID recipe)
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,
            lr_multipliers={
                # Higher LR for action heads — they init fresh from base checkpoint
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # overridden by TOML
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # overridden by TOML
            # Fast early stopping needs a frequent, capped validation signal:
            # max_val_iter caps each validation pass to ~20 batches (full eval
            # split is ~5390 frames / chunk32 ≈ 160 windows / batch~8 ≈ 20
            # batches) so it stays cheap enough to run often.
            max_val_iter=20,
            run_validation=True,
            run_validation_on_start=True,  # baseline point before any fine-tuning
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            # 200, not 50: at a ~25k-iteration ceiling (50-epoch safety net, see
            # session notes), checkpoint writes (~190s each, ~83GB) would otherwise
            # dominate wall-clock time. 200 still gives fine-enough early-stopping
            # resolution relative to the much longer run.
            validation_iter=200,  # overridden by TOML if a different cadence is needed
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip EMA and action heads → init fresh from Cosmos3-Nano base
            # (base has AV pretraining priors but not roboracer-specific action heads)
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # set via TOML env var BASE_CHECKPOINT_PATH
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_roboracer",
            # Count-based batch — start conservative for 48GB A6000s
            # (DROID uses 128 on H200 80GB; scale down proportionally)
            max_samples_per_batch=32,
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                # Was True: with no worker_init_fn, persistent workers' stdlib
                # `random` state (used by mode="joint"'s _choose_mode()) never
                # got reseeded across the run, instead silently carrying over
                # every time the trainer's outer loop calls iter() again at
                # each ~200-iter validation/checkpoint boundary. Suspected
                # (not yet 100% confirmed) cause of a deterministic hang at
                # exactly +144 iterations past every such boundary (v8,
                # 2944/3144/3344). False here forces workers to be killed and
                # freshly spawned (and re-seeded via worker_init_fn below) at
                # every iterator reset instead of carrying stale state.
                persistent_workers=False,
                worker_init_fn=roboracer_worker_init_fn,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=_train_datasets(),
            ),
        ),
        # Held-out eval split (orin10: 5 bags, orin13: 1 bag, zero overlap with
        # train/test — see roboracer_bag_split.json / roboracer_bag_split_orin13.json).
        # No augmentation, deterministic order, so validation loss is comparable
        # across checkpoints/runs.
        dataloader_val=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_roboracer_val",
            # Must satisfy: max_val_iter * max_samples_per_batch <= the SMALLEST
            # eval split, in windows. RankPartitionedDataLoader gives one dataset
            # per rank, so a rank holding the smallest split exhausts its data and
            # leaves the collective early while the others wait -- every GPU pins
            # at 100% with no log output and no timeout.
            # Smallest split is pass_right: 304 frames / 3 episodes = 205 windows.
            # 20 * 32 = 640 deadlocks; 20 * 8 = 160 fits.
            max_samples_per_batch=8,
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=True,
                num_workers=2,
                persistent_workers=False,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=_val_datasets(),
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

# chunk_length=32 → 33 observation frames; match DROID recipe tokenizer duration
action_policy_roboracer_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

for _item in [action_policy_roboracer_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
