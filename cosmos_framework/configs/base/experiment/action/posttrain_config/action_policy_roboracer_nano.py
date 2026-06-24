# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_roboracer_nano`` — Cosmos3-Nano RoboRacer action policy SFT.

Fine-tunes Cosmos3-Nano on the UT Austin RoboRacer 1/10th-scale AV platform
using single-camera video + 9D AV ego-pose actions (domain_id=1).

Leverages Cosmos3's AV pretraining priors (same "av" embodiment, domain_id=1)
to adapt to the roboracer camera view and action distribution with minimal data.

Usage (1 node, 8 GPU)::

    ROBORACER_TRAIN_ROOT=/scratch/tarunrav/roboracer_lerobot_train \\
    ROBORACER_EVAL_ROOT=/scratch/tarunrav/roboracer_lerobot_eval \\
    BASE_CHECKPOINT_PATH=/scratch/tarunrav/cosmos-framework/examples/checkpoints/Cosmos3-Nano \\
    WAN_VAE_PATH=$HOME/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth \\
    IMAGINAIRE_OUTPUT_ROOT=/scratch/tarunrav/cosmos-framework/outputs \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_roboracer_repro.toml

Train/test/eval bag split: see roboracer_bag_split.json (31 train / 4 test /
5 eval bags, 1 bad bag excluded — see scan_roboracer_bag_quality.py). Each
split is converted to its own LeRobot root via convert_roboracer_to_lerobot.py
--bag-split-json/--split so there is no episode overlap between them.

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
from cosmos_framework.data.vfm.action.datasets.roboracer_dataset import get_action_roboracer_sft_dataset

cs = ConfigStore.instance()


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
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    roboracer=dict(
                        ratio=1,
                        dataset=L(get_action_roboracer_sft_dataset)(
                            root="${oc.env:ROBORACER_TRAIN_ROOT}",
                            fps=15.0,
                            chunk_length=32,
                            # joint: per-sample, randomly picks among forward_dynamics/
                            # inverse_dynamics/policy (base_dataset.py _MODE_CHOICES) — matches
                            # the reference DROID recipe's dataset default. inverse_dynamics
                            # (v7) proved the competing-video-objective diagnosis correct (loss
                            # dropped sharply, turn predictions tracked ground truth for the
                            # first time all session) but is NOT deployable: it conditions on
                            # the REAL future video frames, which don't exist yet on a live car
                            # at inference time — the model was reading motion off footage that
                            # already showed the outcome, not predicting before the outcome
                            # exists. "policy" alone (v2-v6) is the deployable mode but failed
                            # outright due to the competing objective. "joint" mixes in real
                            # policy-mode samples (so the model is actually exposed to the
                            # realistic "decide before you know what happens" task) alongside
                            # inverse_dynamics/forward_dynamics samples (which seem to help it
                            # learn the visual-motion<->action relationship) — same approach
                            # DROID's own dataset uses by default.
                            mode="joint",
                            action_normalization="minmax",  # train-split min/max so genuine turns map to exactly [-1,1]
                            use_image_augmentation=True,
                            # Disabled per professor's guidance: isolate variables and verify the
                            # normalization/validation-pipeline fixes on their own first, before
                            # layering oversampling tuning back in. v3/v4 (2.13x) and v5 (2.85x,
                            # 93% turning) both still failed to learn turning, and the failure
                            # mode (near-input-independent default action) wasn't clearly oversampling-
                            # driven — re-add only once this baseline's behavior is understood.
                            oversample_turns=False,
                            resolution="256",  # start at 256p, bump to 480p if memory allows
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        # Held-out eval split (5 bags, zero overlap with train/test — see
        # roboracer_bag_split.json). No augmentation, deterministic order, so
        # validation loss is comparable across checkpoints/runs.
        dataloader_val=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_roboracer_val",
            max_samples_per_batch=32,
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
                datasets=dict(
                    roboracer=dict(
                        ratio=1,
                        dataset=L(get_action_roboracer_sft_dataset)(
                            root="${oc.env:ROBORACER_EVAL_ROOT}",
                            fps=15.0,
                            chunk_length=32,
                            # Deliberately NOT "joint"/"inverse_dynamics" — validation must measure
                            # the actual deployable task (predict from past/current frame only,
                            # no future frames available), or the val loss/early-stopping decision
                            # would be just as misleading as it was for v7's inverse_dynamics runs.
                            mode="policy",
                            action_normalization="minmax",
                            use_image_augmentation=False,
                            resolution="256",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.0,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
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
