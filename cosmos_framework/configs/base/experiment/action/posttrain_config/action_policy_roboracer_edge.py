# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_roboracer_edge`` — Cosmos3-Edge RoboRacer action policy SFT.

Direct port of ``action_policy_roboracer_nano`` onto the Cosmos3-Edge tier
(Nemotron-2B-Dense-VL backbone, 4B total / 2B active) instead of Cosmos3-Nano
(Qwen3-VL-8B, 16B total / 8B active).  Same 8 datasets, same direction-caption
conditioning, same joint-dynamics training / deployable-mode validation split.

NOTE on mode naming: upstream renamed the old ``policy`` mode to ``wam``
(World Action Model). Semantics are identical -- the Nano tree documented
``policy`` as "Predict both actions and video given first frame"; upstream
documents ``wam`` as "Jointly denoise video and actions given the first
frame". ``joint`` is unchanged and still random.choice over
(forward_dynamics, inverse_dynamics, wam). So the v8 decision -- train on
``joint``, validate strictly on the deployable past-only mode -- is
preserved exactly; only the validation mode string changed.

Why Edge:
  * ~4x smaller than Nano.  Measured Nano-distilled-4B peak was 9.20 GB
    allocated for one inference call (6.03 GB weights + ~3.17 GB transient),
    which does not fit the car's 8 GB Jetson budget.  Edge's bf16 weights are
    7.57 GB vs 16 GB, so the INT4 static footprint drops ~4.3 GB -> ~2.2 GB.
  * Unlike the distilled 4B student (whose diffusion expert was cold-copied
    from the text backbone via ``load_weights_from_pretrained=True``, which is
    exactly what forced the teacher/student distillation detour), the
    Cosmos3-Edge-Policy-DROID checkpoint ships a *trained* generation expert
    AND trained action bridges.  Its own config sets
    ``load_weights_from_pretrained=False`` — weights come from the checkpoint.

Base checkpoint:
    nvidia/Cosmos3-Edge-Policy-DROID -> DCP via ``convert_model_to_dcp``.
    Verified 549 keys: net.action2llm(2), net.llm2action(2),
    net.action_modality_embed(1), net.time_embedder(4), net.vae2llm(2),
    net.llm2vae(2), k_norm_und_for_gen(28), *_moe_gen(281).
    (Nano has 1165; the delta is the smaller backbone plus Edge having no
    sound modality — sound2llm/llm2sound/sound_modality_embed are absent.)

Key architectural difference vs Nano — ``use_und_k_norm_for_gen=True``:
    Edge runs ``qk_norm_for_diffusion=True`` with ``qk_norm_for_text=False``
    (the Nemotron-3 configuration).  Without the und-K norm the joint softmax
    computes norm(Q_gen) . K_und_raw^T, where K_und_raw has large uncontrolled
    magnitude and swamps the gen self-attention path.  The 28
    ``k_norm_und_for_gen`` RMSNorms fix that.  They are loaded from the
    checkpoint and deliberately left OUT of ``keys_to_select`` (frozen), so
    Edge's calibrated attention scaling is preserved while only the generation
    and action heads train.

Launch:
    See examples/toml/sft_config/action_policy_roboracer_edge.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.generator.action.datasets.roboracer_dataset import (
    get_action_roboracer_sft_dataset,
    roboracer_worker_init_fn,
)

cs = ConfigStore.instance()

# ── Dataset root paths (all produced by convert_all_datasets.sh) ─────────────
_S = "/scratch/tarunrav"


# Returns a (train_root, eval_root) tuple for a given suffix
def _roots(suffix):
    return f"{_S}/roboracer_lerobot_train{suffix}", f"{_S}/roboracer_lerobot_eval{suffix}"


_NAV_ROOTS = {
    "roboracer":        _roots(""),           # orin10 — large indoor loop CCW (31 ep)
    "roboracer_orin13": _roots("_orin13"),    # orin13 — large indoor loop CW  (6 ep)
    "roboracer_orin02": _roots("_orin02"),    # large oval CCW                  (42 ep)
    "roboracer_orin06": _roots("_orin06"),    # small rectangle CCW             (68 ep)
    "roboracer_orin14": _roots("_orin14"),    # square CCW                      (9 ep)
}
_PEOPLE_ROOTS = {
    "roboracer_pass_right": _roots("_pass_right"),  # pass person on right (24 ep)
    "roboracer_pass_left":  _roots("_pass_left"),   # pass person on left  (24 ep)
    "roboracer_wait":       _roots("_wait"),         # wait for person      (24 ep)
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
        name: dict(ratio=1, dataset=_ds(roots[1], mode="wam", augment=False))
        for name, roots in {**_NAV_ROOTS, **_PEOPLE_ROOTS}.items()
    }


action_policy_roboracer_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
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
            name="action_policy_roboracer_edge",
            wandb_mode="offline",
        ),
        model=dict(
            config=copy.deepcopy(EDGE_MODEL_CONFIG),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Train generation + action heads only (same as the Nano recipe).
            # NOTE: k_norm_und_for_gen is intentionally absent — it ships trained
            # in the Edge checkpoint and keeping it frozen preserves Edge's
            # calibrated gen->und cross-attention scaling.
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
                # Action bridges are domain-indexed (num_embodiment_domains=32).
                # Edge-Policy-DROID trained the DROID domain slot; roboracer's
                # AV slot carries only Edge's general action-dynamics prior, so
                # it still needs a high LR to specialize.
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
            max_val_iter=20,
            run_validation=True,
            run_validation_on_start=True,  # baseline point before any fine-tuning
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=200,  # overridden by TOML
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
            # Skip EMA and action heads -> init fresh, exactly as the Nano recipe
            # does.
            #
            # v1/v2 of this experiment instead LOADED the action bridges from
            # Cosmos3-Edge-Policy-DROID, on the reasoning that trained action
            # heads were the point of using Edge. That was wrong: those heads
            # were trained on DROID, a 7-DoF robot ARM, whose action space has
            # nothing to do with a car's (forward velocity + yaw). Measured on
            # the eval_wait split at v2 iter 2400, the result was a near-constant
            # action -- predicted velocity spanned 0.0083 against ground truth
            # spanning 0.0503, and correlated NEGATIVELY with it (-0.276), while
            # the Nano checkpoint on the identical samples tracked ground truth
            # (spread 0.0474, corr +0.494). The head started in the wrong basin
            # and the 5x lr_multiplier did not pull it out.
            #
            # Edge still contributes what actually matters: a trained GENERATION
            # expert (the 281 *_moe_gen weights) plus the video/text backbone.
            # The action bridges were never the advantage.
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
            # Edge is ~4x smaller than Nano, so this can likely go well above the
            # Nano value (32 here / 96 via supervisor). Left at the Nano baseline
            # so the first run is comparable; raise from observed memory.
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
                # False (not True): persistent workers never reseed stdlib
                # `random` (used by mode="joint"'s _choose_mode()) across the
                # trainer's iterator resets at each validation/checkpoint boundary.
                # Suspected cause of the deterministic +144-iteration hang.
                persistent_workers=False,
                worker_init_fn=roboracer_worker_init_fn,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=_train_datasets(),
            ),
        ),
        dataloader_val=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_roboracer_val",
            # MUST stay small. RankPartitionedDataLoader gives one dataset per
            # rank, and the smallest eval splits are tiny (pass_right 205,
            # pass_left 209, wait 239 windows). Validation consumes
            # max_val_iter * max_samples_per_batch samples per rank; at 32 that
            # is 20*32=640 > 205, so those ranks hit StopIteration and leave the
            # validation loop while the others keep iterating -- a collective-op
            # mismatch that deadlocks NCCL with every GPU spinning at 100%% and
            # nothing logged. At 8: 20*8=160 < 205, so every rank completes the
            # same number of steps.
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

# chunk_length=32 -> 33 observation frames; match the Nano/DROID tokenizer duration.
action_policy_roboracer_edge["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

# ── model.config.resolution: match the Nano recipe (720), do NOT "fix" to 256 ──
#
# This field is NOT the data resolution -- the datasets independently produce
# 256p via _ds(resolution="256"). What it actually selects is two lookups:
#
#   1. the rectified-flow noise schedule shift
#        shift_dict = {"256": 3, "480": 5, "720": 10}
#        shift = shift_dict[config.resolution]        (omni_mot_model.py ~line 456)
#   2. the VAE chunk size
#        chunk_frames = tokenizer.encode_chunk_frames[res_key]
#
# The Nano roboracer recipe leaves this at the Nano tier default of "720" while
# feeding 256p data, i.e. it trains at shift=10. v2 pinned it to "256", which
# silently switched training to shift=3 -- a materially different noise schedule
# from the configuration every validated Nano run used.
#
# Set back to "720" to match. encode_chunk_frames["720"] is aligned to Nano's 12
# below (EDGE_MODEL_CONFIG ships 8) so both lookups agree with the proven recipe.
# The VAE is the same Wan2.2 checkpoint in both trees, so 12 is a known-good
# value for it.
action_policy_roboracer_edge["model"]["config"]["resolution"] = "720"
action_policy_roboracer_edge["model"]["config"]["tokenizer"]["encode_chunk_frames"]["720"] = 12

# Nano sets 1.0; EDGE_MODEL_CONFIG inherits None from Cosmos3-Edge.yaml. Only
# consulted for image batches (`is_image_batch`), which roboracer never produces,
# so this is belt-and-braces rather than a behavioural fix -- but it removes one
# more gratuitous difference from the proven recipe.
action_policy_roboracer_edge["model"]["config"]["rectified_flow_training_config"]["image_loss_scale"] = 1.0

# ── loss balance: match the proven Nano roboracer recipe, NOT the upstream default ──
#
# omni_mot_model.py composes the total loss as:
#     total_loss += fm_loss_vision * loss_scale          # vision  <- loss_scale
#     total_loss += fm_loss_action * action_loss_weight  # action  <- action_loss_weight
# so `loss_scale` weights ONLY the video-reconstruction term.
#
# EDGE_MODEL_CONFIG inherits loss_scale=10.0 from Cosmos3-Edge.yaml, matching the
# DROID/LIBERO recipes where video and action are deliberately balanced 1:1. The
# roboracer Nano recipe instead runs loss_scale=1.0, i.e. action weighted 10x over
# vision.
#
# v1 of this experiment ran at the inherited 10.0 and reproduced exactly the
# failure the Nano line's v7 notes call THE major finding -- "a large competing
# video-reconstruction objective sharing the fine-tuned moe_gen backbone with
# action prediction". Measured on held-out test frames at iter 4200: steering
# still discriminated (straight ~0.20 deg vs turn ~3.22 deg predicted) but
# velocity collapsed to a near-constant, predicting 0.053-0.097 m/frame while
# ground truth spanned 0.0-0.20 -- i.e. it neither stopped when the car was
# stopped nor sped up when it was fast. That breaks the `wait` social behavior,
# which requires actually stopping for a pedestrian.
#
# Restoring 1.0 gives the action head 10x the relative gradient it had in v1 and
# matches the configuration the Nano runs were validated under.
action_policy_roboracer_edge["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 1.0

for _item in [action_policy_roboracer_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
