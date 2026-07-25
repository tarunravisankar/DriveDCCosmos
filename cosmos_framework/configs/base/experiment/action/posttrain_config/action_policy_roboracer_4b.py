# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_roboracer_4b`` — Cosmos3 RoboRacer action policy on Qwen3-VL-4B backbone.

Identical training setup to action_policy_roboracer_nano (8B) except the VLM
backbone is Qwen3-VL-4B-Instruct (hidden=2560, 36 layers) instead of 8B
(hidden=4096).  Trained for Orin Jetson Nano deployment: 4B INT4 ≈ 2.8 GB,
comfortably inside the Orin's 8 GB shared memory budget.

Since no Cosmos3-4B DCP base checkpoint exists, this config loads the VLM
backbone directly from Qwen3-VL-4B-Instruct HuggingFace safetensors via
pretrained_weights.enabled=True.  The diffusion expert is copied from the
backbone at init (load_weights_from_pretrained=True + no DCP).  Action bridge
layers (action2llm / llm2action / action_modality_embed) are randomly
initialised — same as the 8B recipe which always skips them from the base
checkpoint.

No load_path / BASE_CHECKPOINT_PATH needed.

Usage (1 node, 8 GPU)::

    WAN_VAE_PATH=$HOME/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth \\
    IMAGINAIRE_OUTPUT_ROOT=/scratch/tarunrav/cosmos-framework/outputs \\
    ROBORACER_TRAIN_ROOT=/scratch/tarunrav/roboracer_lerobot_train \\
    ROBORACER_EVAL_ROOT=/scratch/tarunrav/roboracer_lerobot_eval \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_roboracer_4b.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.defaults.vlm import (
    create_qwen2_tokenizer_with_download,
    create_vlm_config,
)
from cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
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

_S = "/scratch/tarunrav"

def _roots(suffix): return f"{_S}/roboracer_lerobot_train{suffix}", f"{_S}/roboracer_lerobot_eval{suffix}"

_NAV_ROOTS = {
    "roboracer":        _roots(""),
    "roboracer_orin13": _roots("_orin13"),
    "roboracer_orin02": _roots("_orin02"),
    "roboracer_orin06": _roots("_orin06"),
    "roboracer_orin14": _roots("_orin14"),
}
_PEOPLE_ROOTS = {
    "roboracer_pass_right": _roots("_pass_right"),
    "roboracer_pass_left":  _roots("_pass_left"),
    "roboracer_wait":       _roots("_wait"),
}


def _ds(root, mode, augment):
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


# ── 4B model config — identical to NANO_MODEL_CONFIG except the VLM references ──
_4B_BASE_PATH = "/scratch/tarunrav/cosmos-framework/examples/checkpoints/Qwen3-VL-4B-Instruct"

MODEL_CONFIG_4B = copy.deepcopy(NANO_MODEL_CONFIG)
MODEL_CONFIG_4B["vlm_config"] = dict(
    layer_module="Qwen2MoTDecoderLayer",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    tie_word_embeddings=False,
    use_system_prompt=False,
    pretrained_weights=dict(
        enabled=True,                    # load 4B backbone from local HF safetensors
        backbone_path=_4B_BASE_PATH,
        credentials_path="",
        enable_gcs_patch_in_boto3=False,
        checkpoint_format=None,
    ),
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file=(
                    "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/"
                    "Qwen3-VL-4B-Instruct.json"
                ),
            ),
            freeze_und=False,
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            tie_word_embeddings=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        config_variant="hf",
        pretrained_model_name="Qwen/Qwen3-VL-4B-Instruct",
    ),
)
# With no DCP load_path, load_weights_from_pretrained copies backbone → diffusion expert at init.
MODEL_CONFIG_4B["diffusion_expert_config"] = copy.deepcopy(NANO_MODEL_CONFIG["diffusion_expert_config"])
MODEL_CONFIG_4B["diffusion_expert_config"]["load_weights_from_pretrained"] = True
# Keep 8-way shard like the 8B recipe — simpler, well-tested, and 4B still fits
# easily (4B params × 2 bytes bf16 / 8 ranks = 1 GB weights per rank).
MODEL_CONFIG_4B["parallelism"] = copy.deepcopy(NANO_MODEL_CONFIG["parallelism"])
MODEL_CONFIG_4B["parallelism"]["data_parallel_shard_degree"] = 8
MODEL_CONFIG_4B["parallelism"]["data_parallel_replicate_degree"] = 1


action_policy_roboracer_4b = LazyDict(
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
            project="cosmos3_action",
            group="action_sft",
            name="action_policy_roboracer_4b",
            wandb_mode="offline",
        ),
        model=dict(
            config=MODEL_CONFIG_4B,
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
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
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
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
            max_iter=100,
            max_val_iter=20,
            run_validation=True,
            run_validation_on_start=True,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=200,
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
            # No base DCP to load from — backbone comes from pretrained_weights above.
            # Action bridge layers always start fresh.
            keys_to_skip_loading=[],
            load_ema_to_reg=False,
            load_path="",          # intentionally empty — no DCP base checkpoint
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=200,
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
            dataset_name="action_roboracer_4b",
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
            dataset_name="action_roboracer_4b_val",
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
                datasets=_val_datasets(),
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

action_policy_roboracer_4b["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

for _item in [action_policy_roboracer_4b]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
