# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_roboracer_distill`` — online distillation: 7B teacher → 4B student.

Identical training data and optimizer recipe to ``action_policy_roboracer_4b`` except:
  - Model class is ``DistillOmniMoTModel`` (not ``OmniMoTModel``).
  - A frozen 7B teacher (Cosmos3-Nano v10 iter 7200) runs alongside the 4B student
    every training step; MSE on action predictions is added to the flow-matching loss.
  - Teacher checkpoint path is set via the ``TEACHER_CHECKPOINT_PATH`` env var
    (defaults to the robolidar v10 iter_000007200 path for local runs).

Memory budget on 2×H100 80 GB (TACC):
  - 7B teacher (bf16, non-FSDP, full per rank):  ~14 GB
  - 4B student (bf16, 2-way FSDP):               ~ 4 GB weights + ~ 8 GB optimizer
  - Activations + misc:                           ~ 8 GB
  Total: ~34 GB/rank — comfortable in 80 GB.

Usage (TACC Lonestar6, 1 node 2×H100)::

    TEACHER_CHECKPOINT_PATH=/scratch/11403/tarunrav/outputs/cosmos3_action/action_sft/\\
        action_policy_roboracer_repro_v10/checkpoints/iter_000007200 \\
    BASE_CHECKPOINT_PATH="" \\
    WAN_VAE_PATH=... IMAGINAIRE_OUTPUT_ROOT=... \\
    ROBORACER_TRAIN_ROOT=... ROBORACER_EVAL_ROOT=... \\
    torchrun --nproc_per_node=2 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_roboracer_distill_tacc_h100.toml
"""

import copy
import os

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

# Defaults to the robolidar v10 iter_000007200 checkpoint; override on TACC via env var.
_TEACHER_CHECKPOINT_PATH = os.environ.get(
    "TEACHER_CHECKPOINT_PATH",
    f"{_S}/cosmos-framework/outputs/cosmos3_action/action_sft/"
    "action_policy_roboracer_repro_v10/checkpoints/iter_000007200",
)

_4B_BASE_PATH = os.environ.get(
    "QWEN_4B_PATH",
    f"{_S}/cosmos-framework/examples/checkpoints/Qwen3-VL-4B-Instruct",
)


def _roots(suffix):
    return f"{_S}/roboracer_lerobot_train{suffix}", f"{_S}/roboracer_lerobot_eval{suffix}"


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


# ── 4B student config (identical to action_policy_roboracer_4b) ──────────────
MODEL_CONFIG_4B = copy.deepcopy(NANO_MODEL_CONFIG)
MODEL_CONFIG_4B["vlm_config"] = dict(
    layer_module="Qwen2MoTDecoderLayer",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    tie_word_embeddings=False,
    use_system_prompt=False,
    pretrained_weights=dict(
        enabled=True,
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
MODEL_CONFIG_4B["diffusion_expert_config"] = copy.deepcopy(NANO_MODEL_CONFIG["diffusion_expert_config"])
MODEL_CONFIG_4B["diffusion_expert_config"]["load_weights_from_pretrained"] = True
# Shard degree overridden per TOML (8 for robolidar, 2 for TACC H100)
MODEL_CONFIG_4B["parallelism"] = copy.deepcopy(NANO_MODEL_CONFIG["parallelism"])
MODEL_CONFIG_4B["parallelism"]["data_parallel_shard_degree"] = 8
MODEL_CONFIG_4B["parallelism"]["data_parallel_replicate_degree"] = 1


action_policy_roboracer_distill = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp_distill"},
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
            name="action_policy_roboracer_distill",
            wandb_mode="offline",
        ),
        model=dict(
            config=MODEL_CONFIG_4B,
            teacher_checkpoint_path=_TEACHER_CHECKPOINT_PATH,
            distill_alpha=1.0,
            teacher_experiment_name="action_policy_roboracer_nano",
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
            # Teacher weights are re-loaded every run; skip them in checkpoint.
            keys_to_skip_loading=["_teacher"],
            load_ema_to_reg=False,
            load_path="",   # 4B student starts from HF backbone (pretrained_weights)
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
            dataset_name="action_roboracer_distill",
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
            dataset_name="action_roboracer_distill_val",
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

action_policy_roboracer_distill["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

for _item in [action_policy_roboracer_distill]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
