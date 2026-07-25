#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Slurm submit script: online 7B→4B distillation on TACC Lonestar6 (2×H100 80 GB)
# Allocation: IRI25030  |  Partition: gpu-h100
# Container: cosmos_tacc.sif (NGC pytorch:25.01-py3 + all cosmos deps baked in)
#
# Usage (from TACC login node):
#   sbatch examples/slurm/submit_distill_tacc.sh
#
#SBATCH -J roboracer_distill
#SBATCH -o logs/distill_%j.out
#SBATCH -e logs/distill_%j.err
#SBATCH -p gpu-h100
#SBATCH -A IRI25030
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -t 24:00:00
#SBATCH --exclude=c318-002

set -eo pipefail
mkdir -p logs

# ── Paths ────────────────────────────────────────────────────────────────────
SCRATCH=/scratch/11403/tarunrav
REPO=$SCRATCH/cosmos-framework
SIF=$SCRATCH/cosmos_tacc.sif

# ── Env vars forwarded into the container (APPTAINERENV_ prefix) ─────────────
export APPTAINERENV_TEACHER_CHECKPOINT_PATH=$SCRATCH/outputs/cosmos3_action/action_sft/action_policy_roboracer_repro_v10/checkpoints/iter_000007200
export APPTAINERENV_WAN_VAE_PATH=$SCRATCH/checkpoints/wan22_vae/Wan2.2_VAE.pth
export APPTAINERENV_IMAGINAIRE_OUTPUT_ROOT=$SCRATCH/outputs
export APPTAINERENV_ROBORACER_TRAIN_ROOT=$SCRATCH/roboracer_lerobot_train
export APPTAINERENV_ROBORACER_EVAL_ROOT=$SCRATCH/roboracer_lerobot_eval
export APPTAINERENV_QWEN_4B_PATH=$SCRATCH/checkpoints/Qwen3-VL-4B-Instruct
export APPTAINERENV_PYTHONPATH=$REPO:$SCRATCH/extra_pkgs
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_IB_DISABLE=0
export APPTAINERENV_MASTER_ADDR=$(hostname)
export APPTAINERENV_MASTER_PORT=29500

# ── Run inside NGC container ──────────────────────────────────────────────────
apptainer exec --nv \
    --bind $SCRATCH:$SCRATCH \
    $SIF \
    torchrun \
        --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$(hostname) \
        --master_port=29500 \
        -m cosmos_framework.scripts.train \
        --sft-toml $REPO/examples/toml/sft_config/action_policy_roboracer_distill_tacc_h100.toml
