# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Online knowledge-distillation: frozen 7B teacher → smaller student.

Each training step:
  1. Student (self) runs denoise() on the noised packed sequence → preds_action.
  2. Teacher runs denoise() in torch.no_grad() on the SAME noised inputs → teacher_preds_action.
  3. distill_loss = MSE(student_preds_action, teacher_preds_action) * distill_alpha
     is added on top of the student's normal flow-matching loss.

Teacher loading: each GPU rank loads the full teacher as a non-FSDP model in bf16
(parallelism shard degree = 1).  Memory budget for 2×H100 80 GB:
  - 7B teacher bf16 (per rank, full weights):  ~14 GB
  - 4B student FSDP (per rank, sharded):       ~ 4 GB weights + ~8 GB optimizer states
  - Activations + buffers:                     ~ 8 GB
  Total per rank: ~34 GB — comfortable within 80 GB.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel

log = logging.getLogger(__name__)


class DistillOmniMoTModel(OmniMoTModel):
    """OmniMoTModel subclass that adds online action-prediction distillation."""

    def __init__(
        self,
        config: OmniMoTModelConfig,
        teacher_checkpoint_path: str,
        distill_alpha: float = 1.0,
        teacher_experiment_name: str = "action_policy_roboracer_nano",
    ):
        # Must be set before super().__init__(config): OmniMoTModel.__init__ calls
        # self.set_up_model() internally, which (via dynamic dispatch) resolves to
        # this class's set_up_model() -> _load_teacher(), which reads these attrs.
        self._teacher_checkpoint_path = teacher_checkpoint_path
        self._distill_alpha = distill_alpha
        self._teacher_experiment_name = teacher_experiment_name
        self._teacher: Optional[OmniMoTModel] = None
        # Scratchpad cleared after every _compute_losses call
        self._last_teacher_preds_action: Optional[list[torch.Tensor]] = None
        super().__init__(config)

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------

    def set_up_model(self) -> None:
        """Build student normally, then load frozen teacher on the same device."""
        super().set_up_model()
        self._load_teacher()

    def _load_teacher(self) -> None:
        """Instantiate + load the teacher checkpoint as a non-FSDP model on current GPU."""
        from cosmos_framework.utils.vfm.model_loader import load_model_from_checkpoint

        log.info(
            f"[DistillOmniMoTModel] Loading teacher ({self._teacher_experiment_name}) "
            f"from {self._teacher_checkpoint_path}"
        )

        # Save RNG state; load_model_from_checkpoint calls set_random_seed which
        # would clobber the training RNG if left unrestore.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        # ParallelDims._validate() requires dp_shard * dp_replicate * cp == world_size.
        # We must match the training world_size here or it asserts.
        # The teacher is still frozen (requires_grad=False) — FSDP sharding happens
        # but no reduce-scatter fires during backward since there are no gradients.
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        # load_model_from_checkpoint composes a fresh config for
        # `experiment_name` from scratch — it does NOT inherit this run's
        # TOML overrides. Explicitly re-forward the student's own (already
        # env-resolved) vae_path so the teacher tokenizes with the same VAE
        # instead of falling back to action_policy_roboracer_nano.py's
        # placeholder default path.
        teacher, _ = load_model_from_checkpoint(
            experiment_name=self._teacher_experiment_name,
            checkpoint_path=self._teacher_checkpoint_path,
            parallelism_config={
                "data_parallel_shard_degree": world_size,
                "data_parallel_replicate_degree": 1,
            },
            experiment_opts=[f"model.config.tokenizer.vae_path={self.config.tokenizer.vae_path}"],
        )

        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)

        teacher.eval()
        teacher.requires_grad_(False)
        self._teacher = teacher
        log.info(
            f"[DistillOmniMoTModel] Teacher loaded and frozen "
            f"({sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B params)."
        )

    # ------------------------------------------------------------------
    # Distillation forward hook
    # ------------------------------------------------------------------

    def denoise(
        self,
        net=None,
        data_batch_packed=None,
        fps_vision=None,
        fps_action=None,
        fps_sound=None,
        memory=None,
    ):
        """Run student denoise; also run teacher in no_grad to capture its action preds."""
        out_net = super().denoise(
            net=net,
            data_batch_packed=data_batch_packed,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            memory=memory,
        )

        if self._teacher is not None and self.training:
            with torch.no_grad():
                teacher_out = self._teacher.denoise(
                    data_batch_packed=data_batch_packed,
                    fps_vision=fps_vision,
                    fps_action=fps_action,
                    fps_sound=fps_sound,
                    memory=None,  # teacher has no persistent KV state
                )
            self._last_teacher_preds_action = teacher_out.get("preds_action")

        return out_net

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        out_net,
        data_batch_packed,
        gen_data_noised,
        timesteps,
        is_image_batch,
        timesteps_action=None,
        timesteps_sound=None,
    ):
        """Base flow-matching loss + action distillation MSE from frozen teacher."""
        loss, losses_dict = super()._compute_losses(
            out_net=out_net,
            data_batch_packed=data_batch_packed,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps,
            is_image_batch=is_image_batch,
            timesteps_action=timesteps_action,
            timesteps_sound=timesteps_sound,
        )

        teacher_preds = self._last_teacher_preds_action
        student_preds = out_net.get("preds_action")

        if teacher_preds is not None and student_preds is not None:
            # Both are lists of [T_i, action_dim] tensors — one per sample.
            # Pairs must match: same batch, same noised inputs.
            per_sample_mse = [
                F.mse_loss(s.float(), t.float())
                for s, t in zip(student_preds, teacher_preds)
                if s.numel() > 0 and t.numel() > 0
            ]
            if per_sample_mse:
                distill_loss = torch.stack(per_sample_mse).mean()
                loss = loss + self._distill_alpha * distill_loss
                losses_dict["distill_loss_action"] = distill_loss

        # Clear for next step
        self._last_teacher_preds_action = None

        return loss, losses_dict
