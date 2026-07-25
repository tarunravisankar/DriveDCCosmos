# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference server for the RoboRacer action policy (Cosmos3-Nano).

DRY-RUN BY DESIGN: this server has no ROS2/rclpy dependency and never talks to
the car. It only does one thing — given a single current camera frame, return
the predicted [curvature, velocity] for the next chunk_length (32) steps, in
the same physical units as the car's /ackermann_curvature_drive message
(velocity m/s forward-positive, curvature 1/m left-positive — see
ROBORACER_HANDOFF.md). Whether/how those predictions get published to the car
is entirely the concern of a separate, not-yet-built car-side ROS2 client,
which is expected to run in dry-run mode first (log predictions, never call
the actual publisher) before ever being wired to the real topic.

Protocol (minimal, msgpack over websockets — no external `openpi` dependency):
  - on connection, the server sends an empty msgpack dict (metadata handshake,
    mirroring action_policy_server_robolab.py's convention);
  - each client message is a msgpack dict: {"image": <raw bytes, HxWx3 uint8
    RGB row-major>, "shape": [H, W, 3], "direction": "left" | "right"
    (optional, defaults to "left")};
  - "direction" selects which trained text caption to condition on - the
    model is trained on two datasets (orin10=left turns, orin13=right turns)
    each with a distinct caption, since otherwise visually-similar
    observations can map to different valid actions with no other
    disambiguating signal (see _DIRECTION_CAPTIONS / action_policy_roboracer_nano.py);
  - each response is a msgpack dict:
      {"curvature": [32 floats], "velocity": [32 floats], "raw_action": [32x9 floats]}

NOT YET VALIDATED end-to-end (no GPU headroom to test against a live model as
of writing — training was using all 8 GPUs). Before trusting this:
  1. Run it locally and feed it the exact same test frame used by
     check_roboracer_predictions.py, and confirm the first-step curvature/
     velocity matches that script's output for the same checkpoint.
  2. Only after that, build the car-side client, and run it dry-run (log
     only) before ever calling the real ROS2 publisher.

Example:
  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_roboracer \
    --checkpoint-path /scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_repro_v8/checkpoints/iter_000004000 \
    --port 8765
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import asyncio
import threading
from pathlib import Path

import msgpack
import numpy as np
import torch
import tyro
import websockets

from cosmos_framework.data.vfm.action.action_normalization import denormalize_action, load_action_stats
from cosmos_framework.data.vfm.action.datasets.roboracer_dataset import RoboracerDataset, _STATS_PATH
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import maybe_init_distributed
from cosmos_framework.utils import log

_CHUNK_LENGTH = 32
_RESOLUTION = "256"
_MAX_ACTION_DIM = 64

# Must exactly match the per-dataset captions written by convert_roboracer_to_lerobot.py's
# --task-description for orin10 (left) / orin13 (right) — see action_policy_roboracer_nano.py.
# Mapped from a short client-supplied token rather than accepting freeform text from the
# client, so a typo/mismatch can't silently fall outside the training caption distribution.
_DIRECTION_CAPTIONS = {
    # Navigation — loop/track type (must exactly match convert_all_datasets.sh captions)
    "loop_ccw":      "Drive the roboracer vehicle counter-clockwise around a large indoor loop.",
    "loop_cw":       "Drive the roboracer vehicle clockwise around a large indoor loop.",
    "oval_ccw":      "Drive the roboracer vehicle counter-clockwise around a large oval track.",
    "circle_ccw":    "Drive the roboracer vehicle counter-clockwise around a circular track.",
    "rect_small_ccw": "Drive the roboracer vehicle counter-clockwise around a small rectangular track.",
    "rect_small_cw":  "Drive the roboracer vehicle clockwise around a small rectangular track.",
    "rect_med_ccw":  "Drive the roboracer vehicle counter-clockwise around a medium rectangular track.",
    "square_ccw":    "Drive the roboracer vehicle counter-clockwise around a square track.",
    # Person avoidance — social navigation
    "pass_right":    "Drive the roboracer vehicle and pass the person on the right.",
    "pass_left":     "Drive the roboracer vehicle and pass the person on the left.",
    "wait":          "Drive the roboracer vehicle and wait for the person to pass.",
    # Legacy aliases (pre-v10 clients that send "left"/"right")
    "left":          "Drive the roboracer vehicle counter-clockwise around a large indoor loop.",
    "right":         "Drive the roboracer vehicle clockwise around a large indoor loop.",
}
_DEFAULT_DIRECTION = "loop_ccw"


def to_curvature_velocity(actions_raw: np.ndarray, fps: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Same conversion as check_roboracer_predictions.py — matches the car's
    /ackermann_curvature_drive units exactly (velocity m/s forward-positive,
    curvature 1/m left-positive), so the response can in principle be
    forwarded to that topic as-is once a client exists.
    """
    actions_np = actions_raw.numpy() if isinstance(actions_raw, torch.Tensor) else np.asarray(actions_raw)
    dt = 1.0 / fps
    dx = actions_np[:, 0]
    rot_0, rot_1 = actions_np[:, 3], actions_np[:, 4]
    yaw_delta = np.arctan2(rot_1, rot_0)
    arc_length = np.maximum(np.sqrt(actions_np[:, 0] ** 2 + actions_np[:, 1] ** 2), 1e-3)
    velocity = dx / dt
    curvature = yaw_delta / arc_length
    return curvature, velocity


class RoboracerPolicyService:
    """Loads the checkpoint once; runs mode="policy" (past-only, deployable)
    inference per request. No ROS2 dependency — see module docstring.
    """

    def __init__(self, checkpoint_path: str, eval_root: str, vae_path: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")
        maybe_init_distributed()

        log.info(f"[roboracer-policy-server] loading model from {checkpoint_path}")
        checkpoint_type = CheckpointType.from_path(Path(checkpoint_path))
        if checkpoint_type == CheckpointType.HF:
            # OmniInference.create()'s DCP-bridge loader for HF checkpoints
            # (torch.distributed.checkpoint.hf_storage) raises a spurious
            # "Missing key: lm_head.weight" even though the key is present in
            # the checkpoint's own safetensors index — root cause not yet
            # isolated. Bypass it: load via the plain transformers
            # `from_pretrained` path (proven to work for both bf16 and
            # bitsandbytes-quantized exports in quantize_int4.py), then reach
            # into the wrapper for the actual generation-capable network.
            # This also transparently honors any quantization_config embedded
            # in the checkpoint's own config.json (set automatically by
            # `save_pretrained` after 4-bit loading).
            from cosmos_framework.inference.model import Cosmos3OmniModel

            hf_model = Cosmos3OmniModel.from_pretrained(
                checkpoint_path, device_map={"": 0}, dtype=torch.bfloat16,
            )
            self.model = hf_model.model
        else:
            setup_overrides = OmniSetupOverrides.model_validate({
                "checkpoint_path": checkpoint_path,
                "checkpoint_type": CheckpointType.DCP,
                "experiment": "action_policy_roboracer_nano",
                "experiment_overrides": [f"model.config.tokenizer.vae_path={vae_path}"],
                "output_dir": "/tmp/cosmos3_action_server/roboracer",
                "guardrails": False,
                # Training disabled EMA — see check_roboracer_predictions.py for why.
                "use_ema_weights": False,
            })
            setup_args = setup_overrides.build_setup()
            pipe = OmniInference.create(setup_args)
            self.model = pipe.model
        self.model.eval()

        # Instantiate the real dataset only to read its fixed config (fps,
        # domain_id, viewpoint, action_normalization) consistently with
        # training, rather than re-guessing those values by hand.
        self._ref_dataset = RoboracerDataset(root=eval_root, fps=15.0, chunk_length=_CHUNK_LENGTH,
                                              action_normalization="minmax")

        stats_raw = load_action_stats(str(_STATS_PATH))
        self._stats = {k: torch.from_numpy(v).float() for k, v in stats_raw.items()}

        self._lock = threading.Lock()
        log.info("[roboracer-policy-server] ready")

    def _build_live_sample(self, image_hwc_uint8: np.ndarray, caption: str) -> dict:
        """Build a raw (pre-ActionTransformPipeline) sample dict from a single live frame.

        ``caption`` is the raw text string passed directly to the model as
        ai_caption conditioning.  Use resolve_caption() to go from a direction
        token or free-form text to a caption string before calling this.
        """
        if image_hwc_uint8.ndim != 3 or image_hwc_uint8.shape[-1] != 3:
            raise ValueError(f"image must be [H,W,3], got {image_hwc_uint8.shape}")
        frame = torch.from_numpy(image_hwc_uint8).permute(2, 0, 1).float() / 255.0  # [C,H,W] in [0,1]
        video = frame.unsqueeze(0).repeat(_CHUNK_LENGTH + 1, 1, 1, 1)  # [T,C,H,W]
        action = torch.zeros(_CHUNK_LENGTH, 9)
        ds = self._ref_dataset
        return ds._build_result(
            mode="policy",
            video=video,
            action=action,
            ai_caption=caption,
            additional_view_description=(
                "A single front-facing camera mounted on a 1/10th-scale autonomous vehicle."
            ),
        )

    def predict(self, image_hwc_uint8: np.ndarray, direction: str = _DEFAULT_DIRECTION,
                caption: str | None = None) -> dict:
        """Run inference.  Prefer ``caption`` (raw text) over ``direction`` (token lookup).

        If ``caption`` is provided it is used directly as the model's text
        conditioning — this enables generalization testing with arbitrary prompts
        the model was never explicitly trained on.  If only ``direction`` is
        given it is resolved through _DIRECTION_CAPTIONS as before.
        """
        if caption is None:
            if direction not in _DIRECTION_CAPTIONS:
                raise ValueError(
                    f"direction must be one of {list(_DIRECTION_CAPTIONS)}, got {direction!r}. "
                    "Pass caption= for free-form text."
                )
            caption = _DIRECTION_CAPTIONS[direction]
        with self._lock:
            raw_sample = self._build_live_sample(image_hwc_uint8, caption)
            # ActionSFTDataset/ActionTransformPipeline expects to operate on a
            # raw-dataset-shaped sample; reuse the dataset's own transform via
            # its underlying ActionSFTDataset wiring would require constructing
            # one more layer (resize to _RESOLUTION, padding, sequence packing
            # fields) that check_roboracer_predictions.py gets "for free" from
            # get_action_roboracer_sft_dataset. This direct path is simplified
            # and NOT yet confirmed to produce bit-identical preprocessing —
            # see module docstring's validation step before trusting output.
            from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

            transform = ActionTransformPipeline(max_action_dim=_MAX_ACTION_DIM, cfg_dropout_rate=0.0)
            model_sample = transform(raw_sample, resolution=_RESOLUTION)
            video = model_sample["video"]
            action_zeros = torch.zeros_like(model_sample["action"])
            sequence_plan = build_sequence_plan_from_mode(
                mode="policy", video_length=video.shape[1], action_length=action_zeros.shape[0]
            )
            data_batch = {
                "video": [[video]],
                "action": [[action_zeros]],
                "ai_caption": [model_sample["ai_caption"]],
                "conditioning_fps": [model_sample["conditioning_fps"]],
                "domain_id": [model_sample["domain_id"]],
                "raw_action_dim": [model_sample["raw_action_dim"]],
                "action_processing_record": [model_sample["action_processing_record"]],
                "sequence_plan": [sequence_plan],
            }
            with torch.inference_mode():
                # guidance=1.0 skips the classifier-free-guidance unconditional
                # pass (was 3.0, effectively 2x forward passes per step) to cut
                # per-call latency under shared-host contention; chunk_length/fps
                # gives only ~2.13s of buffer, so inference must stay under that
                # or the buffer runs dry and the client fail-safes mid-turn.
                samples_out = self.model.generate_samples_from_batch(
                    data_batch, guidance=1.0, seed=[0], num_steps=4, shift=5.0
                )
            pred_action_normalized = samples_out["action"][0][:, :9].detach().cpu()
            pred_action_raw = denormalize_action(pred_action_normalized, "minmax", self._stats)
            curvature, velocity = to_curvature_velocity(pred_action_raw)
            return {
                "curvature": curvature.astype(np.float64).tolist(),
                "velocity": velocity.astype(np.float64).tolist(),
                "raw_action": pred_action_raw.numpy().astype(np.float64).tolist(),
            }


async def _serve(service: RoboracerPolicyService, host: str, port: int) -> None:
    async def handler(websocket) -> None:
        await websocket.send(msgpack.packb({}))  # metadata handshake, mirrors robolab server
        async for raw_msg in websocket:
            try:
                obs = msgpack.unpackb(raw_msg, raw=False)
                # "image" is raw bytes (HxWx3 uint8, row-major) + "shape" - far
                # more efficient over the wire than a nested-list encoding.
                image = np.frombuffer(obs["image"], dtype=np.uint8).reshape(obs["shape"])
                # "caption" (raw text) takes priority over "direction" (token).
                # Sending caption= enables free-form generalization testing.
                raw_caption = obs.get("caption", None)
                direction = obs.get("direction", _DEFAULT_DIRECTION)
                log.info(
                    f"[roboracer-policy-server] caption={raw_caption!r} direction={direction!r}"
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    None, service.predict, image, direction, raw_caption
                )
                await websocket.send(msgpack.packb(result))
            except Exception as exc:  # noqa: BLE001 - report errors to the client, keep server alive
                log.warning(f"[roboracer-policy-server] request failed: {exc}")
                await websocket.send(msgpack.packb({"error": str(exc)}))

    async with websockets.serve(handler, host, port, max_size=None):
        log.info(f"[roboracer-policy-server] listening on ws://{host}:{port}")
        await asyncio.Future()  # run forever


def main(
    checkpoint_path: str,
    port: int = 8765,
    host: str = "0.0.0.0",
    eval_root: str = "/scratch/tarunrav/roboracer_lerobot_eval",
    vae_path: str = "/home/tarunrav/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
) -> None:
    service = RoboracerPolicyService(checkpoint_path=checkpoint_path, eval_root=eval_root, vae_path=vae_path)
    asyncio.run(_serve(service, host, port))


if __name__ == "__main__":
    tyro.cli(main)
