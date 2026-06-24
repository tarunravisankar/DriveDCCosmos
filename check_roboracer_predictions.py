#!/usr/bin/env python3
"""
check_roboracer_predictions.py

Loads the fine-tuned Cosmos3-Nano roboracer checkpoint, runs inference on a
handful of real frames from the training set (including known turns and
known straight-driving segments), de-normalizes the predicted actions back
to real units (meters/frame, rot6d), and prints predicted vs. ground truth
side by side.

This answers: "does the model predict something logically consistent with
what's actually happening in the frame (turning vs straight), once we undo
the normalization?"

Usage:
    cd ~/cosmos-framework
    export LD_LIBRARY_PATH=...  (see training launch command)
    python check_roboracer_predictions.py \
        --checkpoint-path /scratch/tarunrav/cosmos-framework/outputs/cosmos3_action/action_sft/action_policy_roboracer_repro/checkpoints/iter_000002000 \
        --num-samples 8
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from cosmos_framework.data.vfm.action.action_normalization import denormalize_action, load_action_stats
from cosmos_framework.data.vfm.action.datasets.roboracer_dataset import (
    RoboracerDataset,
    get_action_roboracer_sft_dataset,
)
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.inference import OmniInference

_STATS_PATH = Path(__file__).parent / "cosmos_framework/data/vfm/action/datasets/stats/roboracer_stats.json"
_ACTION_NAMES = ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "rot_5"]


def find_interesting_samples(ds: RoboracerDataset, num_straight=4, num_turn=4):
    """Scan the dataset for samples with notably high vs near-zero curvature
    (using rot_1 = R[1,0] = sin(yaw_delta), the rot6d component sensitive to yaw
    rotation about the vertical axis), so we test both 'going straight' and
    'turning' cases.

    Note: rot_2 = R[2,0] is always exactly 0 for pure yaw (planar) motion — see
    rotation_matrix_to_rot6d in convert_roboracer_to_lerobot.py — so it cannot be
    used to detect turns; an earlier version of this script used it by mistake.
    """
    # Physically-implausible per-frame deltas (>1 m or >45 deg at 15 fps, i.e. >15 m/s
    # or >675 deg/s) indicate odometry glitches/resets in the raw bags rather than real
    # turns, and must be excluded or they dominate the "highest curvature" ranking.
    max_pos_delta_m = 1.0
    max_yaw_delta_deg = 45.0

    print("Scanning dataset for straight vs turning samples (this may take a minute)...")
    scores = []
    discarded_outliers = 0
    # Sample a subset for speed rather than scanning all 44887 windows
    step = max(1, len(ds) // 2000)
    for i in range(0, len(ds), step):
        row = ds._rows[ds._valid_windows[i][0] + 1]  # first action row after init state
        pos_x = float(row.get("action.pos_x", 0.0))
        pos_y = float(row.get("action.pos_y", 0.0))
        rot_0 = float(row.get("action.rot_0", 1.0))
        rot_1 = float(row.get("action.rot_1", 0.0))
        yaw_delta_deg = math.degrees(math.atan2(rot_1, rot_0))
        if abs(pos_x) > max_pos_delta_m or abs(pos_y) > max_pos_delta_m or abs(yaw_delta_deg) > max_yaw_delta_deg:
            discarded_outliers += 1
            continue
        scores.append((i, abs(rot_1)))

    if discarded_outliers:
        print(f"  Discarded {discarded_outliers} windows with implausible odometry deltas (likely glitches/resets).")
    scores.sort(key=lambda x: x[1])
    straight_idxs = [i for i, _ in scores[:num_straight]]
    turn_idxs = [i for i, _ in scores[-num_turn:]]
    return straight_idxs, turn_idxs


def integrate_trajectory(actions_raw) -> np.ndarray:
    """Integrate per-step body-frame pose deltas into a world-frame (x, y) trajectory.

    Each row is a body-frame delta [dx, dy, dz, rot_0..rot_5] (rot6d), matching the
    convention in convert_roboracer_to_lerobot.py:compute_pose_delta_9d. rot_0=R[0,0],
    rot_1=R[1,0] of the relative rotation, so for planar (z-only) yaw, yaw_delta =
    atan2(rot_1, rot_0).

    Returns an array of shape [T+1, 2] starting at the origin.
    """
    actions_np = actions_raw.numpy() if isinstance(actions_raw, torch.Tensor) else np.asarray(actions_raw)
    x, y, yaw = 0.0, 0.0, 0.0
    traj = [(x, y)]
    for step in actions_np:
        dx, dy = float(step[0]), float(step[1])
        x += math.cos(yaw) * dx - math.sin(yaw) * dy
        y += math.sin(yaw) * dx + math.cos(yaw) * dy
        traj.append((x, y))
        yaw += math.atan2(float(step[4]), float(step[3]))
    return np.array(traj, dtype=np.float32)


def to_curvature_velocity(actions_raw, fps: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw 9D body-frame pose deltas into [curvature, velocity] per step,
    matching the physical representation used by roboracer_ws's av_imitation
    package (see av_imitation/src/main.py: curvature = path curvature from Bezier
    steering, velocity = signed forward speed) so predictions/ground-truth are
    comparable in the same interpretable units, regardless of the differing
    internal action representations/normalization (9D pose-delta + minmax here
    vs. [curvature, velocity] + mean/std there).

    velocity[t] = pos_x[t] / dt (forward body-frame displacement / step duration,
    signed: negative = reversing).
    curvature[t] = yaw_delta[t] / arc_length[t] (path curvature kappa = dtheta/ds,
    the standard kinematic relation for an Ackermann vehicle), with arc_length
    floored to avoid blowing up near-zero-motion steps.
    """
    actions_np = actions_raw.numpy() if isinstance(actions_raw, torch.Tensor) else np.asarray(actions_raw)
    dt = 1.0 / fps
    dx, dy = actions_np[:, 0], actions_np[:, 1]
    rot_0, rot_1 = actions_np[:, 3], actions_np[:, 4]
    yaw_delta = np.arctan2(rot_1, rot_0)
    arc_length = np.maximum(np.sqrt(dx**2 + dy**2), 1e-3)
    velocity = dx / dt
    curvature = yaw_delta / arc_length
    return curvature, velocity


def to_rotation_angular_velocity(actions_raw, fps: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw 9D body-frame pose deltas into [rotation (deg/step), angular
    velocity (deg/s)] per step — a second, simpler diagnostic alongside
    curvature/velocity: rotation is the raw yaw change per step (no coupling to
    distance traveled, unlike curvature = yaw/arc_length), and angular velocity
    is just rotation/dt, i.e. how fast the car is turning regardless of forward
    speed."""
    actions_np = actions_raw.numpy() if isinstance(actions_raw, torch.Tensor) else np.asarray(actions_raw)
    dt = 1.0 / fps
    rot_0, rot_1 = actions_np[:, 3], actions_np[:, 4]
    rotation_deg = np.degrees(np.arctan2(rot_1, rot_0))
    angular_velocity_deg_s = rotation_deg / dt
    return rotation_deg, angular_velocity_deg_s


def plot_rotation_angular_velocity(
    gt_rot: np.ndarray, gt_angvel: np.ndarray,
    pred_rot: np.ndarray, pred_angvel: np.ndarray,
    out_path: Path, title: str,
    rotation_ylim: tuple[float, float] = (-20.0, 20.0),
    angvel_ylim: tuple[float, float] = (-300.0, 300.0),
) -> None:
    """Time-series ground-truth-vs-prediction plot of rotation/angular velocity."""
    steps = np.arange(len(gt_rot))
    fig, (ax_r, ax_w) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_r.plot(steps, gt_rot, color="tab:green", label="ground truth", linewidth=2)
    ax_r.plot(steps, pred_rot, color="tab:red", label="prediction", linewidth=2, linestyle="--")
    ax_r.set_ylabel("rotation (deg/step)")
    ax_r.set_ylim(*rotation_ylim)
    ax_r.set_title(title)
    ax_r.legend(loc="upper right")
    ax_r.axhline(0, color="gray", linewidth=0.5)

    ax_w.plot(steps, gt_angvel, color="tab:green", label="ground truth", linewidth=2)
    ax_w.plot(steps, pred_angvel, color="tab:red", label="prediction", linewidth=2, linestyle="--")
    ax_w.set_ylabel("angular velocity (deg/s)")
    ax_w.set_ylim(*angvel_ylim)
    ax_w.set_xlabel("step (chunk horizon)")
    ax_w.axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_curvature_velocity(
    gt_curv: np.ndarray, gt_vel: np.ndarray,
    pred_curv: np.ndarray, pred_vel: np.ndarray,
    out_path: Path, title: str,
    curvature_ylim: tuple[float, float] = (-2.0, 2.0),
    velocity_ylim: tuple[float, float] = (-2.0, 2.0),
) -> None:
    """Time-series ground-truth-vs-prediction plot over the action chunk horizon,
    in the same [curvature, velocity] units av_imitation's webapp Analysis tab uses.

    Axis limits are fixed (not auto-scaled) to av_imitation's own real-world range
    so the plot is honest about magnitude: curvature bins span [-2, 2] (1/m) in
    av_imitation's main.py oversampling histogram, and velocity tops out at its
    turbo_speed=2.0 m/s config default — auto-scaling would zoom into whatever
    narrow band the data happens to occupy and visually overstate how well
    prediction tracks ground truth.
    """
    steps = np.arange(len(gt_curv))
    fig, (ax_c, ax_v) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_c.plot(steps, gt_curv, color="tab:green", label="ground truth", linewidth=2)
    ax_c.plot(steps, pred_curv, color="tab:red", label="prediction", linewidth=2, linestyle="--")
    ax_c.set_ylabel("curvature (1/m)")
    ax_c.set_ylim(*curvature_ylim)
    ax_c.set_title(title)
    ax_c.legend(loc="upper right")
    ax_c.axhline(0, color="gray", linewidth=0.5)

    ax_v.plot(steps, gt_vel, color="tab:green", label="ground truth", linewidth=2)
    ax_v.plot(steps, pred_vel, color="tab:red", label="prediction", linewidth=2, linestyle="--")
    ax_v.set_ylabel("velocity (m/s)")
    ax_v.set_ylim(*velocity_ylim)
    ax_v.set_xlabel("step (chunk horizon)")
    ax_v.axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_frame_with_arrow(
    video: torch.Tensor,
    pred_rotation_deg: float,
    pred_velocity_mps: float,
    pred_angvel_deg_s: float,
    gt_rotation_deg: float,
    gt_velocity_mps: float,
    gt_angvel_deg_s: float,
    out_path: Path,
    num_history_frames: int = 2,
    angle_exaggeration: float = 8.0,
) -> None:
    """Draw the predicted (red) vs. ground-truth (green) next-step direction as
    an arrow on top of the actual current camera frame, with a strip of the
    preceding frames for context above it, and the raw velocity/angular-
    velocity numbers printed as text.

    video: [C, T, H, W] float in [0,1] (the model-ready resized frames used
    for this sample — same tensor passed to the model, not a separate read).
    A single real-world step's rotation (a few degrees at most) is barely
    visible as an arrow deflection, so the angle is exaggerated by
    angle_exaggeration for legibility — this is a visualization aid only, not
    a claim about the real magnitude (the printed text gives the true values).
    """
    video_perm = video.permute(1, 2, 3, 0)  # [T, H, W, C]
    if video_perm.dtype == torch.uint8:
        video_np = video_perm.numpy()
    else:
        video_np = (video_perm.clamp(0, 1).numpy() * 255).astype(np.uint8)
    current_frame = cv2.cvtColor(video_np[-1], cv2.COLOR_RGB2BGR)
    history_frames = video_np[max(0, video_np.shape[0] - 1 - num_history_frames): -1]

    h, w = current_frame.shape[:2]
    origin = (w // 2, h - 10)
    base_len = 0.15 * h

    def arrow_endpoint(rotation_deg: float, velocity_mps: float) -> tuple[int, int]:
        length = base_len + max(velocity_mps, 0.0) * 0.4 * h
        angle_rad = math.radians(90 + rotation_deg * angle_exaggeration)
        dx = length * math.cos(angle_rad)
        dy = -length * math.sin(angle_rad)
        return (int(origin[0] + dx), int(origin[1] + dy))

    annotated = current_frame.copy()
    cv2.arrowedLine(annotated, origin, arrow_endpoint(gt_rotation_deg, gt_velocity_mps), (0, 160, 0), 3, tipLength=0.25)
    cv2.arrowedLine(annotated, origin, arrow_endpoint(pred_rotation_deg, pred_velocity_mps), (0, 0, 255), 3, tipLength=0.25)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, f"GT:   v={gt_velocity_mps:+.2f} m/s  w={gt_angvel_deg_s:+.1f} deg/s",
                (8, 20), font, 0.45, (0, 160, 0), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"PRED: v={pred_velocity_mps:+.2f} m/s  w={pred_angvel_deg_s:+.1f} deg/s",
                (8, 38), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"(arrow angle exaggerated {angle_exaggeration:.0f}x for visibility)",
                (8, h - 6), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    if len(history_frames) > 0:
        strip = cv2.cvtColor(np.concatenate(list(history_frames), axis=1), cv2.COLOR_RGB2BGR)
        # Pad/resize the strip to the same width as the main frame.
        scale = w / strip.shape[1]
        strip = cv2.resize(strip, (w, int(strip.shape[0] * scale)))
        cv2.putText(strip, "previous frames", (6, 16), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.concatenate([strip, annotated], axis=0)
    else:
        canvas = annotated

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def render_trajectories(gt_traj: np.ndarray, pred_traj: np.ndarray, out_path: Path, canvas_size: int = 600, margin: int = 50) -> None:
    """Draw ground-truth (green) vs predicted (red) trajectories as polylines on one canvas.

    Centering/autoscaling/polyline approach mirrors WheeledLab's mppi/viz.py
    (world_to_centered_map_pixels + draw_trajectory_on_map), reimplemented here
    standalone since cosmos-framework and roboracer_ws are separate workspaces.
    """
    all_pts = np.concatenate([gt_traj, pred_traj], axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    scale = (canvas_size - 2 * margin) / float(span.max())
    center = (min_xy + max_xy) / 2.0

    def to_px(pts: np.ndarray) -> np.ndarray:
        px = (pts - center) * scale
        px[:, 1] *= -1  # image y grows downward
        px += canvas_size / 2
        return px.astype(np.int32)

    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    gt_px, pred_px = to_px(gt_traj), to_px(pred_traj)
    cv2.polylines(canvas, [gt_px], isClosed=False, color=(0, 160, 0), thickness=2)
    cv2.polylines(canvas, [pred_px], isClosed=False, color=(0, 0, 255), thickness=2)
    cv2.circle(canvas, tuple(gt_px[0]), 5, (0, 0, 0), -1)
    cv2.putText(canvas, "GT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
    cv2.putText(canvas, "PRED", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True)
    # Default to the held-out TEST split (not the training data) — this script
    # exists to check generalization, not memorization.
    parser.add_argument("--dataset-root", type=str, default="/scratch/tarunrav/roboracer_lerobot_test")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--plot-dir", type=str, default="/scratch/tarunrav/roboracer_check_predictions/plots")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_root}...")
    ds = RoboracerDataset(root=args.dataset_root, fps=15.0, chunk_length=32, action_normalization="minmax")
    print(f"Dataset has {len(ds)} windows")

    # Model-ready view of the same windows: resized/padded video, sequence_plan,
    # raw_action_dim, etc. — exactly what ActionSFTDataset feeds the model during
    # training (see action_sft_dataset.py / transforms.py:ActionTransformPipeline).
    sft_ds = get_action_roboracer_sft_dataset(
        root=args.dataset_root,
        fps=15.0,
        chunk_length=32,
        action_normalization="minmax",
        resolution="256",
        max_action_dim=64,
    )

    straight_idxs, turn_idxs = find_interesting_samples(ds)
    test_idxs = straight_idxs[: args.num_samples // 2] + turn_idxs[: args.num_samples // 2]
    print(f"Testing on {len(test_idxs)} samples: {len(straight_idxs[:args.num_samples//2])} straight, "
          f"{len(turn_idxs[:args.num_samples//2])} turning")

    print(f"\nLoading model from {args.checkpoint_path}...")
    setup_overrides = OmniSetupOverrides.model_validate({
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_type": CheckpointType.DCP,
        "experiment": "action_policy_roboracer_nano",
        "experiment_overrides": [
            f"model.config.tokenizer.vae_path={Path.home()}/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
        ],
        "output_dir": "/scratch/tarunrav/roboracer_check_predictions",
        "guardrails": False,
        # Training disabled EMA (model.ema.enabled=false in the repro TOML), so the
        # checkpoint has no net_ema.* weights — default use_ema_weights=True would try
        # to load them and fail with "Missing key in checkpoint state_dict: net_ema...".
        "use_ema_weights": False,
    })
    setup_args = setup_overrides.build_setup()
    pipe = OmniInference.create(setup_args)
    model = pipe.model
    model.eval()
    print("Model loaded.\n")

    # Load normalization stats for de-normalization
    stats_raw = load_action_stats(str(_STATS_PATH))
    stats = {k: torch.from_numpy(v).float() for k, v in stats_raw.items()}

    for idx in test_idxs:
        gt_action_normalized = ds[idx]["action"]  # [32, 9], normalized
        gt_action_raw = denormalize_action(gt_action_normalized, "minmax", stats)  # back to real units

        # Model-ready sample: resized/padded video, padded+normalized action,
        # sequence_plan, raw_action_dim. Deliberately tested in "policy" mode
        # (only frame 0 is clean conditioning; frames 1-32 are noised/generated
        # targets, regardless of the real future frames sitting in `video` —
        # the diffusion sampler discards those positions' real values and
        # starts them from noise) — this is the actual deployable task: predict
        # from the current frame only, with no future frames available, the way
        # a real car would at inference time. "inverse_dynamics" mode (used for
        # v7) instead gives the model the REAL future frames as clean
        # conditioning, which made predictions look dramatically better but
        # isn't deployable — see session notes / RoboracerDataset.__init__ for
        # the full reasoning. No action conditioning either way, so the actual
        # action values fed in are irrelevant to generation — zero them out
        # like action_policy_server*.py does.
        model_sample = sft_ds[idx]
        video = model_sample["video"]  # [C, T, H, W]
        action_zeros = torch.zeros_like(model_sample["action"])  # [32, 64]
        sequence_plan = build_sequence_plan_from_mode(
            mode="policy",
            video_length=video.shape[1],
            action_length=action_zeros.shape[0],
        )

        # Build a single-sample batch matching the PackingDataLoader collate
        # convention: multi-item keys ("video", "action") are list[list[Tensor]],
        # per-sequence metadata keys are list[element].
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
            samples_out = model.generate_samples_from_batch(
                data_batch,
                guidance=args.guidance,
                seed=[0],
                num_steps=args.num_steps,
                shift=args.shift,
            )

        pred_action_normalized = samples_out["action"][0][:, :9].detach().cpu()  # [T, 9]
        pred_action_raw = denormalize_action(pred_action_normalized, "minmax", stats)

        # yaw_delta = atan2(rot_1, rot_0) = atan2(R[1,0], R[0,0]) for planar motion.
        gt_yaw_deg = math.degrees(math.atan2(float(gt_action_raw[0, 4]), float(gt_action_raw[0, 3])))
        pred_yaw_deg = math.degrees(math.atan2(float(pred_action_raw[0, 4]), float(pred_action_raw[0, 3])))

        label = "TURN" if idx in turn_idxs else "STRAIGHT"
        print(f"=== sample {idx} ({label}) ===")
        print(f"  GT   pos_x (m/frame): {gt_action_raw[0,0]:+.4f}   pos_y: {gt_action_raw[0,1]:+.4f}   yaw_delta: {gt_yaw_deg:+.3f} deg")
        print(f"  PRED pos_x (m/frame): {pred_action_raw[0,0]:+.4f}   pos_y: {pred_action_raw[0,1]:+.4f}   yaw_delta: {pred_yaw_deg:+.3f} deg")
        print(f"  GT   full first-step: {[round(v,4) for v in gt_action_raw[0].tolist()]}")
        print(f"  PRED full first-step: {[round(v,4) for v in pred_action_raw[0].tolist()]}")

        gt_traj = integrate_trajectory(gt_action_raw)
        pred_traj = integrate_trajectory(pred_action_raw)
        plot_path = Path(args.plot_dir) / f"sample_{idx:05d}_{label}.png"
        render_trajectories(gt_traj, pred_traj, plot_path)
        print(f"  Trajectory plot: {plot_path}")

        gt_curv, gt_vel = to_curvature_velocity(gt_action_raw)
        pred_curv, pred_vel = to_curvature_velocity(pred_action_raw)
        cv_plot_path = Path(args.plot_dir) / f"sample_{idx:05d}_{label}_curvature_velocity.png"
        plot_curvature_velocity(
            gt_curv, gt_vel, pred_curv, pred_vel, cv_plot_path,
            title=f"sample {idx} ({label}) — checkpoint {Path(args.checkpoint_path).name}",
        )
        print(f"  Curvature/velocity plot: {cv_plot_path}")

        gt_rot, gt_angvel = to_rotation_angular_velocity(gt_action_raw)
        pred_rot, pred_angvel = to_rotation_angular_velocity(pred_action_raw)
        rot_plot_path = Path(args.plot_dir) / f"sample_{idx:05d}_{label}_rotation_angvel.png"
        plot_rotation_angular_velocity(
            gt_rot, gt_angvel, pred_rot, pred_angvel, rot_plot_path,
            title=f"sample {idx} ({label}) — checkpoint {Path(args.checkpoint_path).name}",
        )
        print(f"  Rotation/angular-velocity plot: {rot_plot_path}")

        frame_arrow_path = Path(args.plot_dir) / f"sample_{idx:05d}_{label}_frame_arrow.png"
        render_frame_with_arrow(
            video=video,
            pred_rotation_deg=float(pred_rot[0]), pred_velocity_mps=float(pred_vel[0]),
            pred_angvel_deg_s=float(pred_angvel[0]),
            gt_rotation_deg=float(gt_rot[0]), gt_velocity_mps=float(gt_vel[0]),
            gt_angvel_deg_s=float(gt_angvel[0]),
            out_path=frame_arrow_path,
        )
        print(f"  Frame + arrow: {frame_arrow_path}")
        print()


if __name__ == "__main__":
    main()