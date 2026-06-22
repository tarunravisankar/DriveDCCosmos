#!/usr/bin/env python3
"""
convert_roboracer_to_lerobot.py

Converts Roboracer ROS2 SQLite bags to LeRobot v3.0 format for
Cosmos3-Nano action policy fine-tuning using the 'av' embodiment (9D ego-pose).

Action space: AV 9D = [pos_x, pos_y, pos_z, rot_0..rot_5]
  - pos_xyz: frame-to-frame position delta from odom (z always ~0)
  - rot_0..5: rot6d representation of yaw delta from odom quaternion

Usage:
    python convert_roboracer_to_lerobot.py \
        --bags-dir /robodata/fri/spring26/imitation_learning/rosbags/orin10 \
        --output-dir /scratch/tarunrav/roboracer_lerobot \
        --fps 15

Sources this is based on:
    - cosmos-framework/cosmos_framework/data/vfm/action/datasets/base_dataset.py
    - cosmos-framework/cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py
    - cosmos-framework/cosmos_framework/data/vfm/action/domain_utils.py (av: domain_id=1, dim=9)
    - cosmos-framework/cosmos_framework/data/vfm/action/action_spec.py (Pos()+Rot("rot6d") = 9D)
    - LeRobot v3.0 HF docs (LeRobotDataset.create API)
    - LeIsaac docs (must use H.264 encoding for Cosmos)
"""

import argparse
import json
import shutil
import sqlite3
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants matching cosmos-framework domain_utils.py and action_spec.py
# ---------------------------------------------------------------------------
DOMAIN_NAME = "av"
DOMAIN_ID = 1          # from EMBODIMENT_TO_DOMAIN_ID["av"]
ACTION_DIM = 9         # from EMBODIMENT_TO_RAW_ACTION_DIM["av"]
TASK_DESCRIPTION = "Drive the roboracer vehicle to the goal location."

# ROS2 bag topic IDs (confirmed from your bags)
TOPIC_CAMERA = 2       # /camera_0/image_raw/compressed  (sensor_msgs/CompressedImage)
TOPIC_ODOM = 7         # /odom                           (nav_msgs/Odometry)
TOPIC_JOYSTICK = 5     # /joystick                       (sensor_msgs/Joy) - for reference

# Odom pose offset in CDR-serialized nav_msgs/Odometry (confirmed empirically)
ODOM_POSE_OFFSET = 44  # bytes: x(f64), y(f64), z(f64), qx(f64), qy(f64), qz(f64), qw(f64)

# Camera image data offset in CDR-serialized CompressedImage
# Header(4) + stamp(8) + frame_id_len(4) + frame_id + format_len(4) + format + data_len(4) + data
# We'll scan for JPEG SOI marker (0xFF 0xD8) instead of hardcoding

DEFAULT_FPS = 15       # target fps after subsampling (bags are ~30fps)
CHUNK_SIZE = 1000      # max episodes per chunk (LeRobot v3.0 convention)

# LeRobot v3.0 path templates (from lerobot/datasets/utils.py)
INFO_PATH = "meta/info.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"
CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
VIDEO_KEY = "observation.image.camera_0"


# ---------------------------------------------------------------------------
# Rotation utilities (rot6d for Cosmos AV embodiment)
# ---------------------------------------------------------------------------

def quat_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
    xx, xy, xz = qx*qx*s, qx*qy*s, qx*qz*s
    yy, yz, zz = qy*qy*s, qy*qz*s, qz*qz*s
    return np.array([
        [1-(yy+zz),   xy-wz,    xz+wy],
        [  xy+wz,  1-(xx+zz),   yz-wx],
        [  xz-wy,    yz+wx,  1-(xx+yy)],
    ], dtype=np.float32)


def rotation_matrix_to_rot6d(R):
    """Convert 3x3 rotation matrix to rot6d (first two columns, row-major).
    
    rot6d = [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]
    This matches the convention in cosmos-framework/action_spec.py Rot("rot6d").
    """
    return np.array([
        R[0, 0], R[1, 0], R[2, 0],
        R[0, 1], R[1, 1], R[2, 1],
    ], dtype=np.float32)


def compute_pose_delta_9d(pose_t0, pose_t1):
    """Compute 9D AV action = [dx, dy, dz, rot6d] from two absolute poses.
    
    pose = (x, y, z, qx, qy, qz, qw)
    Returns relative delta in the frame of pose_t0.
    """
    x0, y0, z0, qx0, qy0, qz0, qw0 = pose_t0
    x1, y1, z1, qx1, qy1, qz1, qw1 = pose_t1

    # Position delta in world frame → rotate into body frame of t0
    R0 = quat_to_rotation_matrix(qx0, qy0, qz0, qw0)
    dp_world = np.array([x1-x0, y1-y0, z1-z0], dtype=np.float32)
    dp_body = R0.T @ dp_world  # [dx, dy, dz] in body frame

    # Relative rotation R_rel = R0^T @ R1
    R1 = quat_to_rotation_matrix(qx1, qy1, qz1, qw1)
    R_rel = R0.T @ R1
    rot6d = rotation_matrix_to_rot6d(R_rel)

    return np.concatenate([dp_body, rot6d]).astype(np.float32)  # (9,)


# ---------------------------------------------------------------------------
# ROS2 bag reading utilities
# ---------------------------------------------------------------------------

def find_jpeg_start(data: bytes) -> int:
    """Find JPEG SOI marker (0xFF 0xD8) in raw message bytes."""
    for i in range(len(data) - 1):
        if data[i] == 0xFF and data[i+1] == 0xD8:
            return i
    return -1


def decode_compressed_image(data: bytes):
    """Decode a CompressedImage message to a numpy BGR array."""
    raw = bytes(data)
    offset = find_jpeg_start(raw)
    if offset < 0:
        return None
    jpeg_bytes = np.frombuffer(raw[offset:], dtype=np.uint8)
    img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    return img  # BGR, HxWx3


def decode_odom_pose(data: bytes):
    """Decode nav_msgs/Odometry pose: (x, y, z, qx, qy, qz, qw) as float64."""
    raw = bytes(data)
    try:
        vals = struct.unpack_from('<7d', raw, ODOM_POSE_OFFSET)
        x, y, z, qx, qy, qz, qw = vals
        # Sanity check: position reasonable, quaternion unit
        qmag = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
        if abs(x) < 1e6 and abs(y) < 1e6 and 0.8 < qmag < 1.2:
            return (x, y, z, qx, qy, qz, qw)
    except Exception:
        pass
    return None


def read_bag(bag_path: Path):
    """Read all camera frames and odom messages from a bag.
    
    Returns:
        frames: list of (timestamp_ns, jpeg_bytes)
        odom:   list of (timestamp_ns, x, y, z, qx, qy, qz, qw)
    """
    conn = sqlite3.connect(str(bag_path))
    cur = conn.cursor()

    # Read camera frames
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (TOPIC_CAMERA,)
    )
    raw_frames = cur.fetchall()

    # Read odom
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (TOPIC_ODOM,)
    )
    raw_odom = cur.fetchall()
    conn.close()

    frames = []
    for ts, data in raw_frames:
        raw = bytes(data)
        offset = find_jpeg_start(raw)
        if offset >= 0:
            frames.append((ts, raw[offset:]))

    odom = []
    for ts, data in raw_odom:
        pose = decode_odom_pose(data)
        if pose is not None:
            odom.append((ts,) + pose)

    return frames, odom


def subsample_frames(frames, source_fps=30.0, target_fps=15.0):
    """Subsample frames from source_fps to target_fps by keeping every Nth frame."""
    step = max(1, round(source_fps / target_fps))
    return frames[::step]


def sync_odom_to_frames(frames, odom):
    """For each frame timestamp, find the nearest odom reading.
    
    Returns list of poses aligned to frames (same length as frames).
    Falls back to zero delta if no odom within 100ms.
    """
    odom_ts = np.array([o[0] for o in odom], dtype=np.float64)
    synced = []
    for frame_ts, _ in frames:
        if len(odom_ts) == 0:
            synced.append(None)
            continue
        idx = np.argmin(np.abs(odom_ts - frame_ts))
        dt_ms = abs(odom_ts[idx] - frame_ts) / 1e6  # ns → ms
        if dt_ms < 100:
            synced.append(odom[idx][1:])  # (x, y, z, qx, qy, qz, qw)
        else:
            synced.append(None)
    return synced


def compute_actions(poses):
    """Compute 9D AV actions from a list of absolute poses.
    
    Action at frame i = pose delta from frame i to frame i+1.
    Last frame action = zeros (no next frame).
    """
    actions = []
    for i in range(len(poses)):
        if i + 1 < len(poses) and poses[i] is not None and poses[i+1] is not None:
            action = compute_pose_delta_9d(poses[i], poses[i+1])
        else:
            action = np.zeros(9, dtype=np.float32)
        actions.append(action)
    return actions


# ---------------------------------------------------------------------------
# LeRobot v3.0 dataset writer
# ---------------------------------------------------------------------------

class LeRobotV3Writer:
    """Writes a LeRobot v3.0 dataset from scratch.
    
    Based on:
    - lerobot/datasets/utils.py path constants
    - base_dataset.py reader expectations
    - LeRobot v3.0 HF documentation
    """

    def __init__(self, output_dir: Path, fps: int, video_key: str):
        self.output_dir = output_dir
        self.fps = fps
        self.video_key = video_key

        # Create directory structure
        (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
        (output_dir / DATA_DIR / "chunk-000").mkdir(parents=True, exist_ok=True)
        (output_dir / VIDEO_DIR / video_key / "chunk-000").mkdir(parents=True, exist_ok=True)

        self.global_frame_index = 0
        self.episode_index = 0
        self.data_rows = []       # accumulated parquet rows
        self.episode_rows = []    # episode metadata rows

    def write_episode(self, frames_jpeg, actions, episode_name: str):
        """Write one episode: video mp4 + parquet rows.
        
        frames_jpeg: list of JPEG bytes (already subsampled)
        actions:     list of np.ndarray shape (9,)
        episode_name: human-readable name for logging
        """
        n_frames = len(frames_jpeg)
        if n_frames == 0:
            print(f"  Skipping {episode_name}: no frames")
            return

        ep_idx = self.episode_index
        chunk_idx = ep_idx // CHUNK_SIZE
        file_idx = ep_idx % CHUNK_SIZE  # one file per episode within chunk for simplicity

        # --- Write video (H.264 MP4, required by Cosmos/LeIsaac) ---
        video_path = (
            self.output_dir / VIDEO_DIR / self.video_key /
            f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # Decode first frame to get resolution
        first_img = cv2.imdecode(
            np.frombuffer(frames_jpeg[0], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if first_img is None:
            print(f"  Skipping {episode_name}: cannot decode first frame")
            return
        h, w = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_path = video_path.with_suffix('.tmp.mp4')
        writer = cv2.VideoWriter(str(tmp_path), fourcc, self.fps, (w, h))

        frame_start = self.global_frame_index
        for jpeg_bytes in frames_jpeg:
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            writer.write(img)
        writer.release()

        # Re-encode with ffmpeg for proper H.264 (cv2 mp4v is not always H.264)
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-y', '-i', str(tmp_path),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            str(video_path)
        ], capture_output=True)
        tmp_path.unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"  Warning: ffmpeg failed for {episode_name}, using raw mp4v")
            shutil.copy(str(tmp_path), str(video_path))

        # --- Write parquet data rows ---
        for frame_idx, (action) in enumerate(actions[:n_frames]):
            ts = frame_idx / self.fps
            row = {
                "index": self.global_frame_index,
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": float(ts),
                "task_index": 0,
                # AV 9D action: pos_x, pos_y, pos_z, rot_0..rot_5
                "action.pos_x": float(action[0]),
                "action.pos_y": float(action[1]),
                "action.pos_z": float(action[2]),
                "action.rot_0": float(action[3]),
                "action.rot_1": float(action[4]),
                "action.rot_2": float(action[5]),
                "action.rot_3": float(action[6]),
                "action.rot_4": float(action[7]),
                "action.rot_5": float(action[8]),
            }
            self.data_rows.append(row)
            self.global_frame_index += 1

        # --- Episode metadata ---
        self.episode_rows.append({
            "episode_index": ep_idx,
            "tasks": [TASK_DESCRIPTION],
            "length": n_frames,
        })

        self.episode_index += 1
        print(f"  Episode {ep_idx:04d} ({episode_name}): {n_frames} frames → {video_path.name}")

    def finalize(self):
        """Write all parquet files and metadata."""
        print("\nFinalizing dataset...")

        # --- data/chunk-000/file-000.parquet ---
        data_path = self.output_dir / DATA_DIR / "chunk-000" / "file-000.parquet"
        data_table = pa.Table.from_pylist(self.data_rows)
        pq.write_table(data_table, data_path)
        print(f"  Wrote {len(self.data_rows)} rows → {data_path}")

        # --- meta/episodes/chunk-000/file-000.parquet ---
        ep_path = self.output_dir / EPISODES_DIR / "chunk-000" / "file-000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        ep_table = pa.Table.from_pylist(self.episode_rows)
        pq.write_table(ep_table, ep_path)
        print(f"  Wrote {len(self.episode_rows)} episodes → {ep_path}")

        # --- meta/tasks.parquet ---
        tasks_path = self.output_dir / "meta" / "tasks.parquet"
        tasks_table = pa.Table.from_pylist([{"task_index": 0, "task": TASK_DESCRIPTION}])
        pq.write_table(tasks_table, tasks_path)

        # --- meta/info.json ---
        # Schema matches base_dataset.py reader and droid_lerobot_dataset.py conventions
        info = {
            "codebase_version": "v3.0",
            "robot_type": "av",
            "fps": self.fps,
            "domain_name": DOMAIN_NAME,
            "domain_id": DOMAIN_ID,
            "action_dim": ACTION_DIM,
            "total_episodes": self.episode_index,
            "total_frames": self.global_frame_index,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"0:{self.episode_index}"},
            "data_path": DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet",
            "video_path": VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4",
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "action": {
                    "dtype": "float32",
                    "shape": [ACTION_DIM],
                    "names": ["pos_x", "pos_y", "pos_z",
                              "rot_0", "rot_1", "rot_2",
                              "rot_3", "rot_4", "rot_5"],
                },
                self.video_key: {
                    "dtype": "video",
                    "shape": [3, 480, 640],  # will be updated after first frame
                    "names": ["channel", "height", "width"],
                    "video_info": {
                        "video.fps": float(self.fps),
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                    },
                },
            },
        }
        info_path = self.output_dir / INFO_PATH
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Wrote {info_path}")
        print(f"\nDone. Dataset: {self.episode_index} episodes, {self.global_frame_index} frames")
        print(f"Output: {self.output_dir}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(bags_dir: Path, output_dir: Path, fps: int, max_episodes: int = None, bag_names: list[str] | None = None):
    """Convert bags in bags_dir to LeRobot v3.0 format.

    Args:
        bag_names: If given, only convert bags whose parent directory name is in
            this list (e.g. a train/test/eval split) instead of every bag found.
    """

    # Find all .db3 files
    bag_files = sorted(bags_dir.rglob("*.db3"))
    if not bag_files:
        print(f"No .db3 files found in {bags_dir}")
        return

    if bag_names is not None:
        wanted = set(bag_names)
        bag_files = [p for p in bag_files if p.parent.name in wanted]
        missing = wanted - {p.parent.name for p in bag_files}
        if missing:
            raise ValueError(f"Bags listed but not found under {bags_dir}: {sorted(missing)}")
        print(f"Found {len(bag_files)} of {len(wanted)} requested bags in {bags_dir}")
    else:
        print(f"Found {len(bag_files)} bags in {bags_dir}")

    if max_episodes:
        bag_files = bag_files[:max_episodes]
        print(f"Limiting to {max_episodes} episodes")

    if output_dir.exists():
        print(f"Output dir {output_dir} exists, removing...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    writer = LeRobotV3Writer(output_dir, fps=fps, video_key=VIDEO_KEY)
    source_fps = 30.0  # approximate bag camera fps

    for bag_path in bag_files:
        episode_name = bag_path.parent.name
        print(f"\nProcessing {episode_name}...")

        frames, odom = read_bag(bag_path)
        print(f"  Raw: {len(frames)} frames, {len(odom)} odom readings")

        if len(frames) == 0:
            print(f"  Skipping: no camera frames")
            continue

        # Subsample to target fps
        frames = subsample_frames(frames, source_fps=source_fps, target_fps=fps)
        print(f"  After subsampling to {fps}fps: {len(frames)} frames")

        # Sync odom to frame timestamps
        poses = sync_odom_to_frames(frames, odom)
        n_valid_odom = sum(1 for p in poses if p is not None)
        print(f"  Odom sync: {n_valid_odom}/{len(frames)} frames have valid odom")

        # Compute 9D actions
        actions = compute_actions(poses)

        # Extract JPEG bytes
        frames_jpeg = [jpeg for _, jpeg in frames]

        writer.write_episode(frames_jpeg, actions, episode_name)

    writer.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Roboracer bags to LeRobot v3.0")
    parser.add_argument(
        "--bags-dir",
        type=Path,
        default=Path("/robodata/fri/spring26/imitation_learning/rosbags/orin10"),
        help="Directory containing roboracer bag subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/tarunrav/roboracer_lerobot"),
        help="Output LeRobot v3.0 dataset directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Target fps (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes (for testing)"
    )
    parser.add_argument(
        "--bag-split-json",
        type=Path,
        default=None,
        help="Path to a JSON file with {'train': [...], 'test': [...], 'eval': [...]} bag-name lists"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "test", "eval"],
        help="Which key of --bag-split-json to convert (required if --bag-split-json is given)"
    )
    args = parser.parse_args()

    bag_names = None
    if args.bag_split_json is not None:
        if args.split is None:
            raise ValueError("--split is required when --bag-split-json is given")
        import json
        bag_names = json.loads(args.bag_split_json.read_text())[args.split]

    convert(
        bags_dir=args.bags_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_episodes=args.max_episodes,
        bag_names=bag_names,
    )
