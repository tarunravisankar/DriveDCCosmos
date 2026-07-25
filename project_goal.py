"""project_goal.py

Projects a 2D ground-plane goal position into the camera image frame and
draws a goal-dot indicator. Used at training time (baked into video frames
in convert_roboracer_to_lerobot.py) and at inference time (overlaid on live
camera frames in roboracer_chunk_buffered_client.py).

Camera: ArduCam IMX219 8MP on 1/10-scale Orin car
  - Resolution used in bags: 320x240 (4:3 native sensor mode, wider than 1080p crop)
  - Horizontal FOV: ~66 degrees (4:3 native mode; 1080p 16:9 crop is 62.2 deg)
  - Camera height above ground: ~7.3 cm (2.8-3 inches, measured)
  - Camera pitch: 0 degrees (horizontal, not angled downward)

Projection model: simplified pinhole, camera looking forward (+x_robot),
left is +y_robot. Goal is on the ground plane (z=0 in world frame).

Goal dot conventions:
  - GREEN filled circle: goal is within camera FOV (in-image)
  - YELLOW filled circle on image edge: goal is off-screen (compass indicator)
  - Dot radius scales with distance: larger = closer (max at 0.5m, min at 5m+)
"""

from __future__ import annotations

import math
import numpy as np
import cv2


# --- Camera constants (ArduCam IMX219, 320x240 4:3 native mode) ---
IMG_W = 320
IMG_H = 240
CAM_CX = IMG_W / 2.0          # 160.0
CAM_CY = IMG_H / 2.0          # 120.0
CAM_FOV_H_DEG = 66.0           # horizontal FOV, degrees
CAM_HEIGHT_M = 0.073           # camera height above ground, meters
CAM_PITCH_RAD = 0.0            # camera pitch, radians (0 = horizontal)

# Derived focal length (pixels)
CAM_F = (IMG_W / 2.0) / math.tan(math.radians(CAM_FOV_H_DEG / 2.0))

# Dot appearance
DOT_COLOR_INFRAME = (0, 220, 0)     # bright green: goal is visible
DOT_COLOR_OFFSCREEN = (0, 200, 255) # yellow: off-screen compass indicator
DOT_RADIUS_MAX = 10
DOT_RADIUS_MIN = 4
DOT_DIST_CLOSE = 0.5   # meters: at this distance, use max radius
DOT_DIST_FAR   = 5.0   # meters: at this distance, use min radius


def _dot_radius(dist_m: float) -> int:
    """Scale dot radius with proximity — bigger dot = closer goal."""
    t = 1.0 - (dist_m - DOT_DIST_CLOSE) / (DOT_DIST_FAR - DOT_DIST_CLOSE)
    t = max(0.0, min(1.0, t))
    return int(DOT_RADIUS_MIN + t * (DOT_RADIUS_MAX - DOT_RADIUS_MIN))


def _clamp_to_edge(u: float, v: float, cx: float, cy: float, w: int, h: int):
    """
    Given an out-of-bounds projected point (u,v), find where the ray from
    image center (cx,cy) through (u,v) intersects the image border.
    Returns (u_edge, v_edge) clamped to the border.
    """
    du = u - cx
    dv = v - cy
    if du == 0 and dv == 0:
        return int(cx), int(cy)

    # Max scale factors to hit each edge
    scales = []
    if du > 0:
        scales.append((w - 1 - cx) / du)
    elif du < 0:
        scales.append(-cx / du)
    if dv > 0:
        scales.append((h - 1 - cy) / dv)
    elif dv < 0:
        scales.append(-cy / dv)

    s = min(s for s in scales if s > 0)
    return int(cx + s * du), int(cy + s * dv)


def project_goal_onto_image(
    image: np.ndarray,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
    cam_height: float = CAM_HEIGHT_M,
    cam_pitch: float = CAM_PITCH_RAD,
    cam_f: float = CAM_F,
    cam_cx: float = CAM_CX,
    cam_cy: float = CAM_CY,
) -> np.ndarray:
    """
    Draw a goal indicator dot on `image` and return the annotated copy.

    Args:
        image:       HxWx3 uint8 BGR (OpenCV format, as decoded from bag)
        robot_x/y:   robot position in odom frame (meters)
        robot_yaw:   robot heading in odom frame (radians, CCW positive)
        goal_x/y:    goal position in odom frame (meters)
        cam_*:       camera intrinsics (defaults = IMX219 320x240)

    Returns:
        Annotated image (copy, original unchanged).
    """
    out = image.copy()
    h, w = out.shape[:2]

    # --- Step 1: goal in robot-local frame (forward=+x_local, left=+y_local) ---
    dx_world = goal_x - robot_x
    dy_world = goal_y - robot_y
    cos_y = math.cos(-robot_yaw)
    sin_y = math.sin(-robot_yaw)
    dx_local = dx_world * cos_y - dy_world * sin_y   # forward
    dy_local = dx_world * sin_y + dy_world * cos_y   # left

    dist = math.sqrt(dx_local ** 2 + dy_local ** 2)

    # --- Step 2: project into image ---
    # Horizontal: angle to goal from camera axis.
    # dy_local > 0 means LEFT of robot. In image, left = smaller u, so negate.
    alpha = math.atan2(dy_local, max(dx_local, 1e-3))  # positive = goal is LEFT
    u = cam_cx - cam_f * math.tan(alpha)               # LEFT → smaller u ✓

    # Vertical: goal is on ground at distance dist, camera at cam_height,
    # pitched down by cam_pitch from horizontal.
    # Depression angle to goal: atan2(cam_height, dist)
    # Image row: points BELOW horizon are at v > cam_cy
    if dist > 0.01:
        beta = math.atan2(cam_height, dist)  # angle below horizontal
        v = cam_cy + cam_f * math.tan(beta - cam_pitch)
    else:
        # Goal is directly under the camera — show at bottom-center
        v = h - 1.0
        u = cam_cx

    # --- Step 3: draw ---
    radius = _dot_radius(dist)
    in_frame = (0 <= u < w) and (0 <= v < h)

    if dx_local <= 0:
        # Goal is behind the car — always show as compass on bottom edge
        pu = int(cam_cx - cam_f * math.tan(math.atan2(dy_local, 1e-3)))
        eu, ev = _clamp_to_edge(pu, h + 50, cam_cx, cam_cy, w, h)
        cv2.circle(out, (eu, ev), DOT_RADIUS_MIN + 1, DOT_COLOR_OFFSCREEN, -1)
        cv2.circle(out, (eu, ev), DOT_RADIUS_MIN + 1, (255, 255, 255), 1)
    elif in_frame:
        pu, pv = int(u), int(v)
        cv2.circle(out, (pu, pv), radius, DOT_COLOR_INFRAME, -1)
        cv2.circle(out, (pu, pv), radius, (255, 255, 255), 1)  # white border
    else:
        eu, ev = _clamp_to_edge(u, v, cam_cx, cam_cy, w, h)
        cv2.circle(out, (eu, ev), DOT_RADIUS_MIN + 1, DOT_COLOR_OFFSCREEN, -1)
        cv2.circle(out, (eu, ev), DOT_RADIUS_MIN + 1, (255, 255, 255), 1)

    return out


def compute_subgoal_poses(
    odom: list[tuple],
    lookahead_s: float = 5.0,
) -> list[tuple[float, float]]:
    """
    For each odom reading, return the (x, y) position T seconds ahead in the
    same episode as the subgoal. Used for loop-track bags where there is no
    fixed destination — the dot shows where on the loop the car is heading next.

    For the last `lookahead_s` seconds of the episode the final position is used.

    Args:
        odom: list of (timestamp_ns, x, y, ...)
        lookahead_s: lookahead window in seconds (default 5s)

    Returns:
        list of (goal_x, goal_y), same length as odom
    """
    lookahead_ns = int(lookahead_s * 1e9)
    timestamps = np.array([o[0] for o in odom])
    xs = np.array([o[1] for o in odom])
    ys = np.array([o[2] for o in odom])

    goals = []
    for (ts, *_) in odom:
        target_ts = ts + lookahead_ns
        idx = np.searchsorted(timestamps, target_ts)
        idx = min(idx, len(odom) - 1)
        goals.append((xs[idx], ys[idx]))

    return goals


def compute_goal_poses(
    odom: list[tuple],
    goal_x: float,
    goal_y: float,
) -> list[tuple[float, float]]:
    """
    Return a fixed explicit goal for every frame — for point-to-point navigation
    where the destination coordinate is known in advance (future data collection).

    Args:
        odom:   list of (timestamp_ns, x, y, ...)
        goal_x: destination x in odom frame (meters)
        goal_y: destination y in odom frame (meters)

    Returns:
        list of (goal_x, goal_y), same length as odom
    """
    return [(goal_x, goal_y)] * len(odom)


# --- Quick visual test ---
if __name__ == "__main__":
    # Synthetic test: robot at origin, facing +x. Draw goals at various positions.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark grey background

    test_cases = [
        (2.0, 0.0, "straight ahead 2m"),
        (2.0, 1.0, "ahead-left 2m"),
        (2.0, -1.0, "ahead-right 2m"),
        (0.5, 0.0, "very close 0.5m"),
        (10.0, 0.0, "far ahead 10m"),
        (-1.0, 0.0, "behind"),
        (0.0, 5.0, "hard left (off-screen)"),
    ]

    for gx, gy, label in test_cases:
        annotated = project_goal_onto_image(img.copy(), 0, 0, 0, gx, gy)
        outpath = f"/tmp/goal_test_{label.replace(' ', '_')}.png"
        cv2.imwrite(outpath, annotated)
        print(f"{label}: saved {outpath}")
