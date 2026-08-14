"""Chunk-buffered ROS2 client for the RoboRacer action policy.

Decouples the model's inference rate from the car's required command rate:
  - a fast publish loop runs at ~15Hz (matching conditioning_fps), popping the
    next step out of the current predicted chunk and either logging it
    (--dry-run, the default) or publishing it to /ackermann_curvature_drive;
  - a separate inference loop calls the server back-to-back with the latest
    camera frame and replaces the buffer with each fresh 32-step chunk as
    soon as it arrives (receding-horizon style: every new chunk supersedes
    whatever was left of the old one, so the car is always acting on the
    most recent prediction rather than a stale one).

Fail-safe: if the buffer is empty or older than --max-buffer-age-s (e.g. the
server is unreachable or a single inference call stalls), publishes a STOP
command (velocity=0, curvature=0) instead of continuing on stale data.

SAFETY (--dry-run is the default and must be deliberately disabled):
  - --dry-run (default): never constructs or sends an AckermannCurvatureDriveMsg,
    only logs what would have been sent.
  - --live: actually creates the publisher and calls publish(). Even then,
    the car's own vesc_driver.cpp ignores this topic entirely unless a human
    has put the car into autonomous mode via the joystick, and instantly
    overrides per-axis on any joystick input past a small deadzone (confirmed
    by reading vesc_driver.cpp's ackermannCurvatureCallback/isAutonomous()).
  - --max-velocity / --max-curvature: hard clamps applied to every published
    command regardless of what the model predicts, as a second safety net
    independent of the car's own override logic.

Run inside the orin_roboracer container:
    python3 roboracer_chunk_buffered_client.py --server-ip 10.0.0.212 --server-port 18765
    # add --direction right to drive the right-handed (clockwise) course instead
    # of the default left-handed loop - must match the actual course or the
    # model is conditioned on the wrong text caption (see action_policy_server_roboracer.py)
    # add --live only when actually ready to test with a human at the joystick
    # add --record-bag <name> to capture a DAgger-correction episode while driving:
    # records exactly the two topics convert_roboracer_to_lerobot.py needs
    # (/camera_0/image_raw/compressed, /odom) via a `ros2 bag record` subprocess,
    # started on launch and stopped cleanly (SIGINT, so the bag finalizes properly)
    # on Ctrl+C - same bag format as the original training data, so no new
    # conversion code needed.
"""

import argparse
import asyncio
import math
import signal
import subprocess
import threading
import time

import cv2
import msgpack
import numpy as np
import rclpy
import websockets
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

_CONDITIONING_FPS = 15.0
_PUBLISH_PERIOD_S = 1.0 / _CONDITIONING_FPS

# Camera constants (ArduCam IMX219, 320x240 native 4:3 mode — must match training)
_CAM_W, _CAM_H = 320, 240
_CAM_CX, _CAM_CY = 160.0, 120.0
_CAM_FOV_H_DEG = 66.0
_CAM_HEIGHT_M = 0.073   # measured: 2.8–3 inches off floor
_CAM_PITCH_RAD = 0.0    # horizontal
_CAM_F = (_CAM_W / 2.0) / math.tan(math.radians(_CAM_FOV_H_DEG / 2.0))


def _project_goal_onto_image(image_bgr: np.ndarray, rx: float, ry: float, ryaw: float,
                              gx: float, gy: float) -> np.ndarray:
    """Draw 5s-subgoal dot on image_bgr (in-place copy). Mirrors project_goal.py."""
    out = image_bgr.copy()
    h, w = out.shape[:2]

    cos_y, sin_y = math.cos(-ryaw), math.sin(-ryaw)
    dx_w, dy_w = gx - rx, gy - ry
    dx = dx_w * cos_y - dy_w * sin_y   # forward
    dy = dx_w * sin_y + dy_w * cos_y   # left

    dist = math.sqrt(dx * dx + dy * dy)

    if dx <= 0:
        # Behind robot: compass on bottom edge
        alpha = math.atan2(dy, 1e-3)
        eu = int(_CAM_CX - _CAM_F * math.tan(alpha))
        eu = max(0, min(w - 1, eu))
        cv2.circle(out, (eu, h - 1), 5, (0, 200, 255), -1)
        cv2.circle(out, (eu, h - 1), 5, (255, 255, 255), 1)
        return out

    alpha = math.atan2(dy, max(dx, 1e-3))
    u = _CAM_CX - _CAM_F * math.tan(alpha)
    beta = math.atan2(_CAM_HEIGHT_M, dist)
    v = _CAM_CY + _CAM_F * math.tan(beta - _CAM_PITCH_RAD)

    # Dot size: larger = closer
    t = max(0.0, min(1.0, 1.0 - (dist - 0.5) / 4.5))
    radius = int(4 + t * 6)

    in_frame = 0 <= u < w and 0 <= v < h
    if in_frame:
        cv2.circle(out, (int(u), int(v)), radius, (0, 220, 0), -1)
        cv2.circle(out, (int(u), int(v)), radius, (255, 255, 255), 1)
    else:
        # Compass on nearest edge
        du, dv = u - _CAM_CX, v - _CAM_CY
        scales = []
        if du > 0: scales.append((w - 1 - _CAM_CX) / du)
        elif du < 0: scales.append(-_CAM_CX / du)
        if dv > 0: scales.append((h - 1 - _CAM_CY) / dv)
        elif dv < 0: scales.append(-_CAM_CY / dv)
        s = min(sc for sc in scales if sc > 0)
        eu, ev = int(_CAM_CX + s * du), int(_CAM_CY + s * dv)
        cv2.circle(out, (eu, ev), 5, (0, 200, 255), -1)
        cv2.circle(out, (eu, ev), 5, (255, 255, 255), 1)
    return out


class ChunkBufferedPolicyClient(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("roboracer_chunk_buffered_policy_client")
        self._server_uri = f"ws://{args.server_ip}:{args.server_port}"
        self._live = args.live
        self._max_velocity = args.max_velocity
        self._max_curvature = args.max_curvature
        self._max_buffer_age_s = args.max_buffer_age_s
        self._direction = args.direction

        self._latest_image: np.ndarray | None = None
        self._image_lock = threading.Lock()

        # Latest odom: (x, y, yaw, v_lin, v_ang) — used to dead-reckon subgoal
        self._latest_odom: tuple | None = None
        self._odom_lock = threading.Lock()
        self._subgoal_lookahead_s = args.subgoal_lookahead_s
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, 1)

        # Buffer state: list of (curvature, velocity) steps + a cursor + the
        # wallclock time the chunk was produced (for staleness fail-safe).
        self._buffer: list[tuple[float, float]] = []
        self._buffer_idx = 0
        self._buffer_time = 0.0
        self._buffer_lock = threading.Lock()

        self._publisher = None
        if self._live:
            from amrl_msgs.msg import AckermannCurvatureDriveMsg  # local import: only needed in --live mode
            self._AckermannCurvatureDriveMsg = AckermannCurvatureDriveMsg
            self._publisher = self.create_publisher(AckermannCurvatureDriveMsg, "/ackermann_curvature_drive", 1)
            self.get_logger().warn(
                "[LIVE MODE] this node WILL publish to /ackermann_curvature_drive. "
                "The car's vesc_driver only acts on this if autonomous mode is toggled "
                "on the joystick, and joystick input instantly overrides per-axis."
            )
        else:
            self.get_logger().info("[DRY RUN] this node will NEVER publish to any control topic - log-only.")

        self.sub = self.create_subscription(Image, "/camera_0/image_raw", self._on_image, 1)

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        asyncio.run_coroutine_threadsafe(self._inference_loop(), self._loop)

        # Publish/log tick runs on a regular rclpy timer (main thread), independent
        # of inference latency - this is what keeps us under the car's 0.5s timeout.
        self.create_timer(_PUBLISH_PERIOD_S, self._on_publish_tick)

        self.get_logger().info(
            f"roboracer chunk-buffered policy client started. Server: {self._server_uri}. "
            f"live={self._live} max_velocity={self._max_velocity} max_curvature={self._max_curvature} "
            f"direction={self._direction}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        with self._odom_lock:
            self._latest_odom = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                yaw,
                msg.twist.twist.linear.x,   # forward velocity (body frame)
                msg.twist.twist.angular.z,   # yaw rate
            )

    def _annotate_subgoal(self, bgr: np.ndarray) -> np.ndarray:
        """Dead-reckon subgoal_lookahead_s ahead and draw goal dot on bgr frame."""
        # <=0 means "no goal dot", which is what the current checkpoints expect.
        # Without this guard a lookahead of 0 still drew a degenerate dot: the
        # goal collapses onto the robot, dx==0 takes the "behind robot" branch,
        # and a compass dot lands at bottom-centre of every frame.
        if self._subgoal_lookahead_s <= 0.0:
            return bgr
        with self._odom_lock:
            odom = self._latest_odom
        if odom is None:
            return bgr
        x, y, yaw, v_lin, v_ang = odom
        dt = self._subgoal_lookahead_s
        if abs(v_ang) < 1e-4:
            gx = x + v_lin * dt * math.cos(yaw)
            gy = y + v_lin * dt * math.sin(yaw)
        else:
            r = v_lin / v_ang
            new_yaw = yaw + v_ang * dt
            gx = x + r * (math.sin(new_yaw) - math.sin(yaw))
            gy = y - r * (math.cos(new_yaw) - math.cos(yaw))
        return _project_goal_onto_image(bgr, x, y, yaw, gx, gy)

    def _on_image(self, msg: Image) -> None:
        if msg.encoding != "bgr8":
            self.get_logger().warn(f"unexpected encoding {msg.encoding!r}, expected bgr8 - skipping frame")
            return
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        bgr = self._annotate_subgoal(bgr)
        rgb = bgr[:, :, ::-1].copy()
        with self._image_lock:
            self._latest_image = rgb

    async def _inference_loop(self) -> None:
        while True:
            with self._image_lock:
                image = self._latest_image
            if image is None:
                await asyncio.sleep(0.05)
                continue
            t0 = time.time()
            try:
                async with websockets.connect(self._server_uri, max_size=None) as ws:
                    await ws.recv()  # metadata handshake
                    await ws.send(msgpack.packb({
                        "image": image.tobytes(), "shape": list(image.shape), "direction": self._direction,
                    }))
                    response_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                response = msgpack.unpackb(response_raw, raw=False)
                if "error" in response:
                    self.get_logger().error(f"server error: {response['error']}")
                    continue
                curvature = response["curvature"]
                velocity = response["velocity"]
                new_buffer = list(zip(curvature, velocity))
                with self._buffer_lock:
                    self._buffer = new_buffer
                    self._buffer_idx = 0
                    self._buffer_time = time.time()
                self.get_logger().info(
                    f"new chunk ready: first-step curvature={curvature[0]:+.4f} velocity={velocity[0]:+.4f} "
                    f"(inference took {time.time() - t0:.2f}s, {len(new_buffer)} steps)"
                )
            except Exception as exc:  # noqa: BLE001 - keep replanning on any single failure
                self.get_logger().error(f"inference call failed: {exc}")
                await asyncio.sleep(0.5)

    def _on_publish_tick(self) -> None:
        now = time.time()
        with self._buffer_lock:
            buffer_age = now - self._buffer_time if self._buffer else None
            if (
                not self._buffer
                or self._buffer_idx >= len(self._buffer)
                or (buffer_age is not None and buffer_age > self._max_buffer_age_s)
            ):
                curvature, velocity = 0.0, 0.0
                reason = "buffer empty/exhausted/stale -> fail-safe STOP"
            else:
                curvature, velocity = self._buffer[self._buffer_idx]
                self._buffer_idx += 1
                reason = None

        velocity = float(np.clip(velocity, -self._max_velocity, self._max_velocity))
        curvature = float(np.clip(curvature, -self._max_curvature, self._max_curvature))

        if self._live:
            msg = self._AckermannCurvatureDriveMsg()
            msg.velocity = velocity
            msg.curvature = curvature
            self._publisher.publish(msg)
        if reason or self.get_logger().get_effective_level() <= 10:  # DEBUG, or always log fail-safe events
            if reason:
                self.get_logger().warn(f"{reason} (velocity=0, curvature=0)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", type=str, default="10.0.0.212")
    parser.add_argument("--server-port", type=int, default=18765)
    parser.add_argument("--live", action="store_true",
                         help="Actually publish to /ackermann_curvature_drive. Default is dry-run (log only).")
    parser.add_argument("--max-velocity", type=float, default=1.0, help="Hard clamp on |velocity| in m/s.")
    # 1.3 1/m is the car's physical steering limit: car.lua has
    # max_steering_angle=0.4030 rad, so tan(0.4030)/wheelbase(0.32) = 1.33 1/m.
    # The old default of 3.0 was above anything the servo could execute, so the
    # clamp never actually bounded the command -- it just handed vesc_driver a
    # value that saturated the steering servo at full lock.
    parser.add_argument("--max-curvature", type=float, default=1.3, help="Hard clamp on |curvature| in 1/m.")
    parser.add_argument("--max-buffer-age-s", type=float, default=3.0,
                         help="Fail-safe to STOP if the current chunk is older than this.")
    parser.add_argument("--direction", type=str, default="loop_ccw",
                         choices=[
                             # Navigation track types
                             "loop_ccw", "loop_cw",
                             "oval_ccw", "circle_ccw",
                             "rect_small_ccw", "rect_small_cw",
                             "rect_med_ccw", "square_ccw",
                             # Person avoidance
                             "pass_right", "pass_left", "wait",
                             # Legacy aliases
                             "left", "right",
                         ],
                         help="Behavior to condition on. Navigation: loop_ccw (orin10), "
                              "loop_cw (orin13), oval_ccw (orin02), circle_ccw (orin03), "
                              "rect_small_ccw (orin04/06), rect_small_cw (orin08), "
                              "rect_med_ccw (orin05), square_ccw (orin14). "
                              "Person avoidance: pass_right, pass_left, wait. "
                              "See action_policy_server_roboracer.py _DIRECTION_CAPTIONS.")
    parser.add_argument("--subgoal-lookahead-s", type=float, default=0.0,
                         help="Seconds ahead to dead-reckon the subgoal dot on each frame "
                              "(must match the lookahead used at training time). Default 0 "
                              "(disabled) because convert_all_datasets.sh sets LOOKAHEAD=0 -- "
                              "the shipped checkpoints were trained on frames with NO goal dot. "
                              "Only set this if you retrain with dots enabled.")
    parser.add_argument("--record-bag", type=str, default=None,
                         help="If given, record a ros2 bag of this name alongside driving, "
                              "capturing /camera_0/image_raw/compressed and /odom - the same "
                              "topics convert_roboracer_to_lerobot.py expects, for folding this "
                              "session back into the training set (DAgger-style).")
    args = parser.parse_args()

    bag_process = None
    if args.record_bag:
        bag_process = subprocess.Popen([
            "ros2", "bag", "record", "-o", args.record_bag,
            "/camera_0/image_raw/compressed", "/odom",
        ])
        print(f"[record-bag] started ros2 bag record -> {args.record_bag} (pid {bag_process.pid})")

    rclpy.init()
    node = ChunkBufferedPolicyClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if bag_process is not None:
            # SIGINT (not kill -9) so ros2 bag record finalizes the .db3 file cleanly.
            bag_process.send_signal(signal.SIGINT)
            try:
                bag_process.wait(timeout=10)
                print(f"[record-bag] stopped cleanly -> {args.record_bag}")
            except subprocess.TimeoutExpired:
                print(f"[record-bag] did not exit within 10s, sending SIGTERM -> {args.record_bag}")
                bag_process.terminate()


if __name__ == "__main__":
    main()
