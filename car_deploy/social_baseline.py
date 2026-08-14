"""Social navigation baseline: randomly cycles between pass_right, pass_left,
and wait on every inference chunk so we can observe all three behaviors in a
single live run without manually restarting.

Usage (dry run first):
    python3 social_baseline.py --server-ip 10.0.0.211 --server-port 18765

Then live:
    python3 social_baseline.py --server-ip 10.0.0.211 --server-port 18765 --live
"""

import random
import sys
import os

# Patch direction to be re-picked before every inference call.
# We subclass ChunkBufferedPolicyClient and override _inference_loop.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import time
import threading

import msgpack
import websockets

from roboracer_chunk_buffered_client import ChunkBufferedPolicyClient, main as _orig_main

_SOCIAL_DIRECTIONS = ["pass_right", "pass_left", "wait"]


_DIRECTION_HOLD_S = 5.0  # seconds to hold each direction before switching


class SocialRandomClient(ChunkBufferedPolicyClient):
    def __init__(self, args):
        super().__init__(args)
        self._direction_start = time.time()

    async def _inference_loop(self) -> None:
        while True:
            with self._image_lock:
                image = self._latest_image
            if image is None:
                await asyncio.sleep(0.05)
                continue

            # Switch direction only after holding for DIRECTION_HOLD_S seconds.
            if time.time() - self._direction_start >= _DIRECTION_HOLD_S:
                self._direction = random.choice(_SOCIAL_DIRECTIONS)
                self._direction_start = time.time()
                self.get_logger().info(f"[social_baseline] selected direction: {self._direction}")

            t0 = time.time()
            try:
                async with websockets.connect(self._server_uri, max_size=None) as ws:
                    await ws.recv()  # metadata handshake
                    await ws.send(msgpack.packb({
                        "image": image.tobytes(),
                        "shape": list(image.shape),
                        "direction": self._direction,
                    }))
                    response_raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
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
                    f"[{self._direction}] chunk ready: "
                    f"curvature={curvature[0]:+.4f} velocity={velocity[0]:+.4f} "
                    f"(inference {time.time() - t0:.1f}s, {len(new_buffer)} steps)"
                )
                # Wait until buffer is half consumed before fetching next chunk.
                # Prevents thrashing when inference is faster than playback.
                await asyncio.sleep(len(new_buffer) / 2.0 / 15.0)
            except Exception as exc:
                self.get_logger().error(f"inference call failed: {exc}")
                await asyncio.sleep(0.5)


def main() -> None:
    import rclpy

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", type=str, default="10.0.0.211")
    parser.add_argument("--server-port", type=int, default=18765)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--max-curvature", type=float, default=3.0)
    parser.add_argument("--max-buffer-age-s", type=float, default=3.0)
    parser.add_argument("--subgoal-lookahead-s", type=float, default=5.0)
    parser.add_argument("--record-bag", type=str, default=None)
    # direction is ignored — randomized per chunk — but keep it so argparse doesn't error
    parser.add_argument("--direction", type=str, default="pass_right")
    args = parser.parse_args()

    rclpy.init()
    node = SocialRandomClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
