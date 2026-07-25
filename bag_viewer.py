# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal web GUI for browsing/rewatching roboracer rosbags across all orins.

Lists every orin directory under BAGS_ROOT, lets you pick a bag, and plays its
camera feed as an MP4 in the browser. Raw JPEG frames are pulled directly out
of the bag's sqlite3 messages table (same approach as convert_roboracer_to_lerobot.py's
read_bag/find_jpeg_start) and piped straight into ffmpeg - no intermediate
per-frame files, no ROS/rosbags dependency. Generated MP4s are cached on disk
so re-watching a bag is instant after the first view.

Usage:
    python3 bag_viewer.py [--port 5050]
    # then open http://<host>:5050 in a browser
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, render_template_string, send_file

BAGS_ROOT = Path("/robodata/fri/spring26/imitation_learning/rosbags")
CACHE_ROOT = Path("/scratch/tarunrav/roboracer_bag_viewer_cache")
TOPIC_CAMERA_NAME = "/camera_0/image_raw/compressed"
SOURCE_FPS = 30  # matches recording rate used elsewhere in this session

CACHE_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Matches both "roboracer_YYYYMMDD_HHMMSS" and "YYYYMMDD_HHMMSS_description" (orin07)
_BAG_NAME_RES = [
    re.compile(r"roboracer_(\d{8})_(\d{6})"),
    re.compile(r"^(\d{8})_(\d{6})"),
]


def find_jpeg_start(data: bytes) -> int:
    for i in range(len(data) - 1):
        if data[i] == 0xFF and data[i + 1] == 0xD8:
            return i
    return -1


def parse_bag_datetime(bag_name: str) -> datetime | None:
    for pattern in _BAG_NAME_RES:
        m = pattern.search(bag_name)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                continue
    return None


def find_db3(bag_dir: Path) -> Path | None:
    # Check top level first, then one level deep (orin07 may nest the bag in a subdir)
    matches = list(bag_dir.glob("*.db3")) or list(bag_dir.glob("*/*.db3"))
    return matches[0] if matches else None


def list_orins() -> list[str]:
    if not BAGS_ROOT.is_dir():
        return []
    return sorted(p.name for p in BAGS_ROOT.iterdir() if p.is_dir())


def list_bags(orin: str) -> list[dict]:
    orin_dir = BAGS_ROOT / orin
    if not orin_dir.is_dir():
        abort(404)
    bags = []
    for bag_dir in sorted(orin_dir.iterdir()):
        if not bag_dir.is_dir():
            continue
        db3 = find_db3(bag_dir)
        if db3 is None:
            continue
        dt = parse_bag_datetime(bag_dir.name)
        size_mb = db3.stat().st_size / (1024 * 1024)
        cached = (CACHE_ROOT / orin / f"{bag_dir.name}.mp4").exists()
        bags.append(
            {
                "name": bag_dir.name,
                "datetime": dt,
                "size_mb": size_mb,
                "cached": cached,
            }
        )
    bags.sort(key=lambda b: b["datetime"] or datetime.min)
    return bags


def generate_mp4(orin: str, bag_name: str) -> Path:
    """Extract camera JPEG frames from the bag and encode to MP4, cached on disk."""
    out_dir = CACHE_ROOT / orin
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bag_name}.mp4"
    if out_path.exists():
        return out_path

    bag_dir = BAGS_ROOT / orin / bag_name
    db3 = find_db3(bag_dir)
    if db3 is None:
        abort(404, f"no .db3 file found for bag {bag_name!r}")

    conn = sqlite3.connect(str(db3))
    cur = conn.cursor()
    cur.execute("SELECT id FROM topics WHERE name=?", (TOPIC_CAMERA_NAME,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        abort(404, f"bag {bag_name!r} has no {TOPIC_CAMERA_NAME} topic")
    topic_id = row[0]
    cur.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)
    )
    raw_frames = cur.fetchall()
    conn.close()

    if not raw_frames:
        abort(404, f"bag {bag_name!r} has zero camera frames")

    tmp_path = out_path.with_suffix(".mp4.tmp")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "image2pipe",
            "-framerate", str(SOURCE_FPS),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-f", "mp4",  # tmp_path ends in .tmp, not .mp4 - ffmpeg can't infer the muxer from it
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for (data,) in raw_frames:
        raw = bytes(data)
        offset = find_jpeg_start(raw)
        if offset >= 0:
            proc.stdin.write(raw[offset:])
    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0 or not tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
        abort(500, f"ffmpeg failed for bag {bag_name!r}: {stderr.decode(errors='replace')[-2000:]}")

    tmp_path.rename(out_path)
    return out_path


INDEX_TEMPLATE = """
<!doctype html><html><head><title>Roboracer Bag Viewer</title>
<style>body{font-family:sans-serif;margin:2em;background:#111;color:#eee}
a{color:#6cf;text-decoration:none} a:hover{text-decoration:underline}
.card{display:inline-block;background:#1c1c1c;border-radius:8px;padding:1em 1.5em;margin:0.5em;min-width:120px;text-align:center}
.count{color:#888;font-size:0.9em}</style></head><body>
<h1>Roboracer Bag Viewer</h1>
<p>{{ orins|length }} orin device(s) under <code>{{ bags_root }}</code></p>
{% for o in orins %}
<a class="card" href="/orin/{{ o.name }}"><div><b>{{ o.name }}</b></div><div class="count">{{ o.count }} bags</div></a>
{% endfor %}
</body></html>
"""

ORIN_TEMPLATE = """
<!doctype html><html><head><title>{{ orin }} - Roboracer Bag Viewer</title>
<style>body{font-family:sans-serif;margin:2em;background:#111;color:#eee}
a{color:#6cf;text-decoration:none} a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%} td,th{padding:0.4em 0.8em;text-align:left;border-bottom:1px solid #333}
.cached{color:#6f6} .uncached{color:#888}</style></head><body>
<p><a href="/">&larr; all orins</a></p>
<h1>{{ orin }}</h1>
<table><tr><th>Bag</th><th>Recorded</th><th>Size</th><th>Status</th></tr>
{% for b in bags %}
<tr><td><a href="/orin/{{ orin }}/{{ b.name }}">{{ b.name }}</a></td>
<td>{{ b.datetime.strftime('%Y-%m-%d %H:%M:%S') if b.datetime else '?' }}</td>
<td>{{ "%.1f"|format(b.size_mb) }} MB</td>
<td class="{{ 'cached' if b.cached else 'uncached' }}">{{ 'cached' if b.cached else 'not generated yet' }}</td></tr>
{% endfor %}
</table>
</body></html>
"""

PLAYER_TEMPLATE = """
<!doctype html><html><head><title>{{ bag }} - {{ orin }}</title>
<style>body{font-family:sans-serif;margin:2em;background:#111;color:#eee}
a{color:#6cf;text-decoration:none} a:hover{text-decoration:underline}
video{max-width:100%;border-radius:8px}</style></head><body>
<p><a href="/orin/{{ orin }}">&larr; {{ orin }} bags</a></p>
<h1>{{ bag }}</h1>
<video controls autoplay src="/video/{{ orin }}/{{ bag }}.mp4"></video>
</body></html>
"""


@app.route("/")
def index():
    orins = [{"name": o, "count": len(list_bags(o))} for o in list_orins()]
    return render_template_string(INDEX_TEMPLATE, orins=orins, bags_root=str(BAGS_ROOT))


@app.route("/orin/<orin>")
def orin_page(orin: str):
    return render_template_string(ORIN_TEMPLATE, orin=orin, bags=list_bags(orin))


@app.route("/orin/<orin>/<bag>")
def player_page(orin: str, bag: str):
    return render_template_string(PLAYER_TEMPLATE, orin=orin, bag=bag)


@app.route("/video/<orin>/<bag>.mp4")
def video(orin: str, bag: str):
    path = generate_mp4(orin, bag)
    return send_file(path, mimetype="video/mp4", conditional=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
