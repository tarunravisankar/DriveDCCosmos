"""Velocity-stratified eval over the websocket, against a running policy server.

Same sample selection and metrics as velocity_eval.py, but the model is reached
through the deployed server instead of being loaded in-process. That sidesteps
the loader entirely and measures exactly the pipeline the car drives with:
single camera frame + direction token -> curvature/velocity.
"""
import argparse, asyncio, math, statistics as st, sys
from pathlib import Path
import numpy as np, msgpack, websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosmos_framework.data.generator.action.action_normalization import denormalize_action, load_action_stats
from cosmos_framework.data.generator.action.datasets.roboracer_dataset import (
    _STATS_PATH, RoboracerDataset, get_action_roboracer_sft_dataset)
import torch

MAX_POS_DELTA_M, MAX_YAW_DELTA_DEG = 1.0, 45.0

def stratified_by_velocity(ds, n):
    cands, step = [], max(1, len(ds) // 4000)
    for i in range(0, len(ds), step):
        row = ds._rows[ds._valid_windows[i][0] + 1]
        px, py = float(row.get("action.pos_x", 0.0)), float(row.get("action.pos_y", 0.0))
        r0, r1 = float(row.get("action.rot_0", 1.0)), float(row.get("action.rot_1", 0.0))
        if abs(px) > MAX_POS_DELTA_M or abs(py) > MAX_POS_DELTA_M: continue
        if abs(math.degrees(math.atan2(r1, r0))) > MAX_YAW_DELTA_DEG: continue
        cands.append((i, px))
    if not cands: return []
    cands.sort(key=lambda t: t[1])
    picks, seen = [], set()
    for k in range(n):
        j = min(len(cands) - 1, round(k * (len(cands) - 1) / max(1, n - 1)))
        if cands[j][0] not in seen:
            seen.add(cands[j][0]); picks.append(cands[j][0])
    return picks

def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v)
        for p, i in enumerate(o): r[i] = p
        return r
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    den = (sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**0.5
    return num/den if den else float("nan")

TOKEN = {"wait": "wait", "pass_left": "pass_left", "pass_right": "pass_right"}

async def run(root, uri, n, stats, label):
    ds = RoboracerDataset(root=root, fps=15.0, chunk_length=32,
                          action_normalization="minmax", mode="wam")
    sft = get_action_roboracer_sft_dataset(root=root, fps=15.0, chunk_length=32, mode="wam",
        action_normalization="minmax", use_image_augmentation=False, oversample_turns=False,
        resolution="256", max_action_dim=64, cfg_dropout_rate=0.0)
    idxs = stratified_by_velocity(ds, n)
    name = root.rstrip("/").split("/")[-1]
    direction = next((v for k, v in TOKEN.items() if k in name), "loop_ccw")
    print(f"[{label}] {name}: {len(idxs)} samples, token={direction}", flush=True)
    gts, prs, gy, py_ = [], [], [], []
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.recv()
        for c, idx in enumerate(idxs):
            gt = denormalize_action(ds[idx]["action"], "minmax", stats)
            v = sft[idx]["video"]                       # [C,T,H,W], 0..255
            frame = v[:, 0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
            await ws.send(msgpack.packb({"image": frame.tobytes(),
                                         "shape": list(frame.shape), "direction": direction}))
            r = msgpack.unpackb(await ws.recv(), raw=False)
            if "error" in r: raise RuntimeError(r["error"])
            gts.append(float(gt[0, 0])); prs.append(float(r["velocity"][0]) / 15.0)
            gy.append(math.degrees(math.atan2(float(gt[0, 4]), float(gt[0, 3]))))
            py_.append(float(r["curvature"][0]))
            if (c+1) % 12 == 0: print(f"    ...{c+1}/{len(idxs)}", flush=True)
    print(f"\n=== [{label}] {name}, n={len(gts)} ===")
    print(f"  GT   vel range [{min(gts):.4f}, {max(gts):.4f}]")
    print(f"  PRED vel range [{min(prs):.4f}, {max(prs):.4f}]")
    print(f"  velocity Spearman = {spearman(gts, prs):+.3f}")
    print(f"  yaw/curv Spearman = {spearman(gy, py_):+.3f}")
    print(f"  MAE(vel)          = {sum(abs(a-b) for a,b in zip(gts,prs))/len(gts):.4f}", flush=True)

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="ws://127.0.0.1:18767")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--num-samples", type=int, default=48)
    p.add_argument("--label", default="onboard-int4")
    a = p.parse_args()
    sraw = load_action_stats(str(_STATS_PATH))
    stats = {k: torch.from_numpy(v).float() for k, v in sraw.items()}
    for root in [r.strip() for r in a.dataset_root.split(",") if r.strip()]:
        await run(root, a.uri, a.num_samples, stats, a.label)

asyncio.run(main())
