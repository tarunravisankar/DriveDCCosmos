"""Quantize an exported Cosmos3 HF checkpoint to INT4 (bitsandbytes NF4).

Needed only for ON-CAR deployment. The robolidar/robolang inference server has
ample VRAM and should serve the bf16 checkpoint directly -- bf16 is the training
precision, so it is the accuracy ceiling, and it avoids the NF4 dequantization
overhead on every forward pass.

Measured motivation (Edge, 48-sample eval + live server probe):
    bf16 : 8.15 GB weights + 0.19 GB activations = 8.34 GB peak  -> does NOT fit 8 GB Jetson
    INT4 : ~1.7 GB weights + 1.4 GB VAE          = ~3.3 GB peak  -> fits with ~4.5 GB spare

Settings mirror the Nano INT4 checkpoint that was validated end-to-end:
NF4 with double quantization and bfloat16 compute dtype.

NOTE the compute dtype: NF4 stores weights in 4 bits but dequantizes to bf16 on
every forward pass, so INT4 shrinks *weights*, not activations -- and it adds
dequantization work. That is the right trade on an 8 GB board and the wrong one
on a 48/80 GB server.

Usage:
    quantize_int4.py --src <exported_hf_dir> --dst <out_dir>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="exported bf16 HF checkpoint dir")
    ap.add_argument("--dst", required=True, help="output INT4 dir")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not (src / "config.json").is_file():
        print(f"ERROR: {src} has no config.json", file=sys.stderr)
        return 1

    from cosmos_framework.inference.model import Cosmos3OmniModel

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {src} with 4-bit quantization ...", flush=True)
    model = Cosmos3OmniModel.from_pretrained(
        str(src), quantization_config=qcfg, device_map={"": 0}, dtype=torch.bfloat16,
    )

    n4 = sum(p.numel() for p in model.parameters() if "Params4bit" in type(p).__name__)
    nb = sum(p.numel() for p in model.parameters() if "Params4bit" not in type(p).__name__)
    print(f"  quantized params : {n4/1e9:.2f}B")
    print(f"  unquantized      : {nb/1e9:.2f}B")
    if n4 == 0:
        print("ERROR: nothing was quantized -- check bitsandbytes install", file=sys.stderr)
        return 1
    if torch.cuda.is_available():
        print(f"  CUDA allocated   : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {dst} ...", flush=True)
    model.save_pretrained(str(dst))

    # Carry over processor/tokenizer/policy-manifest files that save_pretrained
    # does not emit; the server needs them to build the text tokenizer.
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "preprocessor_config.json", "processor_config.json", "chat_template.jinja",
                 "checkpoint.json", "export_manifest.json"):
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)
    ve = src / "vision_encoder"
    if ve.is_dir() and not (dst / "vision_encoder").exists():
        shutil.copytree(ve, dst / "vision_encoder")

    # Same two deployment gotchas as the bf16 export: an 8-way-sharded degree
    # baked in from training breaks single-process serving, and vae_path must be
    # a real local file.
    cfg_path = dst / "config.json"
    cfg = json.load(open(cfg_path))
    src_cfg = json.load(open(src / "config.json"))

    def find_vae(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "vae_path" and isinstance(v, str):
                    return v
                r = find_vae(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find_vae(v)
                if r:
                    return r
        return None

    vae = find_vae(src_cfg)

    def fix(o):
        n = 0
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "data_parallel_shard_degree" and v != 1:
                    o[k] = 1
                    n += 1
                elif k == "vae_path" and vae and isinstance(v, str) and v != vae:
                    o[k] = vae
                    n += 1
                else:
                    n += fix(v)
        elif isinstance(o, list):
            for v in o:
                n += fix(v)
        return n

    n = fix(cfg)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"  config.json: normalized {n} field(s) for single-process serving")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
