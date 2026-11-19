# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline exporter: MiniMax-H3-Image backbone BF16 -> FP8 (weights per-channel,
activations dynamic per-token at runtime).

Output layout is the `h3_channel_fp8_main_blocks_v1` convention consumed by
sglang multimodal_gen `inspect_fp8_checkpoint`:

- Source: diffusers-layout transformer component (transformer_diffusers/),
  e.g. .../MiniMax--MiniMax-H3-Image/transformer_diffusers
- For each of the 300 main-block matrices
  (transformer_blocks.{i}.attn.to_q / to_k / to_v / to_out.0 /
   ff.net.0.proj / ff.net.2, i in 0..num_blocks-1):
    * weight  -> float8_e4m3fn, symmetric per-output-channel (per-row) scale
    * <name-minus-"weight"> + "weight_scale_inv" -> fp32 [rows, 1]
- All other tensors are copied verbatim (dtype preserved).
- Each shard: metadata {"recipe": "h3_channel_fp8_main_blocks_v1", "variant": ...}
- manifest.json: recipe/status/variant/source_component/profile/shards/tensors.

Run inside the sglang container:
  python3 quant_h3_channel_fp8.py \
    --src  /public/opendas/DL_DATA/llm-models/MiniMax--MiniMax-H3-Image/transformer_diffusers \
    --dst  /public2/home/sunxx/models/minimax-h3-images/quant_weight/MiniMax-H3-Image-CHANNEL-FP8
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FP8_RECIPE = "main-block-only symmetric per-output-channel FP8, no rotation, no calibration"
FP8_FORMAT = "h3_channel_fp8_main_blocks_v1"
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

QUANT_TARGET_RE = re.compile(
    r"^transformer_blocks\.(\d+)\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$"
    r"|^transformer_blocks\.(\d+)\.ff\.net\.(0\.proj|2)\.weight$"
)


def weight_quant_fp8(tensor: torch.Tensor, device: str):
    """Symmetric per-output-channel (per-row) FP8 cast.

    scale_row = absmax(row) / 448; q = clamp(row / scale, -448, 448) -> e4m3.
    Runtime dequant is q * scale (weight_scale_inv semantics).
    """
    assert tensor.dim() == 2
    x = tensor.to(device=device, dtype=torch.float32)
    abs_max = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [rows, 1]
    scale = abs_max / FP8_MAX
    q = torch.clamp(x / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.cpu(), scale.to(torch.float32).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True,
                   help="diffusers-layout transformer component dir (contains *.safetensors)")
    p.add_argument("--native-src", type=Path, default=None,
                   help="native-layout transformer dir; used to source tensors the "
                        "diffusers export is missing (final_layer.video_out -> proj_out, "
                        "rope.inv_freq). Required if --src lacks them.")
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--variant", default="fl2va", choices=["fl2va", "ref2va"])
    p.add_argument("--profile", default="backbone", choices=["backbone", "backbone-adaln"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors under {src}")

    if dst.exists() and any(dst.iterdir()):
        if args.overwrite:
            shutil.rmtree(dst)
        else:
            raise SystemExit(f"{dst} not empty; use --overwrite to replace")
    dst.mkdir(parents=True, exist_ok=True)

    # pass 1: inventory
    shard_files, entries = [], []
    for sp in shards:
        from safetensors import safe_open
        with safe_open(sp, framework="pt") as f:
            names = list(f.keys())
        shard_files.append((sp, names))
        for n in names:
            entries.append(n)

    targets = sorted(n for n in entries if QUANT_TARGET_RE.match(n))
    blocks = sorted({int(m.group(1)) for n in targets
                     for m in [QUANT_TARGET_RE.match(n)] if m and m.group(1)})
    n_blocks = blocks[-1] + 1 if blocks else 0
    expected = 6 * n_blocks
    print(f"shards={len(shards)} tensors={len(entries)} quant targets={len(targets)} "
          f"(blocks={n_blocks}, expected={expected})")
    if len(targets) != expected:
        raise SystemExit("quantization target count mismatch — wrong source layout?")
    if len(set(entries)) != len(entries):
        raise SystemExit("duplicate tensor names across shards")

    # Diffusers layout names the final output conv `proj_out` and omits the RoPE
    # frequency buffer; the sglang loader requires both in the weights dir
    # (final_layer.video_out / rope.inv_freq use missing_param_init='error').
    # Source them from the native layout when the diffusers export lacks them.
    extra_from_native = {}
    if "proj_out.weight" not in entries:
        if args.native_src is None:
            raise SystemExit("diffusers export lacks proj_out; pass --native-src")
        native = {}
        for sp in sorted(args.native_src.glob("*.safetensors")):
            from safetensors import safe_open
            with safe_open(sp, framework="pt") as f:
                for n in ("final_layer.video_out.weight", "final_layer.video_out.bias",
                          "rope.inv_freq"):
                    if n in f.keys():
                        native[n] = f.get_tensor(n)
        extra_from_native = {
            "proj_out.weight": native["final_layer.video_out.weight"],
            "proj_out.bias": native["final_layer.video_out.bias"],
            "rope.inv_freq": native["rope.inv_freq"],
        }
        entries += list(extra_from_native)
        print(f"added from native layout: {sorted(extra_from_native)}")

    # pass 2: quantize + write
    manifest_tensors = []
    for idx, (sp, names) in enumerate(shard_files):
        sd = load_file(sp)
        out = {}
        for n in names:
            w = sd.pop(n)
            if QUANT_TARGET_RE.match(n):
                q, scale = weight_quant_fp8(w, args.device)
                out[n] = q
                out[n[: -len("weight")] + "weight_scale_inv"] = scale
                manifest_tensors.append({"name": n, "shape": list(w.shape), "quantized": True})
            else:
                out[n] = w
                manifest_tensors.append({"name": n, "shape": list(w.shape), "quantized": False})
        if idx == len(shard_files) - 1 and extra_from_native:
            for n, w in extra_from_native.items():
                out[n] = w
                manifest_tensors.append({"name": n, "shape": list(w.shape), "quantized": False})
        outp = dst / sp.name
        save_file(out, str(outp),
                 metadata={"recipe": FP8_FORMAT, "variant": args.variant})
        print(f"wrote {outp.name}: {len(out)} tensors")
        del sd, out

    manifest = {
        "recipe": FP8_RECIPE,
        "status": "complete",
        "variant": args.variant,
        "source_component": "transformer",
        "profile": args.profile,
        "shards": [{"file": sp.name} for sp, _ in shard_files],
        "tensors": manifest_tensors,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest written: {dst / 'manifest.json'} "
          f"({sum(t['quantized'] for t in manifest_tensors)} quantized)")


if __name__ == "__main__":
    main()
