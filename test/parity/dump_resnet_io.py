"""Dump input + reference outputs for a classic timm ResNet variant.

For each variant in `RESNET_VARIANTS_FULL`, writes a small HDF5 fixture:

    /input                       deterministic torch.randn (seeded), NCHW, batch 2
    /output/features             timm forward_features(x)
    /output/logits               timm forward(x), only at in_chans=3
    /state_dict/<no entries>     placeholder group; weights load live from HF

Usage:
    uv run python test/parity/dump_resnet_io.py \\
        --variant resnet18.a1_in1k \\
        --out data/parity/resnet18_a1_in1k_io.h5
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping

import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dump_common import PARITY_BATCH, dump


RESNET_VARIANTS_FULL: Mapping[str, str] = {
    "resnet18_a1_in1k": "resnet18.a1_in1k",
    "resnet34_a1_in1k": "resnet34.a1_in1k",
    "resnet50_a1_in1k": "resnet50.a1_in1k",
    "resnet101_a1_in1k": "resnet101.a1_in1k",
    "resnet152_a1_in1k": "resnet152.a1_in1k",
}


def default_out_path(short_key: str, in_chans: int = 3) -> str:
    suffix = "" if in_chans == 3 else f"_in{in_chans}c"
    parity_dir = os.environ.get("JIMM_PARITY_DIR")
    if parity_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        parity_dir = os.path.join(repo_root, "data", "parity")
    os.makedirs(parity_dir, exist_ok=True)
    return os.path.join(parity_dir, f"{short_key}{suffix}_io.h5")


def run_one(
    short_key: str,
    full_name: str,
    out_path: str,
    seed: int = 0,
    in_chans: int = 3,
) -> None:
    if os.path.exists(out_path):
        print(f"[{short_key}] cached at {out_path}; skipping")
        return
    print(f"[{short_key}] building timm model {full_name!r} "
          f"(in_chans={in_chans}) ...")
    model = timm.create_model(full_name, pretrained=True, in_chans=in_chans).eval()
    _cfg_c, height, width = model.default_cfg["input_size"]
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(PARITY_BATCH, in_chans, height, width, generator=gen)

    with torch.no_grad():
        feats = model.forward_features(x)
        outputs = {"features": feats}
        if in_chans == 3:
            outputs["logits"] = model(x)

    print(f"[{short_key}] input  shape: {tuple(x.shape)}")
    print(f"[{short_key}] feats  shape: {tuple(feats.shape)}")
    if "logits" in outputs:
        print(f"[{short_key}] logits shape: {tuple(outputs['logits'].shape)}")
    print(f"[{short_key}] writing {out_path}")
    dump(out_path, inp=x, state_dict={}, out=outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",
                        help="Full timm variant name, e.g. resnet18.a1_in1k")
    parser.add_argument("--out", help="Output HDF5 path")
    parser.add_argument("--all", action="store_true",
                        help="Dump every currently supported ResNet variant.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random input tensor (default: 0).")
    parser.add_argument("--in-chans", type=int, default=3,
                        help="Input channel count (default: 3).")
    args = parser.parse_args()

    if args.all:
        for short, full in RESNET_VARIANTS_FULL.items():
            run_one(short, full, default_out_path(short, args.in_chans),
                    seed=args.seed, in_chans=args.in_chans)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    if arg in RESNET_VARIANTS_FULL:
        short, full = arg, RESNET_VARIANTS_FULL[arg]
    else:
        short = next((k for k, v in RESNET_VARIANTS_FULL.items() if v == arg), None)
        if short is None:
            parser.error(f"unknown variant: {arg}. "
                         f"Known short keys: {sorted(RESNET_VARIANTS_FULL.keys())}; "
                         f"known full names: {sorted(RESNET_VARIANTS_FULL.values())}")
        full = arg
    out_path = args.out or default_out_path(short, args.in_chans)
    run_one(short, full, out_path, seed=args.seed, in_chans=args.in_chans)


if __name__ == "__main__":
    main()
