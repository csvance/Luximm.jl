"""Dump input + reference outputs for a VGG variant.

For each timm VGG variant in `VGG_VARIANTS_FULL`, writes a small HDF5 fixture
containing:

    /input                       deterministic torch.randn (seeded), NCHW
    /output/features             timm forward_features(x), post-`features`
    /output/logits               timm forward(x); every VGG ships a head
    /state_dict/<no entries>     placeholder group; weights load live from HF

The Julia parity test loads the safetensors directly from HuggingFace and
compares its output against the recorded reference outputs.

Usage:
    uv run python test/parity/dump_vgg_io.py --variant vgg16.tv_in1k
    uv run python test/parity/dump_vgg_io.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping

import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dump_common import dump


# Maps Julia variant key -> full timm model name. Must stay in sync with
# `VGG_VARIANTS` in src/Models/VGG/Config.jl.
VGG_VARIANTS_FULL: Mapping[str, str] = {
    "vgg11_tv_in1k":    "vgg11.tv_in1k",
    "vgg13_tv_in1k":    "vgg13.tv_in1k",
    "vgg16_tv_in1k":    "vgg16.tv_in1k",
    "vgg19_tv_in1k":    "vgg19.tv_in1k",
    "vgg11_bn_tv_in1k": "vgg11_bn.tv_in1k",
    "vgg13_bn_tv_in1k": "vgg13_bn.tv_in1k",
    "vgg16_bn_tv_in1k": "vgg16_bn.tv_in1k",
    "vgg19_bn_tv_in1k": "vgg19_bn.tv_in1k",
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
    in_chans: int | None = None,
) -> None:
    if os.path.exists(out_path):
        print(f"[{short_key}] cached at {out_path}; skipping")
        return
    in_chans_kwarg = {} if in_chans is None else {"in_chans": in_chans}
    print(f"[{short_key}] building timm model {full_name!r}"
          f"{'' if in_chans is None else f' (in_chans={in_chans})'} ...")
    model = timm.create_model(full_name, pretrained=True, **in_chans_kwarg).eval()

    cfg_c, height, width = model.default_cfg["input_size"]
    c = in_chans if in_chans is not None else cfg_c
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, c, height, width, generator=gen)

    with torch.no_grad():
        feats = model.forward_features(x)
        outputs = {"features": feats}
        if in_chans is None:
            outputs["logits"] = model(x)

    print(f"[{short_key}] input  shape: {tuple(x.shape)}")
    print(f"[{short_key}] feats  shape: {tuple(feats.shape)}")
    if "logits" in outputs:
        print(f"[{short_key}] logits shape: {tuple(outputs['logits'].shape)}")
    print(f"[{short_key}] writing {out_path}")
    dump(out_path, inp=x, state_dict={}, out=outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="Full timm name, e.g. vgg16.tv_in1k")
    parser.add_argument("--out", help="Output HDF5 path")
    parser.add_argument("--all", action="store_true",
                        help="Dump every variant in VGG_VARIANTS_FULL.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-chans", type=int, default=None,
                        help="Override input channel count (adds _in<N>c suffix).")
    args = parser.parse_args()
    in_chans = args.in_chans
    out_in_chans = 3 if in_chans is None else in_chans

    if args.all:
        for short, full in VGG_VARIANTS_FULL.items():
            run_one(short, full,
                    default_out_path(short, in_chans=out_in_chans),
                    seed=args.seed, in_chans=in_chans)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    if arg in VGG_VARIANTS_FULL:
        short, full = arg, VGG_VARIANTS_FULL[arg]
    else:
        short = next((k for k, v in VGG_VARIANTS_FULL.items() if v == arg), None)
        if short is None:
            parser.error(f"unknown variant: {arg}. "
                         f"Known short keys: {sorted(VGG_VARIANTS_FULL.keys())}; "
                         f"known full names: {sorted(VGG_VARIANTS_FULL.values())}")
        full = arg
    out_path = args.out or default_out_path(short, in_chans=out_in_chans)
    run_one(short, full, out_path, seed=args.seed, in_chans=in_chans)


if __name__ == "__main__":
    main()
