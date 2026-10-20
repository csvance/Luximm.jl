"""Dump input + reference outputs for a BiT ResNetV2 variant.

For each timm BiT variant in `BIT_VARIANTS_FULL`, writes a small HDF5
fixture containing:

    /input                       deterministic torch.randn (seeded), NCHW, batch 2
    /output/features             timm forward_features(x)
    /output/logits               timm forward(x)
    /state_dict/<no entries>     placeholder group; weights load live from HF

The Julia parity test loads the safetensors directly from HuggingFace and
compares its output against the recorded reference outputs. Keeping the
state_dict out of the fixture keeps each file under a megabyte, since
recording all 47M / 200M weights per variant would be wasteful when the
canonical weights already live on HF.

Usage:
    uv run python test/parity/dump_resnetv2_bit_io.py \\
        --variant resnetv2_50x1_bit.goog_in21k \\
        --out data/parity/resnetv2_50x1_bit_goog_in21k_io.h5

    # All currently supported variants:
    uv run python test/parity/dump_resnetv2_bit_io.py --all
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


# Maps Julia variant key -> full timm model name. Must stay in sync with
# `BIT_VARIANTS` in src/Models/BiT/Config.jl. The Julia key is the timm
# name with the dot rewritten as an underscore (Julia identifier
# constraint); the full name keeps the dot so timm.create_model accepts it.
BIT_VARIANTS_FULL: Mapping[str, str] = {
    "resnetv2_50x1_bit_goog_in21k":  "resnetv2_50x1_bit.goog_in21k",
    "resnetv2_50x3_bit_goog_in21k":  "resnetv2_50x3_bit.goog_in21k",
    "resnetv2_101x1_bit_goog_in21k": "resnetv2_101x1_bit.goog_in21k",
    "resnetv2_101x3_bit_goog_in21k": "resnetv2_101x3_bit.goog_in21k",
    "resnetv2_152x2_bit_goog_in21k": "resnetv2_152x2_bit.goog_in21k",
    "resnetv2_152x4_bit_goog_in21k": "resnetv2_152x4_bit.goog_in21k",

    "resnetv2_50x1_bit_goog_distilled_in1k":
        "resnetv2_50x1_bit.goog_distilled_in1k",

    "resnetv2_50x1_bit_goog_in21k_ft_in1k":
        "resnetv2_50x1_bit.goog_in21k_ft_in1k",
    "resnetv2_50x3_bit_goog_in21k_ft_in1k":
        "resnetv2_50x3_bit.goog_in21k_ft_in1k",
    "resnetv2_101x1_bit_goog_in21k_ft_in1k":
        "resnetv2_101x1_bit.goog_in21k_ft_in1k",
    "resnetv2_101x3_bit_goog_in21k_ft_in1k":
        "resnetv2_101x3_bit.goog_in21k_ft_in1k",
    "resnetv2_152x2_bit_goog_in21k_ft_in1k":
        "resnetv2_152x2_bit.goog_in21k_ft_in1k",
    "resnetv2_152x4_bit_goog_in21k_ft_in1k":
        "resnetv2_152x4_bit.goog_in21k_ft_in1k",

    "resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k":
        "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k",
    "resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k_384":
        "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384",
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
    # in_chans != 3 triggers timm.layers.adapt_input_conv at create_model time,
    # so the stem weight in `model.state_dict()` is already collapsed to the
    # requested channel count and `model.forward_features(x)` is a true
    # reference for the Julia side's adapt_input_conv path.
    model = timm.create_model(full_name, pretrained=True, in_chans=in_chans).eval()

    # timm exposes the variant's native input resolution via default_cfg as a
    # (C, H, W) tuple. The 384 teacher variant is the only BiT entry that
    # trains at a non-224 resolution, but reading from default_cfg avoids a
    # per-variant special case.
    _cfg_c, height, width = model.default_cfg["input_size"]
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(PARITY_BATCH, in_chans, height, width, generator=gen)

    with torch.no_grad():
        feats = model.forward_features(x)
        outputs = {"features": feats}
        # The classifier head is shared across channel counts, so logits stay
        # comparable. Only dump them at the default in_chans=3 to keep the
        # in_chans=1 fixture tightly focused on the stem-adaptation parity.
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
                        help="Full timm variant name, e.g. resnetv2_50x1_bit.goog_in21k")
    parser.add_argument("--out", help="Output HDF5 path (defaults to "
                                       "data/parity/resnetv2_<key>_bit_io.h5)")
    parser.add_argument("--all", action="store_true",
                        help="Dump every variant currently in BIT_VARIANTS_FULL.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random input tensor (default: 0).")
    parser.add_argument("--in-chans", type=int, default=3,
                        help="Input channel count (default: 3). When != 3, "
                             "timm.create_model runs adapt_input_conv on the "
                             "stem and the output path gains an `_in<N>c` "
                             "suffix to keep it distinct from the default "
                             "3-channel fixture.")
    args = parser.parse_args()

    if args.all:
        for short, full in BIT_VARIANTS_FULL.items():
            run_one(short, full,
                    default_out_path(short, in_chans=args.in_chans),
                    seed=args.seed, in_chans=args.in_chans)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    # Accept either the short Julia variant key (e.g. resnetv2_50x1_bit_goog_in21k)
    # or the full timm name with the dot (e.g. resnetv2_50x1_bit.goog_in21k).
    if arg in BIT_VARIANTS_FULL:
        short, full = arg, BIT_VARIANTS_FULL[arg]
    else:
        short = next((k for k, v in BIT_VARIANTS_FULL.items() if v == arg), None)
        if short is None:
            parser.error(f"unknown variant: {arg}. "
                         f"Known short keys: {sorted(BIT_VARIANTS_FULL.keys())}; "
                         f"known full names: {sorted(BIT_VARIANTS_FULL.values())}")
        full = arg
    out_path = args.out or default_out_path(short, in_chans=args.in_chans)
    run_one(short, full, out_path, seed=args.seed, in_chans=args.in_chans)


if __name__ == "__main__":
    main()
