"""Dump input + reference outputs for a ConvNeXt v1 variant.

For each timm ConvNeXt v1 variant in `CONVNEXT_VARIANTS_FULL`, writes a small
HDF5 fixture containing:

    /input                       deterministic torch.randn (seeded), NCHW, batch 2
    /output/features             timm forward_features(x), post-stages
    /output/logits               timm forward(x), only for variants that ship
                                 a trained classification head
    /state_dict/<no entries>     placeholder group; weights load live from HF

The Julia parity test loads the safetensors directly from HuggingFace and
compares its output against the recorded reference outputs.

The input resolution for each variant is taken from `model.default_cfg`'s
`input_size`, so any future variants with non-224 native resolution are
dumped at their native size without per-variant logic here.

Usage:
    uv run python test/parity/dump_convnext_io.py \\
        --variant convnext_tiny.dinov3_lvd1689m \\
        --out data/parity/convnext_tiny_dinov3_lvd1689m_io.h5

    # All currently supported variants:
    uv run python test/parity/dump_convnext_io.py --all
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
# `CONVNEXT_VARIANTS` in src/Models/ConvNeXt/Config.jl. The Julia key is the
# timm name with the dot rewritten as an underscore (Julia identifier
# constraint); the full name keeps the dot so timm.create_model accepts it.
CONVNEXT_VARIANTS_FULL: Mapping[str, str] = {
    # Meta DINOv3 encoders (no usable head).
    "convnext_tiny_dinov3_lvd1689m":  "convnext_tiny.dinov3_lvd1689m",
    "convnext_small_dinov3_lvd1689m": "convnext_small.dinov3_lvd1689m",
    "convnext_base_dinov3_lvd1689m":  "convnext_base.dinov3_lvd1689m",
    "convnext_large_dinov3_lvd1689m": "convnext_large.dinov3_lvd1689m",
    # Facebook AI 2022 ConvNeXt paper checkpoints (T/S/B/L/XL).
    "convnext_tiny_fb_in1k":                "convnext_tiny.fb_in1k",
    "convnext_tiny_fb_in22k":               "convnext_tiny.fb_in22k",
    "convnext_tiny_fb_in22k_ft_in1k":       "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_tiny_fb_in22k_ft_in1k_384":   "convnext_tiny.fb_in22k_ft_in1k_384",
    "convnext_small_fb_in1k":               "convnext_small.fb_in1k",
    "convnext_small_fb_in22k":              "convnext_small.fb_in22k",
    "convnext_small_fb_in22k_ft_in1k":      "convnext_small.fb_in22k_ft_in1k",
    "convnext_small_fb_in22k_ft_in1k_384":  "convnext_small.fb_in22k_ft_in1k_384",
    "convnext_base_fb_in1k":                "convnext_base.fb_in1k",
    "convnext_base_fb_in22k":               "convnext_base.fb_in22k",
    "convnext_base_fb_in22k_ft_in1k":       "convnext_base.fb_in22k_ft_in1k",
    "convnext_base_fb_in22k_ft_in1k_384":   "convnext_base.fb_in22k_ft_in1k_384",
    "convnext_large_fb_in1k":               "convnext_large.fb_in1k",
    "convnext_large_fb_in22k":              "convnext_large.fb_in22k",
    "convnext_large_fb_in22k_ft_in1k":      "convnext_large.fb_in22k_ft_in1k",
    "convnext_large_fb_in22k_ft_in1k_384":  "convnext_large.fb_in22k_ft_in1k_384",
    "convnext_xlarge_fb_in22k":             "convnext_xlarge.fb_in22k",
    "convnext_xlarge_fb_in22k_ft_in1k":     "convnext_xlarge.fb_in22k_ft_in1k",
    "convnext_xlarge_fb_in22k_ft_in1k_384": "convnext_xlarge.fb_in22k_ft_in1k_384",
}

# Variants with a trained classifier head. The DINOv3 checkpoints ship
# num_classes=0 (bare encoders), so logits are never dumped. All FB-paper
# checkpoints ship a head (1000 classes for IN1K, 21841 for IN22K).
HAS_HEAD: Mapping[str, bool] = {
    short: "_fb_" in short for short in CONVNEXT_VARIANTS_FULL
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
    has_head = HAS_HEAD[short_key]
    # When in_chans is None (default), use the variant's native channel count
    # from default_cfg. When set explicitly, pass it to timm.create_model so
    # adapt_input_conv runs server-side. Logits are only dumped for the
    # default channel count and only for variants that actually have a head.
    in_chans_kwarg = {} if in_chans is None else {"in_chans": in_chans}
    print(f"[{short_key}] building timm model {full_name!r} "
          f"({'with head' if has_head else 'encoder only'}"
          f"{'' if in_chans is None else f', in_chans={in_chans}'}) ...")
    if has_head:
        model = timm.create_model(full_name, pretrained=True,
                                   **in_chans_kwarg).eval()
    else:
        model = timm.create_model(full_name, pretrained=True, num_classes=0,
                                   **in_chans_kwarg).eval()

    # timm exposes the variant's native input resolution via default_cfg as a
    # (C, H, W) tuple. After passing in_chans=, default_cfg keeps the original
    # channel count, so we re-derive (C, H, W) from the constructor argument
    # and the spatial dims.
    cfg_c, height, width = model.default_cfg["input_size"]
    c = in_chans if in_chans is not None else cfg_c
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(PARITY_BATCH, c, height, width, generator=gen)

    with torch.no_grad():
        feats = model.forward_features(x)
        outputs = {"features": feats}
        if has_head and in_chans is None:
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
                        help="Full timm variant name, e.g. convnext_tiny.dinov3_lvd1689m")
    parser.add_argument("--out", help="Output HDF5 path (defaults to "
                                       "data/parity/<short_key>_io.h5)")
    parser.add_argument("--all", action="store_true",
                        help="Dump every variant currently in CONVNEXT_VARIANTS_FULL.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random input tensor (default: 0).")
    parser.add_argument("--in-chans", type=int, default=None,
                        help="Override the input channel count. When set, "
                             "timm.create_model receives in_chans=<N> so "
                             "adapt_input_conv runs on the stem, and the "
                             "output path gains an `_in<N>c` suffix to keep "
                             "it distinct from the default 3-channel fixture.")
    args = parser.parse_args()
    in_chans = args.in_chans
    out_in_chans = 3 if in_chans is None else in_chans

    if args.all:
        for short, full in CONVNEXT_VARIANTS_FULL.items():
            run_one(short, full,
                    default_out_path(short, in_chans=out_in_chans),
                    seed=args.seed, in_chans=in_chans)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    # Accept either the short Julia variant key (e.g. convnext_tiny_dinov3_lvd1689m)
    # or the full timm name with the dot (e.g. convnext_tiny.dinov3_lvd1689m).
    if arg in CONVNEXT_VARIANTS_FULL:
        short, full = arg, CONVNEXT_VARIANTS_FULL[arg]
    else:
        short = next((k for k, v in CONVNEXT_VARIANTS_FULL.items() if v == arg), None)
        if short is None:
            parser.error(f"unknown variant: {arg}. "
                         f"Known short keys: {sorted(CONVNEXT_VARIANTS_FULL.keys())}; "
                         f"known full names: {sorted(CONVNEXT_VARIANTS_FULL.values())}")
        full = arg
    out_path = args.out or default_out_path(short, in_chans=out_in_chans)
    run_one(short, full, out_path, seed=args.seed, in_chans=in_chans)


if __name__ == "__main__":
    main()
