"""Dump input + reference outputs for a ConvNeXtV2 variant.

For each timm ConvNeXtV2 variant in `CONVNEXTV2_VARIANTS_FULL`, writes a small
HDF5 fixture containing:

    /input                       deterministic torch.randn (seeded), NCHW, batch 2
    /output/features             timm forward_features(x), post-stages
    /output/logits               timm forward(x), only for variants that ship
                                 a trained classification head
    /state_dict/<no entries>     placeholder group; weights load live from HF

The Julia parity test loads the safetensors directly from HuggingFace and
compares its output against the recorded reference outputs.

The input resolution for each variant is taken from `model.default_cfg`'s
`input_size`, so the `_384` and `_512` checkpoints are dumped at their native
resolution without per-variant logic here.

Usage:
    uv run python test/parity/dump_convnextv2_io.py \\
        --variant convnextv2_atto.fcmae \\
        --out data/parity/convnextv2_atto_fcmae_io.h5

    # All currently supported variants:
    uv run python test/parity/dump_convnextv2_io.py --all
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
# `CONVNEXTV2_VARIANTS` in src/Models/ConvNeXtV2/Config.jl. The Julia key is
# the timm name with the dot rewritten as an underscore (Julia identifier
# constraint); the full name keeps the dot so timm.create_model accepts it.
CONVNEXTV2_VARIANTS_FULL: Mapping[str, str] = {
    # atto
    "convnextv2_atto_fcmae":                       "convnextv2_atto.fcmae",
    "convnextv2_atto_fcmae_ft_in1k":               "convnextv2_atto.fcmae_ft_in1k",
    # femto
    "convnextv2_femto_fcmae":                      "convnextv2_femto.fcmae",
    "convnextv2_femto_fcmae_ft_in1k":              "convnextv2_femto.fcmae_ft_in1k",
    # pico
    "convnextv2_pico_fcmae":                       "convnextv2_pico.fcmae",
    "convnextv2_pico_fcmae_ft_in1k":               "convnextv2_pico.fcmae_ft_in1k",
    # nano
    "convnextv2_nano_fcmae":                       "convnextv2_nano.fcmae",
    "convnextv2_nano_fcmae_ft_in1k":               "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_nano_fcmae_ft_in22k_in1k":         "convnextv2_nano.fcmae_ft_in22k_in1k",
    "convnextv2_nano_fcmae_ft_in22k_in1k_384":     "convnextv2_nano.fcmae_ft_in22k_in1k_384",
    # tiny
    "convnextv2_tiny_fcmae":                       "convnextv2_tiny.fcmae",
    "convnextv2_tiny_fcmae_ft_in1k":               "convnextv2_tiny.fcmae_ft_in1k",
    "convnextv2_tiny_fcmae_ft_in22k_in1k":         "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_tiny_fcmae_ft_in22k_in1k_384":     "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    # base
    "convnextv2_base_fcmae":                       "convnextv2_base.fcmae",
    "convnextv2_base_fcmae_ft_in1k":               "convnextv2_base.fcmae_ft_in1k",
    "convnextv2_base_fcmae_ft_in22k_in1k":         "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_base_fcmae_ft_in22k_in1k_384":     "convnextv2_base.fcmae_ft_in22k_in1k_384",
    # large
    "convnextv2_large_fcmae":                      "convnextv2_large.fcmae",
    "convnextv2_large_fcmae_ft_in1k":              "convnextv2_large.fcmae_ft_in1k",
    "convnextv2_large_fcmae_ft_in22k_in1k":        "convnextv2_large.fcmae_ft_in22k_in1k",
    "convnextv2_large_fcmae_ft_in22k_in1k_384":    "convnextv2_large.fcmae_ft_in22k_in1k_384",
    # huge (no 224 in22k_in1k variant in timm)
    "convnextv2_huge_fcmae":                       "convnextv2_huge.fcmae",
    "convnextv2_huge_fcmae_ft_in1k":               "convnextv2_huge.fcmae_ft_in1k",
    "convnextv2_huge_fcmae_ft_in22k_in1k_384":     "convnextv2_huge.fcmae_ft_in22k_in1k_384",
    "convnextv2_huge_fcmae_ft_in22k_in1k_512":     "convnextv2_huge.fcmae_ft_in22k_in1k_512",
}

# Variants with a trained classifier head. For the bare `.fcmae` encoder,
# `head.fc` was never trained so dumping logits would just record noise.
HAS_HEAD: Mapping[str, bool] = {
    short: not short.endswith("_fcmae")
    for short in CONVNEXTV2_VARIANTS_FULL
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
    # from default_cfg (always 3 for ConvNeXtV2). When set explicitly, pass it
    # to timm.create_model so adapt_input_conv runs server-side. Logits are
    # only dumped for the default channel count: the in_chans=1 fixture is
    # focused on the stem-adaptation parity, not the head.
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
                        help="Full timm variant name, e.g. convnextv2_atto.fcmae")
    parser.add_argument("--out", help="Output HDF5 path (defaults to "
                                       "data/parity/<short_key>_io.h5)")
    parser.add_argument("--all", action="store_true",
                        help="Dump every variant currently in CONVNEXTV2_VARIANTS_FULL.")
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
        for short, full in CONVNEXTV2_VARIANTS_FULL.items():
            run_one(short, full,
                    default_out_path(short, in_chans=out_in_chans),
                    seed=args.seed, in_chans=in_chans)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    # Accept either the short Julia variant key (e.g. convnextv2_atto_fcmae)
    # or the full timm name with the dot (e.g. convnextv2_atto.fcmae).
    if arg in CONVNEXTV2_VARIANTS_FULL:
        short, full = arg, CONVNEXTV2_VARIANTS_FULL[arg]
    else:
        short = next((k for k, v in CONVNEXTV2_VARIANTS_FULL.items() if v == arg), None)
        if short is None:
            parser.error(f"unknown variant: {arg}. "
                         f"Known short keys: {sorted(CONVNEXTV2_VARIANTS_FULL.keys())}; "
                         f"known full names: {sorted(CONVNEXTV2_VARIANTS_FULL.values())}")
        full = arg
    out_path = args.out or default_out_path(short, in_chans=out_in_chans)
    run_one(short, full, out_path, seed=args.seed, in_chans=in_chans)


if __name__ == "__main__":
    main()
