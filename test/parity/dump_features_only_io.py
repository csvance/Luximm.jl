"""Dump timm `features_only=True` reference outputs for the pyramid families.

One fixture per variant, covering every family Luximm builds a feature pyramid
for (ResNet, SE-ResNet, BiT ResNetV2, ConvNeXt, ConvNeXt V2):

    /input                       deterministic torch.randn (seeded), NCHW
    /output/feat_01 ... feat_0K   timm features_only forward, one per tap
    /feature_channels            timm's feature_info.channels()
    /feature_reductions          timm's feature_info.reduction()
    /state_dict/<no entries>     placeholder group; weights load live from HF

The two `/feature_*` vectors are what pins Luximm's `feature_info` tap table
to timm's: a tap list that drifts (wrong channel count, a missed reduction-2
stem tap) fails the test even when every tensor still has a plausible shape.

Usage:
    uv run python test/parity/dump_features_only_io.py --all
    uv run python test/parity/dump_features_only_io.py \\
        --variant resnet18.a1_in1k
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping

import h5py
import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dump_common import dump, to_numpy


# Short key (the Luximm variant symbol) -> full timm name. Kept to one or two
# variants per family: the pyramid taps are a property of the architecture,
# not of the checkpoint, so a representative per family is enough.
FEATURES_ONLY_VARIANTS: Mapping[str, str] = {
    "resnet18_a1_in1k": "resnet18.a1_in1k",
    "resnet50_a1_in1k": "resnet50.a1_in1k",
    "seresnet50_a1_in1k": "seresnet50.a1_in1k",
    "resnetv2_50x1_bit_goog_in21k": "resnetv2_50x1_bit.goog_in21k",
    "convnext_tiny_fb_in1k": "convnext_tiny.fb_in1k",
    "convnextv2_atto_fcmae": "convnextv2_atto.fcmae",
}


def default_out_path(short_key: str) -> str:
    parity_dir = os.environ.get("JIMM_PARITY_DIR")
    if parity_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        parity_dir = os.path.join(repo_root, "data", "parity")
    os.makedirs(parity_dir, exist_ok=True)
    return os.path.join(parity_dir, f"{short_key}_featsonly_io.h5")


def run_one(short_key: str, full_name: str, out_path: str, seed: int = 0) -> None:
    if os.path.exists(out_path):
        print(f"[{short_key}] cached at {out_path}; skipping")
        return
    print(f"[{short_key}] building timm model {full_name!r} (features_only) ...")
    model = timm.create_model(full_name, pretrained=True, features_only=True).eval()

    # `features_only` models drop the classifier, so `default_cfg` is still the
    # place to get the native input resolution.
    _cfg_c, height, width = model.default_cfg["input_size"]
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, height, width, generator=gen)

    with torch.no_grad():
        feats = model(x)

    channels = list(model.feature_info.channels())
    reductions = list(model.feature_info.reduction())
    outputs = {f"feat_{i + 1:02d}": f for i, f in enumerate(feats)}

    print(f"[{short_key}] input      shape: {tuple(x.shape)}")
    print(f"[{short_key}] taps:            {len(feats)}")
    print(f"[{short_key}] channels:        {channels}")
    print(f"[{short_key}] reductions:      {reductions}")
    for name, f in outputs.items():
        print(f"[{short_key}]   {name} shape: {tuple(f.shape)}")

    assert len(channels) == len(feats), "feature_info/channels length mismatch"
    assert len(reductions) == len(feats), "feature_info/reduction length mismatch"
    for f, c, r in zip(feats, channels, reductions):
        assert f.shape[1] == c, f"channel mismatch: {f.shape[1]} != {c}"
        assert f.shape[2] == height // r, f"reduction mismatch at r={r}"

    print(f"[{short_key}] writing {out_path}")
    dump(out_path, inp=x, state_dict={}, out=outputs)

    # `dump` owns /input, /output, /state_dict; append the tap table alongside.
    with h5py.File(out_path, "a") as f:
        f.create_dataset("feature_channels", data=channels)
        f.create_dataset("feature_reductions", data=reductions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",
                        help="Full timm variant name, e.g. resnet18.a1_in1k")
    parser.add_argument("--out", help="Output HDF5 path")
    parser.add_argument("--all", action="store_true",
                        help="Dump every variant in FEATURES_ONLY_VARIANTS.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random input tensor (default: 0).")
    args = parser.parse_args()

    if args.all:
        for short, full in FEATURES_ONLY_VARIANTS.items():
            run_one(short, full, default_out_path(short), seed=args.seed)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    if arg in FEATURES_ONLY_VARIANTS:
        short, full = arg, FEATURES_ONLY_VARIANTS[arg]
    else:
        short = next((k for k, v in FEATURES_ONLY_VARIANTS.items() if v == arg), None)
        if short is None:
            parser.error(
                f"unknown variant: {arg}. "
                f"Known short keys: {sorted(FEATURES_ONLY_VARIANTS.keys())}; "
                f"known full names: {sorted(FEATURES_ONLY_VARIANTS.values())}")
        full = arg
    run_one(short, full, args.out or default_out_path(short), seed=args.seed)


if __name__ == "__main__":
    main()
