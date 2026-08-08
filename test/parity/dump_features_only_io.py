"""Dump timm `features_only=True` reference outputs for the pyramid families.

One fixture per variant, covering every family Luximm builds a feature pyramid
for (ResNet, SE-ResNet, BiT ResNetV2, ConvNeXt, ConvNeXt V2, ViT):

    /input                       deterministic torch.randn (seeded), NCHW
    /output/feat_01 ... feat_0K   timm features_only forward, one per tap
    /feature_channels            timm's feature_info.channels()
    /feature_reductions          timm's feature_info.reduction()
    /state_dict/<no entries>     placeholder group; weights load live from HF

The two `/feature_*` vectors are what pins Luximm's `feature_info` tap table
to timm's: a tap list that drifts (wrong channel count, a missed reduction-2
stem tap) fails the test even when every tensor still has a plausible shape.

For the `vit_*` family the dump passes `out_indices=None` so every encoder
block is dumped: timm's `vit_*` factory defaults to the last three blocks,
while Luximm's default is `nothing` = every tap. Each ViT tap is the raw
post-block token output reshaped to a grid (class token dropped, no final
LayerNorm) — see `timm.VisionTransformer.forward_intermediates`.

Usage:
    uv run python test/parity/dump_features_only_io.py --all
    uv run python test/parity/dump_features_only_io.py --all --family resnet
    uv run python test/parity/dump_features_only_io.py \\
        --variant resnet18.a1_in1k

Existing fixtures are skipped, so `--all` is safe to re-run; delete the
`.h5` to force a re-dump. Honors `JIMM_PARITY_DIR` like the other sidecars.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping, Tuple

import h5py
import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dump_common import dump, to_numpy


# Short key (the Luximm variant symbol) -> (full timm name, Luximm test
# family). Covers every registered variant of the six families that build a
# feature pyramid, so a full CI sweep produces a fixture for each one and no
# variant silently skips its pyramid testset. VGG / CoAtNet are absent
# on purpose: they have no pyramid.
FEATURES_ONLY_VARIANTS: Mapping[str, Tuple[str, str]] = {
    # resnet (5)
    "resnet101_a1_in1k": ("resnet101.a1_in1k", "resnet"),
    "resnet152_a1_in1k": ("resnet152.a1_in1k", "resnet"),
    "resnet18_a1_in1k": ("resnet18.a1_in1k", "resnet"),
    "resnet34_a1_in1k": ("resnet34.a1_in1k", "resnet"),
    "resnet50_a1_in1k": ("resnet50.a1_in1k", "resnet"),
    # seresnet (1)
    "seresnet50_a1_in1k": ("seresnet50.a1_in1k", "seresnet"),
    # bit (15)
    "resnetv2_101x1_bit_goog_in21k": ("resnetv2_101x1_bit.goog_in21k", "bit"),
    "resnetv2_101x1_bit_goog_in21k_ft_in1k": ("resnetv2_101x1_bit.goog_in21k_ft_in1k", "bit"),
    "resnetv2_101x3_bit_goog_in21k": ("resnetv2_101x3_bit.goog_in21k", "bit"),
    "resnetv2_101x3_bit_goog_in21k_ft_in1k": ("resnetv2_101x3_bit.goog_in21k_ft_in1k", "bit"),
    "resnetv2_152x2_bit_goog_in21k": ("resnetv2_152x2_bit.goog_in21k", "bit"),
    "resnetv2_152x2_bit_goog_in21k_ft_in1k": ("resnetv2_152x2_bit.goog_in21k_ft_in1k", "bit"),
    "resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k": ("resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k", "bit"),
    "resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k_384": ("resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384", "bit"),
    "resnetv2_152x4_bit_goog_in21k": ("resnetv2_152x4_bit.goog_in21k", "bit"),
    "resnetv2_152x4_bit_goog_in21k_ft_in1k": ("resnetv2_152x4_bit.goog_in21k_ft_in1k", "bit"),
    "resnetv2_50x1_bit_goog_distilled_in1k": ("resnetv2_50x1_bit.goog_distilled_in1k", "bit"),
    "resnetv2_50x1_bit_goog_in21k": ("resnetv2_50x1_bit.goog_in21k", "bit"),
    "resnetv2_50x1_bit_goog_in21k_ft_in1k": ("resnetv2_50x1_bit.goog_in21k_ft_in1k", "bit"),
    "resnetv2_50x3_bit_goog_in21k": ("resnetv2_50x3_bit.goog_in21k", "bit"),
    "resnetv2_50x3_bit_goog_in21k_ft_in1k": ("resnetv2_50x3_bit.goog_in21k_ft_in1k", "bit"),
    # convnext (23)
    "convnext_base_dinov3_lvd1689m": ("convnext_base.dinov3_lvd1689m", "convnext"),
    "convnext_base_fb_in1k": ("convnext_base.fb_in1k", "convnext"),
    "convnext_base_fb_in22k": ("convnext_base.fb_in22k", "convnext"),
    "convnext_base_fb_in22k_ft_in1k": ("convnext_base.fb_in22k_ft_in1k", "convnext"),
    "convnext_base_fb_in22k_ft_in1k_384": ("convnext_base.fb_in22k_ft_in1k_384", "convnext"),
    "convnext_large_dinov3_lvd1689m": ("convnext_large.dinov3_lvd1689m", "convnext"),
    "convnext_large_fb_in1k": ("convnext_large.fb_in1k", "convnext"),
    "convnext_large_fb_in22k": ("convnext_large.fb_in22k", "convnext"),
    "convnext_large_fb_in22k_ft_in1k": ("convnext_large.fb_in22k_ft_in1k", "convnext"),
    "convnext_large_fb_in22k_ft_in1k_384": ("convnext_large.fb_in22k_ft_in1k_384", "convnext"),
    "convnext_small_dinov3_lvd1689m": ("convnext_small.dinov3_lvd1689m", "convnext"),
    "convnext_small_fb_in1k": ("convnext_small.fb_in1k", "convnext"),
    "convnext_small_fb_in22k": ("convnext_small.fb_in22k", "convnext"),
    "convnext_small_fb_in22k_ft_in1k": ("convnext_small.fb_in22k_ft_in1k", "convnext"),
    "convnext_small_fb_in22k_ft_in1k_384": ("convnext_small.fb_in22k_ft_in1k_384", "convnext"),
    "convnext_tiny_dinov3_lvd1689m": ("convnext_tiny.dinov3_lvd1689m", "convnext"),
    "convnext_tiny_fb_in1k": ("convnext_tiny.fb_in1k", "convnext"),
    "convnext_tiny_fb_in22k": ("convnext_tiny.fb_in22k", "convnext"),
    "convnext_tiny_fb_in22k_ft_in1k": ("convnext_tiny.fb_in22k_ft_in1k", "convnext"),
    "convnext_tiny_fb_in22k_ft_in1k_384": ("convnext_tiny.fb_in22k_ft_in1k_384", "convnext"),
    "convnext_xlarge_fb_in22k": ("convnext_xlarge.fb_in22k", "convnext"),
    "convnext_xlarge_fb_in22k_ft_in1k": ("convnext_xlarge.fb_in22k_ft_in1k", "convnext"),
    "convnext_xlarge_fb_in22k_ft_in1k_384": ("convnext_xlarge.fb_in22k_ft_in1k_384", "convnext"),
    # convnextv2 (26)
    "convnextv2_atto_fcmae": ("convnextv2_atto.fcmae", "convnextv2"),
    "convnextv2_atto_fcmae_ft_in1k": ("convnextv2_atto.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_base_fcmae": ("convnextv2_base.fcmae", "convnextv2"),
    "convnextv2_base_fcmae_ft_in1k": ("convnextv2_base.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_base_fcmae_ft_in22k_in1k": ("convnextv2_base.fcmae_ft_in22k_in1k", "convnextv2"),
    "convnextv2_base_fcmae_ft_in22k_in1k_384": ("convnextv2_base.fcmae_ft_in22k_in1k_384", "convnextv2"),
    "convnextv2_femto_fcmae": ("convnextv2_femto.fcmae", "convnextv2"),
    "convnextv2_femto_fcmae_ft_in1k": ("convnextv2_femto.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_huge_fcmae": ("convnextv2_huge.fcmae", "convnextv2"),
    "convnextv2_huge_fcmae_ft_in1k": ("convnextv2_huge.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_huge_fcmae_ft_in22k_in1k_384": ("convnextv2_huge.fcmae_ft_in22k_in1k_384", "convnextv2"),
    "convnextv2_huge_fcmae_ft_in22k_in1k_512": ("convnextv2_huge.fcmae_ft_in22k_in1k_512", "convnextv2"),
    "convnextv2_large_fcmae": ("convnextv2_large.fcmae", "convnextv2"),
    "convnextv2_large_fcmae_ft_in1k": ("convnextv2_large.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_large_fcmae_ft_in22k_in1k": ("convnextv2_large.fcmae_ft_in22k_in1k", "convnextv2"),
    "convnextv2_large_fcmae_ft_in22k_in1k_384": ("convnextv2_large.fcmae_ft_in22k_in1k_384", "convnextv2"),
    "convnextv2_nano_fcmae": ("convnextv2_nano.fcmae", "convnextv2"),
    "convnextv2_nano_fcmae_ft_in1k": ("convnextv2_nano.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_nano_fcmae_ft_in22k_in1k": ("convnextv2_nano.fcmae_ft_in22k_in1k", "convnextv2"),
    "convnextv2_nano_fcmae_ft_in22k_in1k_384": ("convnextv2_nano.fcmae_ft_in22k_in1k_384", "convnextv2"),
    "convnextv2_pico_fcmae": ("convnextv2_pico.fcmae", "convnextv2"),
    "convnextv2_pico_fcmae_ft_in1k": ("convnextv2_pico.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_tiny_fcmae": ("convnextv2_tiny.fcmae", "convnextv2"),
    "convnextv2_tiny_fcmae_ft_in1k": ("convnextv2_tiny.fcmae_ft_in1k", "convnextv2"),
    "convnextv2_tiny_fcmae_ft_in22k_in1k": ("convnextv2_tiny.fcmae_ft_in22k_in1k", "convnextv2"),
    "convnextv2_tiny_fcmae_ft_in22k_in1k_384": ("convnextv2_tiny.fcmae_ft_in22k_in1k_384", "convnextv2"),
    # vit (4)
    "vit_base_patch16_224_augreg2_in21k_ft_in1k": ("vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit"),
    "vit_base_patch16_clip_224_openai_ft_in1k": ("vit_base_patch16_clip_224.openai_ft_in1k", "vit"),
    "vit_base_patch32_clip_224_openai_ft_in1k": ("vit_base_patch32_clip_224.openai_ft_in1k", "vit"),
    "vit_large_patch14_clip_224_openai_ft_in1k": ("vit_large_patch14_clip_224.openai_ft_in1k", "vit"),
}

# timm's `vit_*` factory pops `out_indices = 3` (last three blocks) as its
# features_only default; Luximm's default is `nothing` = every tap, so the
# ViT dump builds the wrapper directly with `out_indices=None` (see
# `run_one`) to cover the whole block list.
VIT_FAMILY = "vit"


def default_out_path(short_key: str) -> str:
    parity_dir = os.environ.get("JIMM_PARITY_DIR")
    if parity_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        parity_dir = os.path.join(repo_root, "data", "parity")
    os.makedirs(parity_dir, exist_ok=True)
    return os.path.join(parity_dir, f"{short_key}_featsonly_io.h5")


def run_one(
    short_key: str,
    full_name: str,
    out_path: str,
    seed: int = 0,
    family: str = "",
) -> None:
    if os.path.exists(out_path):
        print(f"[{short_key}] cached at {out_path}; skipping")
        return
    print(f"[{short_key}] building timm model {full_name!r} (features_only) ...")
    if family == VIT_FAMILY:
        # `timm.create_model` drops None kwargs (`models/_factory.py`), so
        # `out_indices=None` cannot request every block through the factory
        # — it silently falls back to the `vit_*` default of the last three
        # blocks. Build the bare model and wrap it directly; FeatureGetterNet
        # normalizes `None` to every block index.
        from timm.models._features import FeatureGetterNet

        model = FeatureGetterNet(
            timm.create_model(full_name, pretrained=True).eval(),
            out_indices=None,
        )
    else:
        model = timm.create_model(
            full_name,
            pretrained=True,
            features_only=True,
        ).eval()

    # `features_only` models drop the classifier, so `default_cfg` is still the
    # place to get the native input resolution. The hand-built ViT wrapper
    # carries no `default_cfg`; read it from the wrapped model instead.
    cfg_model = model.model if family == VIT_FAMILY else model
    _cfg_c, height, width = cfg_model.default_cfg["input_size"]
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


PYRAMID_FAMILIES = ("resnet", "seresnet", "bit", "convnext", "convnextv2", "vit")

# Families Luximm registers that have no feature pyramid. A `--variant` from
# one of these exits 0 with a note instead of failing, so the CI builder can
# call this sidecar for every family without special-casing.
NO_PYRAMID_PREFIXES = ("vgg", "coatnet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",
                        help="Variant key or full timm name, e.g. resnet18.a1_in1k")
    parser.add_argument("--out", help="Output HDF5 path")
    parser.add_argument("--all", action="store_true",
                        help="Dump every variant in FEATURES_ONLY_VARIANTS.")
    parser.add_argument("--family", choices=PYRAMID_FAMILIES,
                        help="With --all, restrict the sweep to one family.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random input tensor (default: 0).")
    args = parser.parse_args()

    if args.all:
        selected = [
            (short, full)
            for short, (full, family) in FEATURES_ONLY_VARIANTS.items()
            if args.family is None or family == args.family
        ]
        scope = args.family or "all families"
        print(f"[features_only] sweeping {len(selected)} variant(s) for {scope}")
        for short, full in selected:
            family = FEATURES_ONLY_VARIANTS[short][1]
            run_one(short, full, default_out_path(short), seed=args.seed, family=family)
        return

    if not args.variant:
        parser.error("either --variant or --all is required")

    arg = args.variant
    if arg in FEATURES_ONLY_VARIANTS:
        short, full = arg, FEATURES_ONLY_VARIANTS[arg][0]
        family = FEATURES_ONLY_VARIANTS[arg][1]
    else:
        short = next(
            (k for k, v in FEATURES_ONLY_VARIANTS.items() if v[0] == arg), None)
        if short is None:
            # A variant of a family with no pyramid is a no-op, not an error:
            # its Julia testset skips on the missing fixture by design.
            key = arg.replace(".", "_")
            if key.startswith(NO_PYRAMID_PREFIXES):
                print(f"[{arg}] family has no feature pyramid; nothing to dump")
                return
            parser.error(
                f"unknown variant: {arg}. "
                f"Known short keys: {sorted(FEATURES_ONLY_VARIANTS.keys())}; "
                f"known full names: "
                f"{sorted(v[0] for v in FEATURES_ONLY_VARIANTS.values())}")
        full = arg
        family = FEATURES_ONLY_VARIANTS[short][1]
    run_one(short, full, args.out or default_out_path(short), seed=args.seed, family=family)


if __name__ == "__main__":
    main()
