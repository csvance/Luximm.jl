# Variant catalog for `convnextv2`.
#
# Covers every `convnextv2_*` checkpoint timm ships pretrained weights for:
#   - `.fcmae`                      bare MAE-pretrained encoder, no head.
#   - `.fcmae_ft_in1k`              fine-tuned on ImageNet-1K, 1000-class head.
#   - `.fcmae_ft_in22k_in1k`        pretrained on ImageNet-22k then fine-tuned
#                                   on ImageNet-1K, 1000-class head.
#   - `.fcmae_ft_in22k_in1k_384`    same chain, native 384x384 input.
#   - `.fcmae_ft_in22k_in1k_512`    huge only, native 512x512 input.
#
# `convnextv2_small` is intentionally absent because timm only registers
# `convnextv2_small.untrained` (no pretrained weights to load).
#
# Keys mirror the timm model name with the dot rewritten as an underscore (the
# dot is reserved in Julia identifiers). `hf_repo` keeps the real dot-separated
# timm name.

"""
    ConvNeXtV2Variant

Architectural config for a single ConvNeXtV2 variant.

Fields:
- `name`: lookup key (e.g. `:convnextv2_atto_fcmae`).
- `depths`: per-stage block count, `(d1, d2, d3, d4)`.
- `dims`: per-stage channel widths, `(c1, c2, c3, c4)`. `c1` is also the
  stem output channels. `c4` is `num_features`.
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with.
  `0` for the bare `.fcmae` encoders, `1000` for the ImageNet-1K and
  ImageNet-22k-then-1K fine-tunes.
- `default_input_size`: native training resolution (224, 384, or 512) for
  the released checkpoint. Informational only: the model is fully
  convolutional and accepts any size, so this is not enforced.
"""
struct ConvNeXtV2Variant
    name::Symbol
    depths::NTuple{4, Int}
    dims::NTuple{4, Int}
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
end

# Per-size architecture parameters. Every checkpoint flavour for a given size
# shares depths/dims; only the HF repo, head class count, and native input
# resolution change.
const _CN2_ATTO_DEPTHS = (2, 2, 6, 2)
const _CN2_ATTO_DIMS = (40, 80, 160, 320)
const _CN2_FEMTO_DEPTHS = (2, 2, 6, 2)
const _CN2_FEMTO_DIMS = (48, 96, 192, 384)
const _CN2_PICO_DEPTHS = (2, 2, 6, 2)
const _CN2_PICO_DIMS = (64, 128, 256, 512)
const _CN2_NANO_DEPTHS = (2, 2, 8, 2)
const _CN2_NANO_DIMS = (80, 160, 320, 640)
const _CN2_TINY_DEPTHS = (3, 3, 9, 3)
const _CN2_TINY_DIMS = (96, 192, 384, 768)
const _CN2_BASE_DEPTHS = (3, 3, 27, 3)
const _CN2_BASE_DIMS = (128, 256, 512, 1024)
const _CN2_LARGE_DEPTHS = (3, 3, 27, 3)
const _CN2_LARGE_DIMS = (192, 384, 768, 1536)
const _CN2_HUGE_DEPTHS = (3, 3, 27, 3)
const _CN2_HUGE_DIMS = (352, 704, 1408, 2816)

"""
    CONVNEXTV2_VARIANTS :: Dict{Symbol, ConvNeXtV2Variant}

Lookup table for the ConvNeXtV2 variants this package ports. The
`.fcmae` rows are the bare encoders; all other rows ship a 1000-class
ImageNet head. `convnextv2_small` is not included because timm only
registers it as `.untrained` (no pretrained weights).
"""
const CONVNEXTV2_VARIANTS = Dict{Symbol, ConvNeXtV2Variant}(
    # atto
    :convnextv2_atto_fcmae => ConvNeXtV2Variant(
        :convnextv2_atto_fcmae,
        _CN2_ATTO_DEPTHS,
        _CN2_ATTO_DIMS,
        "timm/convnextv2_atto.fcmae",
        0,
        224,
    ),
    :convnextv2_atto_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_atto_fcmae_ft_in1k,
        _CN2_ATTO_DEPTHS,
        _CN2_ATTO_DIMS,
        "timm/convnextv2_atto.fcmae_ft_in1k",
        1000,
        224,
    ),

    # femto
    :convnextv2_femto_fcmae => ConvNeXtV2Variant(
        :convnextv2_femto_fcmae,
        _CN2_FEMTO_DEPTHS,
        _CN2_FEMTO_DIMS,
        "timm/convnextv2_femto.fcmae",
        0,
        224,
    ),
    :convnextv2_femto_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_femto_fcmae_ft_in1k,
        _CN2_FEMTO_DEPTHS,
        _CN2_FEMTO_DIMS,
        "timm/convnextv2_femto.fcmae_ft_in1k",
        1000,
        224,
    ),

    # pico
    :convnextv2_pico_fcmae => ConvNeXtV2Variant(
        :convnextv2_pico_fcmae,
        _CN2_PICO_DEPTHS,
        _CN2_PICO_DIMS,
        "timm/convnextv2_pico.fcmae",
        0,
        224,
    ),
    :convnextv2_pico_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_pico_fcmae_ft_in1k,
        _CN2_PICO_DEPTHS,
        _CN2_PICO_DIMS,
        "timm/convnextv2_pico.fcmae_ft_in1k",
        1000,
        224,
    ),

    # nano
    :convnextv2_nano_fcmae => ConvNeXtV2Variant(
        :convnextv2_nano_fcmae,
        _CN2_NANO_DEPTHS,
        _CN2_NANO_DIMS,
        "timm/convnextv2_nano.fcmae",
        0,
        224,
    ),
    :convnextv2_nano_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_nano_fcmae_ft_in1k,
        _CN2_NANO_DEPTHS,
        _CN2_NANO_DIMS,
        "timm/convnextv2_nano.fcmae_ft_in1k",
        1000,
        224,
    ),
    :convnextv2_nano_fcmae_ft_in22k_in1k => ConvNeXtV2Variant(
        :convnextv2_nano_fcmae_ft_in22k_in1k,
        _CN2_NANO_DEPTHS,
        _CN2_NANO_DIMS,
        "timm/convnextv2_nano.fcmae_ft_in22k_in1k",
        1000,
        224,
    ),
    :convnextv2_nano_fcmae_ft_in22k_in1k_384 => ConvNeXtV2Variant(
        :convnextv2_nano_fcmae_ft_in22k_in1k_384,
        _CN2_NANO_DEPTHS,
        _CN2_NANO_DIMS,
        "timm/convnextv2_nano.fcmae_ft_in22k_in1k_384",
        1000,
        384,
    ),

    # tiny
    :convnextv2_tiny_fcmae => ConvNeXtV2Variant(
        :convnextv2_tiny_fcmae,
        _CN2_TINY_DEPTHS,
        _CN2_TINY_DIMS,
        "timm/convnextv2_tiny.fcmae",
        0,
        224,
    ),
    :convnextv2_tiny_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_tiny_fcmae_ft_in1k,
        _CN2_TINY_DEPTHS,
        _CN2_TINY_DIMS,
        "timm/convnextv2_tiny.fcmae_ft_in1k",
        1000,
        224,
    ),
    :convnextv2_tiny_fcmae_ft_in22k_in1k => ConvNeXtV2Variant(
        :convnextv2_tiny_fcmae_ft_in22k_in1k,
        _CN2_TINY_DEPTHS,
        _CN2_TINY_DIMS,
        "timm/convnextv2_tiny.fcmae_ft_in22k_in1k",
        1000,
        224,
    ),
    :convnextv2_tiny_fcmae_ft_in22k_in1k_384 => ConvNeXtV2Variant(
        :convnextv2_tiny_fcmae_ft_in22k_in1k_384,
        _CN2_TINY_DEPTHS,
        _CN2_TINY_DIMS,
        "timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        1000,
        384,
    ),

    # base
    :convnextv2_base_fcmae => ConvNeXtV2Variant(
        :convnextv2_base_fcmae,
        _CN2_BASE_DEPTHS,
        _CN2_BASE_DIMS,
        "timm/convnextv2_base.fcmae",
        0,
        224,
    ),
    :convnextv2_base_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_base_fcmae_ft_in1k,
        _CN2_BASE_DEPTHS,
        _CN2_BASE_DIMS,
        "timm/convnextv2_base.fcmae_ft_in1k",
        1000,
        224,
    ),
    :convnextv2_base_fcmae_ft_in22k_in1k => ConvNeXtV2Variant(
        :convnextv2_base_fcmae_ft_in22k_in1k,
        _CN2_BASE_DEPTHS,
        _CN2_BASE_DIMS,
        "timm/convnextv2_base.fcmae_ft_in22k_in1k",
        1000,
        224,
    ),
    :convnextv2_base_fcmae_ft_in22k_in1k_384 => ConvNeXtV2Variant(
        :convnextv2_base_fcmae_ft_in22k_in1k_384,
        _CN2_BASE_DEPTHS,
        _CN2_BASE_DIMS,
        "timm/convnextv2_base.fcmae_ft_in22k_in1k_384",
        1000,
        384,
    ),

    # large
    :convnextv2_large_fcmae => ConvNeXtV2Variant(
        :convnextv2_large_fcmae,
        _CN2_LARGE_DEPTHS,
        _CN2_LARGE_DIMS,
        "timm/convnextv2_large.fcmae",
        0,
        224,
    ),
    :convnextv2_large_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_large_fcmae_ft_in1k,
        _CN2_LARGE_DEPTHS,
        _CN2_LARGE_DIMS,
        "timm/convnextv2_large.fcmae_ft_in1k",
        1000,
        224,
    ),
    :convnextv2_large_fcmae_ft_in22k_in1k => ConvNeXtV2Variant(
        :convnextv2_large_fcmae_ft_in22k_in1k,
        _CN2_LARGE_DEPTHS,
        _CN2_LARGE_DIMS,
        "timm/convnextv2_large.fcmae_ft_in22k_in1k",
        1000,
        224,
    ),
    :convnextv2_large_fcmae_ft_in22k_in1k_384 => ConvNeXtV2Variant(
        :convnextv2_large_fcmae_ft_in22k_in1k_384,
        _CN2_LARGE_DEPTHS,
        _CN2_LARGE_DIMS,
        "timm/convnextv2_large.fcmae_ft_in22k_in1k_384",
        1000,
        384,
    ),

    # huge (no 224 in22k_in1k variant in timm; the in22k chain only ships at
    # 384 and 512).
    :convnextv2_huge_fcmae => ConvNeXtV2Variant(
        :convnextv2_huge_fcmae,
        _CN2_HUGE_DEPTHS,
        _CN2_HUGE_DIMS,
        "timm/convnextv2_huge.fcmae",
        0,
        224,
    ),
    :convnextv2_huge_fcmae_ft_in1k => ConvNeXtV2Variant(
        :convnextv2_huge_fcmae_ft_in1k,
        _CN2_HUGE_DEPTHS,
        _CN2_HUGE_DIMS,
        "timm/convnextv2_huge.fcmae_ft_in1k",
        1000,
        224,
    ),
    :convnextv2_huge_fcmae_ft_in22k_in1k_384 => ConvNeXtV2Variant(
        :convnextv2_huge_fcmae_ft_in22k_in1k_384,
        _CN2_HUGE_DEPTHS,
        _CN2_HUGE_DIMS,
        "timm/convnextv2_huge.fcmae_ft_in22k_in1k_384",
        1000,
        384,
    ),
    :convnextv2_huge_fcmae_ft_in22k_in1k_512 => ConvNeXtV2Variant(
        :convnextv2_huge_fcmae_ft_in22k_in1k_512,
        _CN2_HUGE_DEPTHS,
        _CN2_HUGE_DIMS,
        "timm/convnextv2_huge.fcmae_ft_in22k_in1k_512",
        1000,
        512,
    ),
)
