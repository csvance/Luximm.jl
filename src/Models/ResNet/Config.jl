# Variant catalog for classic timm ResNet.

"""
    ResNetVariant

Architectural config for a classic timm ResNet variant.

Fields:
- `name`: lookup key (e.g. `:resnet50_a1_in1k`).
- `block`: residual block type, either `:basic` (used by r18/r34) or
  `:bottleneck` (used by r50/r101/r152).
- `layers`: per-stage block count `(d1, d2, d3, d4)`.
- `planes`: base channel widths per stage `(64, 128, 256, 512)`.
  Multiplied by 4 inside `:bottleneck` stages to give the actual
  output channel count.
- `num_features`: backbone output channels (`planes[end]` for `:basic`,
  `planes[end] * 4` for `:bottleneck`).
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with.
- `default_input_size`: native training resolution (224 for every
  registered variant). Informational only: the model is fully
  convolutional and accepts any size.
"""
struct ResNetVariant
    name::Symbol
    block::Symbol
    layers::NTuple{4, Int}
    planes::NTuple{4, Int}
    num_features::Int
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
end

"""
    RESNET_VARIANTS :: Dict{Symbol, ResNetVariant}

Lookup table for classic ResNet variants currently ported from timm.
Keys are the timm model names with dots rewritten as underscores.
"""
const RESNET_VARIANTS = Dict{Symbol, ResNetVariant}(
    :resnet18_a1_in1k => ResNetVariant(
        :resnet18_a1_in1k,
        :basic,
        (2, 2, 2, 2),
        (64, 128, 256, 512),
        512,
        "timm/resnet18.a1_in1k",
        1000,
        224,
    ),
    :resnet34_a1_in1k => ResNetVariant(
        :resnet34_a1_in1k,
        :basic,
        (3, 4, 6, 3),
        (64, 128, 256, 512),
        512,
        "timm/resnet34.a1_in1k",
        1000,
        224,
    ),
    :resnet50_a1_in1k => ResNetVariant(
        :resnet50_a1_in1k,
        :bottleneck,
        (3, 4, 6, 3),
        (64, 128, 256, 512),
        2048,
        "timm/resnet50.a1_in1k",
        1000,
        224,
    ),
    :resnet101_a1_in1k => ResNetVariant(
        :resnet101_a1_in1k,
        :bottleneck,
        (3, 4, 23, 3),
        (64, 128, 256, 512),
        2048,
        "timm/resnet101.a1_in1k",
        1000,
        224,
    ),
    :resnet152_a1_in1k => ResNetVariant(
        :resnet152_a1_in1k,
        :bottleneck,
        (3, 8, 36, 3),
        (64, 128, 256, 512),
        2048,
        "timm/resnet152.a1_in1k",
        1000,
        224,
    ),
)
