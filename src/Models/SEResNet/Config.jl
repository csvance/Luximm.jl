# Variant catalog for SE-ResNet (timm's `seresnet*` family).
#
# SE-ResNet is a classic post-activation ResNet whose bottleneck blocks gain a
# squeeze-and-excitation channel-attention module after the final BN. The
# first port target is `seresnet50.a1_in1k`: the ResNet-50 (3,4,6,3) bottleneck
# layout with the plain 7x7 stem, plus SE with reduction 16.

"""
    SEResNetVariant

Architectural config for a single SE-ResNet variant.

Fields:
- `name`: lookup key (e.g. `:seresnet50_a1_in1k`).
- `layers`: per-stage block count `(d1, d2, d3, d4)`.
- `planes`: base channel widths per stage `(64, 128, 256, 512)`. Multiplied by
  4 (the bottleneck expansion) to give the actual stage output channels.
- `num_features`: backbone output channels (`planes[end] * 4 = 2048`).
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with.
- `default_input_size`: native training resolution (224). Informational only.
- `se_reduction`: SE bottleneck reduction divisor (16 for every variant); the
  SE inner width is `se_make_divisible(out_ch / se_reduction, 8)`.
"""
struct SEResNetVariant
    name::Symbol
    layers::NTuple{4,Int}
    planes::NTuple{4,Int}
    num_features::Int
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
    se_reduction::Int
end

"""
    SERESNET_VARIANTS :: Dict{Symbol, SEResNetVariant}

Lookup table for the SE-ResNet variants ported from timm. Keys are the timm
model name with dots rewritten as underscores.
"""
const SERESNET_VARIANTS = Dict{Symbol,SEResNetVariant}(
    :seresnet50_a1_in1k => SEResNetVariant(
        :seresnet50_a1_in1k,
        (3, 4, 6, 3),
        (64, 128, 256, 512),
        2048,
        "timm/seresnet50.a1_in1k",
        1000,
        224,
        16,
    ),
)
