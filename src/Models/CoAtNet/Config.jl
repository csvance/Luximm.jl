# Variant catalog for CoAtNet (timm's `coatnet_*` family in maxxvit.py).
#
# First port target: `coatnet_0_rw_224.sw_in1k` — timm's "rw" recipe (weights on
# HF). A hybrid conv/transformer backbone: a conv stem, two MBConv stages, then
# two transformer stages with relative-position-bias attention.
#
# The "rw" recipe differs from the paper CoAtNet in several timm-specific ways
# (see `_rw_coat_cfg` in maxxvit.py): pre-norm includes an activation, MBConv
# expansion is computed from input channels, shortcut/final 1x1 convs are
# bias-free, SE uses ReLU, MBConv uses SiLU, attention expansion is done in the
# output projection, downsampling uses 2x2 average pooling, and SE sits between
# the depthwise conv and norm2 (`attn_early=True`). `coatnet_0_rw` additionally
# uses a bias-free transformer shortcut.

"""
    CoAtNetVariant

Architectural config for a single CoAtNet variant.

Fields:
- `name`: lookup key (e.g. `:coatnet_0_rw_224_sw_in1k`).
- `depths`: per-stage block count `(d1, d2, d3, d4)`.
- `dims`: per-stage output channel widths `(c1, c2, c3, c4)`; `c4` is
  `num_features`.
- `stem_width`: the two stem conv widths `(s1, s2)`; `s2` feeds stage 1.
- `block_types`: per-stage block kind, `:C` (MBConv) or `:T` (transformer).
- `img_size`: native input resolution (enforced; the transformer
  relative-position bias is sized to the per-stage feature map).
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with.
- `default_input_size`: native training resolution (== `img_size`).
"""
struct CoAtNetVariant
    name::Symbol
    depths::NTuple{4,Int}
    dims::NTuple{4,Int}
    stem_width::NTuple{2,Int}
    block_types::NTuple{4,Symbol}
    img_size::Int
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
end

"""
    coatnet_feat_sizes(cfg) -> NTuple{4,Int}

Per-stage (square) feature-map side length after that stage's stride-2
downsample, given the variant's `img_size`. The stem strides by 2, then each
stage halves: e.g. 224 → 112 → (56, 28, 14, 7). The transformer stages use
these to size the relative-position bias window.
"""
function coatnet_feat_sizes(cfg::CoAtNetVariant)
    feat = cfg.img_size ÷ 2          # after stem (stride 2)
    sizes = Int[]
    for _ = 1:4
        feat = (feat - 1) ÷ 2 + 1    # stage stride-2 downsample
        push!(sizes, feat)
    end
    return (sizes[1], sizes[2], sizes[3], sizes[4])
end

"""
    COATNET_VARIANTS :: Dict{Symbol, CoAtNetVariant}

Lookup table for the CoAtNet variants ported from timm. Keys are the timm model
name with dots rewritten as underscores.
"""
const COATNET_VARIANTS = Dict{Symbol,CoAtNetVariant}(
    :coatnet_0_rw_224_sw_in1k => CoAtNetVariant(
        :coatnet_0_rw_224_sw_in1k,
        (2, 3, 7, 2),
        (96, 192, 384, 768),
        (32, 64),
        (:C, :C, :T, :T),
        224,
        "timm/coatnet_0_rw_224.sw_in1k",
        1000,
        224,
    ),
)
