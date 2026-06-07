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
- `stride_mode`: how MBConv blocks downsample — `:pool` (avg-pool the main
  path, stride-1 convs) or `:dw` (stride the depthwise conv, no pool).
- `attn_early`: MBConv SE placement — `true` puts SE between the depthwise conv
  and norm2 (timm `se_early`), `false` after norm2 (timm `se`).
- `se_act`: MBConv SE bottleneck activation, `:relu` or `:silu`.
- `transformer_shortcut_bias`: whether the transformer downsample shortcut's
  1x1 expand conv carries a bias.
- `layer_scale`: whether transformer blocks apply LayerScale (`ls1`/`ls2`
  per-channel `gamma`) to the attention and MLP residual branches.
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
    stride_mode::Symbol
    attn_early::Bool
    se_act::Symbol
    transformer_shortcut_bias::Bool
    layer_scale::Bool
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
    # coatnet_0: pool-stride MBConv, early SE (ReLU), bias-free transformer
    # shortcut, no LayerScale.
    :coatnet_0_rw_224_sw_in1k => CoAtNetVariant(
        :coatnet_0_rw_224_sw_in1k,
        (2, 3, 7, 2),
        (96, 192, 384, 768),
        (32, 64),
        (:C, :C, :T, :T),
        :pool,
        true,
        :relu,
        false,
        false,
        224,
        "timm/coatnet_0_rw_224.sw_in1k",
        1000,
        224,
    ),
    # coatnet_1: depthwise-stride MBConv, otherwise like coatnet_0.
    :coatnet_1_rw_224_sw_in1k => CoAtNetVariant(
        :coatnet_1_rw_224_sw_in1k,
        (2, 6, 14, 2),
        (96, 192, 384, 768),
        (32, 64),
        (:C, :C, :T, :T),
        :dw,
        true,
        :relu,
        false,
        false,
        224,
        "timm/coatnet_1_rw_224.sw_in1k",
        1000,
        224,
    ),
    # coatnet_2: depthwise stride, late SE (SiLU), transformer shortcut bias.
    :coatnet_2_rw_224_sw_in12k_ft_in1k => CoAtNetVariant(
        :coatnet_2_rw_224_sw_in12k_ft_in1k,
        (2, 6, 14, 2),
        (128, 256, 512, 1024),
        (64, 128),
        (:C, :C, :T, :T),
        :dw,
        false,
        :silu,
        true,
        false,
        224,
        "timm/coatnet_2_rw_224.sw_in12k_ft_in1k",
        1000,
        224,
    ),
    :coatnet_2_rw_224_sw_in12k => CoAtNetVariant(
        :coatnet_2_rw_224_sw_in12k,
        (2, 6, 14, 2),
        (128, 256, 512, 1024),
        (64, 128),
        (:C, :C, :T, :T),
        :dw,
        false,
        :silu,
        true,
        false,
        224,
        "timm/coatnet_2_rw_224.sw_in12k",
        11821,
        224,
    ),
    # coatnet_3: like coatnet_2 plus LayerScale in the transformer blocks.
    :coatnet_3_rw_224_sw_in12k => CoAtNetVariant(
        :coatnet_3_rw_224_sw_in12k,
        (2, 6, 14, 2),
        (192, 384, 768, 1536),
        (96, 192),
        (:C, :C, :T, :T),
        :dw,
        false,
        :silu,
        true,
        true,
        224,
        "timm/coatnet_3_rw_224.sw_in12k",
        11821,
        224,
    ),
)
