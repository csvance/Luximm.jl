# Variant catalog for VGG (timm's `vgg*` family).
#
# Covers the eight torchvision-trained checkpoints timm re-hosts: the four
# classic depths (11/13/16/19) from Simonyan & Zisserman 2014, each in a
# plain and a BatchNorm (`_bn`) flavor. All ship a 1000-class ImageNet head
# at 224x224, under torchvision's BSD-3-Clause license.
#
# The architecture is a stack of 3x3 (pad 1) convolutions and 2x2 stride-2
# max-pools described by the classic VGG "configuration" letters:
#
#   A (vgg11): 1,1,2,2,2 conv blocks per stage
#   B (vgg13): 2,2,2,2,2
#   D (vgg16): 2,2,3,3,3
#   E (vgg19): 2,2,4,4,4
#
# `cfg` is the flat timm layer list: an `Int` is a 3x3 conv producing that
# many channels, the `:M` marker is a 2x2/stride-2 max-pool. `batch_norm`
# inserts a BatchNorm between every conv and its ReLU (the `_bn` variants).

"""
    VGGVariant

Architectural config for a single VGG variant.

Fields:
- `name`: lookup key (e.g. `:vgg16_tv_in1k`).
- `cfg`: flat layer list. Each entry is either an `Int` (a 3x3 pad-1 conv
  with that output-channel count, followed by ReLU) or the `Symbol` `:M`
  (a 2x2 stride-2 max-pool). Matches timm's `cfgs` table in `vgg.py`.
- `batch_norm`: whether a BatchNorm sits between every conv and its ReLU
  (the `*_bn` checkpoints).
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with
  (1000 for every registered variant).
- `default_input_size`: native training resolution (224). Informational
  only: the model accepts any size large enough for the 7x7 `pre_logits`
  conv, but the released head was trained at 224.
"""
struct VGGVariant
    name::Symbol
    cfg::Vector{Any}
    batch_norm::Bool
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
end

# Classic VGG configuration letters. `:M` marks a max-pool boundary; every
# `Int` is a 3x3/pad-1 conv width. These match timm's `cfgs` dict in vgg.py.
const _VGG_CFG_A = Any[64, :M, 128, :M, 256, 256, :M, 512, 512, :M, 512, 512, :M]
const _VGG_CFG_B =
    Any[64, 64, :M, 128, 128, :M, 256, 256, :M, 512, 512, :M, 512, 512, :M]
const _VGG_CFG_D = Any[
    64, 64, :M, 128, 128, :M, 256, 256, 256, :M, 512, 512, 512, :M, 512, 512, 512, :M,
]
const _VGG_CFG_E = Any[
    64, 64, :M, 128, 128, :M, 256, 256, 256, 256, :M, 512, 512, 512, 512, :M, 512, 512,
    512, 512, :M,
]

# Feature-map output width (channels) feeding `pre_logits.fc1`. Every classic
# VGG ends its conv stack at 512 channels.
const _VGG_FEATURE_CH = 512
# `pre_logits` hidden width (the two 4096-wide ConvMlp layers).
const _VGG_MLP_CH = 4096

"""
    VGG_VARIANTS :: Dict{Symbol, VGGVariant}

Lookup table for the VGG variants ported from timm: the four classic depths
(11/13/16/19) in plain and BatchNorm flavors, all torchvision `tv_in1k`
checkpoints. Keys are the timm model name with the dot rewritten as an
underscore.
"""
const VGG_VARIANTS = Dict{Symbol,VGGVariant}(
    :vgg11_tv_in1k =>
        VGGVariant(:vgg11_tv_in1k, _VGG_CFG_A, false, "timm/vgg11.tv_in1k", 1000, 224),
    :vgg13_tv_in1k =>
        VGGVariant(:vgg13_tv_in1k, _VGG_CFG_B, false, "timm/vgg13.tv_in1k", 1000, 224),
    :vgg16_tv_in1k =>
        VGGVariant(:vgg16_tv_in1k, _VGG_CFG_D, false, "timm/vgg16.tv_in1k", 1000, 224),
    :vgg19_tv_in1k =>
        VGGVariant(:vgg19_tv_in1k, _VGG_CFG_E, false, "timm/vgg19.tv_in1k", 1000, 224),
    :vgg11_bn_tv_in1k => VGGVariant(
        :vgg11_bn_tv_in1k,
        _VGG_CFG_A,
        true,
        "timm/vgg11_bn.tv_in1k",
        1000,
        224,
    ),
    :vgg13_bn_tv_in1k => VGGVariant(
        :vgg13_bn_tv_in1k,
        _VGG_CFG_B,
        true,
        "timm/vgg13_bn.tv_in1k",
        1000,
        224,
    ),
    :vgg16_bn_tv_in1k => VGGVariant(
        :vgg16_bn_tv_in1k,
        _VGG_CFG_D,
        true,
        "timm/vgg16_bn.tv_in1k",
        1000,
        224,
    ),
    :vgg19_bn_tv_in1k => VGGVariant(
        :vgg19_bn_tv_in1k,
        _VGG_CFG_E,
        true,
        "timm/vgg19_bn.tv_in1k",
        1000,
        224,
    ),
)
