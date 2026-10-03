# Variant catalog for `bit_resnetv2`.
#
# Each entry captures the architectural knobs that timm's BiT factory
# (`_create_resnetv2_bit` in timm/models/resnetv2.py) passes through to the
# generic `ResNetV2` class, plus the HuggingFace repo to pull pretrained
# weights from and the number of classes the head was trained with. Other
# BiT details (stem padding, GroupNorm groups, bottle ratio, StdConv eps,
# strides) are constants across all six variants and live in the
# constructor.
#
# Keys mirror the timm model name with the dot rewritten as an underscore
# (the dot is reserved in Julia identifiers). `hf_repo` keeps the real
# dot-separated timm name. So future tags such as `.goog_in21k_ft_in1k`
# slot in as `:resnetv2_50x1_bit_goog_in21k_ft_in1k` without colliding
# with the in21k entries below.

"""
    BiTVariant

Architectural config for a single BiT ResNetV2 variant.

Fields:
- `name`: lookup key (e.g. `:resnetv2_50x1_bit_goog_in21k`).
- `layers`: per-stage depth tuple (3,4,6,3) for r50, (3,4,23,3) for r101,
  (3,8,36,3) for r152.
- `width_factor`: integer width multiplier from the timm name suffix
  (`x1`, `x2`, `x3`, `x4`).
- `stem_chs`: stem output channels (`64 * width_factor`).
- `stage_chs`: per-stage output channel tuple (base widths
  `(256,512,1024,2048)` scaled by `width_factor`).
- `num_features`: backbone output channels (`stage_chs[end]`).
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights were trained
  with (21843 for `goog_in21k`, 1000 for the in1k tags).
- `default_input_size`: native training resolution (224 for most tags,
  384 for the `_384` teacher variant). The model itself is fully
  convolutional and accepts any input size; this is just what the
  released weights were tuned at.
"""
struct BiTVariant
    name::Symbol
    layers::NTuple{4, Int}
    width_factor::Int
    stem_chs::Int
    stage_chs::NTuple{4, Int}
    num_features::Int
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
end

"""
    BIT_VARIANTS :: Dict{Symbol, BiTVariant}

Lookup table for the BiT variants this package currently ports. Keys
mirror the timm model name with the dot rewritten as an underscore (the
dot is reserved in Julia identifiers); the full timm name with the dot
lives at `BIT_VARIANTS[key].hf_repo`.
"""
const BIT_VARIANTS = Dict{Symbol, BiTVariant}(
    :resnetv2_50x1_bit_goog_in21k => BiTVariant(
        :resnetv2_50x1_bit_goog_in21k,
        (3, 4, 6, 3),
        1,
        64,
        (256, 512, 1024, 2048),
        2048,
        "timm/resnetv2_50x1_bit.goog_in21k",
        21843,
        224,
    ),
    :resnetv2_50x3_bit_goog_in21k => BiTVariant(
        :resnetv2_50x3_bit_goog_in21k,
        (3, 4, 6, 3),
        3,
        192,
        (768, 1536, 3072, 6144),
        6144,
        "timm/resnetv2_50x3_bit.goog_in21k",
        21843,
        224,
    ),
    :resnetv2_101x1_bit_goog_in21k => BiTVariant(
        :resnetv2_101x1_bit_goog_in21k,
        (3, 4, 23, 3),
        1,
        64,
        (256, 512, 1024, 2048),
        2048,
        "timm/resnetv2_101x1_bit.goog_in21k",
        21843,
        224,
    ),
    :resnetv2_101x3_bit_goog_in21k => BiTVariant(
        :resnetv2_101x3_bit_goog_in21k,
        (3, 4, 23, 3),
        3,
        192,
        (768, 1536, 3072, 6144),
        6144,
        "timm/resnetv2_101x3_bit.goog_in21k",
        21843,
        224,
    ),
    :resnetv2_152x2_bit_goog_in21k => BiTVariant(
        :resnetv2_152x2_bit_goog_in21k,
        (3, 8, 36, 3),
        2,
        128,
        (512, 1024, 2048, 4096),
        4096,
        "timm/resnetv2_152x2_bit.goog_in21k",
        21843,
        224,
    ),
    :resnetv2_152x4_bit_goog_in21k => BiTVariant(
        :resnetv2_152x4_bit_goog_in21k,
        (3, 8, 36, 3),
        4,
        256,
        (1024, 2048, 4096, 8192),
        8192,
        "timm/resnetv2_152x4_bit.goog_in21k",
        21843,
        224,
    ),
    :resnetv2_50x1_bit_goog_distilled_in1k => BiTVariant(
        :resnetv2_50x1_bit_goog_distilled_in1k,
        (3, 4, 6, 3),
        1,
        64,
        (256, 512, 1024, 2048),
        2048,
        "timm/resnetv2_50x1_bit.goog_distilled_in1k",
        1000,
        224,
    ),
    :resnetv2_50x1_bit_goog_in21k_ft_in1k => BiTVariant(
        :resnetv2_50x1_bit_goog_in21k_ft_in1k,
        (3, 4, 6, 3),
        1,
        64,
        (256, 512, 1024, 2048),
        2048,
        "timm/resnetv2_50x1_bit.goog_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_50x3_bit_goog_in21k_ft_in1k => BiTVariant(
        :resnetv2_50x3_bit_goog_in21k_ft_in1k,
        (3, 4, 6, 3),
        3,
        192,
        (768, 1536, 3072, 6144),
        6144,
        "timm/resnetv2_50x3_bit.goog_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_101x1_bit_goog_in21k_ft_in1k => BiTVariant(
        :resnetv2_101x1_bit_goog_in21k_ft_in1k,
        (3, 4, 23, 3),
        1,
        64,
        (256, 512, 1024, 2048),
        2048,
        "timm/resnetv2_101x1_bit.goog_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_101x3_bit_goog_in21k_ft_in1k => BiTVariant(
        :resnetv2_101x3_bit_goog_in21k_ft_in1k,
        (3, 4, 23, 3),
        3,
        192,
        (768, 1536, 3072, 6144),
        6144,
        "timm/resnetv2_101x3_bit.goog_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_152x2_bit_goog_in21k_ft_in1k => BiTVariant(
        :resnetv2_152x2_bit_goog_in21k_ft_in1k,
        (3, 8, 36, 3),
        2,
        128,
        (512, 1024, 2048, 4096),
        4096,
        "timm/resnetv2_152x2_bit.goog_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_152x4_bit_goog_in21k_ft_in1k => BiTVariant(
        :resnetv2_152x4_bit_goog_in21k_ft_in1k,
        (3, 8, 36, 3),
        4,
        256,
        (1024, 2048, 4096, 8192),
        8192,
        "timm/resnetv2_152x4_bit.goog_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k => BiTVariant(
        :resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k,
        (3, 8, 36, 3),
        2,
        128,
        (512, 1024, 2048, 4096),
        4096,
        "timm/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k",
        1000,
        224,
    ),
    :resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k_384 => BiTVariant(
        :resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k_384,
        (3, 8, 36, 3),
        2,
        128,
        (512, 1024, 2048, 4096),
        4096,
        "timm/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384",
        1000,
        384,
    ),
)
