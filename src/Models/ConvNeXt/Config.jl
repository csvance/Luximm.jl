# Variant catalog for `convnext` (timm's ConvNeXt v1 family).
#
# Covers two release lineages:
#
#   - Meta's four DINOv3-pretrained encoders (`*.dinov3_lvd1689m`), which
#     ship `num_classes = 0` (no usable classification head). Licensed
#     under Meta's DINOv3 License.
#   - The 19 Facebook AI checkpoints from the original 2022 ConvNeXt paper
#     (Liu et al., "A ConvNet for the 2020s"): five sizes (T/S/B/L/XL)
#     crossed with `.fb_in1k`, `.fb_in22k`, `.fb_in22k_ft_in1k`, and
#     `.fb_in22k_ft_in1k_384` (XLarge has no `.fb_in1k`). These ship
#     trained classifier heads (1000 classes for IN1K, 21841 for IN22K)
#     and are Apache 2.0, matching timm's own license.
#
# Architectural flags come straight from timm's `convnext_*` factory
# defaults: `conv_mlp=False`, `use_grn=False`, `ls_init_value=1e-6`,
# `stem_type='patch'`, `patch_size=4`, `kernel_sizes=7`, `act_layer='gelu'`
# (the exact erf-based GELU).
#
# Keys mirror the timm model name with the dot rewritten as an underscore (the
# dot is reserved in Julia identifiers). `hf_repo` keeps the real dot-separated
# timm name. Other ConvNeXt v1 lineages (`.in12k_*`, `.clip_laion2b_*`) are
# not registered here; the architecture supports them, only the table needs
# entries.

"""
    ConvNeXtVariant

Architectural config for a single ConvNeXt v1 variant.

Fields:
- `name`: lookup key (e.g. `:convnext_tiny_dinov3_lvd1689m`).
- `depths`: per-stage block count, `(d1, d2, d3, d4)`.
- `dims`: per-stage channel widths, `(c1, c2, c3, c4)`. `c1` is also the
  stem output channels. `c4` is `num_features`.
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with.
  `0` for the DINO encoders (no usable head).
- `default_input_size`: native training resolution (224, 384, …) for
  the released checkpoint. Informational only: the model is fully
  convolutional and accepts any size, so this is not enforced.
- `ls_init`: LayerScale init value (`gamma` parameter in timm). All v1
  variants released so far use `1e-6`; kept as a field in case future
  ports need a different value.
"""
struct ConvNeXtVariant
    name::Symbol
    depths::NTuple{4, Int}
    dims::NTuple{4, Int}
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
    ls_init::Float32
end

# Per-size architecture parameters mirroring timm's `convnext_<size>` factory
# functions in convnext.py. Tiny uses depth 9 in stage 3; small/base/large/
# xlarge all use depth 27. Widths grow as a power of 2.
const _CN_TINY_DEPTHS = (3, 3, 9, 3)
const _CN_TINY_DIMS = (96, 192, 384, 768)
const _CN_SMALL_DEPTHS = (3, 3, 27, 3)
const _CN_SMALL_DIMS = (96, 192, 384, 768)
const _CN_BASE_DEPTHS = (3, 3, 27, 3)
const _CN_BASE_DIMS = (128, 256, 512, 1024)
const _CN_LARGE_DEPTHS = (3, 3, 27, 3)
const _CN_LARGE_DIMS = (192, 384, 768, 1536)
const _CN_XLARGE_DEPTHS = (3, 3, 27, 3)
const _CN_XLARGE_DIMS = (256, 512, 1024, 2048)

# Default LayerScale init for the v1 variants registered here. All
# released v1 checkpoints (FB-paper `.fb_*` and Meta `.dinov3_*`) use the
# same `ls_init_value = 1e-6` from timm's factory defaults.
const _CN_V1_LS_INIT = 1.0f-6

"""
    CONVNEXT_VARIANTS :: Dict{Symbol, ConvNeXtVariant}

Lookup table for the ConvNeXt v1 variants this package ports: the
DINOv3 encoders and the Facebook AI checkpoints from the original
ConvNeXt paper. Additional `convnext_*` lineages (`.in12k_*`, `.clip_*`)
can be registered without touching the constructor or mapping code.
"""
const CONVNEXT_VARIANTS = Dict{Symbol, ConvNeXtVariant}(
    # Meta DINOv3 encoders (no usable head).
    :convnext_tiny_dinov3_lvd1689m => ConvNeXtVariant(
        :convnext_tiny_dinov3_lvd1689m,
        _CN_TINY_DEPTHS,
        _CN_TINY_DIMS,
        "timm/convnext_tiny.dinov3_lvd1689m",
        0,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_small_dinov3_lvd1689m => ConvNeXtVariant(
        :convnext_small_dinov3_lvd1689m,
        _CN_SMALL_DEPTHS,
        _CN_SMALL_DIMS,
        "timm/convnext_small.dinov3_lvd1689m",
        0,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_base_dinov3_lvd1689m => ConvNeXtVariant(
        :convnext_base_dinov3_lvd1689m,
        _CN_BASE_DEPTHS,
        _CN_BASE_DIMS,
        "timm/convnext_base.dinov3_lvd1689m",
        0,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_large_dinov3_lvd1689m => ConvNeXtVariant(
        :convnext_large_dinov3_lvd1689m,
        _CN_LARGE_DEPTHS,
        _CN_LARGE_DIMS,
        "timm/convnext_large.dinov3_lvd1689m",
        0,
        224,
        _CN_V1_LS_INIT,
    ),

    # Facebook AI checkpoints from the original 2022 ConvNeXt paper.
    # Tiny.
    :convnext_tiny_fb_in1k => ConvNeXtVariant(
        :convnext_tiny_fb_in1k,
        _CN_TINY_DEPTHS,
        _CN_TINY_DIMS,
        "timm/convnext_tiny.fb_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_tiny_fb_in22k => ConvNeXtVariant(
        :convnext_tiny_fb_in22k,
        _CN_TINY_DEPTHS,
        _CN_TINY_DIMS,
        "timm/convnext_tiny.fb_in22k",
        21841,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_tiny_fb_in22k_ft_in1k => ConvNeXtVariant(
        :convnext_tiny_fb_in22k_ft_in1k,
        _CN_TINY_DEPTHS,
        _CN_TINY_DIMS,
        "timm/convnext_tiny.fb_in22k_ft_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_tiny_fb_in22k_ft_in1k_384 => ConvNeXtVariant(
        :convnext_tiny_fb_in22k_ft_in1k_384,
        _CN_TINY_DEPTHS,
        _CN_TINY_DIMS,
        "timm/convnext_tiny.fb_in22k_ft_in1k_384",
        1000,
        384,
        _CN_V1_LS_INIT,
    ),

    # Small.
    :convnext_small_fb_in1k => ConvNeXtVariant(
        :convnext_small_fb_in1k,
        _CN_SMALL_DEPTHS,
        _CN_SMALL_DIMS,
        "timm/convnext_small.fb_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_small_fb_in22k => ConvNeXtVariant(
        :convnext_small_fb_in22k,
        _CN_SMALL_DEPTHS,
        _CN_SMALL_DIMS,
        "timm/convnext_small.fb_in22k",
        21841,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_small_fb_in22k_ft_in1k => ConvNeXtVariant(
        :convnext_small_fb_in22k_ft_in1k,
        _CN_SMALL_DEPTHS,
        _CN_SMALL_DIMS,
        "timm/convnext_small.fb_in22k_ft_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_small_fb_in22k_ft_in1k_384 => ConvNeXtVariant(
        :convnext_small_fb_in22k_ft_in1k_384,
        _CN_SMALL_DEPTHS,
        _CN_SMALL_DIMS,
        "timm/convnext_small.fb_in22k_ft_in1k_384",
        1000,
        384,
        _CN_V1_LS_INIT,
    ),

    # Base.
    :convnext_base_fb_in1k => ConvNeXtVariant(
        :convnext_base_fb_in1k,
        _CN_BASE_DEPTHS,
        _CN_BASE_DIMS,
        "timm/convnext_base.fb_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_base_fb_in22k => ConvNeXtVariant(
        :convnext_base_fb_in22k,
        _CN_BASE_DEPTHS,
        _CN_BASE_DIMS,
        "timm/convnext_base.fb_in22k",
        21841,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_base_fb_in22k_ft_in1k => ConvNeXtVariant(
        :convnext_base_fb_in22k_ft_in1k,
        _CN_BASE_DEPTHS,
        _CN_BASE_DIMS,
        "timm/convnext_base.fb_in22k_ft_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_base_fb_in22k_ft_in1k_384 => ConvNeXtVariant(
        :convnext_base_fb_in22k_ft_in1k_384,
        _CN_BASE_DEPTHS,
        _CN_BASE_DIMS,
        "timm/convnext_base.fb_in22k_ft_in1k_384",
        1000,
        384,
        _CN_V1_LS_INIT,
    ),

    # Large.
    :convnext_large_fb_in1k => ConvNeXtVariant(
        :convnext_large_fb_in1k,
        _CN_LARGE_DEPTHS,
        _CN_LARGE_DIMS,
        "timm/convnext_large.fb_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_large_fb_in22k => ConvNeXtVariant(
        :convnext_large_fb_in22k,
        _CN_LARGE_DEPTHS,
        _CN_LARGE_DIMS,
        "timm/convnext_large.fb_in22k",
        21841,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_large_fb_in22k_ft_in1k => ConvNeXtVariant(
        :convnext_large_fb_in22k_ft_in1k,
        _CN_LARGE_DEPTHS,
        _CN_LARGE_DIMS,
        "timm/convnext_large.fb_in22k_ft_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_large_fb_in22k_ft_in1k_384 => ConvNeXtVariant(
        :convnext_large_fb_in22k_ft_in1k_384,
        _CN_LARGE_DEPTHS,
        _CN_LARGE_DIMS,
        "timm/convnext_large.fb_in22k_ft_in1k_384",
        1000,
        384,
        _CN_V1_LS_INIT,
    ),

    # XLarge (no from-scratch IN1K release).
    :convnext_xlarge_fb_in22k => ConvNeXtVariant(
        :convnext_xlarge_fb_in22k,
        _CN_XLARGE_DEPTHS,
        _CN_XLARGE_DIMS,
        "timm/convnext_xlarge.fb_in22k",
        21841,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_xlarge_fb_in22k_ft_in1k => ConvNeXtVariant(
        :convnext_xlarge_fb_in22k_ft_in1k,
        _CN_XLARGE_DEPTHS,
        _CN_XLARGE_DIMS,
        "timm/convnext_xlarge.fb_in22k_ft_in1k",
        1000,
        224,
        _CN_V1_LS_INIT,
    ),
    :convnext_xlarge_fb_in22k_ft_in1k_384 => ConvNeXtVariant(
        :convnext_xlarge_fb_in22k_ft_in1k_384,
        _CN_XLARGE_DEPTHS,
        _CN_XLARGE_DIMS,
        "timm/convnext_xlarge.fb_in22k_ft_in1k_384",
        1000,
        384,
        _CN_V1_LS_INIT,
    ),
)
