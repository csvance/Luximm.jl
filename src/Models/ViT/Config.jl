# Variant catalog for ViT (timm's `vit_*` Vision Transformer family).
#
# First port target: `vit_base_patch16_224.augreg2_in21k_ft_in1k` — the
# canonical ViT-B/16 with a class token, learned absolute position embedding,
# and a plain (GELU) MLP FFN. This is the architectural foundation a wide range
# of checkpoints (DeiT, CLIP, SigLIP, MAE, DINOv2) extend.
#
# CLIP image towers (`vit_*_clip_*`, OpenAI / LAION) are the same
# `VisionTransformer` with three twists, all encoded in `ViTVariant` fields:
#   - `pre_norm = true`: a `norm_pre` LayerNorm runs on the token sequence
#     right after `pos_embed`, before the encoder blocks (timm `pre_norm=True`,
#     which mirrors OpenAI's `ln_pre`). timm's non-CLIP ViTs set this to
#     `nn.Identity()`, so the forward is identical — one captured layer, no
#     branching.
#   - `norm_eps = 1f-5`: CLIP towers normalize with `LayerNorm(eps=1e-5)`,
#     while timm's ImageNet ViTs use `eps=1e-6`. The eps is a per-variant
#     constant so the two families share one constructor.
#   - `stem_bias = false`: the CLIP patch-embed conv is bias-free, so the
#     state dict has no `patch_embed.proj.bias` and the mapping must not
#     reference one.

"""
    ViTVariant

Architectural config for a single Vision Transformer variant.

Fields:
- `name`: lookup key (e.g. `:vit_base_patch16_224_augreg2_in21k_ft_in1k`).
- `depth`: number of transformer encoder blocks.
- `embed_dim`: token / channel width.
- `num_heads`: attention heads (`head_dim = embed_dim ÷ num_heads`).
- `patch`: patch side length (16).
- `img_size`: native input resolution the position embedding was trained at.
  Enforced by the constructor, since absolute pos-embed has no interpolation
  path yet.
- `hf_repo`: HuggingFace repo containing `model.safetensors`.
- `default_num_classes`: head dimension the released weights ship with.
- `default_input_size`: native training resolution (== `img_size`).
- `pre_norm`: whether a `norm_pre` LayerNorm runs after `pos_embed` and
  before the encoder blocks (CLIP image towers; timm `pre_norm=True`).
- `norm_eps`: LayerNorm epsilon for every norm in the model — `1f-6` for
  timm's ImageNet ViTs, `1f-5` for the CLIP towers.
- `stem_bias`: whether the patch-embed conv has a bias (true for timm's
  ImageNet ViTs; false for CLIP, whose `patch_embed.proj` is bias-free).
"""
struct ViTVariant
    name::Symbol
    depth::Int
    embed_dim::Int
    num_heads::Int
    patch::Int
    img_size::Int
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
    pre_norm::Bool
    norm_eps::Float32
    stem_bias::Bool
end

"""
    vit_num_tokens(cfg) -> Int

Number of tokens in the prepended-class-token sequence: one class token plus
`(img_size ÷ patch)^2` patch tokens.
"""
vit_num_tokens(cfg::ViTVariant) = (cfg.img_size ÷ cfg.patch)^2 + 1

"""
    VIT_VARIANTS :: Dict{Symbol, ViTVariant}

Lookup table for the Vision Transformer variants ported from timm. Keys are the
timm model name with dots rewritten as underscores.
"""
const VIT_VARIANTS = Dict{Symbol,ViTVariant}(
    :vit_base_patch16_224_augreg2_in21k_ft_in1k => ViTVariant(
        :vit_base_patch16_224_augreg2_in21k_ft_in1k,
        12,
        768,
        12,
        16,
        224,
        "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        1000,
        224,
        false,
        1.0f-6,
        true,
    ),
    # CLIP image towers fine-tuned on ImageNet (timm `vit_*_clip_*`, OpenAI).
    # All are `pre_norm`, eps 1e-5, bias-free stem, 1000-class head.
    :vit_base_patch32_clip_224_openai_ft_in1k => ViTVariant(
        :vit_base_patch32_clip_224_openai_ft_in1k,
        12,
        768,
        12,
        32,
        224,
        "timm/vit_base_patch32_clip_224.openai_ft_in1k",
        1000,
        224,
        true,
        1.0f-5,
        false,
    ),
    :vit_base_patch16_clip_224_openai_ft_in1k => ViTVariant(
        :vit_base_patch16_clip_224_openai_ft_in1k,
        12,
        768,
        12,
        16,
        224,
        "timm/vit_base_patch16_clip_224.openai_ft_in1k",
        1000,
        224,
        true,
        1.0f-5,
        false,
    ),
    :vit_large_patch14_clip_224_openai_ft_in1k => ViTVariant(
        :vit_large_patch14_clip_224_openai_ft_in1k,
        24,
        1024,
        16,
        14,
        224,
        "timm/vit_large_patch14_clip_224.openai_ft_in1k",
        1000,
        224,
        true,
        1.0f-5,
        false,
    ),
)
