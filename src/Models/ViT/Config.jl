# Variant catalog for ViT (timm's `vit_*` Vision Transformer family).
#
# First port target: `vit_base_patch16_224.augreg2_in21k_ft_in1k` — the
# canonical ViT-B/16 with a class token, learned absolute position embedding,
# and a plain (GELU) MLP FFN. This is the architectural foundation a wide range
# of checkpoints (DeiT, CLIP, SigLIP, MAE, DINOv2) extend.

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
    ),
)
