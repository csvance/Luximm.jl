# Vision Transformer backbone, Lux port of timm's `vit_*` family.
#
# Architecture (matches timm `VisionTransformer` with class token, learned
# absolute pos-embed, plain GELU MLP, `global_pool='token'`, no `fc_norm` /
# `pre_logits`):
#   - `patch_embed`: stride-`patch` conv → token tensor `(embed_dim, T_patch, N)`.
#   - prepend `cls_token`, add `pos_embed` (both captured params).
#   - optional `norm_pre`: for CLIP towers (`pre_norm = true` in the variant
#     config), a LayerNorm over the token sequence right after `pos_embed` and
#     before `blocks` (timm `norm_pre`, mirroring OpenAI's `ln_pre`). Non-CLIP
#     variants capture `identity` here, so the forward is uniform.
#   - `blocks`: `depth` pre-norm transformer encoder blocks (Chain).
#   - final `norm` (LayerNorm over channel axis; eps per-variant: 1e-6 for
#     timm ImageNet ViTs, 1e-5 for CLIP) over the full token sequence.
#
# `num_classes == 0` returns the full normed sequence `(embed_dim, T, N)` —
# exactly timm `forward_features` (not pooled). `num_classes > 0` takes the
# class token (position 1) and applies `head`, returning `(num_classes, N)`,
# matching timm `forward`.
#
# Token layout is `(C, T, N)`; see `src/Layers/{PatchEmbed,Attention,
# TransformerBlock}.jl` for the per-layer numeric notes (qkv split order,
# softmax-over-key axis, channel-axis LayerNorm with eps 1e-6).
#
# Absolute pos-embed is fixed-length, so the constructor enforces the variant's
# native `img_size`; interpolation to other resolutions is a future extension.

include("Config.jl")

# Shared backbone forward, called from both branches so the feature-extractor
# and classifier paths cannot desync. Returns the full normed token sequence.
# `norm_pre` is `identity` for non-CLIP variants, so the norm_pre call is a
# no-op there — one code path for both.
function _vit_features(x, patch, cls_token, pos_embed, norm_pre, blocks, norm)
    x = patch(x)                          # (D, T_patch, N)
    D = size(x, 1)
    N = size(x, 3)
    cls = repeat(cls_token, 1, 1, N)      # (D, 1, N)
    x = cat(cls, x; dims = 2)             # (D, T, N), class token first
    x = x .+ pos_embed                    # pos_embed (D, T, 1) broadcasts over N
    x = norm_pre(x)                       # CLIP towers only; identity otherwise
    x = blocks(x)
    return norm(x)
end

"""
    vit(variant; in_chans=3, num_classes=0) -> @compact block

Build a Vision Transformer. `variant` is a key from [`VIT_VARIANTS`](@ref),
e.g. `:vit_base_patch16_224_augreg2_in21k_ft_in1k`.

When `num_classes == 0`, the forward returns the full normed token sequence
`(embed_dim, T, N)`, matching `timm.forward_features(x)`. When `num_classes >
0`, the class token is selected and passed through `head`, returning
`(num_classes, N)`, matching `timm.forward(x)`.

The input spatial size must equal the variant's native `img_size`; the
absolute position embedding has no interpolation path yet.
"""
function vit(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Int = 0,
    features_only::Bool = false,
    out_indices = nothing,
)
    # No pyramid: a plain ViT is single-scale. timm synthesizes one by
    # reshaping selected block outputs back to a grid, but every level then
    # sits at the same reduction (the patch size), which is not what a
    # UNet/FPN decoder wants.
    _no_feature_pyramid("ViT", variant, features_only, out_indices)
    cfg = get(VIT_VARIANTS, variant) do
        error(
            "Unknown ViT variant: $variant. Known variants: " *
            "$(sort(collect(keys(VIT_VARIANTS))))",
        )
    end
    D = cfg.embed_dim
    T = vit_num_tokens(cfg)
    img = cfg.img_size
    eps = cfg.norm_eps
    # timm uses `nn.Identity()` when `pre_norm=False`, so a captured `identity`
    # keeps one forward for both CLIP and ImageNet variants.
    norm_pre = cfg.pre_norm ? vit_layernorm(D; eps = eps) : identity

    if num_classes == 0
        @compact(
            patch = patch_embed(in_chans, D; patch = cfg.patch, use_bias = cfg.stem_bias),
            cls_token = zeros32(D, 1, 1),
            pos_embed = zeros32(D, T, 1),
            norm_pre = norm_pre,
            blocks = Chain([vit_block(D; num_heads = cfg.num_heads, eps = eps) for _ = 1:cfg.depth]...),
            norm = vit_layernorm(D; eps = eps),
        ) do x
            @assert size(x, 1) == img && size(x, 2) == img "ViT $variant expects " *
                "$(img)x$(img) input; got $(size(x, 1))x$(size(x, 2)). " *
                "Pos-embed interpolation is not implemented."
            @return _vit_features(x, patch, cls_token, pos_embed, norm_pre, blocks, norm)
        end
    else
        nc = num_classes
        @compact(
            patch = patch_embed(in_chans, D; patch = cfg.patch, use_bias = cfg.stem_bias),
            cls_token = zeros32(D, 1, 1),
            pos_embed = zeros32(D, T, 1),
            norm_pre = norm_pre,
            blocks = Chain([vit_block(D; num_heads = cfg.num_heads, eps = eps) for _ = 1:cfg.depth]...),
            norm = vit_layernorm(D; eps = eps),
            head = Dense(D => nc; init_bias = zeros32),
        ) do x
            @assert size(x, 1) == img && size(x, 2) == img "ViT $variant expects " *
                "$(img)x$(img) input; got $(size(x, 1))x$(size(x, 2)). " *
                "Pos-embed interpolation is not implemented."
            x = _vit_features(x, patch, cls_token, pos_embed, norm_pre, blocks, norm)
            cls = reshape(x[:, 1:1, :], size(x, 1), size(x, 3))   # (D, N)
            @return head(cls)
        end
    end
end

# -- Pretrained-weight loading -------------------------------------------

_VIT_MAPPING_ENTRY = Tuple{String,Tuple{Vararg{Symbol}},Function}

"""
    vit_mapping(state_dict, variant; load_classifier=false, in_chans=3,
                prefix=()) -> Vector

Build the `(pytorch_key, lux_path, transform)` triples mapping a timm `vit_*`
state_dict into the Lux tree from [`vit`](@ref).

`cls_token` / `pos_embed` arrive (after axis reversal) as `(D, 1, 1)` /
`(D, T, 1)`, matching the captured params (`identity`). The patch-embed conv is
`identity` (or `adapt_input_conv` for `in_chans != 3`). LayerNorm `(D,)`
parameters are reshaped to `(D, 1, 1)` via `as_token_norm`. Dense/Linear
weights (`qkv`, `proj`, `mlp.fc1/fc2`, `head`) use `axis_reverse`; their biases
and the 1-D norm params use `identity`.
"""
function vit_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(VIT_VARIANTS, variant) do
        error(
            "Unknown ViT variant: $variant. Known variants: " *
            "$(sort(collect(keys(VIT_VARIANTS))))",
        )
    end
    mapping = _VIT_MAPPING_ENTRY[]

    push!(mapping, ("cls_token", (prefix..., :cls_token), identity))
    push!(mapping, ("pos_embed", (prefix..., :pos_embed), identity))

    stem_w_transform = in_chans == 3 ? identity : adapt_input_conv(in_chans)
    push!(
        mapping,
        ("patch_embed.proj.weight", (prefix..., :patch, :proj, :weight), stem_w_transform),
    )
    # CLIP towers have a bias-free stem conv, so their state dict has no
    # `patch_embed.proj.bias`; only reference it when the variant has one.
    if cfg.stem_bias
        push!(
            mapping,
            ("patch_embed.proj.bias", (prefix..., :patch, :proj, :bias), identity),
        )
    end
    # Same for `norm_pre`: only CLIP variants (`pre_norm = true`) ship the
    # pre-encoder LayerNorm parameters.
    if cfg.pre_norm
        push!(mapping, ("norm_pre.weight", (prefix..., :norm_pre, :scale), as_token_norm))
        push!(mapping, ("norm_pre.bias", (prefix..., :norm_pre, :bias), as_token_norm))
    end

    for n = 1:cfg.depth
        blk = (prefix..., :blocks, Symbol("layer_", n))
        py = "blocks.$(n - 1)"
        push!(mapping, ("$(py).norm1.weight", (blk..., :norm1, :scale), as_token_norm))
        push!(mapping, ("$(py).norm1.bias", (blk..., :norm1, :bias), as_token_norm))
        push!(mapping, ("$(py).attn.qkv.weight", (blk..., :attn, :qkv, :weight), axis_reverse))
        push!(mapping, ("$(py).attn.qkv.bias", (blk..., :attn, :qkv, :bias), identity))
        push!(mapping, ("$(py).attn.proj.weight", (blk..., :attn, :proj, :weight), axis_reverse))
        push!(mapping, ("$(py).attn.proj.bias", (blk..., :attn, :proj, :bias), identity))
        push!(mapping, ("$(py).norm2.weight", (blk..., :norm2, :scale), as_token_norm))
        push!(mapping, ("$(py).norm2.bias", (blk..., :norm2, :bias), as_token_norm))
        push!(mapping, ("$(py).mlp.fc1.weight", (blk..., :fc1, :weight), axis_reverse))
        push!(mapping, ("$(py).mlp.fc1.bias", (blk..., :fc1, :bias), identity))
        push!(mapping, ("$(py).mlp.fc2.weight", (blk..., :fc2, :weight), axis_reverse))
        push!(mapping, ("$(py).mlp.fc2.bias", (blk..., :fc2, :bias), identity))
    end

    push!(mapping, ("norm.weight", (prefix..., :norm, :scale), as_token_norm))
    push!(mapping, ("norm.bias", (prefix..., :norm, :bias), as_token_norm))

    if load_classifier
        push!(mapping, ("head.weight", (prefix..., :head, :weight), axis_reverse))
        push!(mapping, ("head.bias", (prefix..., :head, :bias), identity))
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    _load_vit(ps, st, variant; in_chans, num_classes, revision, cache_dir,
              prefix) -> (ps, st)

Private back-end for `create_pretrained` on ViT variants. Loads the timm
`.safetensors` from HuggingFace and applies the params into `ps`; `st` is
returned unchanged (LayerNorm has no running statistics). The classifier-head
handling matches the other families' three cases.
"""
function _load_vit(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = VIT_VARIANTS[variant]
    load_classifier = num_classes > 0 && num_classes == cfg.default_num_classes
    if num_classes > 0 && num_classes != cfg.default_num_classes
        @warn "variant $variant ships $(cfg.default_num_classes)-class pretrained weights, " *
              "but the model has a $num_classes-class head. Loading the backbone only; " *
              "the classifier head is left at its Lux.setup random initialization for you to train."
    end
    path = hf_hub_download(
        cfg.hf_repo,
        "model.safetensors";
        revision = revision,
        cache_dir = cache_dir,
    )
    sd = load_safetensors_state_dict(path)
    ps = apply_state_dict(
        ps,
        sd,
        vit_mapping(
            sd,
            variant;
            load_classifier = load_classifier,
            in_chans = in_chans,
            prefix = prefix,
        ),
    )
    return ps, st
end
