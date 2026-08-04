# ConvNeXtV2 backbone, Lux port of timm's `convnextv2_<variant>` family.
#
# Architecture (matches timm with `use_grn=True`, `conv_mlp=True`,
# `ls_init_value=None`):
#   - Stem: Conv((4,4), in_chans => dims[1]; stride=4, bias) followed by
#     LayerNorm2d(dims[1]).
#   - Four stages with depths from the variant. Stage 0 has no downsample
#     (the stem already strode by 4); stages 1-3 begin with a downsample
#     block of LayerNorm2d(in) + Conv((2,2); stride=2, bias).
#   - Each block: depthwise Conv((7,7); groups=C, pad=3, bias) → LayerNorm2d →
#     1x1 Conv (C => 4C, bias) → GELU → GRN → 1x1 Conv (4C => C, bias) → add
#     residual. No LayerScale (`ls_init_value=None`). DropPath omitted because
#     eval-mode forward is identity; revisit for training-mode parity.
#   - Optional `NormMlpClassifierHead`-style head (used by the
#     `fcmae_ft_in1k` variant): mean-pool over spatial dims → LayerNorm2d →
#     flatten → Dense(num_features => num_classes).
#
# Stem, downsample, head, stage scaffolding, and their mapping-entry
# builders live in `../ConvNeXtCommon/Common.jl`; this file owns only the
# v2-specific block (GRN, no LayerScale) and the top-level constructor and
# mapping function.
#
# `num_classes = 0` returns the post-stage4 feature map (shape
# `(W/32, H/32, dims[4], N)`), matching `timm.forward_features(x)`.
# `num_classes > 0` attaches the head and returns logits `(num_classes, N)`,
# matching `timm.forward(x)`.

include("Config.jl")

# -- Weight transforms -------------------------------------------------------

# Some ConvNeXtV2 FCMAE checkpoints on HuggingFace store the MLP fc1/fc2
# weights as 2D Linear tensors (out, in) rather than 4D Conv2d tensors
# (out, in, 1, 1). After load_safetensors_state_dict's axis reversal the 4D
# case lands as (1, 1, in, out) — exactly what Lux Conv expects — while the
# 2D case lands as (in, out) and must be reshaped. The atto FCMAE checkpoint
# has 4D weights; the huge FCMAE checkpoint has 2D weights. This transform
# handles both so a single mapping entry covers all variants.
function _cn2_fc_weight(w::AbstractArray)
    ndims(w) == 4 && return w          # (1, 1, in, out) — already correct
    ndims(w) == 2 || error("convnextv2 fc weight: expected 2D or 4D, got $(ndims(w))D")
    return reshape(w, 1, 1, size(w, 1), size(w, 2))
end

# -- Custom layers --------------------------------------------------------

function convnextv2_block(C::Int; mlp_ratio::Int = 4, kernel::Int = 7)
    H = mlp_ratio * C
    @compact(
        conv_dw = Conv(
            (kernel, kernel),
            C => C;
            groups = C,
            pad = kernel ÷ 2,
            use_bias = true,
            cross_correlation = true,
            init_weight = _CN_INIT,
            init_bias = zeros32,
        ),
        norm = layernorm2d(C),
        fc1 = Conv(
            (1, 1),
            C => H;
            use_bias = true,
            cross_correlation = true,
            init_weight = _CN_INIT,
            init_bias = zeros32,
        ),
        grn = grn_layer(H),
        fc2 = Conv(
            (1, 1),
            H => C;
            use_bias = true,
            cross_correlation = true,
            init_weight = _CN_INIT,
            init_bias = zeros32,
        ),
    ) do x
        y = conv_dw(x)
        y = norm(y)
        y = fc1(y)
        # timm's ConvNeXtV2 uses nn.GELU() with approximate='none' (the exact
        # erf-based GELU). NNlib.gelu is the tanh approximation; the two
        # diverge by ~1e-4 per call and accumulate visibly over the block depth.
        y = NNlib.gelu_erf.(y)
        y = grn(y)
        y = fc2(y)
        @return y .+ x
    end
end

# -- Top-level constructor -----------------------------------------------

# Shared backbone forward, called from every branch of `convnextv2`.
# Any future change to the backbone path (activation, downsample, stage
# composition, etc.) lives here so the feature-extractor, features-only,
# and classifier branches can't desync.
function _convnextv2_feature_taps(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4)
    x = stem_norm(stem_conv(x))
    f4 = stage1(x)
    f8 = stage2(f4)
    f16 = stage3(f8)
    f32 = stage4(f16)
    return (f4, f8, f16, f32)
end

_convnextv2_features(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4) =
    last(_convnextv2_feature_taps(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4))

"""
    convnextv2_feature_info(cfg) -> FeatureInfo

Ordered feature taps for a ConvNeXtV2, matching timm's `feature_info` for the
`convnextv2_*` family: one tap per stage, at reductions 4/8/16/32. Like v1
there is no reduction-2 tap, since the patch stem strides by 4 in a single
convolution.

The reduction-32 tap is the same tensor the `num_classes = 0` feature
extractor returns.
"""
convnextv2_feature_info(cfg::ConvNeXtV2Variant) =
    FeatureInfo((:stage1, :stage2, :stage3, :stage4), (4, 8, 16, 32), cfg.dims)

# Backbone half, shared by the `num_classes = 0` feature extractor
# (`out_sel = last`) and the features-only pyramid
# (`out_sel = feature_selector(indices)`). Both build the identical slot
# tree, so one pretrained mapping serves both.
function _convnextv2_backbone(cfg::ConvNeXtV2Variant, in_chans::Int, out_sel)
    depths = cfg.depths
    dims = cfg.dims
    strides = (1, 2, 2, 2)
    @compact(
        stem_conv = Conv(
            (4, 4),
            in_chans => dims[1];
            stride = 4,
            pad = 0,
            use_bias = true,
            cross_correlation = true,
            init_weight = _CN_INIT,
            init_bias = zeros32,
        ),
        stem_norm = layernorm2d(dims[1]),
        stage1 = convnext_stage(convnextv2_block, dims[1], dims[1], depths[1], strides[1]),
        stage2 = convnext_stage(convnextv2_block, dims[1], dims[2], depths[2], strides[2]),
        stage3 = convnext_stage(convnextv2_block, dims[2], dims[3], depths[3], strides[3]),
        stage4 = convnext_stage(convnextv2_block, dims[3], dims[4], depths[4], strides[4]),
    ) do x
        @return out_sel(
            _convnextv2_feature_taps(
                x,
                stem_conv,
                stem_norm,
                stage1,
                stage2,
                stage3,
                stage4,
            ),
        )
    end
end

"""
    convnextv2(variant; in_chans=3, num_classes=0,
               features_only=false, out_indices=nothing) -> @compact block

Build a ConvNeXtV2 backbone. `variant` is a key from [`CONVNEXTV2_VARIANTS`]
(e.g. `:convnextv2_atto_fcmae`).

When `num_classes == 0`, the forward pass returns the post-stage4 feature
map shaped `(W/32, H/32, dims[4], N)`, matching
`timm.create_model(..., num_classes=0).forward_features(x)`.

When `features_only == true`, the forward returns a tuple of per-stage
feature maps; see [`feature_info`](@ref) and [`create_model`](@ref).

When `num_classes > 0`, a `NormMlpClassifierHead`-style head is attached
(global mean pool → LayerNorm2d → flatten → Dense) and the forward returns
logits shaped `(num_classes, N)`, matching `timm.forward(x)`.
"""
function convnextv2(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Int = 0,
    features_only::Bool = false,
    out_indices = nothing,
)
    cfg = get(CONVNEXTV2_VARIANTS, variant) do
        error(
            "Unknown ConvNeXtV2 variant: $variant. Known variants: " *
            "$(sort(collect(keys(CONVNEXTV2_VARIANTS))))",
        )
    end
    depths = cfg.depths
    dims = cfg.dims
    strides = (1, 2, 2, 2)

    if features_only
        indices = resolve_out_indices(convnextv2_feature_info(cfg), out_indices, variant)
        return _convnextv2_backbone(cfg, in_chans, feature_selector(indices))
    end
    _check_out_indices_unused(variant, out_indices)

    if num_classes == 0
        _convnextv2_backbone(cfg, in_chans, last)
    else
        nc = num_classes
        @compact(
            stem_conv = Conv(
                (4, 4),
                in_chans => dims[1];
                stride = 4,
                pad = 0,
                use_bias = true,
                cross_correlation = true,
                init_weight = _CN_INIT,
                init_bias = zeros32,
            ),
            stem_norm = layernorm2d(dims[1]),
            stage1 =
                convnext_stage(convnextv2_block, dims[1], dims[1], depths[1], strides[1]),
            stage2 =
                convnext_stage(convnextv2_block, dims[1], dims[2], depths[2], strides[2]),
            stage3 =
                convnext_stage(convnextv2_block, dims[2], dims[3], depths[3], strides[3]),
            stage4 =
                convnext_stage(convnextv2_block, dims[3], dims[4], depths[4], strides[4]),
            head_norm = layernorm2d(dims[4]),
            head_fc = Dense(dims[4] => nc; init_weight = _CN_INIT, init_bias = zeros32),
        ) do x
            x = _convnextv2_features(
                x,
                stem_conv,
                stem_norm,
                stage1,
                stage2,
                stage3,
                stage4,
            )
            # NormMlpClassifierHead with pool='avg': mean pool over spatial,
            # then LN2d, flatten, Dense. NNlib has no adaptive pool; compute
            # the kernel from the actual spatial extent to stay size-agnostic.
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)   # (1, 1, C, N)
            x = head_norm(x)                                # (1, 1, C, N)
            x = reshape(x, (size(x, 3), size(x, 4)))        # (C, N)
            @return head_fc(x)                              # (nc, N)
        end
    end
end

# -- Pretrained-weight loading -------------------------------------------

"""
    convnextv2_mapping(state_dict, variant;
                       load_head_norm=false, load_classifier=false,
                       in_chans=3, prefix=()) -> Vector

Build the `(pytorch_key, lux_path, transform)` triples that move a timm
`convnextv2_<variant>` state_dict into the Lux tree produced by
[`convnextv2`](@ref). The two head pieces are independent:
`load_head_norm=true` adds the `head.norm.*` LayerNorm keys (whose dim
depends on the feature width, not `num_classes`), and
`load_classifier=true` adds the `head.fc.*` Dense keys (whose dim
depends on `num_classes`). Pass `prefix` to address a backbone nested
inside a larger `@compact` model. Pass `in_chans != 3` to adapt the
released 3-channel stem weight to the requested input channel count,
matching timm's behaviour.

Assumes the state dict was loaded with `load_safetensors_state_dict`
(default `reverse_axes=true`) or read from a parity HDF5 fixture: both
deliver conv weights in `(kW, kH, in, out)` order (Lux's `Conv` layout) so
`identity` is correct for them. The one exception is `head.fc.weight`, which
came from `nn.Linear` (2D): after axis-reverse from PyTorch's `(out, in)`
it's `(in, out)`, but Lux `Dense` stores weight as `(out, in)`, so we
apply `axis_reverse` to transpose it.
"""
function convnextv2_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_head_norm::Bool = false,
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(CONVNEXTV2_VARIANTS, variant) do
        error(
            "Unknown ConvNeXtV2 variant: $variant. Known variants: " *
            "$(sort(collect(keys(CONVNEXTV2_VARIANTS))))",
        )
    end
    mapping = _CN_MAPPING_ENTRY[]

    push_stem_mapping!(mapping, prefix, in_chans)

    strides = (1, 2, 2, 2)
    for (i, depth) in enumerate(cfg.depths)
        stride = strides[i]
        stage_sym = Symbol("stage", i)
        py_stage = "stages.$(i - 1)"

        if stride != 1
            push_downsample_mapping!(mapping, prefix, stage_sym, py_stage)
        end

        for j = 1:depth
            block_path = convnext_stage_block_path(i, stride, j)
            py_block = "$(py_stage).blocks.$(j - 1)"

            push!(
                mapping,
                (
                    "$(py_block).conv_dw.weight",
                    (prefix..., block_path..., :conv_dw, :weight),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).conv_dw.bias",
                    (prefix..., block_path..., :conv_dw, :bias),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).norm.weight",
                    (prefix..., block_path..., :norm, :scale),
                    as_channel4d,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).norm.bias",
                    (prefix..., block_path..., :norm, :bias),
                    as_channel4d,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).mlp.fc1.weight",
                    (prefix..., block_path..., :fc1, :weight),
                    _cn2_fc_weight,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).mlp.fc1.bias",
                    (prefix..., block_path..., :fc1, :bias),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).mlp.grn.weight",
                    (prefix..., block_path..., :grn, :scale),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).mlp.grn.bias",
                    (prefix..., block_path..., :grn, :bias),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).mlp.fc2.weight",
                    (prefix..., block_path..., :fc2, :weight),
                    _cn2_fc_weight,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).mlp.fc2.bias",
                    (prefix..., block_path..., :fc2, :bias),
                    identity,
                ),
            )
        end
    end

    if load_head_norm
        push_head_norm_mapping!(mapping, prefix)
    end
    if load_classifier
        push_head_fc_mapping!(mapping, prefix)
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    _load_convnextv2(ps, st, variant; in_chans, num_classes,
                     revision, cache_dir, prefix) -> (ps, st)

Resolve `model.safetensors` for `variant` on HuggingFace (cached on disk
under the standard HF Hub layout, so the same blob is shared with any
`timm` / `huggingface_hub` install on the machine), load it via
`load_safetensors_state_dict`, and rebuild `ps` with the ConvNeXtV2 weights
applied at `ps.<prefix...>`. `st` is returned unchanged (LayerNorm has no
running statistics); the uniform `(ps, st)` shape mirrors the other family
loaders.

`in_chans` and `num_classes` are forwarded from the closure captured
at `create_pretrained` time. The ConvNeXtV2 head bundles a LayerNorm
(`head_norm`) and a Dense classifier (`head_fc`); the LayerNorm is
loaded whenever the model has a head (`num_classes > 0`). The
classifier is loaded only when `num_classes` matches the variant's
`default_num_classes`. Three cases:

- `num_classes == 0`: backbone-only feature extractor.
- `num_classes == default_num_classes(variant)`: full load including
  classifier.
- `num_classes` differs from the variant's default: backbone +
  `head_norm` load, `@warn` is emitted, and the user's custom
  classifier is left at its `Lux.setup` random initialization.

When `in_chans != 3`, the stem weight is adapted from the released
3-channel checkpoint, matching timm's `adapt_input_conv` behaviour at
load time.
"""
function _load_convnextv2(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = CONVNEXTV2_VARIANTS[variant]
    load_head_norm = num_classes > 0
    load_classifier = num_classes > 0 && num_classes == cfg.default_num_classes
    if num_classes > 0 && num_classes != cfg.default_num_classes
        @warn "variant $variant ships $(cfg.default_num_classes)-class pretrained weights, " *
              "but the model has a $num_classes-class head. Loading the backbone " *
              "(and head_norm) only; the classifier is left at its Lux.setup random " *
              "initialization for you to train."
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
        convnextv2_mapping(
            sd,
            variant;
            load_head_norm = load_head_norm,
            load_classifier = load_classifier,
            in_chans = in_chans,
            prefix = prefix,
        ),
    )
    return ps, st
end
