# ConvNeXt v1 backbone, Lux port of timm's `convnext_<variant>` family.
#
# Architecture (matches timm with `use_grn=False`, `conv_mlp=False`,
# `ls_init_value=1e-6`, `stem_type='patch'`, `patch_size=4`,
# `kernel_sizes=7`):
#   - Stem: Conv((4,4), in_chans => dims[1]; stride=4, bias) followed by
#     LayerNorm2d(dims[1]). Same as ConvNeXtV2.
#   - Four stages with depths from the variant. Stage 0 has no downsample
#     (the stem already strode by 4); stages 1-3 begin with a downsample
#     block of LayerNorm2d(in) + Conv((2,2); stride=2, bias). Same as v2.
#   - Each block: depthwise Conv((7,7); groups=C, pad=3, bias) → LayerNorm2d →
#     1x1 Conv (C => 4C, bias) → GELU(erf) → 1x1 Conv (4C => C, bias) →
#     × LayerScale gamma → add residual. **No GRN** (the v2 addition).
#     **LayerScale gamma** is a `(C,)` parameter initialised to `1e-6`.
#     DropPath omitted because eval-mode forward is identity; revisit for
#     training-mode parity.
#   - Optional `NormMlpClassifierHead`-style head (same as v2). The four
#     DINOv3 variants ship `num_classes = 0`, so the head is not exercised
#     by the current test suite; the code path is kept for symmetry with
#     v2 and so that non-DINO v1 checkpoints can be added later.
#
# timm v1 with `conv_mlp=False` stores the MLP as two `nn.Linear` layers in
# channels-last space, but that is mathematically identical to two 1x1 convs
# in NCHW. We build the Julia model with 1x1 `Conv` layers (the v2 path) and
# reshape the 2D Linear weights to 4D at load time via `_linear_to_conv1x1`.
# This keeps the forward pass autodiff- and GPU-friendly, avoids explicit
# permutes inside `@compact`, and lets us share the stem / downsample / stage
# helpers with v2 through `../ConvNeXtCommon/Common.jl`.
#
# `num_classes = 0` returns the post-stage4 feature map (shape
# `(W/32, H/32, dims[4], N)`), matching `timm.forward_features(x)`.
# `num_classes > 0` attaches the head and returns logits `(num_classes, N)`,
# matching `timm.forward(x)`.

include("Config.jl")

# -- Custom layers --------------------------------------------------------

# timm v1 block. Identical to convnextv2_block except: (a) no GRN between
# act and fc2, and (b) a learnable per-channel LayerScale gamma applied
# right before the residual sum. gamma is stored as a 1D (C,) parameter,
# matching the PyTorch state-dict shape so the mapping transform is
# `identity`.
function convnext_block(
    C::Int;
    mlp_ratio::Int = 4,
    kernel::Int = 7,
    ls_init::Float32 = _CN_V1_LS_INIT,
)
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
        fc2 = Conv(
            (1, 1),
            H => C;
            use_bias = true,
            cross_correlation = true,
            init_weight = _CN_INIT,
            init_bias = zeros32,
        ),
        gamma = fill(ls_init, C),
    ) do x
        y = conv_dw(x)
        y = norm(y)
        y = fc1(y)
        # timm v1 uses nn.GELU() with approximate='none' (the exact erf-based
        # GELU). NNlib.gelu is the tanh approximation; the two diverge by
        # ~1e-4 per call and accumulate visibly over the block depth.
        y = NNlib.gelu_erf.(y)
        y = fc2(y)
        y = y .* reshape(gamma, 1, 1, :, 1)
        @return y .+ x
    end
end

# -- Top-level constructor -----------------------------------------------

# Build a thin closure over the variant's `ls_init` so `convnext_stage`'s
# block_ctor signature stays `C -> @compact`. Threading ls_init through every
# call site instead of a closure would require parameterizing convnext_stage
# itself, which would leak v1-specific configuration into shared code.
_convnext_block_for(ls_init::Float32) = C -> convnext_block(C; ls_init = ls_init)

# Shared backbone forward, called from every branch of `convnext`. Any
# future change to the backbone path (activation, downsample, stage
# composition, etc.) lives here so the feature-extractor, features-only,
# and classifier branches can't desync.
function _convnext_feature_taps(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4)
    x = stem_norm(stem_conv(x))
    f4 = stage1(x)
    f8 = stage2(f4)
    f16 = stage3(f8)
    f32 = stage4(f16)
    return (f4, f8, f16, f32)
end

_convnext_features(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4) =
    last(_convnext_feature_taps(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4))

"""
    convnext_feature_info(cfg) -> FeatureInfo

Ordered feature taps for a ConvNeXt v1, matching timm's `feature_info` for
the `convnext_*` family: one tap per stage, at reductions 4/8/16/32. There is
no reduction-2 tap, because the patch stem strides by 4 in a single
convolution; a decoder that needs half resolution must upsample past the
finest tap.

The reduction-32 tap is the same tensor the `num_classes = 0` feature
extractor returns: timm's `norm_pre` is `Identity` for every registered
variant, so nothing sits between the last stage and the head.
"""
convnext_feature_info(cfg::ConvNeXtVariant) =
    FeatureInfo((:stage1, :stage2, :stage3, :stage4), (4, 8, 16, 32), cfg.dims)

# Backbone half, shared by the `num_classes = 0` feature extractor
# (`out_sel = last`) and the features-only pyramid
# (`out_sel = feature_selector(indices)`). Both build the identical slot
# tree, so one pretrained mapping serves both.
function _convnext_backbone(cfg::ConvNeXtVariant, in_chans::Int, out_sel)
    depths = cfg.depths
    dims = cfg.dims
    strides = (1, 2, 2, 2)
    block_ctor = _convnext_block_for(cfg.ls_init)
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
        stage1 = convnext_stage(block_ctor, dims[1], dims[1], depths[1], strides[1]),
        stage2 = convnext_stage(block_ctor, dims[1], dims[2], depths[2], strides[2]),
        stage3 = convnext_stage(block_ctor, dims[2], dims[3], depths[3], strides[3]),
        stage4 = convnext_stage(block_ctor, dims[3], dims[4], depths[4], strides[4]),
    ) do x
        @return out_sel(
            _convnext_feature_taps(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4),
        )
    end
end

"""
    convnext(variant; in_chans=3, num_classes=0,
             features_only=false, out_indices=nothing) -> @compact block

Build a ConvNeXt v1 backbone. `variant` is a key from [`CONVNEXT_VARIANTS`]
(e.g. `:convnext_tiny_dinov3_lvd1689m`).

When `num_classes == 0`, the forward pass returns the post-stage4 feature
map shaped `(W/32, H/32, dims[4], N)`, matching
`timm.create_model(..., num_classes=0).forward_features(x)`.

When `features_only == true`, the forward returns a tuple of per-stage
feature maps; see [`feature_info`](@ref) for the tap table and
[`create_model`](@ref) for the shared semantics.

When `num_classes > 0`, a `NormMlpClassifierHead`-style head is attached
(global mean pool → LayerNorm2d → flatten → Dense) and the forward returns
logits shaped `(num_classes, N)`, matching `timm.forward(x)`. None of the
DINOv3 variants currently registered ship a usable head, so this branch
is exercised only when extending the variant table with future
checkpoints.
"""
function convnext(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Int = 0,
    features_only::Bool = false,
    out_indices = nothing,
)
    cfg = get(CONVNEXT_VARIANTS, variant) do
        error(
            "Unknown ConvNeXt variant: $variant. Known variants: " *
            "$(sort(collect(keys(CONVNEXT_VARIANTS))))",
        )
    end
    depths = cfg.depths
    dims = cfg.dims
    strides = (1, 2, 2, 2)
    block_ctor = _convnext_block_for(cfg.ls_init)

    if features_only
        indices = resolve_out_indices(convnext_feature_info(cfg), out_indices, variant)
        return _convnext_backbone(cfg, in_chans, feature_selector(indices))
    end
    _check_out_indices_unused(variant, out_indices)

    if num_classes == 0
        _convnext_backbone(cfg, in_chans, last)
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
            stage1 = convnext_stage(block_ctor, dims[1], dims[1], depths[1], strides[1]),
            stage2 = convnext_stage(block_ctor, dims[1], dims[2], depths[2], strides[2]),
            stage3 = convnext_stage(block_ctor, dims[2], dims[3], depths[3], strides[3]),
            stage4 = convnext_stage(block_ctor, dims[3], dims[4], depths[4], strides[4]),
            head_norm = layernorm2d(dims[4]),
            head_fc = Dense(dims[4] => nc; init_weight = _CN_INIT, init_bias = zeros32),
        ) do x
            x = _convnext_features(x, stem_conv, stem_norm, stage1, stage2, stage3, stage4)
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)
            x = head_norm(x)
            x = reshape(x, (size(x, 3), size(x, 4)))
            @return head_fc(x)
        end
    end
end

# -- Pretrained-weight loading -------------------------------------------

# v1's MLP in timm is two `nn.Linear` layers (2D weight `(out, in)`); after
# axis-reverse from PyTorch storage that becomes `(in, out)`. We build the
# MLP with Lux `Conv((1,1), in => out)` whose weight is `(1, 1, in, out)`,
# so the transform reshapes the 2D weight into 4D with two leading 1-axes.
# Mathematically equivalent to timm's conv_mlp=False path (a 1x1 conv is a
# Linear acting on flattened (W*H, C) tokens), but lets us reuse the v2
# block scaffolding and avoid permutes in the forward.
_linear_to_conv1x1(w) = reshape(w, 1, 1, size(w, 1), size(w, 2))

"""
    convnext_mapping(state_dict, variant;
                     load_head_norm=false, load_classifier=false,
                     in_chans=3, prefix=()) -> Vector

Build the `(pytorch_key, lux_path, transform)` triples that move a timm
`convnext_<variant>` (v1) state_dict into the Lux tree produced by
[`convnext`](@ref). The two head pieces are independent:
`load_head_norm=true` adds the `head.norm.*` LayerNorm keys (whose dim
depends on the feature width, not `num_classes`), and
`load_classifier=true` adds the `head.fc.*` Dense keys (whose dim
depends on `num_classes`). Pass `prefix` to address a backbone nested
inside a larger `@compact` model. Pass `in_chans != 3` to adapt the
released 3-channel stem weight to the requested input channel count,
matching timm's behaviour.

Conv weights and 1D vectors arrive in their Lux-natural layout already
(via `load_safetensors_state_dict`'s default axis reversal, or `read_parity`
for fixture-driven tests), so most transforms are `identity`. The two
exceptions are: (a) `mlp.fc1.weight` and `mlp.fc2.weight`, which came from
`nn.Linear` (2D `(out, in)` → `(in, out)` after axis reverse), and which
we reshape to `(1, 1, in, out)` to land in a Lux `Conv((1,1))`; (b)
`head.fc.weight`, which is a Lux `Dense` so it needs `axis_reverse` to
transpose back to `(out, in)`.
"""
function convnext_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_head_norm::Bool = false,
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(CONVNEXT_VARIANTS, variant) do
        error(
            "Unknown ConvNeXt variant: $variant. Known variants: " *
            "$(sort(collect(keys(CONVNEXT_VARIANTS))))",
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
                    _linear_to_conv1x1,
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
                    "$(py_block).mlp.fc2.weight",
                    (prefix..., block_path..., :fc2, :weight),
                    _linear_to_conv1x1,
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
            push!(
                mapping,
                ("$(py_block).gamma", (prefix..., block_path..., :gamma), identity),
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
    _load_convnext(ps, st, variant; in_chans, num_classes,
                   revision, cache_dir, prefix) -> (ps, st)

Resolve `model.safetensors` for `variant` on HuggingFace (cached on disk
under the standard HF Hub layout, so the same blob is shared with any
`timm` / `huggingface_hub` install on the machine), load it via
`load_safetensors_state_dict`, and rebuild `ps` with the ConvNeXt v1
weights applied at `ps.<prefix...>`. `st` is returned unchanged
(LayerNorm has no running statistics); the uniform `(ps, st)` shape
mirrors the other family loaders.

`in_chans` and `num_classes` are forwarded from the closure captured
at `create_pretrained` time. The ConvNeXt head bundles a LayerNorm
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

The four DINOv3 variants registered in [`CONVNEXT_VARIANTS`](@ref)
all have `default_num_classes = 0` and ship no pretrained head; any
custom classifier the user adds will trigger the warning case.

When `in_chans != 3`, the stem weight is adapted from the released
3-channel checkpoint, matching timm's `adapt_input_conv` behaviour at
load time.
"""
function _load_convnext(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = CONVNEXT_VARIANTS[variant]
    # `head_norm` exists in the model iff `num_classes > 0` (the
    # constructor's `else` branch). The classifier rule is the same
    # three-way split as the other families.
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
        convnext_mapping(
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
