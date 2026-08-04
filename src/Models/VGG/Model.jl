# VGG backbone, Lux port of timm's `vgg*` family.
#
# Architecture (matches timm `vgg.py`):
#   - `features`: a flat stack of 3x3 (pad 1) convolutions, each followed by
#     ReLU (and, for the `_bn` variants, a BatchNorm between conv and ReLU),
#     with 2x2 stride-2 max-pools at the stage boundaries. The conv widths
#     and pool positions come from the variant's `cfg` list.
#   - `pre_logits`: timm's `ConvMlp` — a 7x7 conv 512=>4096 + ReLU, then a
#     1x1 conv 4096=>4096 + ReLU. **Trap:** timm's VGG head is conv-based,
#     not the torchvision flatten+Linear, so `pre_logits.fc1/fc2` are convs,
#     and only `head.fc` is a Dense.
#   - `head`: global average pool over the `pre_logits` output, flatten, then
#     `head.fc` Dense 4096=>num_classes.
#
# `num_classes == 0` returns the post-`features` map (shape `(W/32, H/32,
# 512, N)`), matching `timm.forward_features(x)`. `num_classes > 0` attaches
# `pre_logits` + `head` and returns logits `(num_classes, N)`, matching
# `timm.forward(x)`.
#
# We build the conv stack as a Lux `Chain`, so its parameters are named
# `layer_1, layer_2, ...` by position (including the pooling layers, which
# occupy a slot with empty params). The weight mapping walks the same `cfg`
# to recover each conv/BN's PyTorch `features.<idx>` name alongside its Lux
# `layer_<idx>` slot, so the two cannot drift.

include("Config.jl")

const _VGG_CONV_INIT = kaiming_normal_fan_out
const _VGG_BN_EPS = 1.0f-5
const _VGG_BN_MOMENTUM = 0.1f0

# Walk the variant `cfg` once, producing one descriptor per Lux `Chain` slot.
# Each conv/BN descriptor records both its Lux `Chain` index (`lux_idx`,
# counting every layer including pools) and its PyTorch `features.<py_idx>`
# index, so the constructor and the weight mapping stay in lockstep.
#
# PyTorch slot accounting (timm builds `nn.Sequential(*layers)`):
#   - plain:  [Conv, ReLU]      per conv  (py advances by 2)
#   - bn:     [Conv, BN, ReLU]  per conv  (py advances by 3)
#   - pool:   [MaxPool]                   (py advances by 1)
function _vgg_feature_plan(cfg, batch_norm::Bool, in_chans::Int)
    plan = NamedTuple[]
    py = 0
    lux = 0
    inch = in_chans
    for v in cfg
        if v === :M
            lux += 1
            push!(plan, (; type = :pool, lux_idx = lux, py_idx = py))
            py += 1
        else
            outch = v::Int
            lux += 1
            push!(
                plan,
                (; type = :conv, lux_idx = lux, py_idx = py, in_ch = inch, out_ch = outch),
            )
            if batch_norm
                lux += 1
                push!(plan, (; type = :bn, lux_idx = lux, py_idx = py + 1, out_ch = outch))
                py += 3
            else
                py += 2
            end
            inch = outch
        end
    end
    return plan
end

# Build the `features` Chain from the plan. Plain-variant convs fold ReLU
# into the Conv activation; `_bn` variants leave the conv linear and fold
# ReLU into the following BatchNorm, matching timm's conv→bn→relu order.
function _vgg_build_features(plan, batch_norm::Bool)
    layers = []
    for d in plan
        if d.type === :conv
            act = batch_norm ? identity : NNlib.relu
            push!(
                layers,
                Conv(
                    (3, 3),
                    d.in_ch => d.out_ch,
                    act;
                    pad = 1,
                    use_bias = true,
                    cross_correlation = true,
                    init_weight = _VGG_CONV_INIT,
                    init_bias = zeros32,
                ),
            )
        elseif d.type === :bn
            push!(
                layers,
                BatchNorm(
                    d.out_ch,
                    NNlib.relu;
                    affine = true,
                    track_stats = true,
                    epsilon = _VGG_BN_EPS,
                    momentum = _VGG_BN_MOMENTUM,
                ),
            )
        else # :pool
            push!(layers, MaxPool((2, 2); stride = 2, pad = 0))
        end
    end
    return Chain(layers...)
end

_vgg_pre_fc1() = Conv(
    (7, 7),
    _VGG_FEATURE_CH => _VGG_MLP_CH,
    NNlib.relu;
    pad = 0,
    use_bias = true,
    cross_correlation = true,
    init_weight = _VGG_CONV_INIT,
    init_bias = zeros32,
)

_vgg_pre_fc2() = Conv(
    (1, 1),
    _VGG_MLP_CH => _VGG_MLP_CH,
    NNlib.relu;
    pad = 0,
    use_bias = true,
    cross_correlation = true,
    init_weight = _VGG_CONV_INIT,
    init_bias = zeros32,
)

# Shared backbone forward. The conv stack is a single Chain, so there is no
# multi-field desync risk, but routing both branches through one helper keeps
# the feature-extractor and classifier paths identical.
_vgg_features(x, features) = features(x)

"""
    vgg(variant; in_chans=3, num_classes=0) -> @compact block

Build a VGG backbone. `variant` is a key from [`VGG_VARIANTS`](@ref), e.g.
`:vgg16_tv_in1k`.

When `num_classes == 0`, the forward returns the post-`features` map shaped
`(W/32, H/32, 512, N)`, matching `timm.forward_features(x)`. When
`num_classes > 0`, the `pre_logits` ConvMlp and the `head.fc` classifier are
attached and the forward returns logits `(num_classes, N)`, matching
`timm.forward(x)`.
"""
function vgg(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Int = 0,
    features_only::Bool = false,
    out_indices = nothing,
)
    # No pyramid yet: the conv stack is one flat Chain, so tapping at the
    # pool boundaries means splitting it into per-stage sub-chains and
    # reworking `vgg_mapping` / `vgg_state_mapping` for the nested paths.
    _no_feature_pyramid("VGG", variant, features_only, out_indices)
    cfg = get(VGG_VARIANTS, variant) do
        error(
            "Unknown VGG variant: $variant. Known variants: " *
            "$(sort(collect(keys(VGG_VARIANTS))))",
        )
    end
    plan = _vgg_feature_plan(cfg.cfg, cfg.batch_norm, in_chans)

    if num_classes == 0
        @compact(features = _vgg_build_features(plan, cfg.batch_norm)) do x
            @return _vgg_features(x, features)
        end
    else
        nc = num_classes
        @compact(
            features = _vgg_build_features(plan, cfg.batch_norm),
            pre_fc1 = _vgg_pre_fc1(),
            pre_fc2 = _vgg_pre_fc2(),
            head_fc = Dense(_VGG_MLP_CH => nc; init_bias = zeros32),
        ) do x
            x = _vgg_features(x, features)
            x = pre_fc1(x)
            x = pre_fc2(x)
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)
            x = reshape(x, (size(x, 3), size(x, 4)))
            @return head_fc(x)
        end
    end
end

# -- Pretrained-weight loading -------------------------------------------

_VGG_MAPPING_ENTRY = Tuple{String,Tuple{Vararg{Symbol}},Function}

"""
    vgg_mapping(state_dict, variant; load_classifier=false, in_chans=3,
                prefix=()) -> Vector

Build the `(pytorch_key, lux_path, transform)` triples mapping a timm
`vgg<variant>` state_dict into the Lux tree from [`vgg`](@ref). Conv weights
and 1D vectors arrive in Lux-natural layout already, so feature/pre_logits
conv transforms are `identity`; the stem conv uses `adapt_input_conv` when
`in_chans != 3`. `head.fc` is a `nn.Linear`, so it needs `axis_reverse`.

BatchNorm running statistics (for `_bn` variants) are state, not params; use
[`vgg_state_mapping`](@ref) for those. When `load_classifier=true`, the
`pre_logits.*` and `head.fc.*` keys are included.
"""
function vgg_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(VGG_VARIANTS, variant) do
        error(
            "Unknown VGG variant: $variant. Known variants: " *
            "$(sort(collect(keys(VGG_VARIANTS))))",
        )
    end
    mapping = _VGG_MAPPING_ENTRY[]
    plan = _vgg_feature_plan(cfg.cfg, cfg.batch_norm, in_chans)

    first_conv = true
    for d in plan
        lux_layer = Symbol("layer_", d.lux_idx)
        if d.type === :conv
            w_transform =
                (first_conv && in_chans != 3) ? adapt_input_conv(in_chans) : identity
            first_conv = false
            push!(
                mapping,
                (
                    "features.$(d.py_idx).weight",
                    (prefix..., :features, lux_layer, :weight),
                    w_transform,
                ),
            )
            push!(
                mapping,
                (
                    "features.$(d.py_idx).bias",
                    (prefix..., :features, lux_layer, :bias),
                    identity,
                ),
            )
        elseif d.type === :bn
            push!(
                mapping,
                (
                    "features.$(d.py_idx).weight",
                    (prefix..., :features, lux_layer, :scale),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "features.$(d.py_idx).bias",
                    (prefix..., :features, lux_layer, :bias),
                    identity,
                ),
            )
        end
    end

    if load_classifier
        push!(
            mapping,
            ("pre_logits.fc1.weight", (prefix..., :pre_fc1, :weight), identity),
        )
        push!(mapping, ("pre_logits.fc1.bias", (prefix..., :pre_fc1, :bias), identity))
        push!(
            mapping,
            ("pre_logits.fc2.weight", (prefix..., :pre_fc2, :weight), identity),
        )
        push!(mapping, ("pre_logits.fc2.bias", (prefix..., :pre_fc2, :bias), identity))
        push!(mapping, ("head.fc.weight", (prefix..., :head_fc, :weight), axis_reverse))
        push!(mapping, ("head.fc.bias", (prefix..., :head_fc, :bias), identity))
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    vgg_state_mapping(state_dict, variant; prefix=()) -> Vector

Build the BatchNorm running-statistics state mapping for a `_bn` VGG variant.
Returns an empty mapping for the plain (non-BatchNorm) variants.
"""
function vgg_state_mapping(
    state_dict::Dict,
    variant::Symbol;
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(VGG_VARIANTS, variant) do
        error(
            "Unknown VGG variant: $variant. Known variants: " *
            "$(sort(collect(keys(VGG_VARIANTS))))",
        )
    end
    mapping = _VGG_MAPPING_ENTRY[]
    cfg.batch_norm || return mapping
    plan = _vgg_feature_plan(cfg.cfg, cfg.batch_norm, 3)
    for d in plan
        d.type === :bn || continue
        lux_layer = Symbol("layer_", d.lux_idx)
        push!(
            mapping,
            (
                "features.$(d.py_idx).running_mean",
                (prefix..., :features, lux_layer, :running_mean),
                identity,
            ),
        )
        push!(
            mapping,
            (
                "features.$(d.py_idx).running_var",
                (prefix..., :features, lux_layer, :running_var),
                identity,
            ),
        )
    end
    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("state mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    _load_vgg(ps, st, variant; in_chans, num_classes, revision, cache_dir,
              prefix) -> (ps, st)

Private back-end for `create_pretrained` on VGG variants. Resolves and loads
the timm `.safetensors` from HuggingFace, applies the conv/Dense params into
`ps`, and (for `_bn` variants) the BatchNorm running statistics into `st`.

Three classifier-head cases mirror the other families:
- `num_classes == 0`: backbone-only feature extractor.
- `num_classes == default_num_classes(variant)`: full load including
  `pre_logits` and the classifier.
- `num_classes` differs from the variant's default: backbone loads, a
  `@warn` is emitted, and the user's custom head (`pre_logits` + `head_fc`)
  is left at its `Lux.setup` random initialization.
"""
function _load_vgg(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = VGG_VARIANTS[variant]
    load_classifier = num_classes > 0 && num_classes == cfg.default_num_classes
    if num_classes > 0 && num_classes != cfg.default_num_classes
        @warn "variant $variant ships $(cfg.default_num_classes)-class pretrained weights, " *
              "but the model has a $num_classes-class head. Loading the backbone only; " *
              "the pre_logits + classifier head are left at their Lux.setup random " *
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
        vgg_mapping(
            sd,
            variant;
            load_classifier = load_classifier,
            in_chans = in_chans,
            prefix = prefix,
        ),
    )
    st = apply_resnet_state_dict(st, sd, vgg_state_mapping(sd, variant; prefix = prefix))
    return ps, st
end
