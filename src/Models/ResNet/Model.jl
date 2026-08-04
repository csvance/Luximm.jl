# Classic ResNet backbone, Lux port of timm's `resnet*.a1_in1k` family.
#
# `num_classes=0` returns the spatial feature map from `forward_features`.
# `num_classes>0` attaches timm's global average pool + Linear classifier and
# returns logits shaped `(num_classes, N)`.

include("Config.jl")

const _RESNET_CONV_INIT = kaiming_normal_fan_out
const _RESNET_BN_EPS = 1.0f-5
const _RESNET_BN_MOMENTUM = 0.1f0

function _resnet_bn(ch::Int; zero_scale::Bool = false)
    if zero_scale
        return BatchNorm(
            ch;
            affine = true,
            track_stats = true,
            epsilon = _RESNET_BN_EPS,
            momentum = _RESNET_BN_MOMENTUM,
            init_scale = zeros32,
        )
    else
        return BatchNorm(
            ch;
            affine = true,
            track_stats = true,
            epsilon = _RESNET_BN_EPS,
            momentum = _RESNET_BN_MOMENTUM,
        )
    end
end

_resnet_expansion(block::Symbol) =
    block == :basic ? 1 :
    block == :bottleneck ? 4 : error("unknown ResNet block type: $block")

function resnet_basic_block(in_ch::Int, out_ch::Int, stride::Int; downsample::Bool)
    if downsample
        @compact(
            conv1 = Conv(
                (3, 3),
                in_ch => out_ch;
                stride = stride,
                pad = 1,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn1 = _resnet_bn(out_ch),
            conv2 = Conv(
                (3, 3),
                out_ch => out_ch;
                stride = 1,
                pad = 1,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn2 = _resnet_bn(out_ch; zero_scale = true),
            downsample_conv = Conv(
                (1, 1),
                in_ch => out_ch;
                stride = stride,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            downsample_bn = _resnet_bn(out_ch),
        ) do x
            shortcut = downsample_bn(downsample_conv(x))
            y = conv1(x)
            y = NNlib.relu.(bn1(y))
            y = conv2(y)
            y = bn2(y)
            @return NNlib.relu.(y .+ shortcut)
        end
    else
        @compact(
            conv1 = Conv(
                (3, 3),
                in_ch => out_ch;
                stride = stride,
                pad = 1,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn1 = _resnet_bn(out_ch),
            conv2 = Conv(
                (3, 3),
                out_ch => out_ch;
                stride = 1,
                pad = 1,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn2 = _resnet_bn(out_ch; zero_scale = true),
        ) do x
            y = conv1(x)
            y = NNlib.relu.(bn1(y))
            y = conv2(y)
            y = bn2(y)
            @return NNlib.relu.(y .+ x)
        end
    end
end

function resnet_bottleneck_block(in_ch::Int, plane_ch::Int, stride::Int; downsample::Bool)
    out_ch = 4 * plane_ch
    if downsample
        @compact(
            conv1 = Conv(
                (1, 1),
                in_ch => plane_ch;
                stride = 1,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn1 = _resnet_bn(plane_ch),
            conv2 = Conv(
                (3, 3),
                plane_ch => plane_ch;
                stride = stride,
                pad = 1,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn2 = _resnet_bn(plane_ch),
            conv3 = Conv(
                (1, 1),
                plane_ch => out_ch;
                stride = 1,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn3 = _resnet_bn(out_ch; zero_scale = true),
            downsample_conv = Conv(
                (1, 1),
                in_ch => out_ch;
                stride = stride,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            downsample_bn = _resnet_bn(out_ch),
        ) do x
            shortcut = downsample_bn(downsample_conv(x))
            y = conv1(x)
            y = NNlib.relu.(bn1(y))
            y = conv2(y)
            y = NNlib.relu.(bn2(y))
            y = conv3(y)
            y = bn3(y)
            @return NNlib.relu.(y .+ shortcut)
        end
    else
        @compact(
            conv1 = Conv(
                (1, 1),
                in_ch => plane_ch;
                stride = 1,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn1 = _resnet_bn(plane_ch),
            conv2 = Conv(
                (3, 3),
                plane_ch => plane_ch;
                stride = stride,
                pad = 1,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn2 = _resnet_bn(plane_ch),
            conv3 = Conv(
                (1, 1),
                plane_ch => 4 * plane_ch;
                stride = 1,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn3 = _resnet_bn(4 * plane_ch; zero_scale = true),
        ) do x
            y = conv1(x)
            y = NNlib.relu.(bn1(y))
            y = conv2(y)
            y = NNlib.relu.(bn2(y))
            y = conv3(y)
            y = bn3(y)
            @return NNlib.relu.(y .+ x)
        end
    end
end

function classic_resnet_stage(
    block::Symbol,
    in_ch::Int,
    plane_ch::Int,
    depth::Int,
    stride::Int,
)
    blocks = []
    expansion = _resnet_expansion(block)
    out_ch = expansion * plane_ch
    first_downsample = stride != 1 || in_ch != out_ch
    if block == :basic
        push!(
            blocks,
            resnet_basic_block(in_ch, out_ch, stride; downsample = first_downsample),
        )
    elseif block == :bottleneck
        push!(
            blocks,
            resnet_bottleneck_block(in_ch, plane_ch, stride; downsample = first_downsample),
        )
    else
        error("unknown ResNet block type: $block")
    end
    for _ = 2:depth
        if block == :basic
            push!(blocks, resnet_basic_block(out_ch, out_ch, 1; downsample = false))
        else
            push!(blocks, resnet_bottleneck_block(out_ch, plane_ch, 1; downsample = false))
        end
    end
    return Chain(blocks...)
end

# Every intermediate map a features-only ResNet can hand back, ordered by
# increasing reduction. `f2` is timm's `act1` tap: post-BN-ReLU, *before* the
# stem maxpool. The three constructor branches all route through here, so the
# feature-extractor, features-only, and classifier paths cannot desync.
function _resnet_feature_taps(x, conv1, bn1, layer1, layer2, layer3, layer4)
    f2 = NNlib.relu.(bn1(conv1(x)))
    x = NNlib.maxpool(f2, (3, 3); stride = 2, pad = 1)
    f4 = layer1(x)
    f8 = layer2(f4)
    f16 = layer3(f8)
    f32 = layer4(f16)
    return (f2, f4, f8, f16, f32)
end

_resnet_features(x, conv1, bn1, layer1, layer2, layer3, layer4) =
    last(_resnet_feature_taps(x, conv1, bn1, layer1, layer2, layer3, layer4))

"""
    resnet_feature_info(cfg) -> FeatureInfo

Ordered feature taps for a classic ResNet, matching timm's `feature_info`
for the `resnet*` family: the post-`act1` stem map at reduction 2, then one
tap per stage. The stem tap is always 64 channels; stage widths carry the
block expansion (1 for `:basic`, 4 for `:bottleneck`).
"""
function resnet_feature_info(cfg::ResNetVariant)
    expansion = _resnet_expansion(cfg.block)
    return FeatureInfo(
        (:act1, :layer1, :layer2, :layer3, :layer4),
        (2, 4, 8, 16, 32),
        (64, ntuple(i -> expansion * cfg.planes[i], 4)...),
    )
end

# The backbone half of a ResNet, shared by the `num_classes = 0` feature
# extractor (`out_sel = last`) and the features-only pyramid
# (`out_sel = feature_selector(indices)`). Both build the identical slot
# tree, so one pretrained mapping serves both.
function _resnet_backbone(cfg::ResNetVariant, in_chans::Int, out_sel)
    depths = cfg.layers
    planes = cfg.planes
    expansion = _resnet_expansion(cfg.block)
    stage_chs = ntuple(i -> expansion * planes[i], 4)
    @compact(
        conv1 = Conv(
            (7, 7),
            in_chans => 64;
            stride = 2,
            pad = 3,
            use_bias = false,
            cross_correlation = true,
            init_weight = _RESNET_CONV_INIT,
        ),
        bn1 = _resnet_bn(64),
        layer1 = classic_resnet_stage(cfg.block, 64, planes[1], depths[1], 1),
        layer2 = classic_resnet_stage(cfg.block, stage_chs[1], planes[2], depths[2], 2),
        layer3 = classic_resnet_stage(cfg.block, stage_chs[2], planes[3], depths[3], 2),
        layer4 = classic_resnet_stage(cfg.block, stage_chs[3], planes[4], depths[4], 2),
    ) do x
        @return out_sel(_resnet_feature_taps(x, conv1, bn1, layer1, layer2, layer3, layer4))
    end
end

"""
    resnet(variant; in_chans=3, num_classes=0,
           features_only=false, out_indices=nothing) -> @compact block

Build a classic timm ResNet. `variant` is a key from [`RESNET_VARIANTS`](@ref),
for example `:resnet18_a1_in1k` or `:resnet50_a1_in1k`.

With `features_only = true` the forward returns a tuple of intermediate
feature maps instead of a single one; see [`feature_info`](@ref) for the tap
table and [`create_model`](@ref) for the shared semantics.
"""
function resnet(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Int = 0,
    features_only::Bool = false,
    out_indices = nothing,
)
    cfg = get(RESNET_VARIANTS, variant) do
        error(
            "Unknown ResNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(RESNET_VARIANTS))))",
        )
    end
    depths = cfg.layers
    planes = cfg.planes
    expansion = _resnet_expansion(cfg.block)
    stage_chs = ntuple(i -> expansion * planes[i], 4)

    if features_only
        indices = resolve_out_indices(resnet_feature_info(cfg), out_indices, variant)
        return _resnet_backbone(cfg, in_chans, feature_selector(indices))
    end
    _check_out_indices_unused(variant, out_indices)

    if num_classes == 0
        _resnet_backbone(cfg, in_chans, last)
    else
        nc = num_classes
        @compact(
            conv1 = Conv(
                (7, 7),
                in_chans => 64;
                stride = 2,
                pad = 3,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn1 = _resnet_bn(64),
            layer1 = classic_resnet_stage(cfg.block, 64, planes[1], depths[1], 1),
            layer2 = classic_resnet_stage(cfg.block, stage_chs[1], planes[2], depths[2], 2),
            layer3 = classic_resnet_stage(cfg.block, stage_chs[2], planes[3], depths[3], 2),
            layer4 = classic_resnet_stage(cfg.block, stage_chs[3], planes[4], depths[4], 2),
            fc = Dense(stage_chs[4] => nc; init_bias = zeros32),
        ) do x
            x = _resnet_features(x, conv1, bn1, layer1, layer2, layer3, layer4)
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)
            x = reshape(x, (size(x, 3), size(x, 4)))
            @return fc(x)
        end
    end
end

_resnet_block_has_downsample(cfg::ResNetVariant, stage::Int, block::Int) =
    block == 1 && (stage > 1 || cfg.block == :bottleneck)

_resnet_last_bn_name(block::Symbol) =
    block == :basic ? :bn2 :
    block == :bottleneck ? :bn3 : error("unknown ResNet block type: $block")

_RESNET_MAPPING_ENTRY = Tuple{String,Tuple{Vararg{Symbol}},Function}

function _push_resnet_bn_param_mapping!(
    mapping,
    py_prefix::String,
    lux_path::Tuple{Vararg{Symbol}},
)
    push!(mapping, ("$(py_prefix).weight", (lux_path..., :scale), identity))
    push!(mapping, ("$(py_prefix).bias", (lux_path..., :bias), identity))
    return mapping
end

function _push_resnet_bn_state_mapping!(
    mapping,
    py_prefix::String,
    lux_path::Tuple{Vararg{Symbol}},
)
    push!(mapping, ("$(py_prefix).running_mean", (lux_path..., :running_mean), identity))
    push!(mapping, ("$(py_prefix).running_var", (lux_path..., :running_var), identity))
    return mapping
end

function _resnet_block_path(stage::Int, block::Int)
    return (Symbol("layer", stage), Symbol("layer_", block))
end

"""
    resnet_mapping(state_dict, variant;
                    load_classifier=false, in_chans=3, prefix=())

Build the parameter mapping for a classic timm ResNet state dict.
BatchNorm running statistics are state, not params; use
[`resnet_state_mapping`](@ref) for those. When `load_classifier=true`,
the `fc.*` keys are also included.
"""
function resnet_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(RESNET_VARIANTS, variant) do
        error(
            "Unknown ResNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(RESNET_VARIANTS))))",
        )
    end
    mapping = _RESNET_MAPPING_ENTRY[]

    stem_w_transform = in_chans == 3 ? identity : adapt_input_conv(in_chans)
    push!(mapping, ("conv1.weight", (prefix..., :conv1, :weight), stem_w_transform))
    _push_resnet_bn_param_mapping!(mapping, "bn1", (prefix..., :bn1))

    for (stage, depth) in enumerate(cfg.layers)
        for block = 1:depth
            block_path = _resnet_block_path(stage, block)
            py_block = "layer$(stage).$(block - 1)"
            push!(
                mapping,
                (
                    "$(py_block).conv1.weight",
                    (prefix..., block_path..., :conv1, :weight),
                    identity,
                ),
            )
            _push_resnet_bn_param_mapping!(
                mapping,
                "$(py_block).bn1",
                (prefix..., block_path..., :bn1),
            )
            push!(
                mapping,
                (
                    "$(py_block).conv2.weight",
                    (prefix..., block_path..., :conv2, :weight),
                    identity,
                ),
            )
            _push_resnet_bn_param_mapping!(
                mapping,
                "$(py_block).bn2",
                (prefix..., block_path..., :bn2),
            )
            if cfg.block == :bottleneck
                push!(
                    mapping,
                    (
                        "$(py_block).conv3.weight",
                        (prefix..., block_path..., :conv3, :weight),
                        identity,
                    ),
                )
                _push_resnet_bn_param_mapping!(
                    mapping,
                    "$(py_block).bn3",
                    (prefix..., block_path..., :bn3),
                )
            end
            if _resnet_block_has_downsample(cfg, stage, block)
                push!(
                    mapping,
                    (
                        "$(py_block).downsample.0.weight",
                        (prefix..., block_path..., :downsample_conv, :weight),
                        identity,
                    ),
                )
                _push_resnet_bn_param_mapping!(
                    mapping,
                    "$(py_block).downsample.1",
                    (prefix..., block_path..., :downsample_bn),
                )
            end
        end
    end

    if load_classifier
        push!(mapping, ("fc.weight", (prefix..., :fc, :weight), axis_reverse))
        push!(mapping, ("fc.bias", (prefix..., :fc, :bias), identity))
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    resnet_state_mapping(state_dict, variant; prefix=())

Build the state mapping for BatchNorm running statistics in a classic timm
ResNet state dict.
"""
function resnet_state_mapping(
    state_dict::Dict,
    variant::Symbol;
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(RESNET_VARIANTS, variant) do
        error(
            "Unknown ResNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(RESNET_VARIANTS))))",
        )
    end
    mapping = _RESNET_MAPPING_ENTRY[]

    _push_resnet_bn_state_mapping!(mapping, "bn1", (prefix..., :bn1))
    for (stage, depth) in enumerate(cfg.layers)
        for block = 1:depth
            block_path = _resnet_block_path(stage, block)
            py_block = "layer$(stage).$(block - 1)"
            _push_resnet_bn_state_mapping!(
                mapping,
                "$(py_block).bn1",
                (prefix..., block_path..., :bn1),
            )
            _push_resnet_bn_state_mapping!(
                mapping,
                "$(py_block).bn2",
                (prefix..., block_path..., :bn2),
            )
            if cfg.block == :bottleneck
                _push_resnet_bn_state_mapping!(
                    mapping,
                    "$(py_block).bn3",
                    (prefix..., block_path..., :bn3),
                )
            end
            if _resnet_block_has_downsample(cfg, stage, block)
                _push_resnet_bn_state_mapping!(
                    mapping,
                    "$(py_block).downsample.1",
                    (prefix..., block_path..., :downsample_bn),
                )
            end
        end
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("state mapping references missing state_dict key: $pykey")
    end
    return mapping
end

function _set_resnet_state_leaf(nt::NamedTuple, path::NTuple{N,Symbol}, leaf) where {N}
    head = path[1]
    haskey(nt, head) || error("state path missing key: $head (have: $(propertynames(nt)))")
    if N == 1
        return merge(nt, (; head => leaf))
    else
        sub = _set_resnet_state_leaf(getfield(nt, head), Base.tail(path), leaf)
        return merge(nt, (; head => sub))
    end
end

function apply_resnet_state_dict(st, state_dict::Dict{String,<:AbstractArray}, mapping)
    out = st
    for (pykey, lux_path, transform) in mapping
        haskey(state_dict, pykey) || error("missing PyTorch state_dict key: $pykey")
        leaf = transform(state_dict[pykey])
        out = _set_resnet_state_leaf(out, lux_path, leaf)
    end
    return out
end

function _validate_resnet_consumed_keys(
    state_dict::Dict,
    param_mapping,
    state_mapping;
    load_classifier::Bool = false,
)
    consumed = Set(first(m) for m in param_mapping)
    union!(consumed, Set(first(m) for m in state_mapping))
    ignored = Set(k for k in keys(state_dict) if endswith(k, ".num_batches_tracked"))
    if !load_classifier
        push!(ignored, "fc.weight")
        push!(ignored, "fc.bias")
    end
    extras = setdiff(Set(keys(state_dict)), union(consumed, ignored))
    isempty(extras) || error("unmapped ResNet state_dict keys: $(sort(collect(extras)))")
    return nothing
end

"""
    _load_resnet(ps, st, variant; in_chans, num_classes, revision,
                 cache_dir, prefix) -> (ps, st)

Private back-end for `create_pretrained` on ResNet variants. Resolves
and loads a timm ResNet `.safetensors` from HuggingFace; returns both
params and state because BatchNorm running statistics live in Lux
state. `in_chans` and `num_classes` are forwarded from the closure
captured at `create_pretrained` time; this loader no longer
introspects `ps`. Three classifier-head cases (matching the three
constructor paths):

- `num_classes == 0`: backbone-only load.
- `num_classes == default_num_classes(variant)`: full load.
- `num_classes` differs from the variant's default: backbone loads,
  `@warn` is emitted, and the user's custom classifier is left at its
  `Lux.setup` random initialization.
"""
function _load_resnet(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = RESNET_VARIANTS[variant]
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
    param_mapping = resnet_mapping(
        sd,
        variant;
        load_classifier = load_classifier,
        in_chans = in_chans,
        prefix = prefix,
    )
    state_mapping = resnet_state_mapping(sd, variant; prefix = prefix)
    _validate_resnet_consumed_keys(
        sd,
        param_mapping,
        state_mapping;
        load_classifier = load_classifier,
    )
    ps = apply_state_dict(ps, sd, param_mapping)
    st = apply_resnet_state_dict(st, sd, state_mapping)
    return ps, st
end
