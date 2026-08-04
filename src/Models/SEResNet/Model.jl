# SE-ResNet backbone, Lux port of timm's `seresnet*` family.
#
# A classic post-activation ResNet bottleneck network with a squeeze-excite
# module inserted after the final BN of each block, before the residual add:
#
#   conv1(1x1)→bn1→relu → conv2(3x3,stride)→bn2→relu → conv3(1x1)→bn3
#     → se_block → (+ downsample shortcut) → relu
#
# Reuses the ResNet stem/stage scaffolding (`_resnet_bn`, the `layer{stage}`
# Chain naming, the BN-running-stats-into-`st` machinery) from
# `../ResNet/Model.jl`, which is included before this file, and the shared
# `se_block` from `src/Layers/SqueezeExcite.jl`.
#
# `num_classes=0` returns the spatial feature map (`forward_features`).
# `num_classes>0` attaches timm's global average pool + Linear classifier and
# returns logits `(num_classes, N)`.

include("Config.jl")

# One SE-bottleneck block. `plane_ch` is the bottleneck width; the block output
# is `4 * plane_ch`. `downsample` adds the 1x1-conv + BN projection shortcut.
function seresnet_bottleneck_block(
    in_ch::Int,
    plane_ch::Int,
    stride::Int;
    downsample::Bool,
    reduction::Int,
)
    out_ch = 4 * plane_ch
    se = se_block(out_ch; rd_ratio = 1 // reduction)
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
            se = se,
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
            y = se(y)
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
                plane_ch => out_ch;
                stride = 1,
                pad = 0,
                use_bias = false,
                cross_correlation = true,
                init_weight = _RESNET_CONV_INIT,
            ),
            bn3 = _resnet_bn(out_ch; zero_scale = true),
            se = se,
        ) do x
            y = conv1(x)
            y = NNlib.relu.(bn1(y))
            y = conv2(y)
            y = NNlib.relu.(bn2(y))
            y = conv3(y)
            y = bn3(y)
            y = se(y)
            @return NNlib.relu.(y .+ x)
        end
    end
end

function seresnet_stage(
    in_ch::Int,
    plane_ch::Int,
    depth::Int,
    stride::Int,
    reduction::Int,
)
    out_ch = 4 * plane_ch
    blocks = []
    push!(
        blocks,
        seresnet_bottleneck_block(
            in_ch,
            plane_ch,
            stride;
            downsample = true,
            reduction = reduction,
        ),
    )
    for _ = 2:depth
        push!(
            blocks,
            seresnet_bottleneck_block(
                out_ch,
                plane_ch,
                1;
                downsample = false,
                reduction = reduction,
            ),
        )
    end
    return Chain(blocks...)
end

# Ordered feature taps, same layout as the classic ResNet: `f2` is timm's
# `act1` map (post-BN-ReLU, before the stem maxpool), then one tap per stage.
# All three constructor branches route through here so they cannot desync.
function _seresnet_feature_taps(x, conv1, bn1, layer1, layer2, layer3, layer4)
    f2 = NNlib.relu.(bn1(conv1(x)))
    x = NNlib.maxpool(f2, (3, 3); stride = 2, pad = 1)
    f4 = layer1(x)
    f8 = layer2(f4)
    f16 = layer3(f8)
    f32 = layer4(f16)
    return (f2, f4, f8, f16, f32)
end

_seresnet_features(x, conv1, bn1, layer1, layer2, layer3, layer4) =
    last(_seresnet_feature_taps(x, conv1, bn1, layer1, layer2, layer3, layer4))

"""
    seresnet_feature_info(cfg) -> FeatureInfo

Ordered feature taps for an SE-ResNet, matching timm's `feature_info` for the
`seresnet*` family. Every registered variant is a bottleneck net, so stage
widths are `4 * planes`.
"""
function seresnet_feature_info(cfg::SEResNetVariant)
    return FeatureInfo(
        (:act1, :layer1, :layer2, :layer3, :layer4),
        (2, 4, 8, 16, 32),
        (64, ntuple(i -> 4 * cfg.planes[i], 4)...),
    )
end

# Backbone half, shared by the `num_classes = 0` feature extractor
# (`out_sel = last`) and the features-only pyramid. Identical slot tree, so
# one pretrained mapping serves both.
function _seresnet_backbone(cfg::SEResNetVariant, in_chans::Int, out_sel)
    depths = cfg.layers
    planes = cfg.planes
    red = cfg.se_reduction
    stage_chs = ntuple(i -> 4 * planes[i], 4)
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
        layer1 = seresnet_stage(64, planes[1], depths[1], 1, red),
        layer2 = seresnet_stage(stage_chs[1], planes[2], depths[2], 2, red),
        layer3 = seresnet_stage(stage_chs[2], planes[3], depths[3], 2, red),
        layer4 = seresnet_stage(stage_chs[3], planes[4], depths[4], 2, red),
    ) do x
        @return out_sel(
            _seresnet_feature_taps(x, conv1, bn1, layer1, layer2, layer3, layer4),
        )
    end
end

"""
    seresnet(variant; in_chans=3, num_classes=0,
             features_only=false, out_indices=nothing) -> @compact block

Build a timm SE-ResNet. `variant` is a key from [`SERESNET_VARIANTS`](@ref),
e.g. `:seresnet50_a1_in1k`.

With `features_only = true` the forward returns a tuple of intermediate
feature maps; see [`feature_info`](@ref) and [`create_model`](@ref).
"""
function seresnet(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Int = 0,
    features_only::Bool = false,
    out_indices = nothing,
)
    cfg = get(SERESNET_VARIANTS, variant) do
        error(
            "Unknown SE-ResNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(SERESNET_VARIANTS))))",
        )
    end
    depths = cfg.layers
    planes = cfg.planes
    red = cfg.se_reduction
    stage_chs = ntuple(i -> 4 * planes[i], 4)

    if features_only
        indices = resolve_out_indices(seresnet_feature_info(cfg), out_indices, variant)
        return _seresnet_backbone(cfg, in_chans, feature_selector(indices))
    end
    _check_out_indices_unused(variant, out_indices)

    if num_classes == 0
        _seresnet_backbone(cfg, in_chans, last)
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
            layer1 = seresnet_stage(64, planes[1], depths[1], 1, red),
            layer2 = seresnet_stage(stage_chs[1], planes[2], depths[2], 2, red),
            layer3 = seresnet_stage(stage_chs[2], planes[3], depths[3], 2, red),
            layer4 = seresnet_stage(stage_chs[3], planes[4], depths[4], 2, red),
            fc = Dense(stage_chs[4] => nc; init_bias = zeros32),
        ) do x
            x = _seresnet_features(x, conv1, bn1, layer1, layer2, layer3, layer4)
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)
            x = reshape(x, (size(x, 3), size(x, 4)))
            @return fc(x)
        end
    end
end

# -- Pretrained-weight loading -------------------------------------------

# Every SE-bottleneck's first block of a stage carries a downsample projection
# (stage 1 changes channels 64→256; stages 2-4 also stride).
_seresnet_block_has_downsample(block::Int) = block == 1

function seresnet_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(SERESNET_VARIANTS, variant) do
        error(
            "Unknown SE-ResNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(SERESNET_VARIANTS))))",
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
            # SE module (1x1 conv fc1/fc2; identity transforms).
            push!(
                mapping,
                (
                    "$(py_block).se.fc1.weight",
                    (prefix..., block_path..., :se, :fc1, :weight),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).se.fc1.bias",
                    (prefix..., block_path..., :se, :fc1, :bias),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).se.fc2.weight",
                    (prefix..., block_path..., :se, :fc2, :weight),
                    identity,
                ),
            )
            push!(
                mapping,
                (
                    "$(py_block).se.fc2.bias",
                    (prefix..., block_path..., :se, :fc2, :bias),
                    identity,
                ),
            )
            if _seresnet_block_has_downsample(block)
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

function seresnet_state_mapping(
    state_dict::Dict,
    variant::Symbol;
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(SERESNET_VARIANTS, variant) do
        error(
            "Unknown SE-ResNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(SERESNET_VARIANTS))))",
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
            _push_resnet_bn_state_mapping!(
                mapping,
                "$(py_block).bn3",
                (prefix..., block_path..., :bn3),
            )
            if _seresnet_block_has_downsample(block)
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

"""
    _load_seresnet(ps, st, variant; in_chans, num_classes, revision,
                   cache_dir, prefix) -> (ps, st)

Private back-end for `create_pretrained` on SE-ResNet variants. Loads the timm
`.safetensors` from HuggingFace, applies conv/SE/Dense params into `ps` and
BatchNorm running statistics into `st`. The classifier-head handling matches
the ResNet loader's three cases.
"""
function _load_seresnet(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = SERESNET_VARIANTS[variant]
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
        seresnet_mapping(
            sd,
            variant;
            load_classifier = load_classifier,
            in_chans = in_chans,
            prefix = prefix,
        ),
    )
    st = apply_resnet_state_dict(st, sd, seresnet_state_mapping(sd, variant; prefix = prefix))
    return ps, st
end
