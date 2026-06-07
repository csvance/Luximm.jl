# CoAtNet backbone, Lux port of timm's `coatnet_*_rw` family (maxxvit.py).
#
# Architecture (coatnet_0_rw recipe):
#   - Stem: conv1 3x3/s2 → BN+SiLU → conv2 3x3/s1 (no act). All convs bias-free.
#   - Stages 1-2 (MBConv): pre_norm(BN+SiLU) → [avgpool if stride] → conv1 1x1
#     expand → BN+SiLU → conv2 3x3 depthwise → SE(ReLU gate) → BN+SiLU →
#     conv3 1x1 project, plus a (pooled, 1x1-expanded) residual shortcut.
#     Expansion is from the *input* channels; SE sits before norm2
#     (`attn_early`); all convs are bias-free.
#   - Stages 3-4 (transformer): pre-norm LayerNorm2d → [avgpool if stride] →
#     relative-position-bias attention → residual, then LayerNorm2d → ConvMlp
#     (1x1 → exact GELU → 1x1) → residual. The downsampling block's shortcut is
#     a pooled, bias-free 1x1 expand.
#   - Final norm: LayerNorm2d(num_features, eps 1e-6).
#   - Head: global average pool → Dense.
#
# `num_classes == 0` returns the final-normed feature map `(W/32, H/32, dims[4],
# N)`, matching `timm.forward_features(x)`. `num_classes > 0` adds the pooled
# Dense head and returns `(num_classes, N)`, matching `timm.forward(x)`.
#
# Reuses `se_block` (SE-ResNet's shared layer) and `rel_pos_attention` /
# `layernorm2d` from `src/Layers/`. BatchNorm running stats (stem + MBConv
# stages) thread into `st` via `apply_resnet_state_dict` + a parallel state
# mapping, mirroring SE-ResNet.

include("Config.jl")

const _COAT_BN_EPS = 1.0f-5
const _COAT_BN_MOMENTUM = 0.1f0

# timm MBConv/stem norms are BatchNormAct2d (BN then SiLU).
_coat_bn(ch::Int) = BatchNorm(
    ch,
    NNlib.swish;
    affine = true,
    track_stats = true,
    epsilon = _COAT_BN_EPS,
    momentum = _COAT_BN_MOMENTUM,
)

_coat_conv(in_ch, out_ch, k; stride = 1, groups = 1) = Conv(
    (k, k),
    in_ch => out_ch;
    stride = stride,
    pad = k ÷ 2,
    groups = groups,
    use_bias = false,
    cross_correlation = true,
)

_coat_avgpool(x) = NNlib.meanpool(x, (2, 2); stride = 2, pad = 0)

# Shared MBConv main-path body. `pool_down`/`attn_early` are build-time
# constants. `attn_early` places SE before norm2 (vs after).
function _coat_mbconv_main(
    x,
    pool_down::Bool,
    attn_early::Bool,
    pre_norm,
    conv1,
    norm1,
    conv2,
    se,
    norm2,
    conv3,
)
    h = pre_norm(x)
    pool_down && (h = _coat_avgpool(h))
    h = norm1(conv1(h))
    h = conv2(h)
    h = attn_early ? norm2(se(h)) : se(norm2(h))
    return conv3(h)
end

# One MBConv block. `mid` (the inverted-bottleneck width) is computed from the
# *input* channels (`expand_output=False`); SE inner width is `int(0.25*mid)`.
# `stride_mode` selects pool-stride (avg-pool the main path, stride-1 depthwise)
# vs depthwise-stride (no pool, strided depthwise conv). The SE field is always
# named `:se`; the mapping picks the matching timm key (`se_early` / `se`).
#
# Downsample shortcut mirrors timm `Downsample2d`: always avg-pool, plus a 1x1
# `shortcut_expand` conv **only when `in_ch != out_ch`** (otherwise the channel
# count already matches and timm uses an Identity expand — no weight).
function coatnet_mbconv_block(
    in_ch::Int,
    out_ch::Int,
    stride::Int;
    stride_mode::Symbol,
    attn_early::Bool,
    se_act,
)
    mid = se_make_divisible(in_ch * 4.0)
    rd = floor(Int, 0.25 * mid)
    se = se_block(mid; rd_channels = rd, act = se_act)
    pool_down = stride_mode === :pool && stride == 2
    conv2_stride = (stride_mode === :dw && stride == 2) ? 2 : 1
    if stride == 2 && in_ch != out_ch
        @compact(
            shortcut_expand = _coat_conv(in_ch, out_ch, 1),
            pre_norm = _coat_bn(in_ch),
            conv1 = _coat_conv(in_ch, mid, 1),
            norm1 = _coat_bn(mid),
            conv2 = _coat_conv(mid, mid, 3; stride = conv2_stride, groups = mid),
            se = se,
            norm2 = _coat_bn(mid),
            conv3 = _coat_conv(mid, out_ch, 1),
        ) do x
            shortcut = shortcut_expand(_coat_avgpool(x))
            @return _coat_mbconv_main(
                x, pool_down, attn_early, pre_norm, conv1, norm1, conv2, se, norm2, conv3,
            ) .+ shortcut
        end
    elseif stride == 2
        @compact(
            pre_norm = _coat_bn(in_ch),
            conv1 = _coat_conv(in_ch, mid, 1),
            norm1 = _coat_bn(mid),
            conv2 = _coat_conv(mid, mid, 3; stride = conv2_stride, groups = mid),
            se = se,
            norm2 = _coat_bn(mid),
            conv3 = _coat_conv(mid, out_ch, 1),
        ) do x
            shortcut = _coat_avgpool(x)
            @return _coat_mbconv_main(
                x, pool_down, attn_early, pre_norm, conv1, norm1, conv2, se, norm2, conv3,
            ) .+ shortcut
        end
    else
        @compact(
            pre_norm = _coat_bn(in_ch),
            conv1 = _coat_conv(in_ch, mid, 1),
            norm1 = _coat_bn(mid),
            conv2 = _coat_conv(mid, mid, 3; groups = mid),
            se = se,
            norm2 = _coat_bn(mid),
            conv3 = _coat_conv(mid, out_ch, 1),
        ) do x
            @return _coat_mbconv_main(
                x, pool_down, attn_early, pre_norm, conv1, norm1, conv2, se, norm2, conv3,
            ) .+ x
        end
    end
end

_coat_mlp_conv(in_ch, out_ch) = Conv(
    (1, 1),
    in_ch => out_ch;
    use_bias = true,
    cross_correlation = true,
    init_bias = zeros32,
)

# One transformer block. `window = (feat, feat)` sizes the relative-position
# bias. The downsampling block (stride 2) pools then 1x1-expands the shortcut
# (with bias iff `shortcut_bias`). The `ls1_gamma`/`ls2_gamma` LayerScale
# vectors are always present (initialized to ones = identity); the mapping loads
# them from the checkpoint only when the variant uses LayerScale, so a fixed
# `@compact` field set covers both cases.
function coatnet_transformer_block(
    in_ch::Int,
    out_ch::Int,
    stride::Int,
    window::Tuple{Int,Int};
    shortcut_bias::Bool,
)
    hidden = out_ch * 4
    attn = rel_pos_attention(in_ch, out_ch; dim_head = 32, window = window)
    _ls(g) = reshape(g, 1, 1, :, 1)
    if stride == 2
        @compact(
            shortcut_expand = Conv(
                (1, 1),
                in_ch => out_ch;
                use_bias = shortcut_bias,
                cross_correlation = true,
                init_bias = zeros32,
            ),
            norm1 = layernorm2d(in_ch),
            attn = attn,
            norm2 = layernorm2d(out_ch),
            mlp_fc1 = _coat_mlp_conv(out_ch, hidden),
            mlp_fc2 = _coat_mlp_conv(hidden, out_ch),
            ls1_gamma = ones32(out_ch),
            ls2_gamma = ones32(out_ch),
        ) do x
            shortcut = shortcut_expand(_coat_avgpool(x))
            h = attn(_coat_avgpool(norm1(x)))
            x = shortcut .+ _ls(ls1_gamma) .* h
            m = mlp_fc2(NNlib.gelu_erf.(mlp_fc1(norm2(x))))
            @return x .+ _ls(ls2_gamma) .* m
        end
    else
        @compact(
            norm1 = layernorm2d(in_ch),
            attn = attn,
            norm2 = layernorm2d(out_ch),
            mlp_fc1 = _coat_mlp_conv(out_ch, hidden),
            mlp_fc2 = _coat_mlp_conv(hidden, out_ch),
            ls1_gamma = ones32(out_ch),
            ls2_gamma = ones32(out_ch),
        ) do x
            x = x .+ _ls(ls1_gamma) .* attn(norm1(x))
            m = mlp_fc2(NNlib.gelu_erf.(mlp_fc1(norm2(x))))
            @return x .+ _ls(ls2_gamma) .* m
        end
    end
end

function coatnet_stage(cfg::CoAtNetVariant, block_type::Symbol, in_ch::Int, out_ch::Int, depth::Int, feat::Int)
    window = (feat, feat)
    se_act = cfg.se_act === :silu ? NNlib.swish : NNlib.relu
    blocks = []
    for b = 1:depth
        stride = b == 1 ? 2 : 1
        ic = b == 1 ? in_ch : out_ch
        if block_type === :C
            push!(
                blocks,
                coatnet_mbconv_block(
                    ic,
                    out_ch,
                    stride;
                    stride_mode = cfg.stride_mode,
                    attn_early = cfg.attn_early,
                    se_act = se_act,
                ),
            )
        else
            push!(
                blocks,
                coatnet_transformer_block(
                    ic,
                    out_ch,
                    stride,
                    window;
                    shortcut_bias = cfg.transformer_shortcut_bias,
                ),
            )
        end
    end
    return Chain(blocks...)
end

function _coatnet_features(
    x,
    stem_conv1,
    stem_norm1,
    stem_conv2,
    stage1,
    stage2,
    stage3,
    stage4,
    norm,
)
    x = stem_conv2(stem_norm1(stem_conv1(x)))
    x = stage1(x)
    x = stage2(x)
    x = stage3(x)
    x = stage4(x)
    return norm(x)
end

"""
    coatnet(variant; in_chans=3, num_classes=0) -> @compact block

Build a CoAtNet backbone. `variant` is a key from [`COATNET_VARIANTS`](@ref),
e.g. `:coatnet_0_rw_224_sw_in1k`.

When `num_classes == 0`, the forward returns the final-normed feature map
`(W/32, H/32, dims[4], N)`, matching `timm.forward_features(x)`. When
`num_classes > 0`, a global-average-pool + Dense head is attached and the
forward returns `(num_classes, N)`, matching `timm.forward(x)`.

The input spatial size must equal the variant's native `img_size`; the
transformer relative-position bias is sized to the per-stage feature map.
"""
function coatnet(variant::Symbol; in_chans::Int = 3, num_classes::Int = 0)
    cfg = get(COATNET_VARIANTS, variant) do
        error(
            "Unknown CoAtNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(COATNET_VARIANTS))))",
        )
    end
    dims = cfg.dims
    sw = cfg.stem_width
    bt = cfg.block_types
    d = cfg.depths
    feats = coatnet_feat_sizes(cfg)
    img = cfg.img_size
    in_for = (sw[2], dims[1], dims[2], dims[3])   # stage input channels

    if num_classes == 0
        @compact(
            stem_conv1 = _coat_conv(in_chans, sw[1], 3; stride = 2),
            stem_norm1 = _coat_bn(sw[1]),
            stem_conv2 = _coat_conv(sw[1], sw[2], 3; stride = 1),
            stage1 = coatnet_stage(cfg, bt[1], in_for[1], dims[1], d[1], feats[1]),
            stage2 = coatnet_stage(cfg, bt[2], in_for[2], dims[2], d[2], feats[2]),
            stage3 = coatnet_stage(cfg, bt[3], in_for[3], dims[3], d[3], feats[3]),
            stage4 = coatnet_stage(cfg, bt[4], in_for[4], dims[4], d[4], feats[4]),
            norm = layernorm2d(dims[4]),
        ) do x
            @assert size(x, 1) == img && size(x, 2) == img "CoAtNet $variant expects " *
                "$(img)x$(img) input; got $(size(x, 1))x$(size(x, 2))."
            @return _coatnet_features(
                x,
                stem_conv1,
                stem_norm1,
                stem_conv2,
                stage1,
                stage2,
                stage3,
                stage4,
                norm,
            )
        end
    else
        nc = num_classes
        @compact(
            stem_conv1 = _coat_conv(in_chans, sw[1], 3; stride = 2),
            stem_norm1 = _coat_bn(sw[1]),
            stem_conv2 = _coat_conv(sw[1], sw[2], 3; stride = 1),
            stage1 = coatnet_stage(cfg, bt[1], in_for[1], dims[1], d[1], feats[1]),
            stage2 = coatnet_stage(cfg, bt[2], in_for[2], dims[2], d[2], feats[2]),
            stage3 = coatnet_stage(cfg, bt[3], in_for[3], dims[3], d[3], feats[3]),
            stage4 = coatnet_stage(cfg, bt[4], in_for[4], dims[4], d[4], feats[4]),
            norm = layernorm2d(dims[4]),
            head_fc = Dense(dims[4] => nc; init_bias = zeros32),
        ) do x
            @assert size(x, 1) == img && size(x, 2) == img "CoAtNet $variant expects " *
                "$(img)x$(img) input; got $(size(x, 1))x$(size(x, 2))."
            x = _coatnet_features(
                x,
                stem_conv1,
                stem_norm1,
                stem_conv2,
                stage1,
                stage2,
                stage3,
                stage4,
                norm,
            )
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)
            x = reshape(x, (size(x, 3), size(x, 4)))
            @return head_fc(x)
        end
    end
end

# -- Pretrained-weight loading -------------------------------------------

_COAT_MAPPING_ENTRY = Tuple{String,Tuple{Vararg{Symbol}},Function}

# Append BN param (scale/bias) or LayerNorm2d (scale/bias via as_channel4d).
function _coat_push_bn!(mapping, py, lux)
    push!(mapping, ("$(py).weight", (lux..., :scale), identity))
    push!(mapping, ("$(py).bias", (lux..., :bias), identity))
end
function _coat_push_ln!(mapping, py, lux)
    push!(mapping, ("$(py).weight", (lux..., :scale), as_channel4d))
    push!(mapping, ("$(py).bias", (lux..., :bias), as_channel4d))
end

function _coat_push_mbconv!(mapping, py, lux, downsample::Bool, has_expand::Bool, attn_early::Bool)
    if downsample && has_expand
        push!(mapping, ("$(py).shortcut.expand.weight", (lux..., :shortcut_expand, :weight), identity))
    end
    _coat_push_bn!(mapping, "$(py).pre_norm", (lux..., :pre_norm))
    push!(mapping, ("$(py).conv1_1x1.weight", (lux..., :conv1, :weight), identity))
    _coat_push_bn!(mapping, "$(py).norm1", (lux..., :norm1))
    push!(mapping, ("$(py).conv2_kxk.weight", (lux..., :conv2, :weight), identity))
    se_key = attn_early ? "se_early" : "se"
    push!(mapping, ("$(py).$(se_key).fc1.weight", (lux..., :se, :fc1, :weight), identity))
    push!(mapping, ("$(py).$(se_key).fc1.bias", (lux..., :se, :fc1, :bias), identity))
    push!(mapping, ("$(py).$(se_key).fc2.weight", (lux..., :se, :fc2, :weight), identity))
    push!(mapping, ("$(py).$(se_key).fc2.bias", (lux..., :se, :fc2, :bias), identity))
    _coat_push_bn!(mapping, "$(py).norm2", (lux..., :norm2))
    push!(mapping, ("$(py).conv3_1x1.weight", (lux..., :conv3, :weight), identity))
end

function _coat_push_transformer!(
    mapping,
    py,
    lux,
    downsample::Bool,
    has_expand::Bool,
    shortcut_bias::Bool,
    layer_scale::Bool,
)
    if downsample
        if has_expand
            push!(mapping, ("$(py).shortcut.expand.weight", (lux..., :shortcut_expand, :weight), identity))
            if shortcut_bias
                push!(mapping, ("$(py).shortcut.expand.bias", (lux..., :shortcut_expand, :bias), identity))
            end
        end
        _coat_push_ln!(mapping, "$(py).norm1.norm", (lux..., :norm1))
    else
        _coat_push_ln!(mapping, "$(py).norm1", (lux..., :norm1))
    end
    push!(mapping, ("$(py).attn.qkv.weight", (lux..., :attn, :qkv, :weight), identity))
    push!(mapping, ("$(py).attn.qkv.bias", (lux..., :attn, :qkv, :bias), identity))
    push!(
        mapping,
        (
            "$(py).attn.rel_pos.relative_position_bias_table",
            (lux..., :attn, :rel_pos_bias_table),
            axis_reverse,
        ),
    )
    push!(mapping, ("$(py).attn.proj.weight", (lux..., :attn, :proj, :weight), identity))
    push!(mapping, ("$(py).attn.proj.bias", (lux..., :attn, :proj, :bias), identity))
    if layer_scale
        push!(mapping, ("$(py).ls1.gamma", (lux..., :ls1_gamma), identity))
    end
    _coat_push_ln!(mapping, "$(py).norm2", (lux..., :norm2))
    push!(mapping, ("$(py).mlp.fc1.weight", (lux..., :mlp_fc1, :weight), identity))
    push!(mapping, ("$(py).mlp.fc1.bias", (lux..., :mlp_fc1, :bias), identity))
    push!(mapping, ("$(py).mlp.fc2.weight", (lux..., :mlp_fc2, :weight), identity))
    push!(mapping, ("$(py).mlp.fc2.bias", (lux..., :mlp_fc2, :bias), identity))
    if layer_scale
        push!(mapping, ("$(py).ls2.gamma", (lux..., :ls2_gamma), identity))
    end
end

"""
    coatnet_mapping(state_dict, variant; load_classifier=false, in_chans=3,
                    prefix=()) -> Vector

Build the `(pytorch_key, lux_path, transform)` triples mapping a timm
`coatnet_*_rw` state_dict into the Lux tree from [`coatnet`](@ref). BatchNorm
running statistics are state, not params; use [`coatnet_state_mapping`](@ref).
"""
function coatnet_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(COATNET_VARIANTS, variant) do
        error(
            "Unknown CoAtNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(COATNET_VARIANTS))))",
        )
    end
    mapping = _COAT_MAPPING_ENTRY[]

    stem_w = in_chans == 3 ? identity : adapt_input_conv(in_chans)
    push!(mapping, ("stem.conv1.weight", (prefix..., :stem_conv1, :weight), stem_w))
    _coat_push_bn!(mapping, "stem.norm1", (prefix..., :stem_norm1))
    push!(mapping, ("stem.conv2.weight", (prefix..., :stem_conv2, :weight), identity))

    stage_in = (cfg.stem_width[2], cfg.dims[1], cfg.dims[2], cfg.dims[3])
    for (s, depth) in enumerate(cfg.depths)
        stage_sym = Symbol("stage", s)
        bt = cfg.block_types[s]
        has_expand = stage_in[s] != cfg.dims[s]   # Downsample2d expand conv present?
        for b = 1:depth
            lux = (prefix..., stage_sym, Symbol("layer_", b))
            py = "stages.$(s - 1).blocks.$(b - 1)"
            downsample = b == 1
            if bt === :C
                _coat_push_mbconv!(mapping, py, lux, downsample, has_expand, cfg.attn_early)
            else
                _coat_push_transformer!(
                    mapping,
                    py,
                    lux,
                    downsample,
                    has_expand,
                    cfg.transformer_shortcut_bias,
                    cfg.layer_scale,
                )
            end
        end
    end

    _coat_push_ln!(mapping, "norm", (prefix..., :norm))

    if load_classifier
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
    coatnet_state_mapping(state_dict, variant; prefix=()) -> Vector

Build the BatchNorm running-statistics state mapping (stem + MBConv stages).
The transformer stages use LayerNorm and contribute no state.
"""
function coatnet_state_mapping(
    state_dict::Dict,
    variant::Symbol;
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(COATNET_VARIANTS, variant) do
        error(
            "Unknown CoAtNet variant: $variant. Known variants: " *
            "$(sort(collect(keys(COATNET_VARIANTS))))",
        )
    end
    mapping = _COAT_MAPPING_ENTRY[]
    push_state(py, lux) = begin
        push!(mapping, ("$(py).running_mean", (lux..., :running_mean), identity))
        push!(mapping, ("$(py).running_var", (lux..., :running_var), identity))
    end

    push_state("stem.norm1", (prefix..., :stem_norm1))
    for (s, depth) in enumerate(cfg.depths)
        cfg.block_types[s] === :C || continue
        stage_sym = Symbol("stage", s)
        for b = 1:depth
            lux = (prefix..., stage_sym, Symbol("layer_", b))
            py = "stages.$(s - 1).blocks.$(b - 1)"
            push_state("$(py).pre_norm", (lux..., :pre_norm))
            push_state("$(py).norm1", (lux..., :norm1))
            push_state("$(py).norm2", (lux..., :norm2))
        end
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("state mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    _load_coatnet(ps, st, variant; in_chans, num_classes, revision, cache_dir,
                  prefix) -> (ps, st)

Private back-end for `create_pretrained` on CoAtNet variants. Loads the timm
`.safetensors` from HuggingFace, applies params into `ps` and BatchNorm running
statistics into `st`. The classifier-head handling matches the other families.
"""
function _load_coatnet(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = COATNET_VARIANTS[variant]
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
        coatnet_mapping(
            sd,
            variant;
            load_classifier = load_classifier,
            in_chans = in_chans,
            prefix = prefix,
        ),
    )
    st = apply_resnet_state_dict(st, sd, coatnet_state_mapping(sd, variant; prefix = prefix))
    return ps, st
end
