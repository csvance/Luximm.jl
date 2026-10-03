# Shared building blocks for ConvNeXt v1 and v2.
#
# Both timm `convnext.py` (v1) and `convnextv2.py` (v2) share the same stem,
# downsample, classifier head, and stage scaffolding; the only architectural
# difference is the block body (v1 has LayerScale, v2 has GRN). To keep the
# two families in their own directories for license isolation while avoiding
# ~80 lines of duplication, this file holds:
#
#   - `_CN_INIT`                       timm's `_init_weights` recipe (Conv +
#                                      Linear → truncated_normal(std=0.02))
#   - `convnext_downsample`            inter-stage LayerNorm2d + 2x2 stride-2
#                                      conv used by every variant
#   - `convnext_stage`                 stage scaffolding parameterized over a
#                                      block-constructor function so v1 and v2
#                                      can pass their own block
#   - `convnext_stage_block_path`      mapping-side helper returning the
#                                      parameter-tree path for block `j` in
#                                      stage `i`, given the stage's stride
#   - `push_stem_mapping!`             mapping-entry builder for `stem.0`,
#                                      `stem.1` (Conv + LayerNorm2d)
#   - `push_downsample_mapping!`       mapping-entry builder for one stage's
#                                      `downsample.0`, `downsample.1`
#   - `push_head_norm_mapping!`        mapping-entry builder for `head.norm`
#   - `push_head_fc_mapping!`          mapping-entry builder for `head.fc`
#
# Everything here is shared *code*, not shared *weights*: v1 and v2 still
# load their own checkpoints from their own HF repos under their own
# licenses (v2: CC-BY-NC-4.0; v1 DINOv3: dinov3-license).

# timm's `_init_weights` for both ConvNeXt families: every Conv2d and every
# Linear gets `trunc_normal_(std=0.02)`, bias zero. LayerNorm/LayerScale
# defaults already match timm. Lux's `truncated_normal` uses absolute bounds
# (lo=-2, hi=2 by default), matching PyTorch's `trunc_normal_(a=-2, b=2)`.
const _CN_INIT = truncated_normal(; mean = 0.0f0, std = 0.02f0)

# Mapping entry type alias shared by both family-level mapping builders.
const _CN_MAPPING_ENTRY = Tuple{String, Tuple{Vararg{Symbol}}, Function}

"""
    convnext_downsample(in_C, out_C) -> @compact block

Inter-stage downsample used by both ConvNeXt v1 and v2: `LayerNorm2d(in_C)`
followed by `Conv((2,2), in_C => out_C; stride=2, bias)`. Matches timm's
`nn.Sequential(LayerNorm2d, Conv2d)` exactly, including the post-norm-then-
conv ordering required for parity.
"""
function convnext_downsample(in_C::Int, out_C::Int)
    return @compact(
        norm = layernorm2d(in_C),
        conv = Conv(
            (2, 2),
            in_C => out_C;
            stride = 2,
            pad = 0,
            use_bias = true,
            cross_correlation = true,
            init_weight = _CN_INIT,
            init_bias = zeros32,
        ),
    ) do x
        @return conv(norm(x))
    end
end

"""
    convnext_stage(block_ctor, in_C, out_C, depth, stride) -> Chain or @compact

Build one ConvNeXt stage. `block_ctor` is a unary function `C -> @compact`
that constructs a single residual block at width `C`; v1 passes its
`convnext_block` (with LayerScale), v2 passes its `convnextv2_block`
(with GRN). `stride==1` (stage 0 only, since the stem already strides by
4) returns a bare `Chain` of blocks; `stride==2` (stages 1-3) wraps the
chain in a `@compact` that prepends an inter-stage `convnext_downsample`.

The parameter tree shape is fixed by this function so that the family-
specific mapping functions can address `(:stage{i}, :layer_{j})` for
stride-1 stages and `(:stage{i}, :blocks, :layer_{j})` for stride-2
stages. See [`convnext_stage_block_path`](@ref).
"""
function convnext_stage(block_ctor, in_C::Int, out_C::Int, depth::Int, stride::Int)
    blocks = [block_ctor(out_C) for _ in 1:depth]
    if stride == 1
        return Chain(blocks...)
    else
        @compact(
            downsample = convnext_downsample(in_C, out_C),
            blocks = Chain(blocks...),
        ) do x
            @return blocks(downsample(x))
        end
    end
end

"""
    convnext_stage_block_path(stage_idx, stride, block_idx) -> Tuple{Vararg{Symbol}}

Path to block `block_idx` of stage `stage_idx` inside a model's parameter
tree, given the stage's `stride` (which determines whether the stage is a
bare `Chain` or a `@compact` with `:downsample` and `:blocks` sub-trees).
Stage 0 (stride 1) is a bare `Chain`, so block `j` sits directly under
`:stage{i}` as `:layer_{j}`. Stages 1-3 (stride 2) are `@compact` with
`:downsample` and `:blocks` sub-trees, so block `j` lives under
`:stage{i}, :blocks, :layer_{j}`.
"""
function convnext_stage_block_path(stage_idx::Int, stride::Int, block_idx::Int)
    stage_sym = Symbol("stage", stage_idx)
    layer_sym = Symbol("layer_", block_idx)
    if stride == 1
        return (stage_sym, layer_sym)
    else
        return (stage_sym, :blocks, layer_sym)
    end
end

"""
    push_stem_mapping!(mapping, prefix, in_chans) -> mapping

Append the four `(pytorch_key, lux_path, transform)` triples for the stem
(`stem.0` = Conv, `stem.1` = LayerNorm2d) to `mapping`. When `in_chans != 3`,
the conv weight transform becomes `adapt_input_conv(in_chans)` so the
released 3-channel weight is collapsed to match the requested input channel
count, mirroring timm's behaviour.
"""
function push_stem_mapping!(mapping::Vector, prefix::Tuple{Vararg{Symbol}}, in_chans::Int)
    stem_w_transform = in_chans == 3 ? identity : adapt_input_conv(in_chans)
    push!(mapping, ("stem.0.weight", (prefix..., :stem_conv, :weight), stem_w_transform))
    push!(mapping, ("stem.0.bias", (prefix..., :stem_conv, :bias), identity))
    push!(mapping, ("stem.1.weight", (prefix..., :stem_norm, :scale), as_channel4d))
    push!(mapping, ("stem.1.bias", (prefix..., :stem_norm, :bias), as_channel4d))
    return mapping
end

"""
    push_downsample_mapping!(mapping, prefix, stage_sym, py_stage) -> mapping

Append the four `(pytorch_key, lux_path, transform)` triples for one
stage's downsample (`downsample.0` = LayerNorm2d, `downsample.1` = Conv) to
`mapping`. `stage_sym` is the Lux-side stage symbol (e.g. `:stage2`);
`py_stage` is the PyTorch-side prefix (e.g. `"stages.1"`).
"""
function push_downsample_mapping!(
        mapping::Vector,
        prefix::Tuple{Vararg{Symbol}},
        stage_sym::Symbol,
        py_stage::String,
    )
    push!(
        mapping,
        (
            "$(py_stage).downsample.0.weight",
            (prefix..., stage_sym, :downsample, :norm, :scale),
            as_channel4d,
        ),
    )
    push!(
        mapping,
        (
            "$(py_stage).downsample.0.bias",
            (prefix..., stage_sym, :downsample, :norm, :bias),
            as_channel4d,
        ),
    )
    push!(
        mapping,
        (
            "$(py_stage).downsample.1.weight",
            (prefix..., stage_sym, :downsample, :conv, :weight),
            identity,
        ),
    )
    push!(
        mapping,
        (
            "$(py_stage).downsample.1.bias",
            (prefix..., stage_sym, :downsample, :conv, :bias),
            identity,
        ),
    )
    return mapping
end

"""
    push_head_norm_mapping!(mapping, prefix) -> mapping

Append the two `head.norm.*` triples (LayerNorm2d) to `mapping`. The
LayerNorm dim depends only on the feature width, not `num_classes`, so
this can be loaded even when the user built the model with a custom
classifier dimension.
"""
function push_head_norm_mapping!(mapping::Vector, prefix::Tuple{Vararg{Symbol}})
    push!(mapping, ("head.norm.weight", (prefix..., :head_norm, :scale), as_channel4d))
    push!(mapping, ("head.norm.bias", (prefix..., :head_norm, :bias), as_channel4d))
    return mapping
end

"""
    push_head_fc_mapping!(mapping, prefix) -> mapping

Append the two `head.fc.*` triples (Linear → Lux Dense) to `mapping`.
The `head.fc` weight came from `nn.Linear` (2D): after axis-reverse
from PyTorch's `(out, in)` it's `(in, out)`, but Lux `Dense` stores
weight as `(out, in)`, so `axis_reverse` is applied to transpose it.
"""
function push_head_fc_mapping!(mapping::Vector, prefix::Tuple{Vararg{Symbol}})
    push!(mapping, ("head.fc.weight", (prefix..., :head_fc, :weight), axis_reverse))
    push!(mapping, ("head.fc.bias", (prefix..., :head_fc, :bias), identity))
    return mapping
end
