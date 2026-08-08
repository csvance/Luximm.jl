module Models

using Lux
using NNlib
using ..Layers
using ..Interop:
    apply_state_dict,
    axis_reverse,
    hf_hub_download,
    hf_hub_cache_dir,
    load_safetensors_state_dict,
    as_channel4d,
    as_token_norm,
    adapt_input_conv

# The feature-pyramid interface (`FeatureInfo`, `resolve_out_indices`,
# `feature_selector`) is referenced by every family's Model.jl, so it comes
# first.
include("FeatureInfo.jl")

# Shared ConvNeXt v1/v2 building blocks must be included before either
# family's Model.jl, since both reference `_CN_INIT`, `convnext_stage`, the
# mapping-entry builders, etc.
include("ConvNeXtCommon/Common.jl")

include("ResNetV2/Model.jl")
include("ResNet/Model.jl")
include("SEResNet/Model.jl")
include("ConvNeXtV2/Model.jl")
include("ConvNeXt/Model.jl")
include("VGG/Model.jl")
include("ViT/Model.jl")
include("CoAtNet/Model.jl")

"""
    create_model(variant; kwargs...) -> model

Family-agnostic random-init model constructor, mirroring
`timm.create_model(..., pretrained=False)`. Dispatches on `variant` to the
matching family constructor and returns the bare `@compact` model — no
parameters, no state, no pretrained weights.

Use this when you want to train from scratch, or as a building block
inside an outer `@compact` when composing a larger model. To load the
released weights for a variant, use [`create_pretrained`](@ref) instead.

```julia
model = create_model(:resnet50_a1_in1k; num_classes = 1000)
ps, st = Lux.setup(rng, model)        # random init, ready for training
```

`kwargs` are forwarded to the family constructor (`in_chans`,
`num_classes`, `features_only`, `out_indices`).

# Feature-pyramid mode

`features_only = true` is the Luximm analog of timm's
`features_only=True`: the forward returns a **tuple** of intermediate
feature maps ordered by increasing reduction (highest resolution first)
instead of a single map. This is what a UNet or FPN decoder consumes.

```julia
model = create_model(:resnet18_a1_in1k; features_only = true)
ps, st = Lux.setup(rng, model)
feats, st = model(x, ps, st)          # NTuple{5, Array{Float32, 4}}
size.(feats, 3)                       # (64, 64, 128, 256, 512)
```

`out_indices` selects a subset of the family's taps. It is **1-based**, so
timm's `out_indices=(1, 2, 3, 4)` is Luximm's `(2, 3, 4, 5)`. Indices must
be strictly increasing; `nothing` (the default) returns every tap. Query
the tap table with [`feature_info`](@ref):

```julia
feature_info(:resnet18_a1_in1k).reductions          # (2, 4, 8, 16, 32)
model = create_model(:resnet18_a1_in1k;
                     features_only = true, out_indices = (2, 3, 4, 5))
```

A features-only model always has `num_classes = 0` (passing anything else
is an error) and builds the **same parameter tree** as the plain
`num_classes = 0` feature extractor, so `create_pretrained` loads released
weights into it unchanged. Selecting fewer taps changes only the forward,
never the tree: every stage is still built and still runs.

Supported families: ResNet, SE-ResNet, BiT ResNetV2, ConvNeXt, ConvNeXt V2,
ViT. VGG and CoAtNet raise an error explaining why.

For the CNN families the taps form a true pyramid (increasing reduction).
A ViT is single-scale, so its taps all sit at the patch-size reduction;
they mirror timm's `features_only=True` for `vit_*` exactly — each tap is a
selected block's raw output with the class token dropped and the patch
tokens reshaped to a `(W, H, embed_dim, N)` grid (timm applies no final
LayerNorm to the intermediates, and neither do we). timm's `vit_*` default
`out_indices = 3` (the last three blocks) is Luximm's
`out_indices = (depth-2, depth-1, depth)`.
"""
function create_model(variant::Symbol; kwargs...)
    if get(kwargs, :features_only, false)
        nc = get(kwargs, :num_classes, 0)
        nc == 0 || error(
            "`features_only = true` requires `num_classes = 0`; got $nc. " *
            "A features-only model returns intermediate feature maps and has " *
            "no classifier head.",
        )
    end
    if haskey(BIT_VARIANTS, variant)
        return bit_resnetv2(variant; kwargs...)
    elseif haskey(RESNET_VARIANTS, variant)
        return resnet(variant; kwargs...)
    elseif haskey(CONVNEXT_VARIANTS, variant)
        return convnext(variant; kwargs...)
    elseif haskey(CONVNEXTV2_VARIANTS, variant)
        return convnextv2(variant; kwargs...)
    elseif haskey(VGG_VARIANTS, variant)
        return vgg(variant; kwargs...)
    elseif haskey(SERESNET_VARIANTS, variant)
        return seresnet(variant; kwargs...)
    elseif haskey(VIT_VARIANTS, variant)
        return vit(variant; kwargs...)
    elseif haskey(COATNET_VARIANTS, variant)
        return coatnet(variant; kwargs...)
    else
        error(
            "Unknown variant: $variant. Not found in any of " *
            "BIT_VARIANTS, RESNET_VARIANTS, CONVNEXT_VARIANTS, " *
            "CONVNEXTV2_VARIANTS, VGG_VARIANTS.",
        )
    end
end

"""
    default_num_classes(variant) -> Int

Head dimension the released checkpoint for `variant` was trained at.
Returns `0` for encoder-only variants (DINOv3 ConvNeXt, ConvNeXtV2
`fcmae` pretrains).
"""
function default_num_classes(variant::Symbol)
    if haskey(BIT_VARIANTS, variant)
        return BIT_VARIANTS[variant].default_num_classes
    elseif haskey(RESNET_VARIANTS, variant)
        return RESNET_VARIANTS[variant].default_num_classes
    elseif haskey(CONVNEXT_VARIANTS, variant)
        return CONVNEXT_VARIANTS[variant].default_num_classes
    elseif haskey(CONVNEXTV2_VARIANTS, variant)
        return CONVNEXTV2_VARIANTS[variant].default_num_classes
    elseif haskey(VGG_VARIANTS, variant)
        return VGG_VARIANTS[variant].default_num_classes
    elseif haskey(SERESNET_VARIANTS, variant)
        return SERESNET_VARIANTS[variant].default_num_classes
    elseif haskey(VIT_VARIANTS, variant)
        return VIT_VARIANTS[variant].default_num_classes
    elseif haskey(COATNET_VARIANTS, variant)
        return COATNET_VARIANTS[variant].default_num_classes
    else
        error(
            "Unknown variant: $variant. Not found in any of " *
            "BIT_VARIANTS, RESNET_VARIANTS, CONVNEXT_VARIANTS, " *
            "CONVNEXTV2_VARIANTS, VGG_VARIANTS.",
        )
    end
end

"""
    feature_info(variant; out_indices=nothing) -> FeatureInfo

Tap table for `variant` in feature-pyramid mode: the reduction (spatial
stride) and channel count of each feature map a
`create_model(variant; features_only = true)` model returns, ordered by
increasing reduction. The Luximm analog of timm's `model.feature_info`.

Call this to size a decoder before building it: a UNet needs the skip
channel counts, an FPN needs them to size its lateral 1×1 convolutions.

```julia
info = feature_info(:resnet18_a1_in1k)
info.reductions              # (2, 4, 8, 16, 32)
info.channels                # (64, 64, 128, 256, 512)

info = feature_info(:resnet18_a1_in1k; out_indices = (2, 3, 4, 5))
info.channels                # (64, 128, 256, 512)
```

`out_indices` is validated exactly as `create_model` validates it, so the
returned info always describes the tuple that model's forward produces.
Families without a pyramid (VGG, CoAtNet) raise an error.
"""
function feature_info(variant::Symbol; out_indices = nothing)
    full = if haskey(BIT_VARIANTS, variant)
        bit_resnetv2_feature_info(BIT_VARIANTS[variant])
    elseif haskey(RESNET_VARIANTS, variant)
        resnet_feature_info(RESNET_VARIANTS[variant])
    elseif haskey(CONVNEXT_VARIANTS, variant)
        convnext_feature_info(CONVNEXT_VARIANTS[variant])
    elseif haskey(CONVNEXTV2_VARIANTS, variant)
        convnextv2_feature_info(CONVNEXTV2_VARIANTS[variant])
    elseif haskey(SERESNET_VARIANTS, variant)
        seresnet_feature_info(SERESNET_VARIANTS[variant])
    elseif haskey(VGG_VARIANTS, variant)
        _no_feature_pyramid("VGG", variant, true, nothing)
    elseif haskey(VIT_VARIANTS, variant)
        vit_feature_info(VIT_VARIANTS[variant])
    elseif haskey(COATNET_VARIANTS, variant)
        _no_feature_pyramid("CoAtNet", variant, true, nothing)
    else
        error(
            "Unknown variant: $variant. Not found in any of " *
            "BIT_VARIANTS, RESNET_VARIANTS, CONVNEXT_VARIANTS, " *
            "CONVNEXTV2_VARIANTS, VGG_VARIANTS.",
        )
    end
    return select_features(full, resolve_out_indices(full, out_indices, variant))
end

"""
    create_pretrained(variant; in_chans=3, num_classes=nothing,
                      features_only=false, out_indices=nothing,
                      revision="main", cache_dir=hf_hub_cache_dir(),
                      prefix=()) -> (model, load)

Family-agnostic pretrained-weight entry point, mirroring
`timm.create_model(..., pretrained=True)`. Returns the model and a
closure that loads the released `model.safetensors` into a `(ps, st)`
pair the caller produced with `Lux.setup`. The closure captures
`variant`, `in_chans`, `num_classes`, and the HF / `prefix` kwargs at
construction time, so calling it is the only place `(ps, st)` need to
be threaded.

```julia
model, load = create_pretrained(:resnet50_a1_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
```

`num_classes = nothing` (the default) builds the head the released
checkpoint ships with — `default_num_classes(variant)`. Pass an
explicit `0` for a features-only model, or any other Int to swap in a
custom-width head (the released classifier is then skipped and the
warning case fires).

For composition, build `model` separately and pass it into an outer
`@compact`, capturing `prefix = (:backbone,)` so the closure writes
into the right subtree:

```julia
backbone, load_backbone = create_pretrained(:resnet50_a1_in1k;
    num_classes = 0, prefix = (:backbone,))
outer = @compact(backbone = backbone,
    head = Dense(2048 => num_outputs)) do x
    head(backbone(x))
end
ps, st = Lux.setup(rng, outer)
ps, st = load_backbone(ps, st)
```

`features_only = true` builds the feature-pyramid model described in
[`create_model`](@ref) and loads the released backbone weights into it. The
parameter tree is identical to the `num_classes = 0` feature extractor, so
this is the same load path; only the forward differs:

```julia
backbone, load = create_pretrained(:resnet18_a1_in1k;
    in_chans = 1, features_only = true, prefix = (:backbone,))
info = feature_info(:resnet18_a1_in1k)   # decoder widths
```
"""
function create_pretrained(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Union{Int,Nothing} = nothing,
    features_only::Bool = false,
    out_indices = nothing,
    revision::AbstractString = "main",
    cache_dir::AbstractString = hf_hub_cache_dir(),
    prefix::Tuple{Vararg{Symbol}} = (),
)
    nc = if features_only
        (num_classes === nothing || num_classes == 0) || error(
            "`features_only = true` requires `num_classes = 0` (or the " *
            "default `nothing`); got $num_classes. A features-only model " *
            "returns intermediate feature maps and has no classifier head.",
        )
        0
    else
        num_classes === nothing ? default_num_classes(variant) : num_classes
    end
    model = create_model(
        variant;
        in_chans = in_chans,
        num_classes = nc,
        features_only = features_only,
        out_indices = out_indices,
    )
    load =
        (ps, st) -> _load_pretrained(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = nc,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    return model, load
end

# Private: family-dispatching loader behind the `create_pretrained`
# closure. Takes `in_chans` and `num_classes` as explicit kwargs and
# forwards them to the per-family loader, which uses them directly
# instead of introspecting `ps`.
function _load_pretrained(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    if haskey(BIT_VARIANTS, variant)
        return _load_bit_resnetv2(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(RESNET_VARIANTS, variant)
        return _load_resnet(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(CONVNEXT_VARIANTS, variant)
        return _load_convnext(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(CONVNEXTV2_VARIANTS, variant)
        return _load_convnextv2(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(VGG_VARIANTS, variant)
        return _load_vgg(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(SERESNET_VARIANTS, variant)
        return _load_seresnet(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(VIT_VARIANTS, variant)
        return _load_vit(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    elseif haskey(COATNET_VARIANTS, variant)
        return _load_coatnet(
            ps,
            st,
            variant;
            in_chans = in_chans,
            num_classes = num_classes,
            revision = revision,
            cache_dir = cache_dir,
            prefix = prefix,
        )
    else
        error(
            "Unknown variant: $variant. Not found in any of " *
            "BIT_VARIANTS, RESNET_VARIANTS, CONVNEXT_VARIANTS, " *
            "CONVNEXTV2_VARIANTS, VGG_VARIANTS.",
        )
    end
end

export BiTVariant,
    BIT_VARIANTS,
    ResNetVariant,
    RESNET_VARIANTS,
    ConvNeXtV2Variant,
    CONVNEXTV2_VARIANTS,
    ConvNeXtVariant,
    CONVNEXT_VARIANTS,
    VGGVariant,
    VGG_VARIANTS,
    SEResNetVariant,
    SERESNET_VARIANTS,
    ViTVariant,
    VIT_VARIANTS,
    CoAtNetVariant,
    COATNET_VARIANTS,
    FeatureInfo,
    create_model,
    create_pretrained,
    default_num_classes,
    feature_info

end # module Models
