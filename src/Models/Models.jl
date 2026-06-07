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
`num_classes`).
"""
function create_model(variant::Symbol; kwargs...)
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
    create_pretrained(variant; in_chans=3, num_classes=nothing,
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
"""
function create_pretrained(
    variant::Symbol;
    in_chans::Int = 3,
    num_classes::Union{Int,Nothing} = nothing,
    revision::AbstractString = "main",
    cache_dir::AbstractString = hf_hub_cache_dir(),
    prefix::Tuple{Vararg{Symbol}} = (),
)
    nc = num_classes === nothing ? default_num_classes(variant) : num_classes
    model = create_model(variant; in_chans = in_chans, num_classes = nc)
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
    create_model,
    create_pretrained,
    default_num_classes

end # module Models
