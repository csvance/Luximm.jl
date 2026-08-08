# Multi-scale feature-pyramid interface: the Luximm analog of timm's
# `features_only=True` / `model.feature_info`.
#
# A features-only model is built from the *same* slot names as the
# `num_classes = 0` feature extractor, so every family's existing pretrained
# weight mapping applies unchanged; only the forward differs, returning a
# tuple of intermediate maps instead of the last one. Each family exposes an
# ordered tap list through `feature_info`, and `out_indices` selects from it.
#
# Ordering is timm's: increasing reduction, so `feats[1]` is the highest
# resolution map and `feats[end]` the coarsest. `out_indices` is 1-based, so
# a timm `out_indices=(0, 1, 2, 3)` becomes `out_indices = (1, 2, 3, 4)`.

"""
    FeatureInfo{K}

Metadata for the `K` feature maps a features-only model returns, ordered by
increasing reduction (highest resolution first). The Luximm analog of timm's
`model.feature_info`.

Fields:
- `names`: the tap's module name in timm, e.g. `(:act1, :layer1, …)`. Purely
  informational; useful when cross-checking against a timm model.
- `reductions`: spatial stride of each map relative to the input, e.g.
  `(2, 4, 8, 16, 32)`. A map at reduction `r` has spatial size
  `(W ÷ r, H ÷ r)`.
- `channels`: channel count of each map, i.e. `size(feats[i], 3)`. This is
  what a UNet/FPN decoder needs to size its skip projections.
- `indices`: which entries of the family's *full* tap list these are. For an
  unfiltered `feature_info(variant)` this is `(1, 2, …, K)`; after selecting
  with `out_indices` it echoes that selection back.

```julia
info = feature_info(:resnet18_a1_in1k)
info.reductions   # (2, 4, 8, 16, 32)
info.channels     # (64, 64, 128, 256, 512)

info = feature_info(:resnet18_a1_in1k; out_indices = (2, 3, 4, 5))
info.channels     # (64, 128, 256, 512)
info.indices      # (2, 3, 4, 5)
```
"""
struct FeatureInfo{K}
    names::NTuple{K,Symbol}
    reductions::NTuple{K,Int}
    channels::NTuple{K,Int}
    indices::NTuple{K,Int}
end

# Full, unfiltered tap list: indices are implicitly 1:K. Every family builds
# its table through this constructor.
function FeatureInfo(
    names::NTuple{K,Symbol},
    reductions::NTuple{K,Int},
    channels::NTuple{K,Int},
) where {K}
    return FeatureInfo{K}(names, reductions, channels, ntuple(identity, K))
end

Base.length(::FeatureInfo{K}) where {K} = K

function Base.show(io::IO, info::FeatureInfo{K}) where {K}
    print(io, "FeatureInfo($K taps:")
    for i = 1:K
        print(
            io,
            " [",
            info.indices[i],
            "] ",
            info.names[i],
            " r",
            info.reductions[i],
            "×",
            info.channels[i],
            "ch",
        )
    end
    print(io, ")")
    return nothing
end

# Human-readable tap table for error messages.
function _feature_tap_table(info::FeatureInfo{K}) where {K}
    rows = [
        "  $(info.indices[i]) => $(info.names[i]) " *
        "(reduction $(info.reductions[i]), $(info.channels[i]) channels)" for i = 1:K
    ]
    return join(rows, "\n")
end

"""
    select_features(info, indices) -> FeatureInfo

Narrow a family's full [`FeatureInfo`](@ref) to the taps at `indices`.
"""
function select_features(info::FeatureInfo, indices::NTuple{K,Int}) where {K}
    return FeatureInfo{K}(
        map(i -> info.names[i], indices),
        map(i -> info.reductions[i], indices),
        map(i -> info.channels[i], indices),
        indices,
    )
end

"""
    resolve_out_indices(info, out_indices, variant) -> NTuple{K,Int}

Validate a user-supplied `out_indices` against the family's full tap list.
`nothing` means "every tap". Indices are 1-based, must be in range, and must
be strictly increasing, since the returned feature tuple is ordered by
increasing reduction and a decoder relies on that ordering.
"""
function resolve_out_indices(info::FeatureInfo{K}, out_indices, variant::Symbol) where {K}
    out_indices === nothing && return ntuple(identity, K)
    idx = Tuple(out_indices)
    isempty(idx) && error(
        "out_indices for $variant is empty; pass at least one tap, or " *
        "`nothing` for all $K.",
    )
    all(i -> i isa Integer, idx) ||
        error("out_indices for $variant must be integers; got $(out_indices).")
    idx = map(Int, idx)
    for i in idx
        1 <= i <= K || error(
            "out_indices entry $i is out of range for $variant, which has " *
            "$K taps (1-based; timm's 0-based index $(i - 1) is Luximm's $i):\n" *
            _feature_tap_table(info),
        )
    end
    all(idx[i] < idx[i+1] for i = 1:(length(idx)-1)) || error(
        "out_indices for $variant must be strictly increasing; got $(idx). " *
        "Returned features are ordered by increasing reduction.",
    )
    return idx
end

# Maps a family's full tap tuple to the selected subset. Captured by the
# features-only `@compact` body, so `indices` is a compile-time-constant
# tuple and the returned tuple stays type-stable under Reactant tracing.
feature_selector(indices::NTuple{K,Int}) where {K} = taps -> map(i -> taps[i], indices)

# `out_indices` only means something in features-only mode; silently ignoring
# it would hand back a single feature map to a caller who asked for four.
function _check_out_indices_unused(variant::Symbol, out_indices)
    out_indices === nothing || error(
        "`out_indices` requires `features_only = true` (variant $variant); " *
        "got out_indices = $(out_indices).",
    )
    return nothing
end

# Guard for the families that do not expose a pyramid yet. Called by their
# constructors so a direct `vgg(...; features_only = true)` fails the same
# way `create_model` does.
function _no_feature_pyramid(
    family::AbstractString,
    variant::Symbol,
    features_only::Bool,
    out_indices,
)
    features_only && error(
        "$family has no feature-pyramid support yet, so `features_only = true` " *
        "is not available for $variant. Families with a pyramid: ResNet, " *
        "SE-ResNet, BiT ResNetV2, ConvNeXt, ConvNeXt V2, ViT.",
    )
    _check_out_indices_unused(variant, out_indices)
    return nothing
end
