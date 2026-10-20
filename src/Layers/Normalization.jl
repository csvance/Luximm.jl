# Normalization statistics and layers that Luximm owns, so the Reactant extension
# (ext/LuximmReactantExt.jl) can give them faster traced methods without overriding
# anything in LuxLib.
#
# On ordinary arrays `layernorm` and `groupnorm` call `LuxLib.layernorm` /
# `LuxLib.groupnorm` unchanged, so the host path keeps LuxLib's kernels and backward
# rules; `mean_var` is `Statistics.mean` / `var`. The layers keep `Lux.LayerNorm` /
# `Lux.GroupNorm`'s parameter names, shapes and field order and their empty state, so
# weight mappings and saved parameters carry over between the two.

"""
    mean_var(x; dims = :, corrected = true) -> (μ, σ²)

Mean and variance of `x` over `dims`, kept as reduced dimensions. The Reactant
extension replaces this on traced arrays with a form whose reverse pass skips the
identically zero ∂σ²/∂μ reduction while staying exact at every derivative order.
"""
function mean_var(x::AbstractArray; dims = :, corrected::Bool = true)
    μ = mean(x; dims)
    return μ, var(x; dims, corrected, mean = μ)
end

"""
    layernorm(x, scale, bias, dims, epsilon)

LayerNorm of `x` over `dims` with an affine `scale` / `bias` already shaped to
broadcast against `x`. `LuxLib.layernorm` on ordinary arrays.
"""
layernorm(x::AbstractArray, scale, bias, dims, epsilon) =
    Lux.LuxLib.layernorm(x, scale, bias, identity, dims, epsilon)

"""
    groupnorm(x, scale, bias, groups, epsilon)

GroupNorm of `x` (channels on axis `ndims(x) - 1`) with `(C,)` `scale` / `bias`.
`LuxLib.groupnorm` on ordinary arrays.
"""
groupnorm(x::AbstractArray, scale, bias, groups::Int, epsilon) =
    Lux.LuxLib.groupnorm(x, scale, bias, groups, identity, epsilon)

"""
    LuximmLayerNorm(shape, dims; epsilon = 1f-5)

Affine LayerNorm over `dims`, with `:bias` and `:scale` of size `(shape..., 1)`:
the same parameter tree as `Lux.LayerNorm(shape; dims, epsilon)`, routed through
[`layernorm`](@ref Luximm.Layers.layernorm).
"""
struct LuximmLayerNorm{S <: Tuple, D, E} <: Lux.AbstractLuxLayer
    shape::S
    dims::D
    epsilon::E
end

LuximmLayerNorm(shape, dims; epsilon = 1.0f-5) = LuximmLayerNorm(Tuple(shape), dims, epsilon)

# Same field order as Lux.LayerNorm's parameters: bias, then scale.
Lux.initialparameters(rng::AbstractRNG, l::LuximmLayerNorm) =
    (; bias = zeros32(rng, l.shape..., 1), scale = ones32(rng, l.shape..., 1))

(l::LuximmLayerNorm)(x, ps, st::NamedTuple) =
    (layernorm(x, ps.scale, ps.bias, l.dims, l.epsilon), st)

"""
    LuximmGroupNorm(chs, groups; epsilon = 1f-5)

Affine GroupNorm with `(chs,)` `:scale` and `:bias`: the same parameter tree as
`Lux.GroupNorm(chs, groups; affine = true, epsilon)`, routed through
[`groupnorm`](@ref Luximm.Layers.groupnorm).
"""
struct LuximmGroupNorm{E} <: Lux.AbstractLuxLayer
    chs::Int
    groups::Int
    epsilon::E
end

function LuximmGroupNorm(chs::Int, groups::Int; epsilon = 1.0f-5)
    chs % groups == 0 || throw(ArgumentError("chs = $chs is not divisible by groups = $groups"))
    return LuximmGroupNorm(chs, groups, epsilon)
end

# Same field order as Lux.GroupNorm's parameters: scale, then bias.
Lux.initialparameters(rng::AbstractRNG, l::LuximmGroupNorm) =
    (; scale = ones32(rng, l.chs), bias = zeros32(rng, l.chs))

(l::LuximmGroupNorm)(x, ps, st::NamedTuple) =
    (groupnorm(x, ps.scale, ps.bias, l.groups, l.epsilon), st)
