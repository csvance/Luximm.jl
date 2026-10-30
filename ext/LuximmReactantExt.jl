module LuximmReactantExt

# Traced-array methods for Luximm's normalization statistics (src/Layers/Normalization.jl), and
# the `@trace` call-cache key of its normalization layers. Only Luximm's own functions and types
# get methods here; nothing in LuxLib is overridden, so this composes with any LuxLib, forked or
# not.

using EnzymeCore: EnzymeCore
using Reactant: Reactant, AnyTracedRArray
using Statistics: mean, var

using Luximm.Layers: Layers

# LuxCore's Reactant extension returns every AbstractLuxLayer unchanged from `make_tracer`, which
# leaves the layer out of the key of Reactant's `@trace` call cache: two layers of one type with
# different settings would share one traced call. Put our layers into the key; identical layers
# still share a call.
function Reactant.make_tracer(
        seen, @nospecialize(l::Union{Layers.LuximmGroupNorm, Layers.LuximmLayerNorm}),
        @nospecialize(path), mode; kwargs...
    )
    mode == Reactant.TracedToTypes || return l
    push!(path, l)
    return nothing
end

# Reverse-mode AD of `var(x; mean = μ)` differentiates through `μ` as well as `x`, and the term it
# gets through `μ` is ∂σ²/∂μ = -2/(M - c) Σ(x - μ), which is identically zero because `μ` is the mean
# of `x`. Enzyme cannot see that, so every normalization's backward pass spends a full-size
# multiply-and-reduce computing it. We compute the variance about a detached copy μ₀ of the mean
# instead, and subtract the König-Huygens correction, which is exact for any constant shift:
#
#     σ² = Σ(x - μ₀)² / (M - c) - M / (M - c) * (μ - μ₀)²
#
# The correction is zero in value (μ₀ = μ) and only involves the reduced-size `μ`, so the full-size
# reduction is gone. Unlike detaching `μ` alone, this is still an exact formula for the variance as a
# function of `x`, so derivatives of every order (nested AD, gradient penalties) stay correct
# regardless of how `ignore_derivatives` behaves under nesting.
#
# Proposed upstream as LuxDL/Lux.jl#1762.
function Layers.mean_var(x::AnyTracedRArray; dims = :, corrected::Bool = true)
    μ = mean(x; dims)
    μ₀ = EnzymeCore.ignore_derivatives(μ)
    M = length(x) ÷ length(μ)  # elements per reduction; sizes are static at trace time
    k = Reactant.unwrapped_eltype(x)(M / (M - corrected))
    return μ, var(x; dims, corrected, mean = μ₀) .- k .* abs2.(μ .- μ₀)
end

# Both normalizations fold the statistics into a per-group scale and shift first, so the full-size
# work is one multiply-add, as in LuxLib's affine normalization.
function Layers.layernorm(x::AnyTracedRArray, scale, bias, dims, epsilon)
    μ, σ² = Layers.mean_var(x; dims, corrected = false)
    γ′ = scale ./ sqrt.(σ² .+ epsilon)
    return x .* γ′ .+ (bias .- μ .* γ′)
end

function Layers.groupnorm(x::AnyTracedRArray, scale, bias, groups::Int, epsilon)
    sz = size(x)
    nd = ndims(x)
    C = sz[nd - 1]
    lead = ntuple(_ -> 1, nd - 2)
    xg = reshape(x, sz[1:(nd - 2)]..., C ÷ groups, groups, sz[nd])
    μ, σ² = Layers.mean_var(xg; dims = ntuple(identity, nd - 1), corrected = false)
    γ′ = reshape(scale, lead..., C ÷ groups, groups, 1) ./ sqrt.(σ² .+ epsilon)
    β′ = reshape(bias, lead..., C ÷ groups, groups, 1) .- μ .* γ′
    return reshape(xg .* γ′ .+ β′, sz)
end

end # module LuximmReactantExt
