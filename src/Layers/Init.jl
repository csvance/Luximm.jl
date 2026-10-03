# Initializers that mirror timm's `_init_weights` recipes. Lux ships
# `kaiming_normal` with fan-in semantics and lacks a parametric normal init,
# so we provide the fan-out and fixed-std variants here. `truncated_normal`
# from Lux already matches PyTorch's `trunc_normal_` semantics (absolute
# bounds), so no helper is needed for that recipe.

using Random: AbstractRNG, randn

"""
    kaiming_normal_fan_out(rng, [T,] dims...) -> Array{T}

PyTorch's `nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')`.

`std = sqrt(2 / fan_out)` where `fan_out` follows PyTorch's
`_calculate_fan_in_and_fan_out`: `out_channels * prod(receptive_field)`.
Lux `Conv` weight shape is `(kW, kH, in, out)`, so
`fan_out = dims[end] * prod(dims[1:end-2])`. For `Dense` weight shape
`(out, in)`, `fan_out = dims[1]`.
"""
function kaiming_normal_fan_out(
        rng::AbstractRNG,
        ::Type{T},
        dims::Integer...,
    ) where {T <: Real}
    fan_out = length(dims) <= 2 ? dims[1] : dims[end] * prod(dims[1:(end - 2)])
    std = sqrt(T(2) / T(fan_out))
    return std .* randn(rng, T, dims...)
end
kaiming_normal_fan_out(rng::AbstractRNG, dims::Integer...) =
    kaiming_normal_fan_out(rng, Float32, dims...)

"""
    normal_init(; std) -> (rng, dims...) -> Array{Float32}

Closure that produces `Normal(0, std)` samples in `Float32`, mirroring
PyTorch's `nn.init.normal_(weight, mean=0., std=std)`. Used for `timm`'s
BiT classifier head (`std = 0.01`).
"""
normal_init(; std::Real = 0.01f0) =
    (rng::AbstractRNG, dims::Integer...) -> Float32(std) .* randn(rng, Float32, dims...)
