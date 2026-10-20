# Correctness of the Reactant extension (ext/LuximmReactantExt.jl). Its traced methods of
# Luximm's normalization statistics must give the values, gradients and second derivatives of the
# definitions, written here in plain Statistics on the host. Tiny inputs, no fixtures, no weights.
#
# Gradients and Hessians are checked in Float64 against ForwardDiff, with non-uniform loss weights
# so every derivative path through the statistics matters: a uniform weight makes the
# mean-subtraction term vanish and would hide a wrong gradient.

using Test
using Luximm
using Lux
using Random
using Statistics
using LinearAlgebra: norm
using Reactant, Enzyme, ForwardDiff

const NormLayers = Luximm.Layers

function ref_layernorm(x, γ, β, dims, ϵ)
    μ = mean(x; dims)
    σ² = var(x; dims, corrected = false, mean = μ)
    return γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β
end

function ref_groupnorm(x, γ, β, groups, ϵ)
    W, H, C, N = size(x)
    xr = reshape(x, W, H, C ÷ groups, groups, N)
    μ = mean(xr; dims = (1, 2, 3))
    σ² = var(xr; dims = (1, 2, 3), corrected = false, mean = μ)
    x̂ = reshape((xr .- μ) ./ sqrt.(σ² .+ ϵ), W, H, C, N)
    return x̂ .* reshape(γ, 1, 1, C, 1) .+ reshape(β, 1, 1, C, 1)
end

# StdConv's weight standardization: statistics over each output filter.
function standardize(w, ϵ)
    μ, σ² = NormLayers.mean_var(w; dims = 1, corrected = false)
    return (w .- μ) ./ sqrt.(σ² .+ ϵ)
end
function ref_standardize(w, ϵ)
    μ = mean(w; dims = 1)
    return (w .- μ) ./ sqrt.(var(w; dims = 1, corrected = false, mean = μ) .+ ϵ)
end

# Each case: the loss through Luximm's function, the same loss through the reference, and its
# arguments. The first argument is the input; the last is the loss weight.
const rng = Xoshiro(0)
const x_ln, w_ln = randn(rng, 6, 5, 3) .+ 1.0, randn(rng, 6, 5, 3)
const γ_ln, β_ln = randn(rng, 6, 1, 1), randn(rng, 6, 1, 1)
const x_gn, w_gn = randn(rng, 4, 5, 8, 2) .+ 1.0, randn(rng, 4, 5, 8, 2)
const γ_gn, β_gn = randn(rng, 8), randn(rng, 8)
const x_sc, w_sc = randn(rng, 27, 4) .+ 0.5, randn(rng, 27, 4)

const CASES = (
    layernorm = (
        (x, γ, β, w) -> sum(NormLayers.layernorm(x, γ, β, 1, 1.0e-5) .* w),
        (x, γ, β, w) -> sum(ref_layernorm(x, γ, β, 1, 1.0e-5) .* w),
        (x_ln, γ_ln, β_ln, w_ln),
    ),
    groupnorm = (
        (x, γ, β, w) -> sum(NormLayers.groupnorm(x, γ, β, 4, 1.0e-5) .* w),
        (x, γ, β, w) -> sum(ref_groupnorm(x, γ, β, 4, 1.0e-5) .* w),
        (x_gn, γ_gn, β_gn, w_gn),
    ),
    weight_standardization = (
        (x, w) -> sum(standardize(x, 1.0e-8) .* w),
        (x, w) -> sum(ref_standardize(x, 1.0e-8) .* w),
        (x_sc, w_sc),
    ),
)

∇ₓ(loss, x, rest...) = Enzyme.gradient(Reverse, Const(loss), x, map(Const, rest)...)[1]

# Forward over reverse.
hvp_forward_over_reverse(loss, x, v, rest...) =
    Enzyme.autodiff(Forward, Const(z -> ∇ₓ(loss, z, rest...)), Duplicated(x, v))[1]

# Reverse over reverse, the shape of a gradient penalty: ∇ₓ⟨∇ₓ loss, v⟩.
∇ₓ_dot(loss, x, v, rest...) = sum(∇ₓ(loss, x, rest...) .* v)
hvp_reverse_over_reverse(loss, x, v, rest...) = Enzyme.gradient(
    Reverse, Const(∇ₓ_dot), Const(loss), x, Const(v), map(Const, rest)...
)[2]

@testset "mean_var statistics (dims = $dims, corrected = $corrected)" for dims in
        (1, (1, 2), (1, 2, 3), 1:2, :), corrected in (false, true)

    x = randn(Xoshiro(1), Float32, 5, 6, 8, 3) .+ 2.0f0
    f = x -> NormLayers.mean_var(x; dims, corrected)
    μ, σ² = @jit f(Reactant.to_rarray(x))
    # `dims = :` reduces to scalars, which `Array` does not take; compare directly.
    @test μ ≈ mean(x; dims) atol = 1.0f-5 rtol = 1.0f-5
    @test σ² ≈ var(x; dims, corrected) atol = 1.0f-5 rtol = 1.0f-5
end

@testset "$(name) gradient" for (name, (loss, ref_loss, args)) in pairs(CASES)
    ∂ = @jit Enzyme.gradient(Reverse, Const(loss), Reactant.to_rarray.(args)...)
    for i in 1:(length(args) - 1)   # every argument but the loss weight
        ref = ForwardDiff.gradient(a -> ref_loss(Base.setindex(args, a, i)...), args[i])
        @test Array(∂[i]) ≈ ref atol = 1.0e-10 rtol = 1.0e-10
    end
end

# A variance that merely detached the mean would lose the -1/M term of its Hessian wherever the
# detach held at every derivative order; the correction keeps it exact. The reductions here are
# small (M = 6, 40, 27), which makes that term large.
@testset "$(name) Hessian-vector products" for (name, (loss, ref_loss, args)) in pairs(CASES)
    x, rest = first(args), Base.tail(args)
    v = randn(Xoshiro(2), size(x)...)
    ref = ForwardDiff.derivative(
        t -> ForwardDiff.gradient(z -> ref_loss(z, rest...), x .+ t .* v), 0.0
    )
    args_ra = Reactant.to_rarray.((x, v, rest...))
    @test Array(@jit hvp_forward_over_reverse(loss, args_ra...)) ≈ ref atol = 1.0e-10 rtol =
        1.0e-10
    @test Array(@jit hvp_reverse_over_reverse(loss, args_ra...)) ≈ ref atol = 1.0e-10 rtol =
        1.0e-10
end

# End to end: a compiled forward of two representative families (ConvNeXt V2's LayerNorm,
# BiT's GroupNorm and StdConv) matches the host forward. Random init, small input.
@testset "compiled $(v) matches the host forward" for v in
    (:convnextv2_atto_fcmae, :resnetv2_50x1_bit_goog_in21k)

    model = create_model(v; num_classes = 0)
    ps, st = Lux.setup(Xoshiro(0), model)
    st = Lux.testmode(st)
    x = randn(Xoshiro(3), Float32, 64, 64, 3, 2)
    y_host, _ = model(x, ps, st)
    y_ra, _ = @jit model(Reactant.to_rarray(x), Reactant.to_rarray(ps), Reactant.to_rarray(st))
    @test size(y_ra) == size(y_host)
    @test norm(Array(y_ra) .- y_host) / norm(y_host) < 1.0f-4
end
