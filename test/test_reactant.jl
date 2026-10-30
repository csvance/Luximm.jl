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

# Run on Reactant's CPU client only. Without this, Reactant's first array call probes CUDA as
# well, and on a machine with an NVIDIA driver but no visible GPU it logs a long, harmless
# "Failed to initialize client: cuda" error before falling back to the CPU.
Reactant.set_default_backend("cpu")

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

# The layer forwards are `@trace function`s, so Reactant traces each one once per compile and reuses
# it through a cached `func.call`. The cache key comes from `make_tracer`, which LuxCore's Reactant
# extension makes skip every AbstractLuxLayer; without the extension's method for Luximm's layers,
# two layers of one type with different settings on same-shaped inputs would share one traced call,
# and the second would silently return the first one's result. The input's channels have different
# means and scales, because on a normalized input 1 and 32 groups agree and the check is vacuous.
const x_mixed = randn(Xoshiro(4), Float32, 8, 8, 64, 2) .* reshape(1:64, 1, 1, :, 1) .+
    reshape(1:64, 1, 1, :, 1)

# Both layers applied, in order, to the same input and parameters inside one compiled function.
function jit_pair(l1, l2, x, ps, st)
    f(x, ps) = (first(l1(x, ps, st)), first(l2(x, ps, st)))
    return Array.(@jit f(Reactant.to_rarray(x), Reactant.to_rarray(ps)))
end

@testset "$(name): two settings in one compile" for (name, l1, l2) in (
        ("LuximmGroupNorm groups", NormLayers.LuximmGroupNorm(64, 32), NormLayers.LuximmGroupNorm(64, 1)),
        (
            "LuximmGroupNorm epsilon",
            NormLayers.LuximmGroupNorm(64, 32; epsilon = 1.0f-5),
            NormLayers.LuximmGroupNorm(64, 32; epsilon = 10.0f0),
        ),
        (
            "LuximmLayerNorm epsilon",
            NormLayers.LuximmLayerNorm((8, 1, 1), 1; epsilon = 1.0f-5),
            NormLayers.LuximmLayerNorm((8, 1, 1), 1; epsilon = 10.0f0),
        ),
        ("LuximmLayerNorm dims", NormLayers.LuximmLayerNorm((8, 1, 1), 1), NormLayers.LuximmLayerNorm((8, 1, 1), (1, 2))),
    )

    ps, st = Lux.setup(Xoshiro(0), l1)
    y1, y2 = jit_pair(l1, l2, x_mixed, ps, st)
    @test y1 ≈ first(l1(x_mixed, ps, st)) atol = 1.0f-4 rtol = 1.0f-4
    @test y2 ≈ first(l2(x_mixed, ps, st)) atol = 1.0f-4 rtol = 1.0f-4
    @test maximum(abs, y1 .- y2) > 1.0f-2   # the input tells the two settings apart
end

# The same check on Lux's own GroupNorm, which LuxCore leaves out of the key: the negative control
# for the testset above, showing that it detects the bug. When Lux ships a fix this reports an
# Unexpected Pass; then raise the LuxCore lower bound and consider dropping the extension's
# `make_tracer` method for Luximm's layers.
@testset "Lux.GroupNorm: two settings in one compile (upstream bug)" begin
    l1, l2 = Lux.GroupNorm(64, 32), Lux.GroupNorm(64, 1)
    ps, st = Lux.setup(Xoshiro(0), l1)
    _, y2 = jit_pair(l1, l2, x_mixed, ps, st)
    @test_broken isapprox(y2, first(l2(x_mixed, ps, st)); atol = 1.0f-4, rtol = 1.0f-4)
end

# The key separates settings without defeating the cache: identical layers share one traced
# function, different ones get one each. Counted in the unoptimized module, where the calls are
# not yet inlined.
@testset "traced calls are shared between identical layers" begin
    function n_traced(l1, l2)
        ps, st = Lux.setup(Xoshiro(0), l1)
        f(x, ps) = (first(l1(x, ps, st)), first(l2(x, ps, st)))
        hlo = string(
            @code_hlo optimize = false f(Reactant.to_rarray(x_mixed), Reactant.to_rarray(ps))
        )
        return count(l -> occursin("func.func", l) && occursin("LuximmGroupNorm", l), split(hlo, '\n'))
    end
    @test n_traced(NormLayers.LuximmGroupNorm(64, 32), NormLayers.LuximmGroupNorm(64, 32)) == 1
    @test n_traced(NormLayers.LuximmGroupNorm(64, 32), NormLayers.LuximmGroupNorm(64, 1)) == 2
end

# Gradients and a forward-over-reverse Hessian-vector product through the traced layer call, which
# now holds the extension's statistics and their `ignore_derivatives`. Each case: the layer, the
# reference forward of `(x, scale, bias)`, the input, the parameters and the loss weight.
const LAYER_CASES = (
    LuximmLayerNorm = (
        NormLayers.LuximmLayerNorm((6,), 1; epsilon = 1.0e-5),
        (x, γ, β) -> ref_layernorm(x, γ, β, 1, 1.0e-5),
        x_ln, (; bias = β_ln, scale = γ_ln), w_ln,
    ),
    LuximmGroupNorm = (
        NormLayers.LuximmGroupNorm(8, 4; epsilon = 1.0e-5),
        (x, γ, β) -> ref_groupnorm(x, γ, β, 4, 1.0e-5),
        x_gn, (; scale = γ_gn, bias = β_gn), w_gn,
    ),
)

@testset "$(name) layer gradient and Hessian-vector product" for (name, (layer, ref, x, ps, w)) in
    pairs(LAYER_CASES)

    loss = (x, ps, w) -> sum(first(layer(x, ps, (;))) .* w)
    ref_loss = (x, γ, β) -> sum(ref(x, γ, β) .* w)
    x_ra, ps_ra, w_ra = Reactant.to_rarray.((x, ps, w))

    ∂x, ∂ps = @jit Enzyme.gradient(Reverse, Const(loss), x_ra, ps_ra, Const(w_ra))
    @test Array(∂x) ≈ ForwardDiff.gradient(z -> ref_loss(z, ps.scale, ps.bias), x) atol = 1.0e-10 rtol =
        1.0e-10
    @test Array(∂ps.scale) ≈ ForwardDiff.gradient(γ -> ref_loss(x, γ, ps.bias), ps.scale) atol =
        1.0e-10 rtol = 1.0e-10
    @test Array(∂ps.bias) ≈ ForwardDiff.gradient(β -> ref_loss(x, ps.scale, β), ps.bias) atol =
        1.0e-10 rtol = 1.0e-10

    v = randn(Xoshiro(2), size(x)...)
    hvp_ref = ForwardDiff.derivative(
        t -> ForwardDiff.gradient(z -> ref_loss(z, ps.scale, ps.bias), x .+ t .* v), 0.0
    )
    @test Array(@jit hvp_forward_over_reverse(loss, x_ra, Reactant.to_rarray(v), ps_ra, w_ra)) ≈
        hvp_ref atol = 1.0e-10 rtol = 1.0e-10
end
