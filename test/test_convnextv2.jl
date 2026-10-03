# ConvNeXtV2 parity tests against timm reference outputs.
#
# Each test:
#   1. Reads a Python-dumped HDF5 fixture
#      (data/parity/<variant>_io.h5) containing the deterministic input and
#      timm's forward_features (and logits, for variants with a trained head).
#   2. Downloads the variant's model.safetensors from HuggingFace (cached on
#      disk so reruns are fast and offline).
#   3. Builds the Luximm model, applies the weights, and asserts max-abs-diff
#      against the timm reference is under `LOGITS_ATOL`.
#
# Skipped if the fixture file is missing or HF_OFFLINE=1 is set (so CI without
# network or fixtures can still pass the rest of the suite).

using Test
using Luximm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

# Test every registered ConvNeXtV2 variant. Each entry is gated on its
# fixture file existing under data/parity/, so machines without the full
# set of dumps simply skip the missing variants. Deriving the list from
# CONVNEXTV2_VARIANTS keeps it in sync as new variants land in
# src/Models/ConvNeXtV2/Config.jl without a second edit here.
const VARIANTS_TO_TEST = Tuple(sort(collect(keys(Luximm.CONVNEXTV2_VARIANTS))))

@testset "ConvNeXtV2 parity" begin
    for variant in variant_filter(VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = load_parity_fixture(variant)
            if hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end
            # Gated on its own fixture, so it still runs when the
            # forward_features fixture is absent.
            run_variant_feature_pyramid_parity(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(parity_fixture_path(variant))"
                continue
            end
            run_variant_parity(variant, fixture)
        end
    end
end

# `conv_mlp = false`: timm's channels-last block layout (LayerNorm over a contiguous channel axis,
# GEMM pointwise layers). No fixtures or downloads: it is checked against the default layout,
# which the parity tests above pin to timm.
@testset "ConvNeXtV2 conv_mlp = false" begin
    v = :convnextv2_tiny_fcmae_ft_in22k_in1k
    a = Luximm.create_model(v; num_classes = 0)
    b = Luximm.create_model(v; num_classes = 0, conv_mlp = false)
    ps_a, st_a = Lux.setup(Xoshiro(0), a)
    ps_b, st_b = Lux.setup(Xoshiro(0), b)
    @test ps_a == ps_b                     # names, shapes and values: weights load unchanged
    @test st_a == st_b
    # GRN is zero-initialized (identity); give it non-trivial parameters so its path is exercised.
    ps = Lux.Functors.fmap_with_path(ps_a) do kp, x
        any(==(:grn), kp) ? randn(Xoshiro(hash(kp)), Float32, size(x)) .* 0.1f0 : x
    end
    x = randn(Xoshiro(1), Float32, 64, 64, 3, 2)
    y_a, _ = a(x, ps, st_a)
    y_b, _ = b(x, ps, st_b)
    @test size(y_a) == size(y_b)
    @test isapprox(y_b, y_a; rtol = 1.0f-4, atol = 1.0f-5)
end
