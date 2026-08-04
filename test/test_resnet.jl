# Classic ResNet parity tests against timm reference outputs.

using Test
using Luximm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

# Test every registered ResNet variant. Each entry is gated on its
# fixture file existing under data/parity/, so machines without the full
# set of dumps simply skip the missing variants. Deriving the list from
# RESNET_VARIANTS keeps it in sync as new variants land in
# src/Models/ResNet/Config.jl without a second edit here.
const RESNET_VARIANTS_TO_TEST = Tuple(sort(collect(keys(Luximm.RESNET_VARIANTS))))

@testset "ResNet parity" begin
    for variant in variant_filter(RESNET_VARIANTS_TO_TEST)
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
