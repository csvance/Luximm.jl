# BiT ResNetV2 parity tests against timm reference outputs.
#
# Each test:
#   1. Reads a Python-dumped HDF5 fixture (data/parity/resnetv2_<key>_bit_io.h5)
#      containing the deterministic input and timm's forward_features /
#      forward outputs for that variant.
#   2. Downloads the variant's model.safetensors from HuggingFace (cached on
#      disk so reruns are fast and offline).
#   3. Builds the Luximm model, applies the weights, and asserts max-abs-diff
#      against the timm reference is under `LOGITS_ATOL`.
#
# Skipped if the fixture file is missing or HF_OFFLINE=1 is set in env (so
# CI without network or fixtures can still pass the rest of the suite).

using Test
using Luximm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

# Test every registered BiT variant. Each entry is gated on its fixture
# file existing under data/parity/, so machines without the full set of
# dumps simply skip the missing variants. Deriving the list from
# BIT_VARIANTS keeps it in sync as new variants land in
# src/Models/ResNetV2/Config.jl without a second edit here.
const VARIANTS_TO_TEST = Tuple(sort(collect(keys(Luximm.BIT_VARIANTS))))

@testset "BiT parity" begin
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
