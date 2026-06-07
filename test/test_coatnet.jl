# CoAtNet parity tests against timm reference outputs.
#
# Each test reads a Python-dumped HDF5 fixture (data/parity/<variant>_io.h5),
# downloads the variant's model.safetensors from HuggingFace (cached on disk),
# builds the Luximm model, applies the weights, and asserts the forward output
# matches timm within tolerance. Skipped if the fixture is missing or
# HF_OFFLINE=1.

using Test
using Luximm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

const VARIANTS_TO_TEST = Tuple(sort(collect(keys(Luximm.COATNET_VARIANTS))))

@testset "CoAtNet parity" begin
    for variant in variant_filter(VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = load_parity_fixture(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(parity_fixture_path(variant))"
                continue
            end
            if hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end
            run_variant_parity(variant, fixture)
        end
    end
end
