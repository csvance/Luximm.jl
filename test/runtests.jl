# Pkg.test() entry point. Runs every family selected by JIMM_TEST_FAMILIES /
# JIMM_TEST_VARIANTS (see `_filter.jl`) in one process, so compiled code is
# shared across backbones that use the same layers.

using Test
using Luximm

include("_filter.jl")

let unknown = setdiff(families_enabled(), _JIMM_DEFAULT_FAMILIES)
    isempty(unknown) || error(
        "unknown family in JIMM_TEST_FAMILIES: $(join(sort(collect(unknown)), ", ")); " *
            "known: $(join(_JIMM_DEFAULT_FAMILIES, ", "))",
    )
end

@testset "Luximm" begin
    for (family, files) in FAMILY_FILES
        family_enabled(family) || continue
        @testset "$(family)" begin
            for file in files
                include(joinpath(@__DIR__, file))
            end
        end
    end
end
