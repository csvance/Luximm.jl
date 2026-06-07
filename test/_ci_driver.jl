# Multi-family test driver — runs every family selected by JIMM_TEST_FAMILIES
# inside a single Julia process so JIT/compile caches are reused across
# backbones that share layers.
#
# This is invoked from two paths:
#
#   1. `Pkg.test()` (local dev): `runtests.jl` is a one-line shim that
#      includes this file. `JIMM_TEST_FAMILIES` / `JIMM_TEST_VARIANTS`
#      gate which families run.
#
#   2. The CI builder (`ci/JimmCI/src/Builder.jl`): bypasses Pkg.test
#      entirely and invokes Julia with
#      `julia --project=. -e 'include("test/_ci_driver.jl")'`, then watches
#      stdout for the structured markers below to drive per-family GitHub
#      check_runs.
#
# Marker contract (kept stable; the CI builder greps for these literals):
#
#   ==> JIMM_FAMILY_BEGIN: family=<name>
#   ==> JIMM_FAMILY_END:   family=<name> rc=<0|1>
#
# `rc=0` iff every `@test` in the family passed and the test file ran to
# completion without throwing. Any uncaught error during `include` is
# reported as `rc=1` and the next family still runs.

using Test
using Luximm

isdefined(@__MODULE__, :family_enabled) || include("_filter.jl")

# Canonical run order. Matches PathFilter.ALL_FAMILIES so the CI's check_run
# list shows up in the same sequence regardless of how families were passed
# in via JIMM_TEST_FAMILIES.
const _DRIVER_ORDER = (
    "infra",
    "bit",
    "resnet",
    "convnext",
    "convnextv2",
    "vgg",
    "seresnet",
    "vit",
    "coatnet",
)

function _scaffold_testset()
    @testset "Luximm scaffold" begin
        @test isdefined(Luximm, :Interop)
        @test isdefined(Luximm, :Layers)
        @test isdefined(Luximm, :Models)
        @test isdefined(Luximm.Interop, :read_parity)
        @test isdefined(Luximm.Interop, :apply_state_dict)
        @test isdefined(Luximm.Interop, :hf_download)
        @test isdefined(Luximm.Interop, :hf_hub_download)
        @test isdefined(Luximm.Interop, :hf_hub_cache_dir)
        @test isdefined(Luximm.Interop, :load_safetensors_state_dict)
        @test isdefined(Luximm.Interop, :adapt_input_conv)
        @test isdefined(Luximm.Layers, :std_conv)
        @test isdefined(Luximm.Layers, :layernorm2d)
        @test isdefined(Luximm.Layers, :grn_layer)
        @test isdefined(Luximm.Models, :bit_resnetv2)
        @test isdefined(Luximm.Models, :bit_resnetv2_mapping)
        @test isdefined(Luximm.Models, :_load_bit_resnetv2)
        @test isdefined(Luximm.Models, :BIT_VARIANTS)
        @test isdefined(Luximm.Models, :resnet)
        @test isdefined(Luximm.Models, :resnet_mapping)
        @test isdefined(Luximm.Models, :resnet_state_mapping)
        @test isdefined(Luximm.Models, :_load_resnet)
        @test isdefined(Luximm.Models, :RESNET_VARIANTS)
        @test isdefined(Luximm.Models, :convnextv2)
        @test isdefined(Luximm.Models, :convnextv2_mapping)
        @test isdefined(Luximm.Models, :_load_convnextv2)
        @test isdefined(Luximm.Models, :CONVNEXTV2_VARIANTS)
        @test isdefined(Luximm.Models, :convnext)
        @test isdefined(Luximm.Models, :convnext_mapping)
        @test isdefined(Luximm.Models, :_load_convnext)
        @test isdefined(Luximm.Models, :CONVNEXT_VARIANTS)
        @test isdefined(Luximm.Models, :vgg)
        @test isdefined(Luximm.Models, :vgg_mapping)
        @test isdefined(Luximm.Models, :_load_vgg)
        @test isdefined(Luximm.Models, :VGG_VARIANTS)
        @test isdefined(Luximm.Models, :seresnet)
        @test isdefined(Luximm.Models, :seresnet_mapping)
        @test isdefined(Luximm.Models, :seresnet_state_mapping)
        @test isdefined(Luximm.Models, :_load_seresnet)
        @test isdefined(Luximm.Models, :SERESNET_VARIANTS)
        @test isdefined(Luximm.Layers, :se_block)
        @test isdefined(Luximm.Models, :vit)
        @test isdefined(Luximm.Models, :vit_mapping)
        @test isdefined(Luximm.Models, :_load_vit)
        @test isdefined(Luximm.Models, :VIT_VARIANTS)
        @test isdefined(Luximm.Layers, :patch_embed)
        @test isdefined(Luximm.Layers, :mhsa)
        @test isdefined(Luximm.Layers, :vit_block)
        @test isdefined(Luximm.Models, :coatnet)
        @test isdefined(Luximm.Models, :coatnet_mapping)
        @test isdefined(Luximm.Models, :coatnet_state_mapping)
        @test isdefined(Luximm.Models, :_load_coatnet)
        @test isdefined(Luximm.Models, :COATNET_VARIANTS)
        @test isdefined(Luximm.Layers, :rel_pos_attention)
        @test isdefined(Luximm.Models, :create_model)
        @test isdefined(Luximm.Models, :create_pretrained)
        @test isdefined(Luximm.Models, :default_num_classes)
        @test isdefined(Luximm.Models, :_load_pretrained)
    end
end

function _dispatch_family(name::AbstractString)
    if name == "infra"
        _scaffold_testset()
        include(joinpath(@__DIR__, "test_hf_download.jl"))
        include(joinpath(@__DIR__, "test_hf_hub_download.jl"))
        include(joinpath(@__DIR__, "test_init.jl"))
    elseif name == "bit"
        include(joinpath(@__DIR__, "test_bit_resnet.jl"))
    elseif name == "resnet"
        include(joinpath(@__DIR__, "test_resnet.jl"))
    elseif name == "convnext"
        include(joinpath(@__DIR__, "test_convnext.jl"))
    elseif name == "convnextv2"
        include(joinpath(@__DIR__, "test_convnextv2.jl"))
    elseif name == "vgg"
        include(joinpath(@__DIR__, "test_vgg.jl"))
    elseif name == "seresnet"
        include(joinpath(@__DIR__, "test_seresnet.jl"))
    elseif name == "vit"
        include(joinpath(@__DIR__, "test_vit.jl"))
    elseif name == "coatnet"
        include(joinpath(@__DIR__, "test_coatnet.jl"))
    else
        error("unknown family in JIMM_TEST_FAMILIES: $(name)")
    end
end

# Run one family inside its own outer `@testset`. `@testset` rethrows
# `TestSetException` at end-of-block if any inner `@test` failed; anything
# else escaping `_dispatch_family` is a service-level error (e.g. uncaught
# exception during include). Both map to rc=1.
function _run_family(name::AbstractString)
    failed = false
    try
        @testset "$(name)" begin
            _dispatch_family(name)
        end
    catch e
        failed = true
        if !(e isa Test.TestSetException)
            println(stderr, "ERROR in family $(name): ", sprint(showerror, e))
            Base.show_backtrace(stderr, catch_backtrace())
            println(stderr)
        end
    end
    return failed ? 1 : 0
end

function _driver_main()
    requested = families_enabled()
    ordered = [f for f in _DRIVER_ORDER if f in requested]
    overall = 0
    for fam in ordered
        println(stdout, "==> JIMM_FAMILY_BEGIN: family=$(fam)");
        flush(stdout)
        rc = _run_family(fam)
        rc == 0 || (overall = 1)
        println(stdout, "==> JIMM_FAMILY_END: family=$(fam) rc=$(rc)");
        flush(stdout)
    end
    return overall
end

exit(_driver_main())
