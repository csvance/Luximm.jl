# Shared scaffolding for the per-family parity test files
# (test_convnextv2.jl, test_convnext.jl, test_resnet.jl, test_bit_resnet.jl).
#
# Each family file is a thin shell: a variant list, an outer @testset for the
# family name, and a loop that loads the fixture, skips when missing or
# HF_OFFLINE=1, and then delegates the actual forward/load/assert work to
# run_variant_parity below.
#
# Include with:
#   isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")

using HDF5: HDF5

parity_fixture_path(variant::Symbol; in_chans::Int = 3) = begin
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    base = get(ENV, "JIMM_PARITY_DIR", joinpath(@__DIR__, "..", "data", "parity"))
    joinpath(base, "$(variant)$(suffix)_io.h5")
end

# HDF5 fixtures store PyTorch-layout tensors; read_parity reverses axes to
# Lux-natural (W, H, C, N) for features and (K, N) for logits, which matches
# every family's Luximm output layout.
function load_parity_fixture(variant::Symbol; in_chans::Int = 3)
    path = parity_fixture_path(variant; in_chans = in_chans)
    isfile(path) || return nothing
    return Luximm.Interop.read_parity(path)
end

hf_offline() = get(ENV, "HF_OFFLINE", "") == "1"

# -- features_only / feature-pyramid parity -------------------------------

featsonly_fixture_path(variant::Symbol) = begin
    base = get(ENV, "JIMM_PARITY_DIR", joinpath(@__DIR__, "..", "data", "parity"))
    joinpath(base, "$(variant)_featsonly_io.h5")
end

# Fixtures come from test/parity/dump_features_only_io.py, which writes the
# usual /input + /output/feat_NN groups plus timm's own tap table under
# /feature_channels and /feature_reductions.
function load_featsonly_fixture(variant::Symbol)
    path = featsonly_fixture_path(variant)
    isfile(path) || return nothing
    parity = Luximm.Interop.read_parity(path)
    channels, reductions = HDF5.h5open(path, "r") do f
        (Int.(read(f["feature_channels"])), Int.(read(f["feature_reductions"])))
    end
    expected = [parity.output[k] for k in sort(collect(keys(parity.output)))]
    return (; input = parity.input, expected = expected, channels, reductions)
end

# Parity for `features_only = true`: every tap must match timm's
# corresponding `features_only=True` output, and the declared tap table must
# match timm's `feature_info`. Skips when the fixture is missing, like the
# other parity paths.
#
# Also asserts the invariant the whole design rests on: an `out_indices`
# subset shares the parameter tree with the full pyramid, so the *same*
# loaded `(ps, st)` drives both and returns the very same tensors.
function run_variant_feature_pyramid_parity(variant::Symbol)
    fixture = load_featsonly_fixture(variant)
    if fixture === nothing
        @info "skipping $(variant) feature pyramid: fixture missing at " *
              featsonly_fixture_path(variant)
        return nothing
    end

    @testset "features_only" begin
        info = feature_info(variant)
        @test collect(info.channels) == fixture.channels
        @test collect(info.reductions) == fixture.reductions
        @test length(info) == length(fixture.expected)

        model, load = create_pretrained(variant; features_only = true)
        ps, st = Lux.setup(Xoshiro(0), model)
        st = Lux.testmode(st)
        ps, st = load(ps, st)
        feats, _ = model(fixture.input, ps, st)

        @test feats isa Tuple
        @test length(feats) == length(fixture.expected)
        for i in eachindex(fixture.expected)
            expected = fixture.expected[i]
            @test size(feats[i]) == size(expected)
            diff = maximum(abs.(feats[i] .- expected))
            ref_scale = max(maximum(abs.(expected)), eps(Float32))
            rel = diff / ref_scale
            @info "$(variant) tap $i ($(info.names[i]), reduction " *
                  "$(info.reductions[i])) max-abs-diff = $diff, rel = $rel"
            @test rel < FEATURES_RTOL
        end

        # Dropping the finest tap must not disturb the tree or the tensors.
        sub_indices = Tuple(2:length(info))
        sub_model = create_model(
            variant;
            features_only = true,
            num_classes = 0,
            out_indices = sub_indices,
        )
        sub_feats, _ = sub_model(fixture.input, ps, st)
        @test length(sub_feats) == length(sub_indices)
        # Bitwise `==` is deliberate: the subset re-runs the identical
        # arithmetic on the identical input, so the tensors should be equal to
        # the last bit, not merely close. Describe any mismatch before
        # asserting, so a failing log says whether it is a last-bit wobble or a
        # genuinely wrong tap. The assertion itself is on a `Bool` rather than
        # on `got == ref`, so a failure prints `false` instead of dumping both
        # feature maps into the log.
        mismatched = false
        for i in eachindex(sub_indices)
            ref = feats[sub_indices[i]]
            got = sub_feats[i]
            tap_equal = size(got) == size(ref) && got == ref
            if size(got) != size(ref)
                @info "$(variant) subset tap $i: size $(size(got)) != full tap " *
                      "$(sub_indices[i]) size $(size(ref))"
            elseif !tap_equal
                d = maximum(abs.(got .- ref))
                scale = maximum(abs.(ref))
                @info "$(variant) subset tap $i (full tap $(sub_indices[i])): " *
                      "$(count(got .!= ref)) of $(length(ref)) elements differ, " *
                      "max-abs-diff = $d, ref scale = $scale, " *
                      "rel = $(d / max(scale, eps(Float32)))"
            end
            mismatched |= !tap_equal
            @test tap_equal
        end

        # Only when the above already failed, and only once: is the full model
        # reproducible on this machine at all? If a second identical pass
        # disagrees with the first, the mismatch is BLAS/thread nondeterminism
        # rather than anything `out_indices` did.
        if mismatched
            rerun, _ = model(fixture.input, ps, st)
            stable = true
            for i in eachindex(feats)
                rerun[i] == feats[i] && continue
                stable = false
                @info "$(variant) rerun control tap $i: " *
                      "$(count(rerun[i] .!= feats[i])) of $(length(feats[i])) " *
                      "elements differ between two identical full passes, " *
                      "max-abs-diff = $(maximum(abs.(rerun[i] .- feats[i])))"
            end
            stable && @info "$(variant) rerun control: two identical full passes " *
                  "agree bitwise, so the subset mismatch is not run-to-run noise"
        end

        sub_info = feature_info(variant; out_indices = sub_indices)
        @test sub_info.indices == sub_indices
        @test sub_info.channels == map(i -> info.channels[i], sub_indices)
        @test sub_info.reductions == map(i -> info.reductions[i], sub_indices)
    end
    return nothing
end

# Runs the three parity sub-tests for one variant: forward_features,
# forward (logits) when the fixture ships them, and forward_features at
# in_chans=1 when a 1-channel fixture is available. `fixture` is the
# 3-channel fixture already loaded by the caller.
#
# The logits sub-test keys on `haskey(fixture.output, "logits")` alone:
# the timm dumper only writes that key for variants with a trained head,
# which matches the `default_num_classes(variant) > 0` rule the per-family
# files used previously.
function run_variant_parity(variant::Symbol, fixture)
    x = fixture.input
    expected_features = fixture.output["features"]

    @testset "forward_features" begin
        model, load = create_pretrained(variant; num_classes = 0)
        ps, st = Lux.setup(Xoshiro(0), model)
        st = Lux.testmode(st)
        ps, st = load(ps, st)
        y, _ = model(x, ps, st)
        @test size(y) == size(expected_features)
        diff = maximum(abs.(y .- expected_features))
        ref_scale = max(maximum(abs.(expected_features)), eps(Float32))
        rel = diff / ref_scale
        @info "$(variant) features max-abs-diff = $diff, rel = $rel"
        @test rel < FEATURES_RTOL
    end

    if haskey(fixture.output, "logits")
        expected_logits = fixture.output["logits"]
        @testset "forward (logits)" begin
            model, load = create_pretrained(variant)
            ps, st = Lux.setup(Xoshiro(0), model)
            st = Lux.testmode(st)
            ps, st = load(ps, st)
            y, _ = model(x, ps, st)
            @test size(y) == size(expected_logits)
            diff = maximum(abs.(y .- expected_logits))
            @info "$(variant) logits max-abs-diff = $diff"
            @test diff < LOGITS_ATOL
        end
    end

    fixture_in1c = load_parity_fixture(variant; in_chans = 1)
    if fixture_in1c === nothing
        @info "skipping $(variant) in_chans=1: fixture missing at " *
              parity_fixture_path(variant; in_chans = 1)
    else
        @testset "forward_features (in_chans=1)" begin
            x1 = fixture_in1c.input
            expected1 = fixture_in1c.output["features"]
            model, load = create_pretrained(variant; in_chans = 1, num_classes = 0)
            ps, st = Lux.setup(Xoshiro(0), model)
            st = Lux.testmode(st)
            ps, st = load(ps, st)
            y, _ = model(x1, ps, st)
            @test size(y) == size(expected1)
            diff = maximum(abs.(y .- expected1))
            ref_scale = max(maximum(abs.(expected1)), eps(Float32))
            rel = diff / ref_scale
            @info "$(variant) features (in_chans=1) max-abs-diff = $diff, rel = $rel"
            @test rel < FEATURES_RTOL
        end
    end
end
