# Interface behavior of the model API, with no fixtures, weights or downloads: every registered
# variant builds; one small variant per family produces the shapes and tap tables it advertises
# and treats batch elements independently; the vendored normalization layers keep Lux's
# parameter trees; and the two ConvNeXt block layouts agree. Numerical agreement with timm is
# the parity families' job.

using Test
using Luximm
using Lux
using Random

const VARIANT_TABLES = (
    Luximm.BIT_VARIANTS,
    Luximm.RESNET_VARIANTS,
    Luximm.SERESNET_VARIANTS,
    Luximm.CONVNEXT_VARIANTS,
    Luximm.CONVNEXTV2_VARIANTS,
    Luximm.VGG_VARIANTS,
    Luximm.VIT_VARIANTS,
    Luximm.COATNET_VARIANTS,
)
variant_entry(v::Symbol) = first(t[v] for t in VARIANT_TABLES if haskey(t, v))

# One small variant per family. VGG and CoAtNet have no feature pyramid.
const REPRESENTATIVES = (
    :resnet18_a1_in1k,
    :seresnet50_a1_in1k,
    :resnetv2_50x1_bit_goog_in21k_ft_in1k,
    :convnext_tiny_fb_in1k,
    :convnextv2_atto_fcmae_ft_in1k,
    :vgg11_tv_in1k,
    :vit_base_patch32_clip_224_openai_ft_in1k,
    :coatnet_0_rw_224_sw_in1k,
)
const NO_PYRAMID = (:vgg11_tv_in1k, :coatnet_0_rw_224_sw_in1k)

function forward(model, x; rng = Xoshiro(0))
    ps, st = Lux.setup(rng, model)
    return first(model(x, ps, Lux.testmode(st)))
end

@testset "vendored norm layers keep Lux's parameter trees" begin
    cases = (
        (layernorm2d(8), Lux.LayerNorm((1, 1, 8); dims = 3, epsilon = 1.0f-6), (5, 4, 8, 2)),
        (vit_layernorm(8), Lux.LayerNorm((8, 1); dims = 1, epsilon = 1.0f-6), (8, 7, 2)),
        (
            Luximm.Layers.LuximmGroupNorm(8, 4; epsilon = 1.0f-5),
            Lux.GroupNorm(8, 4; affine = true, epsilon = 1.0f-5),
            (5, 4, 8, 2),
        ),
    )
    for (ours, lux, xsize) in cases
        ps, st = Lux.setup(Xoshiro(0), ours)
        ps_lux, st_lux = Lux.setup(Xoshiro(0), lux)
        @test ps == ps_lux                  # names, field order, shapes and values
        @test st == st_lux
        ps = map(p -> randn(Xoshiro(1), Float32, size(p)), ps)
        x = randn(Xoshiro(2), Float32, xsize)
        @test first(ours(x, ps, st)) == first(lux(x, ps, st_lux))   # the same LuxLib call
    end
end

@testset "every registered variant builds" begin
    for table in VARIANT_TABLES, v in keys(table)
        @test create_model(v) isa Lux.AbstractLuxLayer
        @test default_num_classes(v) == table[v].default_num_classes
    end
    @test_throws ErrorException create_model(:not_a_variant)
    @test_throws ErrorException create_model(
        :resnet18_a1_in1k; features_only = true, num_classes = 10
    )
end

@testset "$(v)" for v in REPRESENTATIVES
    s = variant_entry(v).default_input_size
    x = randn(Xoshiro(1), Float32, s, s, 3, 2)

    @testset "classifier head" begin
        model = create_model(v)
        ps, st = Lux.setup(Xoshiro(0), model)
        st = Lux.testmode(st)
        y = first(model(x, ps, st))
        @test size(y) == (default_num_classes(v), 2)
        # Each sample's output must not depend on the rest of the batch.
        y2 = first(model(x[:, :, :, 2:2], ps, st))
        @test isapprox(y2, y[:, 2:2]; rtol = 1.0f-4)
    end

    @testset "num_classes = 0 and in_chans = 1" begin
        f3 = forward(create_model(v; num_classes = 0), x)
        f1 = forward(create_model(v; num_classes = 0, in_chans = 1), x[:, :, 1:1, :])
        @test size(f3, ndims(f3)) == 2
        @test size(f1) == size(f3)
    end

    if v in NO_PYRAMID
        @test_throws ErrorException create_model(v; features_only = true)
    else
        @testset "features_only matches feature_info" begin
            info = feature_info(v)
            feats = forward(create_model(v; features_only = true), x)
            @test feats isa Tuple
            @test length(feats) == length(info)
            for (i, f) in enumerate(feats)
                @test size(f) == (s ÷ info.reductions[i], s ÷ info.reductions[i], info.channels[i], 2)
            end
        end
    end
end

# The default ConvNeXt block layout (timm's `conv_mlp = False`: LayerNorm over a contiguous
# channel axis, GEMM pointwise layers) against the 1x1-convolution form (`conv_mlp = true`). The
# parity families pin the default to timm; this pins the two layouts to each other.
@testset "ConvNeXt conv_mlp = true matches the default layout" begin
    v = :convnext_tiny_dinov3_lvd1689m
    a = Luximm.create_model(v; num_classes = 0, conv_mlp = true)
    b = Luximm.create_model(v; num_classes = 0)
    ps_a, st_a = Lux.setup(Xoshiro(0), a)
    ps_b, st_b = Lux.setup(Xoshiro(0), b)
    @test ps_a == ps_b                     # names, shapes and values: weights load unchanged
    @test st_a == st_b
    # LayerScale is initialized to 1e-6, which would hide the block behind the residual; give it
    # non-trivial values so the block's own output is compared.
    ps = Lux.Functors.fmap_with_path(ps_a) do kp, x
        last(kp) == :gamma ? randn(Xoshiro(hash(kp)), Float32, size(x)) : x
    end
    x = randn(Xoshiro(1), Float32, 64, 64, 3, 2)
    y_a, _ = a(x, ps, st_a)
    y_b, _ = b(x, ps, st_b)
    @test size(y_a) == size(y_b)
    @test isapprox(y_b, y_a; rtol = 1.0f-4, atol = 1.0f-5)
end

@testset "ConvNeXtV2 conv_mlp = true matches the default layout" begin
    v = :convnextv2_tiny_fcmae_ft_in22k_in1k
    a = Luximm.create_model(v; num_classes = 0, conv_mlp = true)
    b = Luximm.create_model(v; num_classes = 0)
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
