module PathFilter

export families_for_paths, ALL_FAMILIES, REPRESENTATIVE_VARIANT

const _SHARED_PREFIXES =
    ("src/Layers/", "src/Interop/", "src/Models/ConvNeXtCommon/", "ci/")

const _SHARED_EXACT = Set([
    "src/Luximm.jl",
    "src/Models/Models.jl",
    "src/Models/FeatureInfo.jl",
    "Project.toml",
    "Manifest.toml",
    "test/runtests.jl",
    "test/_ci_driver.jl",
    "test/_filter.jl",
    "test/_parity_helpers.jl",
    "test/_parity_tol.jl",
])

const _FAMILY_PREFIXES = Dict{String,Tuple{Vararg{String}}}(
    "bit" => ("src/Models/ResNetV2/",),
    "resnet" => ("src/Models/ResNet/",),
    "convnext" => ("src/Models/ConvNeXt/",),
    "convnextv2" => ("src/Models/ConvNeXtV2/",),
    "vgg" => ("src/Models/VGG/",),
    "seresnet" => ("src/Models/SEResNet/",),
    "vit" => ("src/Models/ViT/",),
    "coatnet" => ("src/Models/CoAtNet/",),
    "infra" => (),
)

const _FAMILY_EXACT = Dict{String,Set{String}}(
    "bit" => Set(["test/test_bit_resnet.jl"]),
    "resnet" => Set(["test/test_resnet.jl"]),
    "convnext" => Set(["test/test_convnext.jl"]),
    "convnextv2" => Set(["test/test_convnextv2.jl"]),
    "vgg" => Set(["test/test_vgg.jl"]),
    "seresnet" => Set(["test/test_seresnet.jl"]),
    "vit" => Set(["test/test_vit.jl"]),
    "coatnet" => Set(["test/test_coatnet.jl"]),
    "infra" => Set([
        "test/test_hf_download.jl",
        "test/test_hf_hub_download.jl",
        "test/test_init.jl",
    ]),
)

const ALL_FAMILIES = (
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

const REPRESENTATIVE_VARIANT = Dict{String,String}(
    "infra" => "",
    "bit" => "resnetv2_50x1_bit_goog_in21k_ft_in1k",
    "resnet" => "resnet50_a1_in1k",
    "convnext" => "convnext_tiny_fb_in1k",
    "convnextv2" => "convnextv2_atto_fcmae_ft_in1k",
    "vgg" => "vgg16_tv_in1k",
    "seresnet" => "seresnet50_a1_in1k",
    # Two representatives: the ViT family now spans two architectural flavors —
    # timm's ImageNet ViTs (`pre_norm = false`, eps 1e-6, biased stem) and the
    # CLIP image towers (`pre_norm = true`, eps 1e-5, bias-free stem). A single
    # representative would leave one flavor untested on PR-scope runs; the
    # builder splits this on commas and dumps/runs both.
    "vit" => "vit_base_patch16_224_augreg2_in21k_ft_in1k," *
        "vit_base_patch32_clip_224_openai_ft_in1k",
    "coatnet" => "coatnet_0_rw_224_sw_in1k",
)

_is_shared(path::AbstractString) =
    path in _SHARED_EXACT || any(startswith(path, p) for p in _SHARED_PREFIXES)

function _family_for(path::AbstractString)
    for fam in keys(_FAMILY_PREFIXES)
        if any(startswith(path, p) for p in _FAMILY_PREFIXES[fam])
            return fam
        end
        if path in _FAMILY_EXACT[fam]
            return fam
        end
    end
    return nothing
end

"""
    families_for_paths(paths) -> Vector{String}

Map a list of changed file paths to the set of Luximm test families they
affect, returned in canonical order. Returns an empty vector when no
recognized file changed.
"""
function families_for_paths(paths)
    paths = collect(paths)
    if any(_is_shared(p) for p in paths)
        return collect(ALL_FAMILIES)
    end
    touched = Set{String}()
    for p in paths
        fam = _family_for(p)
        fam === nothing || push!(touched, fam)
    end
    return [f for f in ALL_FAMILIES if f in touched]
end

end # module
