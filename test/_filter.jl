# Test-subset gating shared by every test file.
#
# Families come in two kinds:
#
#   unit    `infra`, `interface`, `reactant`: small, self-contained checks with
#           no parity fixtures and no pretrained weights. A plain `Pkg.test()`
#           runs these, and so does CI.
#   parity  one family per backbone, comparing every variant against a timm
#           fixture. Heavy (weights for every variant), so they only run when
#           asked for: by scripts/parity_sweep.sh, scripts/test_variant.sh, or
#           the variables below.
#
# Two environment variables select what runs:
#
#   JIMM_TEST_FAMILIES  comma-separated family names (the keys of `FAMILY_FILES`),
#                       or `parity` for every parity family, or `all`.
#   JIMM_TEST_VARIANTS  comma-separated variant keys (Symbol names) to keep
#                       inside each parity family. When JIMM_TEST_FAMILIES is
#                       unset, the families are inferred from these variants.
#
# Neither set: the unit families. Examples:
#
#   julia --project -e 'using Pkg; Pkg.test()'                        # unit
#   JIMM_TEST_VARIANTS=convnextv2_atto_fcmae julia --project -e '...'  # one variant
#   JIMM_TEST_FAMILIES=infra,convnextv2 julia --project -e '...'       # mixed

# Every test family and the files it runs, in run order. `runtests.jl` walks this
# table; adding a parity family means adding its row here, to `PARITY_FAMILIES`,
# and a prefix in `_jimm_variant_family` below.
const FAMILY_FILES = (
    "infra" => ("test_hf_download.jl", "test_hf_hub_download.jl", "test_init.jl"),
    "interface" => ("test_interface.jl",),
    "reactant" => ("test_reactant.jl",),
    "bit" => ("test_bit_resnet.jl",),
    "resnet" => ("test_resnet.jl",),
    "convnext" => ("test_convnext.jl",),
    "convnextv2" => ("test_convnextv2.jl",),
    "vgg" => ("test_vgg.jl",),
    "seresnet" => ("test_seresnet.jl",),
    "vit" => ("test_vit.jl",),
    "coatnet" => ("test_coatnet.jl",),
)

const _JIMM_ALL_FAMILIES = first.(FAMILY_FILES)
const PARITY_FAMILIES = ("bit", "resnet", "convnext", "convnextv2", "vgg", "seresnet", "vit", "coatnet")
const _JIMM_DEFAULT_FAMILIES = Tuple(f for f in _JIMM_ALL_FAMILIES if !(f in PARITY_FAMILIES))

function _jimm_env_csv(name::AbstractString)
    raw = get(ENV, name, "")
    isempty(raw) && return String[]
    return [strip(s) for s in split(raw, ',') if !isempty(strip(s))]
end

# Map a variant Symbol to the test family it belongs to. Returns "" when the
# variant doesn't match a known family prefix; callers treat that as "include
# every family" so unknown variants do not silently filter the whole suite.
function _jimm_variant_family(v::Symbol)
    s = String(v)
    # Order matters: check `convnextv2_` before `convnext_` because both
    # start with the same prefix. (Julia's `startswith` requires an exact
    # prefix match, so `"convnextv2_atto_fcmae"` does not start with
    # `"convnext_"`, but explicit ordering still documents the intent.)
    startswith(s, "convnextv2_") && return "convnextv2"
    startswith(s, "convnext_") && return "convnext"
    startswith(s, "resnetv2_") && occursin("_bit_", s) && return "bit"
    startswith(s, "seresnet") && return "seresnet"
    startswith(s, "resnet") && return "resnet"
    startswith(s, "vgg") && return "vgg"
    startswith(s, "vit") && return "vit"
    startswith(s, "coatnet") && return "coatnet"
    return ""
end

"""
    families_enabled() -> Set{String}

The set of test families that should run this invocation. Resolution order:

1. `JIMM_TEST_FAMILIES` set: exactly that set, with `parity` expanding to
   every parity family and `all` to every family.
2. `JIMM_TEST_FAMILIES` unset, `JIMM_TEST_VARIANTS` set: the parity families
   of the requested variants (all parity families if any variant is
   unrecognized, rather than silently dropping it).
3. Both unset: the unit families, `_JIMM_DEFAULT_FAMILIES`.
"""
function families_enabled()
    explicit = _jimm_env_csv("JIMM_TEST_FAMILIES")
    if !isempty(explicit)
        fams = Set{String}()
        for f in explicit
            f == "all" ? union!(fams, _JIMM_ALL_FAMILIES) :
                f == "parity" ? union!(fams, PARITY_FAMILIES) : push!(fams, f)
        end
        return fams
    end

    variants = _jimm_env_csv("JIMM_TEST_VARIANTS")
    if !isempty(variants)
        fams = [_jimm_variant_family(Symbol(raw)) for raw in variants]
        return any(isempty, fams) ? Set(PARITY_FAMILIES) : Set(fams)
    end

    return Set(_JIMM_DEFAULT_FAMILIES)
end

family_enabled(name::AbstractString) = name in families_enabled()

"""
    variant_filter(variants) -> Tuple

Filter an iterable of variant `Symbol`s by `JIMM_TEST_VARIANTS`. When that
env var is unset or empty, returns the input unchanged.
"""
function variant_filter(variants)
    raw = _jimm_env_csv("JIMM_TEST_VARIANTS")
    isempty(raw) && return Tuple(variants)
    keep = Set(Symbol.(raw))
    return Tuple(v for v in variants if v in keep)
end
