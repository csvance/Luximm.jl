# Test-subset gating shared by every test file.
#
# Two environment variables control what runs:
#
#   JIMM_TEST_FAMILIES  comma-separated family names. Recognized values are
#                       "infra", "bit", "resnet", "convnextv2", and "convnext".
#                       Empty / unset means
#                       all families run, unless JIMM_TEST_VARIANTS is set
#                       (in which case the family set is inferred from the
#                       requested variants and infra is dropped).
#   JIMM_TEST_VARIANTS  comma-separated variant keys (Symbol names) to keep
#                       inside each family's parity sweep. Empty / unset means
#                       all variants for the enabled families run.
#
# Intent: compute-constrained workflows can run a single backbone, or a single
# variant of a single backbone, without paying for the rest. The common case
# only needs one env var:
#
#   JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
#       julia --project -e 'using Pkg; Pkg.test()'
#
# That runs *only* the ConvNeXtV2 parity testset for `convnextv2_atto_fcmae`;
# the infra and BiT families are skipped automatically. To run the infra
# checks too, list them explicitly:
#
#   JIMM_TEST_FAMILIES=infra,convnextv2 \
#   JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
#       julia --project -e 'using Pkg; Pkg.test()'

const _JIMM_DEFAULT_FAMILIES = (
    "infra",
    "bit",
    "resnet",
    "convnextv2",
    "convnext",
    "vgg",
    "seresnet",
    "vit",
    "coatnet",
)

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

1. `JIMM_TEST_FAMILIES` set: use exactly that set.
2. `JIMM_TEST_FAMILIES` unset, `JIMM_TEST_VARIANTS` set: infer the family
   set from the requested variants. Infra is dropped — when the user asks
   for a specific backbone variant, they almost always want only that
   variant's parity test, not the infra checks too.
3. Both unset: every family in `_JIMM_DEFAULT_FAMILIES`.
"""
function families_enabled()
    explicit = _jimm_env_csv("JIMM_TEST_FAMILIES")
    isempty(explicit) || return Set(explicit)

    variants = _jimm_env_csv("JIMM_TEST_VARIANTS")
    if !isempty(variants)
        fams = String[]
        for raw in variants
            f = _jimm_variant_family(Symbol(raw))
            isempty(f) || push!(fams, f)
        end
        # If every listed variant resolved to a known family, return just
        # those. If any are unknown, fall back to all families (safer than
        # silently dropping the unrecognized name).
        if length(fams) == length(variants)
            return Set(fams)
        end
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
