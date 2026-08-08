# Shared parity tolerance constants for all family test files.
# Include with: isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")

const LOGITS_ATOL = 1.0f-3
# Features parity uses a relative bar: max-abs diff divided by max-abs of the
# timm reference. Deep backbones accumulate FP32 rounding through many stages,
# inflating raw pre-norm feature diffs even when downstream logits stay tight,
# so an absolute ceiling there gives false negatives.
const FEATURES_RTOL = 1.0f-4

# Deep variants whose feature rel-diff sits within ~2x of FEATURES_RTOL in
# float32. At depth 24+ the accumulated rounding moves the measured rel diff a
# few percent run to run (BLAS thread-count nondeterminism), so the shared bar
# flakes even though logits stay an order of magnitude under LOGITS_ATOL. The
# override widens the bar only for these paths; it is evidence-based, not a
# blanket relaxation — the variant's logits sub-test still holds the model to
# the tight absolute bar.
const FEATURE_RTOL_OVERRIDES = Dict{Symbol,Float32}(
    # ViT-L/14 CLIP: 24 pre-norm blocks; in1c features measured 9.3e-5 .. 1.1e-4.
    :vit_large_patch14_clip_224_openai_ft_in1k => 2.0f-4,
)

feature_rtol(variant::Symbol) = get(FEATURE_RTOL_OVERRIDES, variant, FEATURES_RTOL)
