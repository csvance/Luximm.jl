#!/usr/bin/env bash
#
# Targeted parity test for a single backbone variant.
#
# Resolves the variant's family, ensures its HDF5 fixture exists under
# `data/parity/` (dumping it via the Python sidecar if not), and runs only
# that variant's testset via JIMM_TEST_VARIANTS.
#
# Usage:
#   scripts/test_variant.sh <variant_key> [--in-chans N] [--force]
#
# Examples:
#   scripts/test_variant.sh convnextv2_atto_fcmae
#   scripts/test_variant.sh convnextv2_atto_fcmae --in-chans 1
#   scripts/test_variant.sh resnet18_a1_in1k
#   scripts/test_variant.sh resnetv2_50x1_bit_goog_in21k
#
# `--force` re-dumps the fixture even if it already exists.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <variant_key> [--in-chans N] [--force]

Runs the Julia parity test for a single backbone variant, dumping the timm
reference fixture first if it is missing. Variant keys match Luximm's
<FAMILY>_VARIANTS table (the timm name with the dot rewritten as
underscore), e.g. convnextv2_atto_fcmae, resnetv2_50x1_bit_goog_in21k.
EOF
}

variant=""
in_chans=3
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --in-chans) in_chans="$2"; shift 2 ;;
        --in-chans=*) in_chans="${1#*=}"; shift ;;
        --force) force=1; shift ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -z "$variant" ]]; then
                variant="$1"; shift
            else
                echo "Unexpected positional arg: $1" >&2; exit 2
            fi
            ;;
    esac
done

if [[ -z "$variant" ]]; then
    usage >&2
    exit 2
fi

# Accept either the Julia variant key (e.g. `convnext_tiny_dinov3_lvd1689m`)
# or the canonical timm name (`convnext_tiny.dinov3_lvd1689m`). Normalize
# dots to underscores so the fixture path and JIMM_TEST_VARIANTS env var both
# reach the Julia side as the underscore-form key.
variant="${variant//./_}"

featsonly_sidecar="test/parity/dump_features_only_io.py"

# Run from the repo root so all relative paths resolve consistently.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

# Family resolution mirrors test/_filter.jl::_jimm_variant_family. Order
# matters here: convnextv2_* must come before convnext_*, because both
# patterns share the "convnext" prefix.
# `has_pyramid` gates the second fixture: the five families with a feature
# pyramid also need `<variant>_featsonly_io.h5`, dumped by a shared sidecar.
# Without it the pyramid testset silently skips on "fixture missing".
has_pyramid=0
case "$variant" in
    convnextv2_*)
        sidecar="test/parity/dump_convnextv2_io.py"
        has_pyramid=1
        ;;
    convnext_*)
        sidecar="test/parity/dump_convnext_io.py"
        has_pyramid=1
        ;;
    resnetv2_*_bit_*)
        sidecar="test/parity/dump_resnetv2_bit_io.py"
        has_pyramid=1
        ;;
    seresnet*)
        sidecar="test/parity/dump_seresnet_io.py"
        has_pyramid=1
        ;;
    resnet*)
        sidecar="test/parity/dump_resnet_io.py"
        has_pyramid=1
        ;;
    vgg*)
        sidecar="test/parity/dump_vgg_io.py"
        ;;
    vit*)
        sidecar="test/parity/dump_vit_io.py"
        ;;
    coatnet*)
        sidecar="test/parity/dump_coatnet_io.py"
        ;;
    *)
        echo "Could not resolve family for variant '$variant'." >&2
        echo "Known prefixes: convnextv2_*, convnext_*, resnetv2_*_bit_*, seresnet*, resnet*, vgg*, vit*, coatnet*." >&2
        exit 2
        ;;
esac

if [[ "$in_chans" -lt 1 ]]; then
    echo "--in-chans must be >= 1 (got $in_chans)" >&2
    exit 2
fi

suffix=""
if [[ "$in_chans" -ne 3 ]]; then
    suffix="_in${in_chans}c"
fi
fixture="data/parity/${variant}${suffix}_io.h5"

if [[ -f "$fixture" && "$force" -ne 1 ]]; then
    echo "[test_variant] fixture $fixture exists; reusing (pass --force to re-dump)"
else
    if [[ "$force" -eq 1 && -f "$fixture" ]]; then
        echo "[test_variant] --force set; re-dumping $fixture"
    else
        echo "[test_variant] fixture $fixture missing; dumping via $sidecar"
    fi
    dump_args=(--variant "$variant")
    if [[ "$in_chans" -ne 3 ]]; then
        dump_args+=(--in-chans "$in_chans")
    fi
    uv run python "$sidecar" "${dump_args[@]}"
fi

# The features_only fixture has no in_chans variants: the pyramid parity test
# only exercises the 3-channel path.
if [[ "$has_pyramid" -eq 1 && "$in_chans" -eq 3 ]]; then
    fo_fixture="data/parity/${variant}_featsonly_io.h5"
    if [[ -f "$fo_fixture" && "$force" -ne 1 ]]; then
        echo "[test_variant] fixture $fo_fixture exists; reusing"
    else
        # The sidecar skips any fixture that already exists, so --force has to
        # remove it first or the re-dump is a silent no-op.
        [[ "$force" -eq 1 ]] && rm -f "$fo_fixture"
        echo "[test_variant] dumping $fo_fixture via $featsonly_sidecar"
        uv run python "$featsonly_sidecar" --variant "$variant"
    fi
fi

echo "[test_variant] running Julia test for $variant"
JIMM_TEST_VARIANTS="$variant" julia --project=. -e 'using Pkg; Pkg.test()'
