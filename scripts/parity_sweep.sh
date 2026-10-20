#!/usr/bin/env bash
#
# Full parity sweep: dump every missing timm fixture, run the Julia test suite,
# and keep the log and a per-check CSV report. Runs on CPU in FP32.
#
# Usage:
#   scripts/parity_sweep.sh
#
# Normally launched through scripts/parity_env.sh (a git-ignored copy of
# scripts/parity_env.example.sh) which sets the environment below first.
#
# Environment:
#   LUXIMM_DATA         directory on a large disk. When set, everything below that
#                       is unset defaults to a path inside it. When unset, the
#                       usual defaults apply (~/.cache/huggingface, data/parity).
#   HF_HOME             Hugging Face cache; timm and Luximm share $HF_HOME/hub
#   JIMM_PARITY_DIR     parity fixtures
#   JIMM_PARITY_REPORT  CSV report path (default: inside this run's directory)
#   UV_PROJECT_ENVIRONMENT, UV_CACHE_DIR
#                       PyTorch venv and uv's wheel cache
#   HF_TOKEN            optional; read from $HF_HOME/token when unset
#   JIMM_TEST_FAMILIES, JIMM_TEST_VARIANTS
#                       scope the run, for the dumps as well as the tests
#                       (see test/_filter.jl). Neither set: the parity
#                       families only; CI runs the unit families. Set
#                       JIMM_TEST_FAMILIES=all to run both.
#
# Each run writes runs/<timestamp>-<sha>/ (under LUXIMM_DATA, or the repo's
# data/ when it is unset) with sweep.log and parity_report.csv. Exits non-zero
# when a dump or a test failed, or when any check skipped on a missing fixture:
# a complete sweep has none, so a skip is lost coverage.

set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

if [[ "${HF_OFFLINE:-}" == "1" ]]; then
    echo "[sweep] HF_OFFLINE=1 makes every parity test skip; unset it first." >&2
    exit 2
fi

if [[ -n "${LUXIMM_DATA:-}" ]]; then
    export HF_HOME="${HF_HOME:-$LUXIMM_DATA/hf}"
    export JIMM_PARITY_DIR="${JIMM_PARITY_DIR:-$LUXIMM_DATA/parity}"
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$LUXIMM_DATA/venv}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$LUXIMM_DATA/uv-cache}"
    runs_root="$LUXIMM_DATA/runs"
else
    runs_root="$repo_root/data/runs"
fi

# huggingface_hub reads HF_TOKEN; Luximm reads the legacy HUGGING_FACE_HUB_TOKEN.
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -z "${HF_TOKEN:-}" && -f "$hf_home/token" ]]; then
    HF_TOKEN=$(<"$hf_home/token")
    export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
# CoAtNet's in12k repos use Xet storage, which stalls on some networks.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# A plain Pkg.test() runs only the unit families, which CI covers; a sweep runs the
# parity families, so their results do not wait behind the unit tests.
if [[ -z "${JIMM_TEST_FAMILIES:-}" && -z "${JIMM_TEST_VARIANTS:-}" ]]; then
    export JIMM_TEST_FAMILIES=parity
fi

sha=$(git rev-parse --short HEAD)
run_dir="$runs_root/$(date +%Y%m%d-%H%M%S)-$sha"
mkdir -p "$run_dir"
export JIMM_PARITY_REPORT="${JIMM_PARITY_REPORT:-$run_dir/parity_report.csv}"
: >"$JIMM_PARITY_REPORT"
[[ -n "${JIMM_PARITY_DIR:-}" ]] && mkdir -p "$JIMM_PARITY_DIR"

log="$run_dir/sweep.log"
exec > >(tee -a "$log") 2>&1

echo "[sweep] commit $(git rev-parse HEAD)"
[[ -n "$(git status --porcelain --untracked-files=no)" ]] &&
    echo "[sweep] WARNING: working tree has uncommitted changes"
echo "[sweep] run dir       $run_dir"
echo "[sweep] HF_HOME       ${HF_HOME:-<default>}"
echo "[sweep] fixtures      ${JIMM_PARITY_DIR:-$repo_root/data/parity}"
echo "[sweep] families      ${JIMM_TEST_FAMILIES:-<from variants>}"
echo "[sweep] variants      ${JIMM_TEST_VARIANTS:-<all>}"
echo "[sweep] HF token      $([[ -n "${HF_TOKEN:-}" ]] && echo set || echo "not set (anonymous downloads)")"

echo "[sweep] uv sync"
uv sync || { echo "[sweep] uv sync failed" >&2; exit 1; }

# The family list and the scoping both come from test/_filter.jl, so the dumps
# cover exactly the parity families the tests will run. The file has no
# dependencies.
mapfile -t families < <(julia --startup-file=no -e '
    include("test/_filter.jl")
    foreach(println, (f for f in PARITY_FAMILIES if family_enabled(f)))')
mapfile -t pyramid_families < <(uv run python -c '
import sys; sys.path.insert(0, "test/parity")
from dump_features_only_io import PYRAMID_FAMILIES; print("\n".join(PYRAMID_FAMILIES))')

sidecar_for() {
    case "$1" in
        bit) echo "test/parity/dump_resnetv2_bit_io.py" ;;
        *) echo "test/parity/dump_$1_io.py" ;;
    esac
}

dump_failures=()
dump() {
    local label="$1"
    shift
    echo "[sweep] dump: $label"
    "$@" || dump_failures+=("$label")
}

if [[ -n "${JIMM_TEST_VARIANTS:-}" ]]; then
    IFS=',' read -ra variants <<<"$JIMM_TEST_VARIANTS"
    for v in "${variants[@]}"; do
        v="${v// /}"
        [[ -z "$v" ]] && continue
        dump "$v" scripts/test_variant.sh "$v" --dump-only
        dump "$v in_chans=1" scripts/test_variant.sh "$v" --in-chans 1 --dump-only
    done
else
    for f in "${families[@]}"; do
        sidecar=$(sidecar_for "$f")
        dump "$f" uv run python "$sidecar" --all
        dump "$f in_chans=1" uv run python "$sidecar" --all --in-chans 1
        if printf '%s\n' "${pyramid_families[@]}" | grep -qx "$f"; then
            dump "$f features_only" uv run python test/parity/dump_features_only_io.py --all --family "$f"
        fi
    done
fi

echo "[sweep] running Julia tests"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
test_rc=$?

echo
echo "[sweep] ================ summary ================"
echo "[sweep] log     $log"
echo "[sweep] report  $JIMM_PARITY_REPORT"
checks=$(($(wc -l <"$JIMM_PARITY_REPORT") - 1))
failed=$(awk -F, 'NR > 1 && $6 != "true"' "$JIMM_PARITY_REPORT")
echo "[sweep] parity checks: $((checks < 0 ? 0 : checks)), failed: $(grep -c . <<<"$failed")"
[[ -n "$failed" ]] && echo "$failed" | sed 's/^/[sweep]   FAIL /'
if [[ $checks -gt 0 ]]; then
    worst=$(awk -F, 'NR > 1' "$JIMM_PARITY_REPORT" | sort -t, -k3 -g | tail -1)
    echo "[sweep] worst rel-L2: $worst"
fi
skipped=$(grep -c "^\[ Info: skipping" "$log")
echo "[sweep] skipped checks (missing fixture): $skipped"
[[ $skipped -gt 0 ]] && grep "^\[ Info: skipping" "$log" | sed 's/^\[ Info: /[sweep]   /'
if [[ ${#dump_failures[@]} -gt 0 ]]; then
    echo "[sweep] failed dumps: ${#dump_failures[@]}"
    printf '[sweep]   %s\n' "${dump_failures[@]}"
fi
echo "[sweep] Pkg.test exit code: $test_rc"

[[ $test_rc -eq 0 && ${#dump_failures[@]} -eq 0 && $skipped -eq 0 ]]
