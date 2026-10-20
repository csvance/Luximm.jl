#!/usr/bin/env bash
#
# Machine-specific settings for the parity sweep. Copy to scripts/parity_env.sh
# (git-ignored), edit, and run that:
#
#   cp scripts/parity_env.example.sh scripts/parity_env.sh
#   scripts/parity_env.sh                                  # full sweep
#   JIMM_TEST_FAMILIES=convnextv2 scripts/parity_env.sh    # one backbone family
#
# Anything left unset here gets a default inside LUXIMM_DATA; see
# scripts/parity_sweep.sh for the full list.

# A directory on the large disk. Weights, fixtures, the venv and every run's log
# and report go under it.
export LUXIMM_DATA=/path/to/big/disk/luximm
if [[ "$LUXIMM_DATA" == /path/to/* ]]; then
    echo "Set LUXIMM_DATA in $0 to a directory on the large disk first." >&2
    exit 2
fi

# Optional: a Hugging Face token avoids anonymous rate limits across ~90
# downloads. Leave unset to read $HF_HOME/token (written by `hf auth login`).
# export HF_TOKEN=hf_...

exec "$(dirname "${BASH_SOURCE[0]}")/parity_sweep.sh" "$@"
