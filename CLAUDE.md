# Working in this repo

## Formatting

This repo uses **Runic.jl** via the `runic` CLI as the Julia formatter. Runic is
opinionated like `gofmt` — there is no config file and output is deterministic.

- Format in place: `runic -i <path>` (files or directories, e.g. `runic -i src test`)
- Check without writing: `runic -c <path>` (exits non-zero if anything is unformatted)

After editing Julia code, run `runic -i` on the touched files (or at minimum verify with
`runic -c`) before reporting the task complete. Do not hand-format Julia code to match
Runic style — let the tool do it. Likewise, do not use JuliaFormatter-based tooling
(including the Kaimon `format_code` tool); `runic` is the canonical formatter here.

CI enforces this: `.github/workflows/format.yml` runs Runic 1.13 over every `.jl` file
and fails on any diff. Keep its `version` in sync with the local `runic --version`.
`scripts/pre-commit-runic` formats `src test docs utils` and re-stages the result on
every commit; install it in a fresh clone with
`cp scripts/pre-commit-runic .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`.

## After any non-trivial change, check these two files

1. **`README.md`** (and the duplicate backbone table in `docs/src/index.md`) — the README enumerates supported backbones, families, and variant counts. After adding, removing, or renaming a model family, variant, or top-level public API, re-read README.md and update any list, table, or example that drifted. A change is "non-trivial" if a reader looking at README.md would now see something that is no longer true.

2. **The new-family checklist** — Phase 7 of `docs/src/porting.md` lists every place a new model family must be registered: the four `haskey` dispatch chains and the exports in `src/Models/Models.jl` / `src/Luximm.jl`; `FAMILY_FILES`, `PARITY_FAMILIES` and `_jimm_variant_family` in `test/_filter.jl`; `VARIANT_TABLES` and `REPRESENTATIVES` in `test/test_interface.jl`; the `case` block in `scripts/test_variant.sh`; `PYRAMID_FAMILIES` / `FEATURES_ONLY_VARIANTS` in `test/parity/dump_features_only_io.py` for a pyramid family; and the docs tables. The sidecar must be named `test/parity/dump_<family>_io.py`. A test file missing from `FAMILY_FILES` never runs, and a parity family missing from `PARITY_FAMILIES` runs in CI and gets no fixtures dumped by the sweep.

When in doubt, run through this checklist explicitly before reporting a task complete.
