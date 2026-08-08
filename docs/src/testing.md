```@meta
CurrentModule = Luximm
```

# Testing

Luximm's correctness story rests on parity tests: for every registered
variant, the Julia forward must match `timm`'s forward on the same
input and weights. The bar is two-tier:

- **Logits** are checked at an absolute max-abs-diff under
  `TOL = 1f-3`. The classifier head is shallow and tightly bounded,
  so an absolute ceiling is the meaningful end-to-end guarantee.
- **Features** (`forward_features` and the `in_chans=1` companion)
  are checked at a relative bar, `max-abs-diff / max-abs(timm ref)`
  under `FEATURES_RTOL = 1f-3`. Deep backbones accumulate FP32
  rounding through dozens of stages, which inflates raw pre-norm
  feature diffs by a factor that scales with depth and channel
  width, even when downstream logits stay tight. A relative bar
  keeps the check scale-free across tiny through huge variants.

This page covers the layout of the test suite, how to scope a run
to a single variant, how to dump the HDF5 fixture a parity test
consumes, and how production CI runs the sweep.

## Test layout

Everything lives under `test/`:

```
test/
├── runtests.jl         # entry point
├── _filter.jl          # env-var-driven family/variant filtering
├── parity/             # Python sidecars that produce HDF5 fixtures
│   ├── _dump_common.py
│   ├── dump_resnet_io.py
│   ├── dump_resnetv2_bit_io.py
│   ├── dump_convnext_io.py
│   └── dump_convnextv2_io.py
├── test_resnet.jl      # ResNet parity sweep
├── test_bit_resnet.jl  # BiT ResNetV2 parity sweep
├── test_convnext.jl    # ConvNeXt v1 parity sweep
├── test_convnextv2.jl  # ConvNeXtV2 parity sweep
├── test_init.jl        # Init-recipe parity (random-init)
├── test_hf_download.jl # Raw HF download path
└── test_hf_hub_download.jl  # HF Hub cache layout
```

`runtests.jl` consults `_filter.jl` to decide which family files to
include. Each family file iterates over a tuple of variant keys and
either runs a parity test against an existing HDF5 fixture or skips
the variant if its fixture is missing under `data/parity/`.

The three "infra" test files (`test_init.jl`, `test_hf_download.jl`,
`test_hf_hub_download.jl`) cover the cross-cutting concerns: init
recipes, raw HuggingFace downloads, and the Hub cache layout. They
run as the `infra` family.

## Scoping a run with environment variables

A full parity sweep downloads every released checkpoint and runs a
forward through every variant. That is the right thing to do on the
CI server, but on a developer machine it is overkill. Two
environment variables narrow what runs without editing files:

- `JIMM_TEST_VARIANTS`: comma-separated variant keys (e.g.
  `convnextv2_atto_fcmae`). Setting this alone also restricts the
  active families to whichever ones contain the listed variants and
  drops the `infra` family, so a single variant key is enough to
  scope a run.
- `JIMM_TEST_FAMILIES`: comma-separated list of families. Recognized
  values are `infra`, `bit`, `resnet`, `convnext`, `convnextv2`.
  When set, this is authoritative and overrides the family
  inference from `JIMM_TEST_VARIANTS`. Unset and
  `JIMM_TEST_VARIANTS` also unset means every family runs.

```
# Just the ConvNeXtV2 atto fcmae parity tests, nothing else:
JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
    julia --project -e 'using Pkg; Pkg.test()'

# Include the infra checks alongside a single backbone variant:
JIMM_TEST_FAMILIES=infra,convnextv2 \
JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
    julia --project -e 'using Pkg; Pkg.test()'
```

Parity tests also skip when their HDF5 fixture is missing under
`data/parity/`, so a contributor can dump one variant's fixture
(and optionally its `_in1c` companion) and run just that one
without touching the test code.

## Parity fixtures

A parity fixture is an HDF5 file produced by one of the Python
sidecars under `test/parity/`. It contains:

- `/input`: deterministic `torch.randn` input in PyTorch NCHW
  layout.
- `/state_dict/<key>`: every PyTorch parameter, keyed by its
  `state_dict` name.
- `/output/features`: the result of `model.forward_features(input)`.
- `/output/logits` (when the variant ships a trained head): the
  result of `model.forward(input)`.

Fixtures live in `data/parity/` and follow the naming convention
`<variant_key>_io.h5` (and `<variant_key>_in1c_io.h5` for
single-channel variants). The directory is gitignored; fixtures are
not redistributed.

The Julia side consumes fixtures via `Luximm.Interop.read_parity`, which
returns a NamedTuple `(input, state_dict, output)` with all arrays
already axis-reversed from PyTorch NCHW into Lux's WHCN layout. The
mapping function for the family then routes each state-dict key
into the corresponding Lux parameter path via
[`apply_state_dict`](@ref).

### The `features_only` fixture

Variants in a family with a feature pyramid (ResNet, SE-ResNet, BiT
ResNetV2, ConvNeXt, ConvNeXt V2, ViT) have a **second** fixture,
`<variant_key>_featsonly_io.h5`, produced by a single shared sidecar
rather than a per-family one. It holds timm's `features_only=True`
outputs, one dataset per tap, plus timm's own tap table:

- `/output/feat_01` ... `/output/feat_0K`: one per feature map, ordered
  by increasing reduction.
- `/feature_channels`, `/feature_reductions`: timm's
  `feature_info.channels()` and `feature_info.reduction()`. These pin
  Luximm's [`feature_info`](@ref) table to timm's, so a tap list that
  drifts (a wrong channel count, a missed reduction-2 stem tap) fails
  the test even when every tensor still has a plausible shape.

The ViT dump is special-cased: `timm.create_model` drops `None` kwargs,
so `out_indices=None` (every block) cannot go through the factory — the
sidecar builds the bare model and wraps it in `FeatureGetterNet` directly
instead, since Luximm's default is `nothing` = every tap while timm's
`vit_*` default is the last three blocks.

Dump one variant, one family, or everything:

```
# One variant
uv run python test/parity/dump_features_only_io.py \
    --variant resnet50.a1_in1k

# One family. The six pyramid families are: resnet, seresnet, bit,
# convnext, convnextv2, vit.
uv run python test/parity/dump_features_only_io.py --all --family bit

# Every registered variant of all six families (74 fixtures)
uv run python test/parity/dump_features_only_io.py --all
```

Existing fixtures are skipped, so `--all` is safe to re-run and resumes
after an interruption; delete the `.h5` to force a re-dump. Like the
other sidecars it honors `JIMM_PARITY_DIR`, so on the CI box:

```
JIMM_PARITY_DIR=/var/lib/jimm-ci/parity \
    uv run python test/parity/dump_features_only_io.py --all
```

There is no `--in-chans` companion: the pyramid parity test only
exercises the 3-channel path, since the `adapt_input_conv` stem is
already covered by the `forward_features` fixtures.

`jimm-ci` dumps these automatically (`Builder.jl` invokes the sidecar
per family alongside the per-family one), and `scripts/test_variant.sh`
dumps a variant's pyramid fixture along with its `forward_features` one.
Both paths are gated on the fixture existing, so a missing one shows up
as `skipping <variant> feature pyramid: fixture missing at ...` rather
than a failure.

### Dumping a fixture

```
uv run python test/parity/dump_<family>_io.py \
    --variant <timm_name> \
    --out data/parity/<variant_key>_io.h5
```

`<timm_name>` is the dot-separated `timm` model name (e.g.
`convnextv2_atto.fcmae`). `<variant_key>` is the Julia symbol form
with the dot rewritten as an underscore. Pass `--in-chans 1` to
produce the single-channel companion fixture; the output filename
suffix changes from `_io` to `_in1c_io`.

The first dump for a family materializes the Python sidecar
environment (PyTorch plus `timm` plus the small HDF5 helpers).
`uv sync` against the repo's `pyproject.toml` is the supported
provisioning path.

## The `scripts/test_variant.sh` wrapper

For the common case (one variant, dump-if-missing then test),
`scripts/test_variant.sh` chains the fixture dump and the Julia
invocation:

```
# Resolve family, dump fixture if absent, run only this variant:
scripts/test_variant.sh convnextv2_atto_fcmae

# Classic ResNet18:
scripts/test_variant.sh resnet18_a1_in1k

# Single-channel parity test (dumps the _in1c fixture):
scripts/test_variant.sh convnextv2_atto_fcmae --in-chans 1

# Force a fresh fixture dump even if one already exists:
scripts/test_variant.sh convnextv2_atto_fcmae --force
```

The script resolves the family from the variant prefix, calls the
appropriate Python sidecar under `test/parity/` via `uv run` if the
HDF5 fixture is missing, then runs the Julia test suite with
`JIMM_TEST_VARIANTS=<variant>` set. Requires both `uv` and `julia`
on `PATH`.

## CI

Production CI runs on a self-hosted Linux VM via a Julia TUI driver
named `jimm-ci` (under `ci/JimmCI/`). The setup is unusual on
purpose: the parity test suite needs hundreds of gigabytes of
`timm` reference weights and per-variant HDF5 fixtures that each
take a few minutes of CPU plus a PyTorch environment to produce.
Hosted CI runners would re-download and re-generate that material
on every run; a self-hosted machine with a persistent state
directory hits the cache instead.

See [`ci/README.md`](https://github.com/csvance/Luximm.jl/blob/master/ci/README.md)
for the full deployment story (App registration, secrets layout,
the TUI keybindings, and how PR runs map paths to test families).
The short version, from a contributor's perspective:

- PRs are reviewed by a maintainer and approved by selecting their
  row in the `jimm-ci` TUI on the VM. Fork PRs are filtered out
  client-side and never produce a Check Run.
- The runner consults `ci/JimmCI/src/PathFilter.jl` to map changed
  paths to families. If you add a new family or rename a shared
  module, that file must be updated; otherwise CI silently skips
  tests for the new code.
- Per-PR Check Runs come back through the standard GitHub Checks
  API. Look at the `jimm-ci / <family>` checks on the PR.

The separate documentation workflow under
`.github/workflows/docs.yml` is a hosted GitHub Actions
job. It builds and deploys this documentation site without needing
parity weights or fixtures, so the regular runners are sufficient
for that path.
