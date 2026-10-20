```@meta
CurrentModule = Luximm
```

# Testing

Luximm's correctness story rests on parity tests: for every registered
variant, the Julia forward must match `timm`'s forward on the same
input and weights. Every comparison (features, logits, the
`in_chans=1` companion and each `features_only` tap) uses one metric,
the relative L2 error `‖y - ref‖ / ‖ref‖`, against one bar,
`PARITY_RTOL = 1f-3` in `test/_parity_tol.jl`. The suite runs on CPU
in FP32. Correct ports land at `1e-7` to `1e-5`, and porting bugs at
`1e-2` and above; [Porting](porting.md) explains the choice.

Set `JIMM_PARITY_REPORT` to a file path to also append one CSV row
per check (`variant,check,rel_l2,cosine,max_abs,pass`), which makes a
full sweep easy to review.

Those parity tests are heavy (every variant's weights), so they are
not what a plain `Pkg.test()` runs. The suite has two kinds of test
family:

- **Unit families**, run by `Pkg.test()` and by the GitHub Actions
  `Tests` workflow. No fixtures and no pretrained weights.
  - `infra`: init recipes, raw HuggingFace downloads, and the Hub
    cache layout.
  - `interface`: every registered variant builds; one small variant
    per family produces the advertised output shapes and
    `feature_info` tap table, handles `num_classes = 0` and
    `in_chans = 1`, and treats batch elements independently; the
    vendored normalization layers keep Lux's parameter trees; and
    the two ConvNeXt block layouts agree.
  - `reactant`: the Reactant extension. Its normalization statistics,
    their gradients and their Hessian-vector products match plain
    `Statistics` definitions differentiated with ForwardDiff (Float64,
    to `1e-10`), and a compiled ConvNeXt V2 and BiT forward matches
    the host forward.
- **Parity families**, one per backbone (`bit`, `resnet`, `convnext`,
  `convnextv2`, `vgg`, `seresnet`, `vit`, `coatnet`), run only when
  asked for: by the full sweep, by `scripts/test_variant.sh`, or by
  the variables below.

This page covers the layout of the test suite, how to scope a run
to a single variant, how to dump the HDF5 fixture a parity test
consumes, and how the full sweep is run.

## Test layout

Everything lives under `test/`:

```
test/
├── runtests.jl          # entry point
├── _filter.jl           # families, and env-var-driven family/variant filtering
├── _parity_helpers.jl   # shared parity scaffolding
├── _parity_tol.jl       # PARITY_RTOL and check_parity
├── parity/              # Python sidecars that produce HDF5 fixtures
│   ├── _dump_common.py
│   ├── dump_<family>_io.py
│   └── dump_features_only_io.py
├── test_interface.jl    # interface family
├── test_reactant.jl     # reactant family
├── test_init.jl         # infra: init recipes (random init)
├── test_hf_download.jl  # infra: raw HF download path
├── test_hf_hub_download.jl  # infra: HF Hub cache layout
└── test_<family>.jl     # one parity family per backbone
```

`runtests.jl` walks the `FAMILY_FILES` table in `_filter.jl` and
includes the files of every enabled family. Each parity file iterates
over its family's variants and either runs the parity checks against
the variant's HDF5 fixture or skips the variant if its fixture is
missing.

## Scoping a run with environment variables

Two environment variables choose what runs without editing files:

- `JIMM_TEST_FAMILIES`: comma-separated list of families, `parity`
  for every parity family, or `all`. When set, this is authoritative
  and overrides the family inference from `JIMM_TEST_VARIANTS`.
- `JIMM_TEST_VARIANTS`: comma-separated variant keys (e.g.
  `convnextv2_atto_fcmae`) to keep inside each parity family. Setting
  this alone selects the parity families that contain the listed
  variants, so a single variant key is enough to scope a run.

Neither set means the unit families.

```
# The unit families (what CI runs):
julia --project -e 'using Pkg; Pkg.test()'

# Just the ConvNeXtV2 atto fcmae parity tests, nothing else:
JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
    julia --project -e 'using Pkg; Pkg.test()'

# The unit checks alongside a single backbone variant:
JIMM_TEST_FAMILIES=infra,interface,reactant,convnextv2 \
JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
    julia --project -e 'using Pkg; Pkg.test()'
```

Parity tests also skip when their HDF5 fixture is missing under
`data/parity/` (or `JIMM_PARITY_DIR`), so a contributor can dump one
variant's fixture (and optionally its `_in1c` companion) and run just
that one without touching the test code. They skip entirely when
`HF_OFFLINE=1` is set, since they need the HuggingFace weights;
`scripts/parity_sweep.sh` refuses to run with it set.

## Parity fixtures

A parity fixture is an HDF5 file produced by one of the Python
sidecars under `test/parity/`. It contains:

- `/input`: deterministic `torch.randn` input in PyTorch NCHW
  layout, with a batch of two distinct samples (`PARITY_BATCH` in
  `test/parity/_dump_common.py`). At batch 1, a forward that mixes
  the batch axis into tokens, heads or a reduction would still match
  exactly; at batch 2 it fails.
- `/state_dict`: an empty placeholder group. The tests load the
  released `model.safetensors` from HuggingFace through
  `create_pretrained`, so a parity run needs network access or a warm
  HuggingFace cache.
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
parity helpers in `test/_parity_helpers.jl` build the model with
[`create_pretrained`](@ref), load the HuggingFace weights, run
`fixture.input` through it and compare against `fixture.output` with
`check_parity`.

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
other sidecars it honors `JIMM_PARITY_DIR`, so fixtures can live on a
larger disk:

```
JIMM_PARITY_DIR=/data/luximm/parity \
    uv run python test/parity/dump_features_only_io.py --all
```

There is no `--in-chans` companion: the pyramid parity test only
exercises the 3-channel path, since the `adapt_input_conv` stem is
already covered by the `forward_features` fixtures.

`scripts/test_variant.sh` dumps a variant's pyramid fixture along with
its `forward_features` one. The test is gated on the fixture existing,
so a missing one shows up as
`skipping <variant> feature pyramid: fixture missing at ...` rather
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

# Dump the fixtures without running the tests:
scripts/test_variant.sh convnextv2_atto_fcmae --dump-only
```

The script resolves the family from the variant prefix, calls the
appropriate Python sidecar under `test/parity/` via `uv run` if the
HDF5 fixture is missing, then runs the Julia test suite with
`JIMM_TEST_VARIANTS=<variant>` set. Requires both `uv` and `julia`
on `PATH`.

## CI

The `Tests` workflow (`.github/workflows/tests.yml`) runs the unit
families on pushes to master and on pull requests that touch code. The parity suite does not run
on hosted CI: a full sweep needs every variant's `timm` weights and a
PyTorch environment to dump the fixtures, which is more disk and CPU
than a hosted runner offers. The maintainer runs the full sweep by
hand on a machine with a large disk, on CPU in FP32, before each
release.

`scripts/parity_sweep.sh` runs it: `uv sync`, dump every missing
fixture (both `in_chans` and the `features_only` pyramids), run
`Pkg.test()` over every family, and write the log and the CSV report
to a per-run directory, ending with a summary of failed checks,
skipped checks and failed dumps. Machine-specific settings go in a
git-ignored launcher:

```
cp scripts/parity_env.example.sh scripts/parity_env.sh
# edit LUXIMM_DATA (a directory on the large disk), optionally HF_TOKEN
scripts/parity_env.sh                                  # full sweep
JIMM_TEST_FAMILIES=convnextv2 scripts/parity_env.sh    # one backbone family
```

`JIMM_TEST_FAMILIES` and `JIMM_TEST_VARIANTS` scope the dumps as well
as the tests, so a scoped sweep only downloads the weights it uses.

The other hosted workflows need neither weights nor fixtures either:
`.github/workflows/format.yml` checks Runic formatting, and
`.github/workflows/docs.yml` builds and deploys this documentation
site.
