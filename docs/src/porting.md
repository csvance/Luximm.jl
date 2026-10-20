```@meta
CurrentModule = Luximm
```

# Porting Backbones

This page is a contributor guide for adding a new `timm` backbone to
Luximm.jl. The bar is numeric parity with `timm` on the released
weights; the rest is mechanics.

If you are using Claude Code to drive the port, the same workflow is
encoded as an agent-facing skill at `skills/timm-to-lux/SKILL.md`,
installed with the `luximm` Claude Code plugin (see the README). This
page covers the same ground for human contributors without those
tools.

## Acceptance criteria

A new backbone is mergeable when all four hold:

1. **Pretrained parity.** Forward output of the Lux model with weights
   loaded via the closure returned by
   [`create_pretrained`](@ref) matches `timm`'s forward
   output on the same input: features (`num_classes = 0`, and the
   `in_chans = 1` companion), logits and every `features_only` tap
   are each within a relative L2 error `‖y - ref‖ / ‖ref‖` of
   `PARITY_RTOL = 1f-3`. Existing models land far inside this, at
   `1e-6` to `1e-5`; ConvNeXt V2 Tiny, for example, sits near `2e-6`
   on features and logits.
2. **Random-init recipe.** `Lux.setup` reproduces the distributions
   of the `_init_weights` recipe `timm` uses for the family, so a model
   trained from scratch starts where timm's would. `test/test_init.jl`
   checks this statistically (per-layer standard deviations). See
   `_CN_INIT` in `src/Models/ConvNeXtCommon/Common.jl`
   (`truncated_normal(mean = 0f0, std = 0.02f0)`, mirroring timm's
   `trunc_normal_(std = 0.02)`) for a worked example.
3. **State-dict round-trip.** The mapping function consumes every
   PyTorch `state_dict` key. Mappings raise on a missing key so
   silent random-init leaks cannot pass parity by coincidence; see
   the assertion block at the end of `convnextv2_mapping` in
   `src/Models/ConvNeXtV2/Model.jl`.
4. **Variant table entry and bookkeeping.** New variants are
   registered in `<FAMILY>_VARIANTS` (`src/Models/<Family>/Config.jl`)
   with their HuggingFace repo id and default class count, and the
   Phase 7 checklist below is complete.

## Why the bar is `1e-3` relative L2

The suite runs on CPU in FP32, and its question is whether a model is
close enough to `timm` that its pretrained weights transfer, not
whether it reproduces `timm` to the last bit. Float32 round-off
accumulates through the depth of a network, and summation order, BLAS
thread counts and CPU kernels all move it, so a correct port lands
between `1e-7` and `1e-5` relative L2 and varies a little from run to
run. Real porting bugs (cross-correlation vs convolution, a wrong
weight mapping, an axis permutation, a missing layer, the wrong
activation) land at `1e-2` and above. The bar sits between the two,
with room for the spread between machines.

Relative L2 is used everywhere, not max-abs-diff: pre-norm features
on deep or wide backbones reach magnitudes in the thousands, and
transformers carry a few outlier activations, so an absolute bar or
one normalized by the largest element misjudges them.

The bar is loose enough to pass a few subtle mismatches, such as a
norm `epsilon` of `1e-5` against `1e-6` or a population against a
sample variance on large maps. Those do not change what a downstream
head learns, but they are still wrong, so read the logged number, not
just the pass: if a port that ought to be exact reports `1e-4` or
more, find out why before calling it done. Bisect with the per-stage
debugging fixtures described in Phase 1.

## Reference example

`src/Models/ResNetV2/` is the canonical, fully-worked port. Read it
first. It exercises most of the shared utilities in one place:

- Pre-activation residual blocks.
- Weight-standardized convolutions (`std_conv` from `Luximm.Layers`).
- `Layers.LuximmGroupNorm` with explicit epsilon (`gn_act`).
- `adapt_input_conv` stem adaptation for non-RGB inputs.
- The full mapping/loader pattern that every other family follows.

If your port can be expressed as parameter-table changes on top of an
existing family, do that. Adding a new variant to `BIT_VARIANTS` or
`CONVNEXTV2_VARIANTS` is a one-row PR. Adding a new family is the
multi-file workflow below.

## Workflow

### Phase 1: Capture a parity fixture

Each port starts with an HDF5 fixture produced by a small Python
sidecar in `test/parity/`. The sidecar uses the shared
`_dump_common.py` helpers and must be named `dump_<family>_io.py`
(`scripts/parity_sweep.sh` finds it by that name). Look at
`test/parity/dump_resnet_io.py` for the template.

The fixture stores `/input` (deterministic random input, PyTorch NCHW
layout, a batch of two samples) and `/output/features` plus optional
`/output/logits`. Its `/state_dict` group is an empty placeholder: the
parity tests load the released weights from HuggingFace through
`create_pretrained`, so they cover the production loading path. The
Julia side reads the fixture via `Luximm.Interop.read_parity`, which
reverses the axes so tensors arrive in Lux's WHCN layout.

Run the dump once per variant:

```
uv run python test/parity/dump_<family>_io.py --variant <timm_name> --out data/parity/<key>_io.h5
```

Optionally dump a single-channel companion fixture (`--in-chans 1`,
output `data/parity/<key>_in1c_io.h5`) so the `in_chans = 1` parity
test can run as well.

For a brand-new family, also dump local debugging fixtures while
you port: the end-to-end check gives a pass/fail signal with no
localization power. A debugging fixture can carry the full
`model.state_dict()` (so you can test the forward without the
HuggingFace download), a few random seeds, and per-stage
intermediates captured with forward hooks on `model.stages[i]` and
stored as `/output/stage0`, `/output/stage1`, ... These are something
you add for the port, not something the committed sidecars write;
they are the bisection tool when the end-to-end test fails.

### Phase 2: Reuse the shared utilities

Do not re-implement anything that already lives under `src/Layers/`
or `src/Interop/`. The load-bearing helpers and what they do:

- `Luximm.Interop.read_parity` returns `(input, state_dict, output)` as
  `Float32` arrays in WHCN layout (the reverse of PyTorch's logical
  NCHW). Conv weight `(out, in, kH, kW)` becomes `(kW, kH, in,
  out)`, which is exactly Lux's Conv layout. For most parameters the
  layout is what you want and the per-key transform is `identity`.
- [`apply_state_dict`](@ref) rebuilds the parameter `NamedTuple` by
  setting leaves from the dict. Mapping entries are triples
  `(pytorch_key, lux_path_tuple, transform)`. Non-mutating; bind the
  result.
- [`load_safetensors_state_dict`](@ref) reads HuggingFace weights
  into the same reversed layout (`reverse_axes = true` is the
  default), so one mapping function serves both fixture `state_dict`s
  and production loading.
- `Luximm.Interop.axis_reverse` and `Luximm.Interop.pyperm` are ready-made
  transforms for the cases where the HDF5-natural layout is not what
  you want, typically Dense weights. `as_channel4d` and
  `as_token_norm` reshape a `(C,)` norm parameter for `layernorm2d`
  and `vit_layernorm`; `adapt_input_conv` adapts the stem weight to
  `in_chans`.
- [`std_conv`](@ref), [`layernorm2d`](@ref), [`vit_layernorm`](@ref),
  [`grn_layer`](@ref) are the building blocks that match `timm`'s
  `StdConv2dSame`, channel-axis `LayerNorm2d`, ViT LayerNorm and
  Global Response Norm. The normalization primitives
  (`Layers.LuximmLayerNorm`, `Layers.LuximmGroupNorm`,
  `Layers.layernorm`, `Layers.groupnorm`, `Layers.mean_var`) are
  covered in the [Layers](api/layers.md) reference.

Keep the per-family weight mapping in `src/Models/<Family>/Model.jl`
as `<family>_mapping(state_dict, variant; load_classifier, in_chans,
prefix)`, returning a vector of `(pytorch_key, lux_path, transform)`
triples. The `prefix` argument lets a backbone be nested under a
wrapper model. BatchNorm families also need a
`<family>_state_mapping` for the running statistics.

### Phase 3: Implement the model with `@compact`

Lux's `@compact` is the right primitive for composing layers. The
pattern is fixed:

```julia
@compact(
    conv1 = Conv((3, 3), in_ch => out_ch; pad = 1, cross_correlation = true),
    norm1 = Layers.LuximmGroupNorm(out_ch, 32; epsilon = 1f-5),
) do x
    @return NNlib.relu.(norm1(conv1(x)))
end
```

Numeric conventions that bite if you forget them, in rough order of
how often they cost real debugging time:

- **Cross-correlation, always.** PyTorch's `Conv2d` is
  cross-correlation; Lux's `Conv` defaults to true convolution
  (kernel-flipped). Pass `cross_correlation = true` to every `Conv`.
  Without it, weights load with the right shape but produce mirrored
  outputs. When you must drop into `NNlib.conv` directly (weight
  standardization is the canonical case), pass `flipkernel = true`
  on `NNlib.DenseConvDims`. Same semantic, two flags.
- **Explicit padding when zero-padding matters.** `Conv((k, k), ...;
  pad = p)` works for symmetric same-value padding. For pooling that
  must pad with zeros instead of `-Inf` (timm's BiT stem is the
  canonical case), call `NNlib.pad_zeros(x, (l, r, t, b, 0, 0, 0,
  0))` first and use `pad = 0` on the op.
- **Luximm's norms, not Lux's.** Build LayerNorm and GroupNorm with
  `layernorm2d`, `vit_layernorm`, `Layers.LuximmLayerNorm`,
  `Layers.LuximmGroupNorm` (or the `Layers.layernorm` /
  `Layers.groupnorm` / `Layers.mean_var` functions), never
  `Lux.LayerNorm`, `Lux.GroupNorm` or `Statistics.var` directly. They
  have Lux's parameter trees, and they are what the Reactant
  extension accelerates. BatchNorm stays `Lux.BatchNorm`.
- **Pass `epsilon` explicitly on every norm.** Lux and PyTorch both
  default to `1e-5`, but timm overrides it in places: `LayerNorm2d`
  (ConvNeXt, CoAtNet) and ViT use `1e-6`. A mismatched epsilon
  silently shifts activations by an amount small enough to pass the
  parity bar.
- **Variance corrections.** Sample variance (Bessel-corrected, the
  Julia default) and population variance (`corrected = false`, what
  PyTorch uses for BN-style stats and for weight standardization)
  differ by a factor of `N / (N - 1)`. Pass `corrected = false` to
  `var` whenever you are matching a BN- or WS-style operation. See
  `std_conv` in `src/Layers/StdConv.jl`.
- **`Lux.testmode(st)` for parity tests.** Otherwise BatchNorm
  running stats update and any dropout activates, neither of which
  is what `model.eval()` does on the PyTorch side.

The forward should look like math: a sequence of broadcasts and
tensor ops with no scalar control flow. No `x[i] = ...`, no
`Array(x)` inside the forward, no `if`/`else` on tensor values, no
scalar indexing.

### Phase 4: Wire the variant table and mapping

`src/Models/<Family>/Config.jl` holds the variant catalog:

- The `<Family>Variant` struct captures the architectural knobs
  (depths, dims, stem channels) plus `hf_repo`,
  `default_num_classes`, and `default_input_size`.
- `<FAMILY>_VARIANTS :: Dict{Symbol, <Family>Variant}` lists every
  registered variant.

Variant keys are the `timm` model name with the dot rewritten as an
underscore (so the key is a single Julia identifier). The full
HuggingFace repo id (`timm/<timm name>`) lives at
`<FAMILY>_VARIANTS[key].hf_repo`.

Pick one variant and finish it before adding the second. When a
model family has many variants, port the smallest first: it surfaces
every shared numeric trap with the fastest test loop, and the
mapping function written for one variant typically generalizes by
changing only `depths` or `widths`. Adding a second variant before
the first passes parity creates code paths nothing has exercised and
that the first failed test cannot localize.

### Phase 5: Add the loader

Each family has a private `_load_<family>(ps, st, variant; in_chans,
num_classes, revision, cache_dir, prefix) -> (ps, st)`, reached
through the [`create_pretrained`](@ref) closure. The flow is
identical across families:

1. Look up the variant in `<FAMILY>_VARIANTS`.
2. Load the classifier head only when `num_classes` matches
   `default_num_classes`; otherwise warn and leave the head at its
   random init (or omit it for `num_classes = 0`).
3. Call [`hf_hub_download`](@ref) to resolve the snapshot path under
   the HuggingFace Hub cache layout.
4. Load via [`load_safetensors_state_dict`](@ref), which returns the
   same WHCN layout the HDF5 fixtures produce.
5. Apply via [`apply_state_dict`](@ref), after checking that every
   checkpoint key was consumed.
6. For families with BatchNorm running stats (ResNet, SE-ResNet,
   CoAtNet, `vgg*_bn`), additionally load the statistics into the
   model state via `<family>_state_mapping`. LayerNorm and GroupNorm
   families return `st` unchanged.

A new variant of an existing family needs only its table row. A new
family also needs a branch in each of the four dispatch chains in
`src/Models/Models.jl`: [`create_model`](@ref),
[`default_num_classes`](@ref), [`feature_info`](@ref) and
`_load_pretrained` (each is a `haskey(<FAMILY>_VARIANTS, variant)`
chain).

Keep the constructor and the weight loading separate. The
constructor returns a `@compact` block; the loader takes the result
of `Lux.setup` and returns a new `(ps, st)`. Mixing the two inside
`@compact` makes the model unusable in tests that want a random
init.

### Phase 6: Verify parity end-to-end

The verification loop is layered. Run the cheap gates between
meaningful edits.

End to end first, through the production loading path (what
`run_variant_parity` in `test/_parity_helpers.jl` does):

```julia
data = Luximm.Interop.read_parity(fixture_path)
model, load = create_pretrained(variant; num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
st = Lux.testmode(st)
ps, st = load(ps, st)
y, _ = model(data.input, ps, st)
check_parity(variant, "features", y, data.output["features"])
```

`check_parity` (in `test/_parity_tol.jl`) tests the relative L2
error against `PARITY_RTOL` and logs it. To isolate the forward from
the weight download, run the same comparison against a debugging
fixture's `state_dict` with
`ps = apply_state_dict(ps, data.state_dict, <family>_mapping(data.state_dict, variant))`.

The convenience wrapper `scripts/test_variant.sh` runs these checks
(features, logits, `in_chans = 1` and the pyramid taps) for one
variant, dumping the fixtures first if absent. See
[Testing](testing.md) for the variant-test workflow.

When parity fails, the per-stage and per-block fixtures pay off.
Walk the forward by hand: run the partial forward up to each stage
and compare against `data.output["stage0"]`, `data.output["stage1"]`, ... The first stage where
parity breaks localizes the bug. Inside the failing stage, splice
the matching state-dict entries into a single block in isolation
and compare against the per-block fixture.

### Phase 7: Update the bookkeeping files

After parity passes, the rest of the repo has to learn about the
family. Each item missed is a silent gap: code that is never tested,
or a table that is no longer true.

1. **Dispatch and exports.** A branch in `create_model`,
   `default_num_classes`, `feature_info` and `_load_pretrained` in
   `src/Models/Models.jl`, and the family's constructor, mapping and
   variant table in the export lists of `src/Models/Models.jl` and
   `src/Luximm.jl`.
2. **`test/_filter.jl`.** The family's row in `FAMILY_FILES`, its
   name in `PARITY_FAMILIES`, and its variant prefix in
   `_jimm_variant_family`. A test file that is not listed in
   `FAMILY_FILES` never runs.
3. **`test/test_interface.jl`.** The variant table in
   `VARIANT_TABLES` and one small variant in `REPRESENTATIVES` (plus
   `NO_PYRAMID` if it has no feature pyramid).
4. **`scripts/test_variant.sh`.** The family's prefix and sidecar in
   the `case "$variant"` block.
5. **Feature pyramid.** For a family with `features_only` support,
   its name in `PYRAMID_FAMILIES` and its variants in
   `FEATURES_ONLY_VARIANTS` in `test/parity/dump_features_only_io.py`.
6. **Tables and docs.** The backbone tables in `README.md` and
   `docs/src/index.md`, the family list in
   `docs/src/getting_started.md` ("Switching families"), and the API
   reference page (`docs/src/api/models.md`).

## Pitfalls reference

A running checklist. Every item here has cost real debugging time
on a previous port.

- **Cross-correlation.** Every `Conv` needs `cross_correlation =
  true`. Every `NNlib.conv` needs `flipkernel = true` on its
  `DenseConvDims`.
- **Zero-padded pooling.** `NNlib.maxpool(pad = 1)` pads with
  `-Inf`. `nn.MaxPool2d(padding = 1)` pads with zero. When the
  input has negative values (post-norm activations, leaky-relu
  outputs), this diverges. Use `pad_zeros` first, then pool with
  `pad = 0`.
- **Norm epsilons.** Always pass `epsilon` explicitly; timm's
  `LayerNorm2d` and ViT use `1e-6`, not the `1e-5` default.
- **Luximm norms, not Lux norms.** `Lux.LayerNorm` and `Lux.GroupNorm`
  work, but miss the Reactant extension.
- **Variance correction.** Pass `corrected = false` when matching
  anything BN-style or WS-style.
- **Axis order on `apply_state_dict`.** HDF5 fixtures and
  [`load_safetensors_state_dict`](@ref) (with its default
  `reverse_axes = true`) both arrive reversed (Lux-natural), so both
  sources share one mapping function.
- **`forward_features` vs `forward`.** The fixture and the Julia
  forward must agree on which one was dumped. Mismatch shows up as
  a shape error if you are lucky, a silent wrong number if you are
  not.
- **Pre-activation block order.** norm then activation then conv vs
  conv then norm then activation. Both compile. Only one matches
  the upstream architecture.
- **`timm`'s `adapt_input_conv` for `in_chans != 3`.** The stem
  weight in the safetensors file is the 3-channel version. The
  loader applies `adapt_input_conv` at load time to collapse it.
  Do not re-collapse on the Julia side.
- **`Lux.testmode(st)`.** Required for parity. Forgetting it makes
  the test occasionally pass and occasionally fail depending on
  RNG state inside dropout.
- **Batch of one.** A single-sample input cannot catch a reshape that
  mixes the batch axis into tokens or channels; keep the fixtures at
  `PARITY_BATCH = 2`.
- **Generalize only after one variant passes.** Adding a second
  variant before the first lands creates code paths that the first
  failed test cannot localize.
