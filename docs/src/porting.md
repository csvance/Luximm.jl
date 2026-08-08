```@meta
CurrentModule = Luximm
```

# Porting Backbones

This page is a contributor guide for adding a new `timm` backbone to
Luximm.jl. The bar is numeric parity with `timm` on the released
weights; the rest is mechanics.

If you are using Claude Code to drive the port, the same workflow is
encoded as an agent-facing skill at
`skills/timm-to-lux/SKILL.md` and loads automatically inside
this repo. The skill assumes the Kaimon REPL workflow described in
`.claude/skills/kaimon-julia/SKILL.md`. This page covers the same
ground for human contributors without those tools.

## Acceptance criteria

A new backbone is mergeable when all four hold:

1. **Pretrained parity.** Forward output of the Lux model with weights
   loaded via the closure returned by
   [`create_pretrained`](@ref) matches `timm`'s forward
   output on the same input. The bar is two-tier: **logits** are
   checked at an absolute max-abs-diff under `LOGITS_ATOL = 1f-3`, and
   **features** (`num_classes = 0`, and the `in_chans = 1` companion)
   are checked at a relative bar `max-abs-diff / max-abs(timm ref)`
   under `FEATURES_RTOL = 1f-4`. Existing models land well inside
   this: BiT ResNetV2-50 features around `1.5e-4` absolute (well
   under the relative bar at typical feature magnitudes), its logits
   around `2e-5`; ConvNeXtV2 atto comparable.
2. **Random-init parity.** With the Lux model initialized using the
   same `_init_weights` recipe `timm` uses for the family, and `timm`
   initialized with the same RNG seed, forward outputs match the
   same logits and features bars. See `_CN2_INIT` at
   `src/Models/ConvNeXtV2/Model.jl` (`truncated_normal(mean = 0f0,
   std = 0.02f0)` mirroring timm's `trunc_normal_(std = 0.02)`) for a
   worked example.
3. **State-dict round-trip.** The mapping function consumes every
   PyTorch `state_dict` key. Mappings raise on a missing key so
   silent random-init leaks cannot pass parity by coincidence; see
   the assertion block at the end of `convnextv2_mapping` in
   `src/Models/ConvNeXtV2/Model.jl`.
4. **Variant table entry.** New variants are registered in
   `<FAMILY>_VARIANTS` (`src/Models/<Family>/Config.jl`) with their
   HuggingFace repo id and default class count, and listed in the
   README backbone table.

## Why the bars are `~1e-3` (logits) and `~1e-4` (features), not `1e-5`

Float32 round-off accumulates through the depth of a network. Each
conv, norm, and weight-standardization stage reorders sums in ways
that differ from PyTorch's BLAS or cuDNN kernels even when the math
is semantically identical, so a 50-layer ResNet against the `timm`
reference reliably lands near `1e-4` absolute diff on order-one
activations. The shallower head (one pool plus one Dense) sits
closer to `1e-5` because almost no new accumulation happens after
the backbone.

The test gates are `LOGITS_ATOL = 1f-3` (absolute, logits) and
`FEATURES_RTOL = 1f-4` (relative, features). Both are tight enough
to catch the silent-divergence failure modes that actually matter
(cross-correlation vs convolution, population vs sample variance,
GELU approximation mismatch, axis permutations, missing norm
`epsilon`), and loose enough not to fail on legitimate float32
reordering. The features bar is relative because raw pre-norm
features on deep / wide backbones (large and xlarge ConvNeXt, huge
ConvNeXtV2) drift by `~1e-3` to `~2e-3` absolute even when their
downstream logits stay near `1e-5` after the LayerNorm + classifier
squashes them; a relative bar keeps the check scale-free across
tiny through huge variants. If your port reports a logits
max-abs-diff in the `1e-2` range, or a features relative diff in
the `1e-2` range or higher, that is a real bug, not round-off;
bisect it with the per-stage parity fixtures described below.

## Reference example

`src/Models/ResNetV2/` is the canonical, fully-worked port. Read it
first. It exercises most of the shared utilities in one place:

- Pre-activation residual blocks.
- Weight-standardized convolutions (`std_conv` from `Luximm.Layers`).
- GroupNorm with explicit epsilon.
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
`_dump_common.py` helpers and follows the `dump_<family>_io.py`
convention. Look at `test/parity/dump_resnetv2_bit_io.py` for the
template.

The fixture stores `/input` (deterministic random input, PyTorch NCHW
layout), `/state_dict/<key>` for every PyTorch parameter, and
`/output/features` plus optional `/output/logits`. The Julia side
reads it via `Luximm.Interop.read_parity`, which reverses the axes so
tensors arrive in Lux's WHCN layout.

Run the dump once per variant:

```
uv run python test/parity/dump_<family>_io.py --variant <timm_name> --out data/parity/<key>_io.h5
```

Optionally dump a single-channel companion fixture (`--in-chans 1`,
output `data/parity/<key>_in1c_io.h5`) so the `in_chans = 1` parity
test can run as well.

For a brand-new family, capture more than one fixture: the
end-to-end one gives a pass/fail signal with no localization power.
Add at least three random seeds and per-stage intermediates by
registering forward hooks on `model.stages[i]`. The per-stage
fixtures are the bisection tool when the end-to-end test fails.

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
- `Luximm.Interop.axis_reverse` and `Luximm.Interop.pyperm` are ready-made
  transforms for the cases where the HDF5-natural layout is not what
  you want, typically Dense weights and LayerNorm scale/bias.
- [`std_conv`](@ref), [`layernorm2d`](@ref), [`grn_layer`](@ref) are
  the building blocks that match `timm`'s `StdConv2dSame`,
  channel-axis `LayerNorm2d`, and Global Response Norm.

Keep the per-family weight mapping in `src/Models/<Family>/Model.jl`
as `<family>_mapping(state_dict, variant; prefix, num_classes,
in_chans)`, returning a `Vector{Tuple{String, Tuple{Vararg{Symbol}},
Function}}`. The `prefix` argument lets a backbone be nested under a
wrapper model.

### Phase 3: Implement the model with `@compact`

Lux's `@compact` is the right primitive for composing layers. The
pattern is fixed:

```julia
@compact(
    conv1 = Conv((3, 3), in_ch => out_ch; pad = 1, cross_correlation = true),
    norm1 = GroupNorm(out_ch, 32; affine = true, epsilon = 1f-5),
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
- **Norm defaults are not portable.** Always pass `epsilon` and
  `affine` explicitly on `GroupNorm`, `LayerNorm`, `BatchNorm`.
  PyTorch's `nn.GroupNorm` uses `eps = 1e-5`; Lux's default differs.
  Mismatched epsilons silently shift activations and look like a
  flaky parity failure.
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
dot-separated name lives at `<FAMILY>_VARIANTS[key].hf_repo`.

Pick one variant and finish it before adding the second. When a
model family has many variants, port the smallest first: it surfaces
every shared numeric trap with the fastest test loop, and the
mapping function written for one variant typically generalizes by
changing only `depths` or `widths`. Adding a second variant before
the first passes parity creates code paths nothing has exercised and
that the first failed test cannot localize.

### Phase 5: Add the loader

Each family exposes a `load_<family>_pretrained(ps, st, variant;
num_classes, in_chans, revision, cache_dir, prefix) -> (ps, st)`
function. All four share this signature: stateless families
(BiT/ConvNeXt/ConvNeXtV2) take `st` and return it unchanged; ResNet
mutates it to merge BatchNorm running stats. The flow is identical
across families:

1. Look up the variant in `<FAMILY>_VARIANTS`.
2. Validate `num_classes` against `default_num_classes` (or allow 0
   for backbone-only).
3. Call [`hf_hub_download`](@ref) to resolve the snapshot path under
   the HuggingFace Hub cache layout.
4. Load via [`load_safetensors_state_dict`](@ref). Pass
   `reverse_axes = true` so the safetensors arrays end up in the
   same WHCN layout the HDF5 fixtures produce; this lets one
   `<family>_mapping` function serve both fixture-driven tests and
   production loading.
5. Apply via [`apply_state_dict`](@ref).
6. For families with BatchNorm running stats (ResNet only),
   additionally apply the state dict to the model state via
   `<family>_state_mapping` and `apply_state_dict`.

Once the variant is registered in `<FAMILY>_VARIANTS`, it is reachable
through the family-agnostic [`create_pretrained`](@ref) and
[`create_model`](@ref) dispatchers automatically — no extra wiring.

Keep the constructor and the weight loading separate. The
constructor returns a `@compact` block; the loader takes the result
of `Lux.setup` and returns a new `(ps, st)`. Mixing the two inside
`@compact` makes the model unusable in tests that want a random
init.

### Phase 6: Verify parity end-to-end

The verification loop is layered. Run the cheap gates between
meaningful edits.

End-to-end first, against the fixture's `state_dict` (not the HF
download) so the test isolates the forward pass:

```julia
data = Luximm.Interop.read_parity(_FIXTURE_PATH)
model = <family>(variant; in_chans = 3, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
st = Lux.testmode(st)
ps = Luximm.Interop.apply_state_dict(ps, data.state_dict,
                                   <family>_mapping(data.state_dict, variant))
y, _ = model(data.input, ps, st)
expected = data.output["features"]
diff = maximum(abs.(y .- expected))
rel  = diff / max(maximum(abs.(expected)), eps(Float32))
@test rel < 1f-3
```

The convenience wrapper `scripts/test_variant.sh` runs exactly this
for one variant, dumping the fixture first if absent. See
[Testing](testing.md) for the variant-test workflow.

When parity fails, the per-stage and per-block fixtures pay off.
Walk the forward by hand: run the partial forward up to each stage
and compare against `data.output["stage_i"]`. The first stage where
parity breaks localizes the bug. Inside the failing stage, splice
the matching state-dict entries into a single block in isolation
and compare against the per-block fixture.

### Phase 7: Update the bookkeeping files

After parity passes, two repo files need to stay in sync. Forgetting
either is a silent regression in CI's path filtering or the
user-facing variant table.

1. `README.md` and the API reference. Update the backbone table at
   the top of `README.md`, and confirm the API reference page
   (`docs/src/api/models.md`) lists the new constructor and loader.
2. `ci/JimmCI/src/PathFilter.jl`. Update `_FAMILY_PREFIXES`,
   `_FAMILY_EXACT`, `ALL_FAMILIES`, `REPRESENTATIVE_VARIANT`, and
   the `_SHARED_*` lists as appropriate. Without this, CI will
   silently skip tests for the new code on PR-scope runs; only a
   full `jimm-ci --master` sweep would catch it.

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
- **GN/BN/LN defaults.** Always pass `epsilon` and `affine`
  explicitly. Never trust the Lux default to match the PyTorch
  default.
- **Variance correction.** Pass `corrected = false` when matching
  anything BN-style or WS-style.
- **Axis order on `apply_state_dict`.** HDF5 fixtures arrive
  reversed (Lux-natural). SafeTensors arrive PyTorch-natural by
  default; pass `reverse_axes = true` to
  [`load_safetensors_state_dict`](@ref) so both sources share one
  mapping function.
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
- **Generalize only after one variant passes.** Adding a second
  variant before the first lands creates code paths that the first
  failed test cannot localize.
