---
name: timm-to-lux
description: Workflow guide for porting a PyTorch `timm` model to a numerically-equivalent Lux.jl implementation in Luximm.jl. Load this skill whenever the task involves porting, converting, or translating a PyTorch model (especially anything from `timm.create_model`, `forward_features`, ResNet/ViT/EfficientNet/ConvNeXt/etc. backbones) to Lux; writing or editing Lux `@compact` blocks that must match a PyTorch reference; producing or consuming HDF5 parity fixtures, `read_parity`, `apply_state_dict`, or `check_parity` relative-L2 parity tests (`PARITY_RTOL`); loading `.safetensors` weights from HuggingFace Hub in a Julia context; or reasoning about PyTorch-vs-Lux numeric differences (cross-correlation vs convolution, padding semantics, norm epsilons, weight standardization, NCHW vs WHCN).
---

# Porting `timm` models to Lux.jl

A reference for converting a PyTorch `timm` backbone, layer, or building block into a numerically-equivalent Luximm.jl implementation that loads pretrained weights from HuggingFace.

`src/Models/ResNetV2/` (BiT ResNetV2: `bit_resnetv2`, `bit_resnetv2_mapping`, `_load_bit_resnetv2`) is the canonical, parity-verified port. The supporting utilities live in `src/Interop/` (`Parity.jl`, `SafeTensors.jl`, `HFHub.jl`), the shared layers in `src/Layers/`, and the Python sidecars in `test/parity/`. Use them as the pattern. Do not re-implement what is already there.

## 1. Workflow at a glance

Seven phases, run in order. Each phase has a cheap correctness gate before you move on.

1. Capture parity fixtures from `timm` in Python: end-to-end input/output, plus per-stage intermediates and one block per stage while debugging.
2. Stub the family's files following the existing families: `src/Models/<Family>/Config.jl` (variant table), `src/Models/<Family>/Model.jl`, `test/test_<family>.jl`, `test/parity/dump_<family>_io.py`.
3. Implement layers with `@compact`, matching PyTorch's numeric conventions explicitly.
4. Keep the forward pass autodiff- and Reactant-safe.
5. Wire pretrained `.safetensors` loading from HuggingFace.
6. Verify parity end-to-end, then narrow on any divergence using the per-stage and per-block fixtures.
7. Do the bookkeeping: dispatch chains, exports, test tables, README. See [Phase 7 of the porting guide](../../docs/src/porting.md).

Iterate in one Kaimon + Revise REPL throughout, restarting only when struct or include topology changes (section 9).

**Pick one variant and finish it before starting the next.** When a model family has many variants (ResNet-50/101/152, ViT-B/L/H, EfficientNet-B0..B7), porting the smallest variant first is almost always the right call: it surfaces every shared numeric trap (cross-correlation, padding, norm defaults, axis order) with the fastest test loop, and the weight-mapping function written for one variant typically generalizes by changing only `depths`/`widths`. Resist the temptation to architect for all variants up front. Land parity on one. Then add the second by parameterizing what differs, not by writing a second implementation.

## 2. Phase 1: capture parity fixtures from `timm`

A fixture is an HDF5 file built by a small Python sidecar in `test/parity/`, using `dump` from `test/parity/_dump_common.py`. It writes `/input`, a `/state_dict` group, and either `/output` as a single dataset or `/output/<name>` as a group of named outputs. The Julia side reads it with `Luximm.Interop.read_parity(path)`.

There are two kinds of fixture, and they differ in what they carry:

- **The committed parity fixture** (`data/parity/<variant_key>_io.h5`, written by `dump_<family>_io.py`) holds only the input and timm's outputs: `/output/features`, plus `/output/logits` for variants with a trained head. Its `/state_dict` group is an empty placeholder (`state_dict={}`). The parity tests load the released weights from HuggingFace through `create_pretrained`, so the test covers the production loading path. Model the new family's sidecar on an existing one, such as `dump_resnet_io.py`.
- **Debugging fixtures you dump while porting** can and should carry the full `model.state_dict()`, per-stage outputs and per-block I/O, so you can test the forward in isolation from the HF download and bisect a divergence. They stay local (`data/parity/` is git-ignored).

Every fixture input is a batch of `PARITY_BATCH = 2` distinct samples (from `_dump_common.py`). At batch 1, a forward that mixes the batch axis into tokens, heads or a reduction still matches exactly; at batch 2 it fails.

For debugging, capture more than the end-to-end output. The end-to-end check alone gives a pass/fail signal with no localization power. Add:

- **Multiple inputs.** A few random seeds and at least one off-default shape, dumped as `/output/seed0`, `/output/seed1`, etc. Catches seed-tuned bugs and shape-dependent errors.
- **Per-stage intermediates.** Register a `forward_hook` on each top-level stage module (`model.stages[0]`, `model.stages[1]`, ...) and store its output under `/output/stage0`, `/output/stage1`, etc. Required for bisecting divergence.
- **Per-block I/O.** For one representative block in each stage, dump its input and output as a separate fixture file (`<model>_block_stage1.h5`). Lets you test the block in isolation without standing up the whole forward.

A debugging dump looks like this:

```python
from _dump_common import PARITY_BATCH, dump

model = timm.create_model("resnetv2_50x1_bit.goog_in21k",
                          pretrained=True, in_chans=1, num_classes=0)
model.eval()

stage_outs = {}
hooks = [s.register_forward_hook(
            lambda m, _, o, k=f"stage{i}": stage_outs.__setitem__(k, o.detach()))
         for i, s in enumerate(model.stages)]

gen = torch.Generator().manual_seed(0)
inp = torch.randn(PARITY_BATCH, 1, 224, 224, generator=gen)
with torch.no_grad():
    final = model.forward_features(inp)
for h in hooks: h.remove()

dump(OUT_PATH, inp=inp, state_dict=model.state_dict(),
     out={"features": final, **stage_outs})
```

`in_chans=1` triggers `timm.layers.adapt_input_conv` so the stem weight in `state_dict` is already collapsed to one input channel. Do not re-collapse on the Julia side when loading a fixture's `state_dict`; when loading the 3-channel HF weights, use `adapt_input_conv(in_chans)` as the stem transform instead. Use `forward_features` when porting the backbone, `forward` when porting the full classifier; the fixture and the Julia forward must agree on which one.

## 3. Phase 2: reuse the existing utilities, do not re-implement

The interop functions are load-bearing (`Luximm.Interop.*`; `apply_state_dict` is also exported from `Luximm`):

- `read_parity(path)` returns `(input, state_dict, output)` as `Float32` arrays. Tensors come back in their **HDF5-natural Julia layout**, which is the **reverse** of the PyTorch logical axis order. PyTorch `(N, C, H, W)` becomes Julia `(W, H, C, N)`, which is exactly Lux's WHCN. Conv weight `(out, in, kH, kW)` becomes `(kW, kH, in, out)`, which is exactly Lux's Conv layout. Bias `(C,)` stays `(C,)`. For most parameters and activations, the HDF5-natural layout is what you want and the transform is `identity`.
- `load_safetensors_state_dict(path)` reads a `.safetensors` file into the same reversed layout by default (`reverse_axes = true`), so one mapping function serves both fixture `state_dict`s and HF weights.
- `apply_state_dict(ps, state_dict, mapping)` rebuilds the parameter `NamedTuple` by setting leaves from the dict. Mapping entries are `(pytorch_key, lux_path_tuple, transform)`. Non-mutating; bind the result.
- Transforms for the cases where the HDF5-natural layout is *not* what you want: `axis_reverse(a)` and `pyperm(perm)` for permutations, `as_channel4d` for a `(C,)` norm parameter feeding `layernorm2d`'s `(1, 1, C, 1)` leaves, `as_token_norm` for `vit_layernorm`'s `(C, 1, 1)` leaves, and `adapt_input_conv(in_chans)` for the stem weight.

Keep the weight mapping function in the family's `Model.jl`, named `<family>_mapping(state_dict, variant; load_classifier = false, in_chans = 3, prefix = ())`, returning a vector of `(pytorch_key, lux_path, transform)` triples. The `prefix` arg lets a backbone be nested under a wrapper model (e.g. `prefix = (:backbone,)`). `bit_resnetv2_mapping` in `src/Models/ResNetV2/Model.jl` is the reference. BatchNorm families also need a `<family>_state_mapping` for the running statistics in `st` (see `resnet_state_mapping`).

## 4. Phase 3: implement layers with `@compact`

Lux's `@compact` is the right primitive for composing layers (https://lux.csail.mit.edu/stable/api/Lux/utilities#Lux.@compact). The pattern is fixed:

```julia
@compact(
    conv1 = Conv((3, 3), in_ch => out_ch; pad = 1, cross_correlation = true),
    norm1 = Layers.LuximmGroupNorm(out_ch, 32; epsilon = 1f-5),
) do x
    @return NNlib.relu.(norm1(conv1(x)))
end
```

**Build normalization with Luximm's own layers**, never `Lux.LayerNorm`, `Lux.GroupNorm` or `Statistics.var` directly: `Layers.LuximmGroupNorm`, `Layers.LuximmLayerNorm`, `layernorm2d`, `vit_layernorm`, and the functions `Layers.layernorm`, `Layers.groupnorm`, `Layers.mean_var`. They have the same parameter trees as Lux's layers, and they are what the Reactant extension (`ext/LuximmReactantExt.jl`) accelerates; a raw Lux norm or `var` call misses it. BatchNorm stays `Lux.BatchNorm`, which LuxLib already specializes for Reactant.

Numeric conventions that bite if you forget them:

- **Cross-correlation, always.** PyTorch's `Conv2d` is cross-correlation; Lux's `Conv` defaults to true convolution (kernel-flipped). Pass `cross_correlation = true` to every `Conv`. Without it, weights load with the right shape but produce mirrored outputs. When you must drop into `NNlib.conv` directly (weight standardization is the canonical case), pass `flipkernel = true` to `NNlib.DenseConvDims`. They are the same semantic.
- **Explicit padding when zero-padding matters.** `Conv((k, k), ...; pad = p)` works for symmetric same-value padding. For asymmetric padding, or for pooling that must pad with zeros instead of `-Inf`, call `NNlib.pad_zeros(x, (l, r, t, b, 0, 0, 0, 0))` *before* the op and use `pad = 0` on the op itself. timm's BiT stem is the canonical case: `ConstantPad2d(value=0)` then `MaxPool` with no padding; see the stem in `src/Models/ResNetV2/Model.jl`.
- **Pass `epsilon` explicitly on every norm.** Lux's norms and PyTorch's `GroupNorm`/`BatchNorm`/`LayerNorm` all default to `1e-5`, but timm overrides it in places: `LayerNorm2d` (ConvNeXt, CoAtNet) and ViT use `1e-6`. A mismatched epsilon silently shifts activations by an amount small enough to pass the parity bar, so read the logged rel-L2, not just the pass.
- **Variance corrections.** Sample variance (Bessel-corrected, the Julia default) and population variance (`corrected = false`, what PyTorch uses for BN-style stats and for weight standardization) differ by a factor of `N/(N-1)`. Pass `corrected = false` whenever you're matching a BN/WS-style operation. See `std_conv` in `src/Layers/StdConv.jl`, which calls `mean_var(w; dims = 1, corrected = false)`.
- **`Lux.testmode(st)` for parity tests.** Otherwise BN running stats update and dropout activates, neither of which is what `model.eval()` does on the PyTorch side.

**Multi-variant architectures.** Even though you port one variant first, design the constructor so the next variant is a parameter change. Each family keeps a `<FAMILY>_VARIANTS` table in `Config.jl` mapping the variant key (the timm name with the dot rewritten as an underscore, e.g. `:resnet50_a1_in1k`) to a config struct holding the architecture (`widths`, `depths`, ...), `hf_repo` (`"timm/resnet50.a1_in1k"`), `default_num_classes` and `default_input_size`. The family constructor takes the variant key:

```julia
function resnet(variant::Symbol; in_chans::Int = 3, num_classes::Int = 0,
                features_only::Bool = false, out_indices = nothing)
    cfg = RESNET_VARIANTS[variant]
    ...
end
```

But do not add the bigger variants to the table until the first passes parity. Adding them earlier creates code paths nothing has exercised and that the first failed parity test will not localize. Each new variant is its own small port: dump a fresh fixture, run the parity test, add the entry.

**User-facing knobs every constructor exposes:**

- `in_chans::Int = 3`.
- `num_classes::Int = 0`. Zero means the classifier head is omitted and the forward returns features. (`create_model` and `create_pretrained` default to the variant's `default_num_classes` instead.)
- `features_only` / `out_indices` for families with a feature pyramid.
- Anything else the architecture genuinely exposes (e.g. ConvNeXt's `conv_mlp`). Do not invent knobs for parity completeness; only mirror what `timm.create_model` accepts.
- The constructor never takes `pretrained`. Weight loading is a separate private `_load_<family>(ps, st, variant; ...) -> (ps, st)`, reached through the `create_pretrained(variant) -> (model, load)` closure. Keep the constructor pure so tests can build random-init models without touching the network and so the model composes cleanly inside a larger `@compact` block.

## 5. Phase 4: autodiff- and Reactant-safe code

Every forward must work under Reactant + Enzyme (the `reactant` test family exercises this) and must not assume CPU arrays. The rules:

- **No in-place writes to params or activations.** No `x[i] = ...`, no `setindex!`, no `.+=`. Build new arrays with `cat`, `reshape`, `permutedims`, broadcasting, and `vcat`/`hcat`.
- **No `Array(x)`, `collect(x)`, or `cpu(x)` inside the forward.** They move device arrays back to host and break the gradient graph. Constants the forward needs (coordinate grids, fixed buffers) belong in the `@compact` capture list.
- **No scalar indexing.** Calls like `x[1, 2, 3]` will throw or warn on device arrays and Reactant tracers. Express the same access through broadcasting, `getindex` with ranges, or a reshape.
- **No `if`/`else` on tensor values in the forward.** Use masks, `clamp`, `NNlib.relu`, `ifelse.(...)`, or the appropriate activation primitive. Branching on sizes and configuration is fine; they are static at trace time.
- **Keep the structure type-stable.** Tuples of taps, `Val` for static choices; `src/Models/FeatureInfo.jl` shows the pattern for `out_indices`.

The forward should *look* like math: a sequence of broadcasts and tensor ops with no scalar control flow.

## 6. Phase 5: HuggingFace `.safetensors` weights

Production users build the model and load the released weights in one call: `model, load = create_pretrained(:resnet50_a1_in1k); ps, st = Lux.setup(rng, model); ps, st = load(ps, st)`. The parity tests use the same path.

Steps:

1. **Register the variant** in `<FAMILY>_VARIANTS` with its `hf_repo` (`"timm/<timm name>"`). That is the only place the weights' location is written.
2. **Write the loader** with the shared signature:

   ```julia
   function _load_resnet(ps, st, variant::Symbol;
           in_chans::Int, num_classes::Int, revision::AbstractString,
           cache_dir::AbstractString, prefix::Tuple{Vararg{Symbol}} = ())
       cfg = RESNET_VARIANTS[variant]
       # Load the head only when it matches the checkpoint's; otherwise warn and leave it at
       # its random init for fine-tuning.
       load_classifier = num_classes > 0 && num_classes == cfg.default_num_classes
       path = hf_hub_download(cfg.hf_repo, "model.safetensors";
                              revision = revision, cache_dir = cache_dir)
       sd = load_safetensors_state_dict(path)     # axes already reversed
       param_mapping = resnet_mapping(sd, variant; load_classifier, in_chans, prefix)
       state_mapping = resnet_state_mapping(sd, variant; prefix)
       _validate_resnet_consumed_keys(sd, param_mapping, state_mapping; load_classifier)
       ps = apply_state_dict(ps, sd, param_mapping)
       st = apply_resnet_state_dict(st, sd, state_mapping)
       return ps, st
   end
   ```

   (Abridged from `src/Models/ResNet/Model.jl`.) LayerNorm / GroupNorm families thread `st` through unchanged. BatchNorm families (ResNet, SE-ResNet, CoAtNet, `vgg*_bn`) also load the running statistics into `st` through a `<family>_state_mapping`. Validate that every checkpoint key was consumed, so a misspelled key fails loudly instead of silently leaving a parameter at its random init. `hf_hub_download` uses the standard HuggingFace cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or `~/.cache/huggingface/hub`), shared with Python's `huggingface_hub`, and sends `HUGGING_FACE_HUB_TOKEN` when set.
3. **Wire the family into the dispatch chains.** `src/Models/Models.jl` has a manual `haskey(<FAMILY>_VARIANTS, variant)` branch per family in `create_model`, `default_num_classes`, `feature_info` and `_load_pretrained`; a new family needs a branch in each. A new *variant* of an existing family needs only its table row.

## 7. Phase 6: verify parity, narrow on divergence

The verification loop is layered. Run the cheap gates between every meaningful edit.

Every check, logits and features alike, is the relative L2 error `‖y - ref‖ / ‖ref‖` under `PARITY_RTOL = 1f-3`, through `check_parity(variant, check, y, ref)` in `test/_parity_tol.jl`. A correct port lands at `1e-7` to `1e-5`; real porting bugs land at `1e-2` and above. A pass at `1e-4` or more on a port that should be exact still deserves a look (a norm `epsilon`, a variance correction).

End to end, through the production loading path (what `run_variant_parity` in `test/_parity_helpers.jl` does):

```julia
data = Luximm.Interop.read_parity(fixture_path)
model, load = create_pretrained(:resnet50_a1_in1k)          # trained head
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, Lux.testmode(st))
y, _ = model(data.input, ps, st)
check_parity(:resnet50_a1_in1k, "logits", y, data.output["logits"])

model_f, load_f = create_pretrained(:resnet50_a1_in1k; num_classes = 0)
ps_f, st_f = Lux.setup(Xoshiro(0), model_f)
ps_f, st_f = load_f(ps_f, Lux.testmode(st_f))
y_f, _ = model_f(data.input, ps_f, st_f)
check_parity(:resnet50_a1_in1k, "features", y_f, data.output["features"])
```

`scripts/test_variant.sh <variant_key>` dumps the committed fixtures if missing and runs exactly these checks for one variant.

To isolate the forward from the weight download, run the same comparison against a debugging fixture's `state_dict`: `ps = apply_state_dict(ps, data.state_dict, resnet_mapping(data.state_dict, variant; load_classifier = true))`.

When it fails, the per-stage and per-block debugging fixtures earn their keep. Walk the forward by hand:

- For each stage, run the partial forward up to that stage (the simplest way is a transient `@compact` that drops in the relevant sub-models, or peeling layers off the full `Chain`). Compare against `data.output["stage0"]`, `data.output["stage1"]`, ... The first stage where parity breaks localizes the bug to that stage's blocks.
- Inside the failing stage, instantiate a single block (`preact_bottleneck(in, out, stride; ...)`), call `Lux.setup`, splice in the matching `state_dict` entries via a small mapping, and check against the per-block fixture. This is faster than printing intermediate tensors and survives random init.

Divergence-pattern playbook:

- **Off by a little, structured** (rel-L2 around `1e-5` to `1e-3`). Float32 cast on the Python side is already in `_dump_common.to_numpy` so that's not it. Suspect `corrected = false` or an `epsilon` mismatch on a norm.
- **Off by a lot, structured.** Suspect (a) `cross_correlation = true` missing on a `Conv`, (b) `-Inf` vs zero padding on a `maxpool`, (c) flipped axes from a mapping transform mismatch.
- **Off by a lot, random-looking.** Suspect that you applied `apply_state_dict` to the wrong sub-tree (`prefix` wrong) or that a `state_dict` key spelled differently in your mapping silently fell back to random init. Make your mapping function `error` on any `state_dict` key it did not consume; `convnextv2_mapping` in `src/Models/ConvNeXtV2/Model.jl` ends with that check.
- **Only one sample of the batch is wrong.** A reshape or reduction mixes the batch axis into another axis.
- **Stages 0..k pass, stage k+1 fails.** Confirms the bug is local to stage k+1's first block; bisect within that block (norm before conv? conv first vs norm first?). The order of operations in pre-activation ResNet blocks differs from post-activation; both compile, only one matches.

## 8. Phase 7: bookkeeping

A ported family is not done until the repo knows about it. The full checklist is Phase 7 of `docs/src/porting.md`: exports in `src/Models/Models.jl` and `src/Luximm.jl`, the dispatch chains above, the test tables in `test/_filter.jl` (`FAMILY_FILES`, `PARITY_FAMILIES`, `_jimm_variant_family`) and `test/test_interface.jl` (`VARIANT_TABLES`, `REPRESENTATIVES`), the `case` block in `scripts/test_variant.sh`, `PYRAMID_FAMILIES` / `FEATURES_ONLY_VARIANTS` in `test/parity/dump_features_only_io.py` for a pyramid family, and the backbone tables in `README.md` and `docs/src/index.md`. Missing any of them makes the new code silently untested.

## 9. Iterate via Kaimon + Revise

Stand up one `start_session()`-attached REPL and keep it alive across the whole port. Edit `.jl` files; Revise picks up function-body changes between `ex` calls. Re-run the parity test via `ex(e="include(\"test/test_<family>.jl\")", ses=<key>)`. Restart only when:

- You added or removed a struct field, including `@compact` layer fields.
- You added or removed an `include`.
- A long-lived REPL state (a cached `Lux.setup` result, a fixture loaded into a global) is stale.

The per-block parity tests are well-suited to be a `smoke_test()`-style gate: a few microseconds per call, structured small output, runs after every edit. The full end-to-end parity is more expensive and goes at the end of each iteration.

## 10. Pitfalls reference

A running checklist; every item below has cost real time on a previous port.

- **Cross-correlation.** Every `Conv` needs `cross_correlation = true`. Every `NNlib.conv` needs `flipkernel = true` on its `DenseConvDims`. Same semantic, two flags.
- **Zero-padded pooling.** `NNlib.maxpool(pad=1)` pads with `-Inf`. `nn.MaxPool2d` with `padding=1` pads with zero. When the input has negative values (post-norm activations, leaky-relu outputs), this diverges. `pad_zeros` first, then pool with `pad = 0`.
- **Norm epsilons.** Always pass `epsilon` explicitly; timm's `LayerNorm2d` and ViT use `1e-6`, not the `1e-5` default.
- **Luximm norms, not Lux norms.** `Lux.LayerNorm` / `Lux.GroupNorm` work but miss the Reactant extension; use `layernorm2d`, `vit_layernorm`, `Layers.LuximmLayerNorm`, `Layers.LuximmGroupNorm`.
- **Variance correction.** Pass `corrected = false` when matching anything BN-style or WS-style.
- **Weight axis order.** `read_parity` and `load_safetensors_state_dict` both return the reversed (Lux-natural) layout. Write mapping transforms against that layout, not PyTorch's.
- **`forward_features` vs `forward`.** The fixture and the Julia forward must agree on which one was dumped. Mismatch shows up as a `(N, num_classes)` vs `(N, C, H, W)` shape error if you're lucky, a silent wrong number if you're not.
- **WHCN vs NCHW.** `read_parity` returns WHCN-natural already. If you bypass it and call `HDF5.read` directly, you get reversed-PyTorch (which is WHCN), which is what you want, but it's easy to convince yourself you need to permute "back" and break it.
- **Pre-activation block order.** norm -> activation -> conv vs conv -> norm -> activation. Both compile. Only one matches the upstream architecture.
- **timm stem adaptation for `in_chans != 3`.** A fixture dumped with `in_chans=1` already has the collapsed stem in its `state_dict`. HF weights are 3-channel; adapt them with `adapt_input_conv(in_chans)` in the mapping.
- **`Lux.testmode(st)`.** Required for parity. Forgetting it makes the test occasionally pass and occasionally fail depending on RNG state inside dropout.
- **Batch of one.** A fixture with a single sample cannot catch batch-mixing bugs; keep `PARITY_BATCH = 2`.
- **Don't generalize until one variant works.** Adding `:resnet101_a1_in1k` to the table before `:resnet50_a1_in1k` passes parity creates code paths that the first failed test cannot localize.
