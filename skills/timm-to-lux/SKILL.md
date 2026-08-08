---
name: timm-to-lux
description: Workflow guide for porting a PyTorch `timm` model to a numerically-equivalent Lux.jl implementation. Load this skill whenever the task involves porting, converting, or translating a PyTorch model (especially anything from `timm.create_model`, `forward_features`, ResNet/ViT/EfficientNet/ConvNeXt/etc. backbones) to Lux; writing or editing Lux `@compact` blocks that must match a PyTorch reference; producing or consuming HDF5 parity fixtures, `read_parity`, `apply_state_dict`, or `@test isapprox` parity tests; loading `.safetensors` weights from HuggingFace Hub in a Julia context; or reasoning about PyTorch-vs-Lux numeric differences (cross-correlation vs convolution, padding semantics, GroupNorm/BatchNorm defaults, weight standardization, NCHW vs WHCN). This skill layers on top of `kaimon-julia`, which remains the source of truth for driving the Julia REPL.
---

# Porting `timm` models to Lux.jl

A reference for converting a PyTorch `timm` backbone, layer, or building block into a numerically-equivalent Lux.jl implementation that loads pretrained weights from HuggingFace. Read `kaimon-julia` first; this skill assumes the REPL workflow it describes.

The repository at `campfire/lib/MMILux.jl` contains a worked, parity-verified port (`BiTResNet.jl`) plus the supporting utilities (`Parity.jl`, the `test/parity/` Python sidecars). Use it as the canonical pattern. Do not re-implement what is already there.

## 1. Workflow at a glance

Seven phases, run in order. Each phase has a cheap correctness gate before you move on.

1. Capture parity fixtures from `timm` in Python: end-to-end input/output, per-stage intermediates, one block per stage.
2. Stub the host Julia package layout (`src/<Model>.jl`, `test/test_<model>.jl`, `test/parity/dump_<model>.py`, `data/parity/<model>.h5`) following the MMILux conventions. The host may be MMILux.jl (for internal layers) or a fresh standalone package intended for public release; see section 3 for which utilities to vendor in the latter case.
3. Implement layers with `@compact`, matching PyTorch's numeric conventions explicitly.
4. Keep the forward pass autodiff- and GPU-safe by default.
5. Wire pretrained `.safetensors` loading from HuggingFace as the production weight path.
6. Verify parity end-to-end, then narrow on any divergence using the per-stage and per-block fixtures.
7. Iterate via Kaimon + Revise, restarting the REPL only when struct or include topology changes.

**Pick one variant and finish it before starting the next.** When a model family has many variants (ResNet-50/101/152, ViT-B/L/H, EfficientNet-B0..B7), porting the smallest variant first is almost always the right call: it surfaces every shared numeric trap (cross-correlation, padding, norm defaults, axis order) with the fastest test loop, and the weight-mapping function written for one variant typically generalizes by changing only `depths`/`widths`. Resist the temptation to architect for all variants up front. Land parity on one. Then add the second by parameterizing what differs, not by writing a second implementation.

## 2. Phase 1: capture parity fixtures from `timm`

The fixture is an HDF5 file built by a small Python sidecar in `test/parity/`. Reuse `_dump_common.py` from `campfire/lib/MMILux.jl/test/parity/_dump_common.py` verbatim (copy it into the target project, or symlink it if the project shares a workspace with campfire). It writes `/input`, `/state_dict/<key>` for every PyTorch parameter, and either `/output` as a single dataset or `/output/<name>` as a group of named outputs. The Julia side reads it via `read_parity(path)` (imported from MMILux or vendored into the host package, per section 3).

Capture more than one fixture per port. The end-to-end one alone gives a pass/fail signal with no localization power. Add:

- **Multiple inputs.** Three to five different random seeds and at least one off-default shape, dumped as `/output/seed0`, `/output/seed1`, etc. Catches seed-tuned bugs and shape-dependent errors.
- **Per-stage intermediates.** Register a `forward_hook` on each top-level stage module (`model.stages[0]`, `model.stages[1]`, ...) and store its output under `/output/stage0`, `/output/stage1`, etc. Required for bisecting divergence.
- **Per-block I/O.** For one representative block in each stage, dump its input and output as a separate fixture file (`<model>_block_stage1.h5`). Lets you test the block in isolation without standing up the whole forward.

The Python loop looks like this:

```python
model = timm.create_model("resnetv2_50x1_bit.goog_in21k",
                          pretrained=True, in_chans=1, num_classes=0)
model.eval()

stage_outs = {}
hooks = [s.register_forward_hook(
            lambda m, _, o, k=f"stage{i}": stage_outs.__setitem__(k, o.detach()))
         for i, s in enumerate(model.stages)]

inp = torch.randn(1, 1, 224, 224)
with torch.no_grad():
    final = model.forward_features(inp)
for h in hooks: h.remove()

out = {"final": final, **stage_outs}
dump(OUT_PATH, inp=inp, state_dict=model.state_dict(), out=out)
```

`in_chans=1` triggers `timm.layers.adapt_input_conv` so the stem weight in `state_dict` is already collapsed to one input channel. Do not re-collapse on the Julia side. Use `forward_features` when porting the backbone, `forward` when porting the full classifier; the fixture and the Julia forward must agree on which one.

## 3. Phase 2: reuse the existing utilities, do not re-implement

First, pick the host package. Two cases:

- **The port lives in MMILux.jl** (the existing campfire-internal package). Import the utilities below directly: `using MMILux: read_parity, apply_state_dict, axis_reverse, pyperm`. This is the right home for internal layers and one-off ports, not for code intended for public release.
- **The port lives in a new standalone Julia package**, intended for eventual publication of the backbone library separate from MMILux. In this case, vendor the same four utilities into the new package's `src/Parity.jl` verbatim (it is ~100 lines of permissively-structured code), preserving the function names and signatures so the rest of this skill's guidance applies unchanged. Do not take a hard dependency on MMILux from a package that is meant to be public.

Either way, the four functions below are load-bearing and must be reachable by these exact names:

- `read_parity(path)` returns `(input, state_dict, output)` as `Float32` arrays. Tensors come back in their **HDF5-natural Julia layout**, which is the **reverse** of the PyTorch logical axis order. PyTorch `(N, C, H, W)` becomes Julia `(W, H, C, N)`, which is exactly Lux's WHCN. Conv weight `(out, in, kH, kW)` becomes `(kW, kH, in, out)`, which is exactly Lux's Conv layout. Bias `(C,)` stays `(C,)`. For most parameters and activations, the HDF5-natural layout is what you want and the transform is `identity`.
- `apply_state_dict(ps, state_dict, mapping)` rebuilds the parameter `NamedTuple` by setting leaves from the dict. Mapping entries are `(pytorch_key, lux_path_tuple, transform)`. Non-mutating; bind the result.
- `axis_reverse(a)` and `pyperm(perm)` are the two ready-made transforms for the cases where the HDF5-natural layout is *not* what you want (a custom tensor whose Julia axes were designed in PyTorch order, or a layer like DSNT2D whose output axes are hand-arranged).

Keep the weight mapping function in the model file, named `<model>_mapping(state_dict; prefix::Tuple{Vararg{Symbol}} = ())`, returning a `Vector{Tuple{String, Tuple{Vararg{Symbol}}, Function}}`. The `prefix` arg lets a backbone be nested under a wrapper model (e.g. `prefix = (:backbone,)`). `BiTResNet.jl:162-204` is the reference.

## 4. Phase 3: implement layers with `@compact`

Lux's `@compact` is the right primitive for composing layers (https://lux.csail.mit.edu/stable/api/Lux/utilities#Lux.@compact). The pattern is fixed:

```julia
@compact(
    conv1 = Conv((3, 3), in_ch => out_ch; pad = 1, cross_correlation = true),
    norm1 = GroupNorm(out_ch, 32; affine = true, epsilon = 1f-5),
) do x
    @return NNlib.relu.(norm1(conv1(x)))
end
```

Numeric conventions that bite if you forget them:

- **Cross-correlation, always.** PyTorch's `Conv2d` is cross-correlation; Lux's `Conv` defaults to true convolution (kernel-flipped). Pass `cross_correlation = true` to every `Conv`. Without it, weights load with the right shape but produce mirrored outputs. When you must drop into `NNlib.conv` directly (weight standardization is the canonical case), pass `flipkernel = true` to `NNlib.DenseConvDims`. They are the same semantic.
- **Explicit padding when zero-padding matters.** `Conv((k, k), ...; pad = p)` works for symmetric same-value padding. For asymmetric padding, or for pooling that must pad with zeros instead of `-Inf`, call `NNlib.pad_zeros(x, (l, r, t, b, 0, 0, 0, 0))` *before* the op and use `pad = 0` on the op itself. timm's BiT stem is the canonical case: `ConstantPad2d(value=0)` then `MaxPool` with no padding. The Lux equivalent is in `BiTResNet.jl:38-39`.
- **Norm defaults are not portable.** Always pass `epsilon` and `affine` explicitly on `GroupNorm`/`LayerNorm`/`BatchNorm`. PyTorch's `nn.GroupNorm` uses `eps=1e-5`; Lux's default differs. Mismatched epsilons silently shift activations and look like a flaky parity failure.
- **Variance corrections.** Sample variance (`Bessel-corrected`, the Julia default) and population variance (`corrected = false`, what PyTorch uses for BN-style stats and for weight standardization) differ by a factor of `N/(N-1)`. Pass `corrected = false` to `var` whenever you're matching a BN/WS-style operation. See `std_conv` at `BiTResNet.jl:73-75`.
- **`Lux.testmode(st)` for parity tests.** Otherwise BN running stats update and dropout activates, neither of which is what `model.eval()` does on the PyTorch side.

**Multi-variant architectures.** Even though you port one variant first, design the constructor so the next variant is a parameter change. Lift `widths`, `depths`, `strides` into a tuple inside one shared constructor. Expose family variants through a top-level dispatcher:

```julia
const _RESNET_SPECS = Dict(
    :resnet50  => (widths = (256, 512, 1024, 2048), depths = (3, 4, 6, 3)),
    :resnet101 => (widths = (256, 512, 1024, 2048), depths = (3, 4, 23, 3)),
    :resnet152 => (widths = (256, 512, 1024, 2048), depths = (3, 8, 36, 3)),
)

function resnet(family::Symbol; in_chans::Int = 3, num_classes::Int = 0,
                pretrained::Bool = false)
    spec = _RESNET_SPECS[family]
    return _resnet_impl(; spec..., in_chans, num_classes, pretrained, family)
end
```

But do not add `:resnet101` and `:resnet152` to the dict until `:resnet50` passes parity. Adding them earlier creates code paths nothing has exercised and that the first failed parity test will not localize. Each new variant is its own small port: dump a fresh fixture, run the parity test, add the entry.

**User-facing knobs every constructor exposes:**

- `in_chans::Int` (default matches `timm`'s default for the family, typically 3).
- `num_classes::Int = 0`. Zero means the classifier head is omitted and the forward returns features.
- Anything else the architecture genuinely exposes: `drop_rate`, `stem_type`, etc. Do not invent knobs for parity completeness; only mirror what `timm.create_model` accepts.
- The family constructor itself never takes `pretrained`. Weight loading is a separate `load_<family>_pretrained(ps, st, variant; ...) -> (ps, st)` function, reachable through the family-agnostic `load_pretrained(ps, st, variant; ...)` dispatcher in `src/Models/Models.jl`. Keep the constructor pure so tests can build random-init models without touching the network and so the model composes cleanly inside a larger `@compact` block.

## 5. Phase 4: autodiff- and GPU-safe code by default

Every forward must work under at least Zygote and Reactant, and must not assume CPU arrays. The rules:

- **No in-place writes to params or activations.** No `x[i] = ...`, no `setindex!`, no `.+=`. Build new arrays with `cat`, `reshape`, `permutedims`, broadcasting, and `vcat`/`hcat`. `DSNT.jl:60-66` shows the cat-based stacking pattern instead of an indexed assignment.
- **No `Array(x)`, `collect(x)`, or `cpu(x)` inside the forward.** They move GPU arrays back to host and break the gradient graph. Constants the forward needs (coordinate grids, normalized linspaces, fixed buffers) belong in the `@compact` capture list. Lux's `setup` will move them to the active device automatically.
- **No scalar indexing.** Calls like `x[1, 2, 3]` in a kernel context will throw or warn on `CuArray` and on Reactant tracers. Express the same access through broadcasting, `getindex` with ranges, or a reshape.
- **No `if`/`else` on tensor values in the forward.** Branching on `if x > 0` against an `AbstractArray` is a non-starter under autodiff. Use masks, `clamp`, `NNlib.relu`, `ifelse.(...)`, or the appropriate activation primitive.
- **Use `Lux.testmode(st)` for parity tests.** Otherwise BN running stats update and any dropout activates.

The forward should *look* like math: a sequence of broadcasts and tensor ops with no scalar control flow.

## 6. Phase 5: HuggingFace `.safetensors` for the production weight path

Parity fixtures bake the `state_dict` into the HDF5 file so the test can run offline. Production users should not pay that round trip; they build the model with `create_model(:resnet50)`, run `Lux.setup`, then stream the weights in from HuggingFace via `load_pretrained(ps, st, :resnet50; ...)`.

Steps:

1. **Add SafeTensors.jl to the host package.** From inside Kaimon: `pkg_add(packages=["SafeTensors", "Downloads"], ses=<key>)`. Downloads is stdlib but must be in `Project.toml` if a package module uses it.
2. **Pin the URL per variant in source.** Use the `timm` Hugging Face repos under `https://huggingface.co/timm/`. Resolve the actual `.safetensors` URL with `/resolve/main/model.safetensors`. Comment the timm canonical name next to the constant so the URL can be re-derived if it ever drifts:

   ```julia
   # timm: resnet50.a1_in1k
   const RESNET50_URL =
       "https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/model.safetensors"
   ```

3. **Define one download helper per package, not per model:**

   ```julia
   function _hf_download(url::AbstractString, dest::AbstractString)
       isfile(dest) && return dest
       mkpath(dirname(dest))
       headers = Pair{String,String}[]
       token = get(ENV, "HUGGING_FACE_HUB_TOKEN", "")
       isempty(token) || push!(headers, "Authorization" => "Bearer $token")
       Downloads.download(url, dest; headers = headers)
       return dest
   end
   ```

   The token is optional. Public `timm` weights download anonymously; the header only matters for gated or private repos. Cache under `joinpath(@__DIR__, "..", "weights", "<file>")` to match the BiT convention, or under `~/.cache/<pkg>/` if the host package prefers a user-scoped cache.

4. **Define one loader per family with the shared signature `(ps, st, variant; kwargs...) -> (ps, st)`:**

   ```julia
   function load_resnet_pretrained(ps, st, variant::Symbol;
           num_classes::Int = 0, in_chans::Int = 3,
           revision::AbstractString = "main",
           cache_dir::AbstractString = hf_hub_cache_dir(),
           prefix::Tuple{Vararg{Symbol}} = ())
       cfg = RESNET_VARIANTS[variant]
       path = hf_hub_download(cfg.hf_repo, "model.safetensors";
                              revision = revision, cache_dir = cache_dir)
       sd = load_safetensors_state_dict(path)
       ps = apply_state_dict(ps, sd, resnet_mapping(sd, variant; prefix = prefix))
       # If the family has BatchNorm running stats, also apply them into `st`.
       return ps, st
   end
   ```

   Stateless families (GroupNorm / LayerNorm) just thread `st` through
   unchanged. ResNet-style families mutate `st` via a parallel
   `<family>_state_mapping` and `apply_state_dict`. The uniform return
   shape is what lets `load_pretrained(ps, st, variant)` dispatch on the
   symbol.

   **Axis-order gotcha.** `SafeTensors.load` returns arrays in PyTorch logical axis order (NCHW for activations, `(out, in, kH, kW)` for conv weights), not the reversed layout that `read_parity` produces from HDF5. Two acceptable resolutions:
   - **Recommended:** normalize the SafeTensors dict to the HDF5-natural reversed layout once at load time by applying `axis_reverse` to every tensor whose Julia layout in `apply_state_dict` is reversed. Then the same `<model>_mapping` works for both fixture-driven tests and production loading.
   - **Alternative:** keep two mapping functions, one for fixtures and one for safetensors. More code; easier to silently drift.

5. **Family-agnostic dispatch is already wired.** Once the family's
   `<FAMILY>_VARIANTS` dict and `load_<family>_pretrained` exist, the
   variant becomes reachable through `create_model(variant; ...)` and
   `load_pretrained(ps, st, variant; ...)` in `src/Models/Models.jl`
   automatically — no extra plumbing per port. Both entry points are
   type-stable on the Union axis (`create_model` always returns just
   the model, `load_pretrained` always returns `(ps, st)`). Keep the
   constructor and the weight loading separate so test code can build
   random-init models without network access and so the model
   composes cleanly inside a larger `@compact` block.

## 7. Phase 6: verify parity, narrow on divergence

The verification loop is layered. Run the cheap gates between every meaningful edit.

End-to-end first, against the fixture's `state_dict` (not the HF download) so the test isolates the forward pass:

Logits (`num_classes > 0`): absolute max-abs-diff under `LOGITS_ATOL`.

```julia
data = read_parity(_FIXTURE_PATH)
model = resnet(:resnet50; in_chans = 3)
rng = Xoshiro(0)
ps, st = Lux.setup(rng, model)
st = Lux.testmode(st)
ps = apply_state_dict(ps, data.state_dict, resnet50_mapping(data.state_dict))
y, _ = model(data.input, ps, st)
diff = maximum(abs.(y .- data.output["logits"]))
@test diff < 1f-3  # LOGITS_ATOL
```

Features (`num_classes = 0`): relative bar — absolute diff divided by the max absolute value of the timm reference. This keeps the check scale-free across tiny-through-huge variants whose raw pre-norm activations span very different magnitudes.

```julia
model_f = resnet(:resnet50; in_chans = 3, num_classes = 0)
ps_f, st_f = Lux.setup(rng, model_f)
st_f = Lux.testmode(st_f)
ps_f = apply_state_dict(ps_f, data.state_dict, resnet50_mapping(data.state_dict))
y_f, _ = model_f(data.input, ps_f, st_f)
diff = maximum(abs.(y_f .- data.output["features"]))
rel  = diff / max(maximum(abs.(data.output["features"])), eps(Float32))
@test rel < 1f-4  # FEATURES_RTOL
```

When it fails, the per-stage and per-block fixtures earn their keep. Walk the forward by hand:

- For each stage, run the partial forward up to that stage (the simplest way is a transient `@compact` that drops in the relevant sub-models, or peeling layers off the full `Chain`). Compare against `data.output["stage_i"]`. The first stage where parity breaks localizes the bug to that stage's blocks.
- Inside the failing stage, instantiate a single block (`preact_bottleneck(in, out, stride; ...)`), call `Lux.setup`, splice in the matching `state_dict` entries via a small mapping, and check against the per-block fixture. This is faster than printing intermediate tensors and survives random init.

Divergence-pattern playbook:

- **Off by a few ULP, structured.** Float32 cast on the Python side is already in `_dump_common.to_numpy` so that's not it. Suspect `cross_correlation`, `corrected = false`, or an `epsilon` mismatch on a norm.
- **Off by a lot, structured.** Suspect (a) `cross_correlation = true` missing on a `Conv`, (b) `-Inf` vs zero padding on a `maxpool`, (c) flipped axes from a mapping transform mismatch.
- **Off by a lot, random-looking.** Suspect that you applied `apply_state_dict` to the wrong sub-tree (`prefix` wrong) or that a `state_dict` key spelled differently in your mapping silently fell back to random init. Add an `error` in your mapping function that fails on any key absent from `state_dict.keys()`; `BiTResNet.jl:199-202` does this at the bottom of the mapping builder.
- **Stages 0..k pass, stage k+1 fails.** Confirms the bug is local to stage k+1's first block; bisect within that block (norm before conv? conv first vs norm first?). The order of operations in pre-activation ResNet blocks differs from post-activation; both compile, only one matches.

## 8. Phase 7: iterate via Kaimon + Revise

Stand up one `start_session()`-attached REPL the way `kaimon-julia` describes, then keep it alive across the whole port. Edit `.jl` files; Revise picks up function-body changes between `ex` calls. Re-run the parity test via `ex(e="include(\"test/test_<model>.jl\")", ses=<key>)`. Restart only when:

- You added or removed a struct field, including `@compact` layer fields.
- You added or removed an `include`.
- A long-lived REPL state (a cached `Lux.setup` result, a fixture loaded into a global) is stale.

The per-block parity tests are well-suited to be a `smoke_test()`-style gate: a few microseconds per call, structured small output, runs after every edit. The full end-to-end parity is more expensive and goes at the end of each iteration.

## 9. Pitfalls reference

A running checklist; every item below has cost real time on a previous port.

- **Cross-correlation.** Every `Conv` needs `cross_correlation = true`. Every `NNlib.conv` needs `flipkernel = true` on its `DenseConvDims`. Same semantic, two flags.
- **Zero-padded pooling.** `NNlib.maxpool(pad=1)` pads with `-Inf`. `nn.MaxPool2d` with `padding=1` pads with zero. When the input has negative values (post-norm activations, leaky-relu outputs), this diverges. `pad_zeros` first, then pool with `pad = 0`.
- **GN/BN/LN defaults.** Always pass `epsilon` and `affine` explicitly. Never trust the Lux default to match the PyTorch default.
- **Variance correction.** Pass `corrected = false` when matching anything BN-style or WS-style.
- **Weight axis order on `apply_state_dict`.** HDF5 fixtures arrive reversed (Lux-natural). SafeTensors arrive PyTorch-natural and must be reversed in the transform (or normalized at load time).
- **`forward_features` vs `forward`.** The fixture and the Julia forward must agree on which one was dumped. Mismatch shows up as a `(N, num_classes)` vs `(N, C, H, W)` shape error if you're lucky, a silent wrong number if you're not.
- **WHCN vs NCHW.** `read_parity` returns WHCN-natural already. If you bypass it and call `HDF5.read` directly, you get reversed-PyTorch (which is WHCN), which is what you want, but it's easy to convince yourself you need to permute "back" and break it.
- **Pre-activation block order.** norm -> activation -> conv vs conv -> norm -> activation. Both compile. Only one matches the upstream architecture.
- **timm stem adaptation for `in_chans != 3`.** `adapt_input_conv` collapses the stem at `state_dict` time. Do not re-collapse on the Julia side.
- **`Lux.testmode(st)`.** Required for parity. Forgetting it makes the test occasionally pass and occasionally fail depending on RNG state inside dropout.
- **`pretrained` URLs go stale.** Pin them next to a comment giving the `timm` canonical name. Public timm weights live at `https://huggingface.co/timm/<name>/resolve/main/model.safetensors`.
- **Don't generalize until one variant works.** Adding `:resnet101` to the dispatcher before `:resnet50` passes parity creates code paths that the first failed test cannot localize.

Read `kaimon-julia` first; this skill layers on top of that workflow.
