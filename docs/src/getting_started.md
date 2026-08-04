```@meta
CurrentModule = Luximm
```

# Getting Started

This page walks through Luximm.jl end-to-end: installing the package,
loading a pretrained ResNet50 and predicting an ImageNet class,
switching to feature-extractor mode, working with non-RGB inputs, and
the HuggingFace cache layout.

## Installation

Luximm targets Julia 1.12 or newer and is registered in the Julia
General registry, so it installs with `Pkg.add` directly:

```julia
using Pkg
Pkg.add("Luximm")
```

For local hacking, clone the repo and `Pkg.develop` the path:

```julia
using Pkg
Pkg.develop(path = "/path/to/Luximm.jl")
```

The Python sidecar (`pyproject.toml` with `timm`, `torch`, `h5py`,
`safetensors`, `huggingface-hub`) is only needed to regenerate parity
fixtures for new backbones. Inference and pretrained-weight loading
work from Julia alone. See [Testing](testing.md) for the contributor
setup.

## Loading ResNet50 and predicting an ImageNet class

```julia
using Luximm, Lux, Random

# `create_pretrained` is family-agnostic; the symbol selects the
# family. It returns the model and a closure that loads the released
# weights into a `(ps, st)` pair you produce with `Lux.setup`.
model, load = create_pretrained(:resnet50_a1_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)              # (1000, 1)

top5 = partialsortperm(vec(logits), 1:5; rev = true)
```

A few things worth noting in the snippet:

- Variant keys are Julia symbols. `:resnet50_a1_in1k` is the `timm`
  model name `resnet50.a1_in1k` with the dot rewritten as an
  underscore. The full name with the dot is at
  `RESNET_VARIANTS[:resnet50_a1_in1k].hf_repo`.
- `create_pretrained` defaults `num_classes` to the variant's
  `default_num_classes` (1000 for the `resnet50.a1_in1k` ImageNet
  checkpoint). Pass an explicit `num_classes = 0` for a features-only
  model, or any other Int for a custom head — see the next two
  sections.
- `Lux.setup` produces a `ps` (parameters) and `st` (state)
  NamedTuple. The closure returns a new `(ps, st)` with the
  HuggingFace weights merged in. Stateless families (BiT, ConvNeXt,
  ConvNeXtV2, ViT, and the plain VGG variants) return `st` unchanged;
  the BatchNorm families (ResNet, SE-ResNet, CoAtNet, and the `vgg*_bn`
  variants) merge BatchNorm running statistics into `st`. Call
  `Lux.testmode(st)` before inference so BatchNorm uses those running
  statistics instead of the current batch's statistics; for the
  stateless families it is a no-op but still a safe default.
- The input is shaped `(W, H, C, N)`, Lux's convention. PyTorch's
  `(N, C, H, W)` is read-reversed at load time so most weights land
  in the layout Lux expects directly.
- `x` here is random noise. For a real prediction, replace it with a
  preprocessed image: resize to 224x224, scale to `Float32` in
  `[0, 1]`, then normalize with the ImageNet mean and std.
- `create_model(variant; ...)` (without weight loading) is also
  exported for from-scratch training and for embedding into an outer
  `@compact` — see [Composing into a larger
  model](#composing-into-a-larger-model) below.

To map the top-5 indices to class names, pair the variant with any
ImageNet class label list (Luximm ships none of its own; the timm
`imagenet_classes.txt` works directly because the class index ordering
is unchanged).

## Feature extractor mode

Pass `num_classes = 0` to drop the classifier head and get the
post-stage feature map back instead of logits. This matches
`timm.create_model(..., num_classes=0).forward_features(x)`:

```julia
model, load = create_pretrained(:resnet50_a1_in1k; num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)            # (7, 7, 2048, 1)
```

The output is shaped `(W/32, H/32, num_features, N)` for every
registered backbone. For ResNet50 that is `(7, 7, 2048, 1)` on a
224x224 input; for `convnextv2_atto_fcmae` it is `(7, 7, 320, 1)`.

Use this mode when you want to attach Luximm's pretrained encoder to
your own downstream head (regression, segmentation, neural ODE, etc.)
without carrying around the 1000-class classifier the released
checkpoint would otherwise initialize.

## Feature-pyramid (multi-scale) mode

`num_classes = 0` gives you one map, at the coarsest reduction. A
dense-prediction decoder, UNet or FPN, needs the intermediate maps too,
so it can recover the spatial detail the stride-32 map has thrown away.
Pass `features_only = true` to get all of them; this is the Luximm
analog of timm's `features_only=True`:

```julia
model, load = create_pretrained(:resnet18_a1_in1k; features_only = true)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
feats, _ = model(x, ps, st)               # a 5-tuple
size.(feats)
# ((112, 112,  64, 1),      reduction 2
#  ( 56,  56,  64, 1),      reduction 4
#  ( 28,  28, 128, 1),      reduction 8
#  ( 14,  14, 256, 1),      reduction 16
#  (  7,   7, 512, 1))      reduction 32
```

The forward returns a **tuple** ordered by increasing reduction, so
`feats[1]` is the highest-resolution map and `feats[end]` the coarsest.
That ordering is fixed, and a decoder can rely on it.

[`feature_info`](@ref) is the tap table. Call it to size a decoder
before you build it:

```julia
info = feature_info(:resnet18_a1_in1k)
info.reductions                           # (2, 4, 8, 16, 32)
info.channels                             # (64, 64, 128, 256, 512)
info.names                                # (:act1, :layer1, :layer2, :layer3, :layer4)
```

`out_indices` selects a subset. It is **1-based**, so timm's
`out_indices=(1, 2, 3, 4)` is Luximm's `(2, 3, 4, 5)`; indices must be
strictly increasing, and the default `nothing` returns every tap:

```julia
model = create_model(:resnet18_a1_in1k;
                     features_only = true, out_indices = (2, 3, 4, 5))
feature_info(:resnet18_a1_in1k; out_indices = (2, 3, 4, 5)).channels
# (64, 128, 256, 512)
```

Selecting fewer taps changes only the forward, never the parameter
tree: every stage is still built and still runs. A features-only model
has the *same* tree as the plain `num_classes = 0` extractor, which is
why `create_pretrained` loads released weights into it unchanged.

Two things to know about the tap tables:

- ConvNeXt and ConvNeXt V2 have **no reduction-2 tap**. Their patch stem
  strides by 4 in a single convolution, so the finest map is quarter
  resolution and a decoder has to upsample past it. The ResNet families
  do expose reduction 2.
- For BiT ResNetV2 the reduction-32 tap is the raw `stage4` output,
  which is **pre-activation**: `final_norm` is not applied, matching
  timm, whose `norm` module sits after the last hooked stage. The same
  model at `num_classes = 0` *does* apply it, so those two outputs
  differ. The parameters are still there in `ps.final_norm` if you want
  the normalized map.

Families with a pyramid: ResNet, SE-ResNet, BiT ResNetV2, ConvNeXt,
ConvNeXt V2. VGG, ViT, and CoAtNet raise an error explaining why.

## Single-channel and other non-RGB inputs

Pass `in_chans` to `create_pretrained`. The closure adapts the
released 3-channel weight at load time, matching timm's
`adapt_input_conv` semantics: sum across input channels for
`in_chans = 1`, tile and rescale by `3 / in_chans` for other counts.

```julia
model, load = create_pretrained(:convnextv2_atto_fcmae; in_chans = 1)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)                     # BatchNorm/Dropout in eval mode

x = randn(Float32, 224, 224, 1, 1)
features, _ = model(x, ps, st)            # (7, 7, 320, 1)
```

This is the right path for grayscale medical or scientific imagery,
where collapsing the RGB stem is preferable to repeating the
single-channel input three times.

## Composing into a larger model

`create_pretrained` returns the backbone and a closure that already
knows the variant; pass `prefix = (:backbone,)` so the closure writes
into the right subtree of the outer `Lux.setup` result. The backbone
itself goes into the outer `@compact` block as a value:

```julia
backbone, load_backbone = create_pretrained(:resnet50_a1_in1k;
    num_classes = 0, prefix = (:backbone,))

outer = @compact(
    backbone = backbone,
    head     = Dense(2048 => num_outputs),
) do x
    head(backbone(x))
end

ps, st = Lux.setup(Xoshiro(0), outer)
ps, st = load_backbone(ps, st)
```

The mapping function prefixes every leaf path with `prefix...`, so
`apply_state_dict` overwrites just the `ps.backbone.*` and `st.backbone.*`
subtrees. Anything you added (`:head`, sibling slots, deeper nestings)
keeps its `Lux.setup` initialization.

### Multiple backbones of the same variant

Each `create_pretrained` call returns its own closure that captures its
own `prefix`, so two backbones of the same variant compose by giving
them different prefixes. The two closures are independent and write
into disjoint subtrees of `ps` and `st`:

```julia
backbone_a, load_a = create_pretrained(:resnet50_a1_in1k;
    num_classes = 0, prefix = (:resnet50_a,))
backbone_b, load_b = create_pretrained(:resnet50_a1_in1k;
    num_classes = 0, prefix = (:resnet50_b,))

outer = @compact(
    resnet50_a = backbone_a,
    resnet50_b = backbone_b,
    head       = Dense(4096 => num_outputs),
) do x
    feats = cat(resnet50_a(x), resnet50_b(x); dims = 3)
    head(feats)
end

ps, st = Lux.setup(Xoshiro(0), outer)
ps, st = load_a(ps, st)
ps, st = load_b(ps, st)
```

Loading is opt-in per backbone. Calling `load_a` and skipping `load_b`
leaves `resnet50_b` at its `Lux.setup` random initialization, which is
useful when one tower is pretrained and the other is trained from
scratch. The prefix tuple must match the slot name used in the outer
`@compact`; otherwise the loader raises `leaf path missing key` and
names the slot it expected to find.

### Backbones nested deeper than one level

The `prefix` tuple's length must match the depth of the backbone slot
inside the outer model, because the closure prepends `prefix...` to
every Lux leaf path. For a backbone wrapped in an inner `@compact`
that is itself slotted into an outer `@compact`, the prefix is the
sequence of slot names from the outermost block inward:

```julia
backbone, load_backbone = create_pretrained(:resnet50_a1_in1k;
    num_classes = 0, prefix = (:tower, :backbone))

encoder = @compact(backbone = backbone) do x
    backbone(x)
end

outer = @compact(
    tower = encoder,
    head  = Dense(2048 => num_outputs),
) do x
    head(tower(x))
end

ps, st = Lux.setup(Xoshiro(0), outer)
ps, st = load_backbone(ps, st)
```

The weights land in `ps.tower.backbone.*` because that is where the
outer `Lux.setup` placed the parameter subtree. There is no length
limit on the tuple, so the same rule scales to any depth.

## Switching families

`create_pretrained` (released weights) and `create_model` (random
init) both dispatch on the variant symbol, so picking a different
family is just picking a different symbol — no per-family entry
points to learn. Each family registers its variants under one
`<FAMILY>_VARIANTS` dict, which is enumerable for variant discovery:

- [`RESNET_VARIANTS`](@ref). 5 variants.
- [`BIT_VARIANTS`](@ref). 15 variants.
- [`CONVNEXT_VARIANTS`](@ref). 23 variants (19 Apache 2.0 Facebook
  AI checkpoints plus 4 Meta DINOv3 encoders).
- [`CONVNEXTV2_VARIANTS`](@ref). 26 variants.

See the [API Reference](api/models.md) for the full signatures.

## Pretrained weights and the HuggingFace cache

The `create_pretrained` closure resolves `model.safetensors` against
the standard HuggingFace Hub cache layout
(`HF_HUB_CACHE` → `$HF_HOME/hub` → `~/.cache/huggingface/hub`), so the
same blob is shared with `timm` and `huggingface_hub`: whichever tool
downloads first, the other sees a cache hit. Subsequent calls
short-circuit on the cached snapshot symlink.

For gated repos, set `HUGGING_FACE_HUB_TOKEN` in the environment
before calling the loader. The token is forwarded as a bearer header
on the resolve and download requests.

For lower-level access, [`hf_hub_download`](@ref) returns the
resolved snapshot path directly and mirrors the cache semantics of
`huggingface_hub.hf_hub_download`:

```julia
path = hf_hub_download("timm/resnet50.a1_in1k",
                       "model.safetensors";
                       revision = "main")
state_dict = load_safetensors_state_dict(path)
```

This is the escape hatch for cases where you want to inspect or
transform the raw PyTorch state dict before applying it.
