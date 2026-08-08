```@meta
CurrentModule = Luximm
```

# Models

## Family-agnostic interface

`create_pretrained` is the symbol-dispatched entry point for loading
released weights. It returns the model and a closure that loads the
HuggingFace checkpoint into a `(ps, st)` pair you produce with
`Lux.setup`:

```julia
model, load = create_pretrained(variant)
ps, st = Lux.setup(rng, model)
ps, st = load(ps, st)
```

The closure captures `variant`, `in_chans`, `num_classes`, and the HF
/ `prefix` kwargs at construction time, so the loader body no longer
needs to introspect `ps` to recover what you already told it.
`create_model` is the random-init counterpart: it returns the bare
`@compact` model with no weights loaded. See
[Getting Started](../getting_started.md#composing-into-a-larger-model)
for the nested pattern with `prefix`.

```@docs
create_pretrained
create_model
default_num_classes
```

## Feature pyramids

`features_only = true` turns a supported backbone into a multi-scale
feature extractor whose forward returns a tuple of intermediate maps
ordered by increasing reduction, which is what a UNet or FPN decoder
consumes. `feature_info` is the accompanying tap table (reductions and
channel counts), so a decoder can be sized before it is built. See
[Getting Started](../getting_started.md#Feature-pyramid-(multi-scale)-mode)
for the walkthrough.

Supported families: ResNet, SE-ResNet, BiT ResNetV2, ConvNeXt, ConvNeXt V2,
and ViT. VGG and CoAtNet raise an error explaining why. A ViT is
single-scale (every tap at the patch-size reduction, one per block), so a
decoder that needs multiple resolutions must upsample past the taps it
selects.

```@docs
feature_info
FeatureInfo
```

## Per-family namespaces

Each family exports its variant config struct and the
`<FAMILY>_VARIANTS` registry dict. The remaining family internals
(per-family constructors, weight mappings, state mappings) live in
`Luximm.Models.*` for callers who need to escape the
`create_pretrained` / `create_model` front door.

## ResNet

```@docs
ResNetVariant
RESNET_VARIANTS
```

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.RESNET_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.num_features) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## SE-ResNet

```@docs
SEResNetVariant
SERESNET_VARIANTS
```

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.SERESNET_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.num_features) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## BiT ResNetV2

```@docs
BiTVariant
BIT_VARIANTS
```

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.BIT_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.num_features) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## ConvNeXt

```@docs
ConvNeXtVariant
CONVNEXT_VARIANTS
```

!!! warning "DINOv3 weights are not Apache 2.0"
    The four `:convnext_*_dinov3_lvd1689m` encoders are released by Meta
    under the
    [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/),
    which imposes obligations on outputs derived from the weights that
    differ from a standard permissive open-source license. Read the
    license before using the weights for any downstream task. This
    applies only to the weights; the Julia code in this package is
    Apache 2.0. The Facebook AI `.fb_*` checkpoints carry the upstream
    Apache 2.0 license and are unaffected.

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.CONVNEXT_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.dims[end]) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## ConvNeXt V2

```@docs
ConvNeXtV2Variant
CONVNEXTV2_VARIANTS
```

!!! warning "ConvNeXtV2 weights are non-commercial"
    Every ConvNeXtV2 checkpoint is released by Meta under
    [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
    **Commercial use of these weights is not permitted.** This applies
    to every row in the variant table below and is independent of
    Luximm.jl's own Apache 2.0 code license. If commercial use matters,
    BiT (Apache 2.0) or the ConvNeXt v1 `.fb_*` checkpoints (Apache 2.0)
    are the alternatives.

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.CONVNEXTV2_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.dims[end]) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## VGG

```@docs
VGGVariant
VGG_VARIANTS
```

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.VGG_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | 512 | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## ViT

```@docs
ViTVariant
VIT_VARIANTS
```

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.VIT_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.embed_dim) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## CoAtNet

```@docs
CoAtNetVariant
COATNET_VARIANTS
```

### Registered variants

```@eval
using Markdown, Luximm
rows = sort(collect(Luximm.COATNET_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | num_classes | num_features | input size |")
println(io, "|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| [`:$(k)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.dims[end]) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```
