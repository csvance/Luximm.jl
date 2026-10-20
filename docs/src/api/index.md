```@meta
CurrentModule = Luximm
```

# API Reference

Luximm.jl exposes three groups of public symbols.

| Group | Purpose | Page |
|---|---|---|
| Models | The family-agnostic `create_model` / `create_pretrained` pair, variant tables, and per-variant config structs. | [Models](models.md) |
| Layers | Building blocks shared across families: weight-standardized conv, channel-axis LayerNorm, Global Response Norm, Luximm's own LayerNorm / GroupNorm (accelerated under Reactant), and timm-equivalent initializers. | [Layers](layers.md) |
| Interop | PyTorch and HuggingFace plumbing: HDF5 parity fixtures, `state_dict` application, axis transforms, HuggingFace Hub cache, SafeTensors loading. | [Interop](interop.md) |

The top-level `Luximm` module re-exports most public symbols from
these three groups, so end users normally just call `using Luximm`.
Two sets are reached through their submodule instead: the Interop
fixture and transform helpers (`Luximm.Interop.read_parity`,
`axis_reverse`, `pyperm`, `as_channel4d`, `as_token_norm`,
`adapt_input_conv`) and the normalization primitives
(`Luximm.Layers.LuximmLayerNorm`, `LuximmGroupNorm`, `layernorm`,
`groupnorm`, `mean_var`). Submodule-qualified names work for every
symbol, which helps when a short top-level name would collide with
something else in the caller's namespace.

## Module map

```
Luximm
├── Luximm.Models    # family constructors, loaders, variant tables
├── Luximm.Layers    # std_conv, layernorm2d, grn_layer, normalization, init recipes
└── Luximm.Interop   # parity, HF Hub, SafeTensors
```

Internal block constructors (`resnet_basic_block`,
`convnextv2_block`, the stage scaffolding under `ConvNeXtCommon`,
and the per-family `resnet` / `bit_resnetv2` / `convnext` /
`convnextv2` constructors themselves) are deliberately unexported.
They are implementation details of `create_model` and
`create_pretrained`; rely on those two as the public construction
surface.
