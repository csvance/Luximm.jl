```@meta
CurrentModule = Luximm
```

# Layers

Reusable building blocks shared across model families. These exist
to match specific `timm` constructs in semantics, layout, and
default parameters. They are not intended as a general-purpose
layer library.

## Building blocks

```@docs
std_conv
layernorm2d
grn_layer
se_block
patch_embed
mhsa
vit_block
vit_layernorm
rel_pos_attention
```

## Normalization

Luximm's own normalization layers and statistics, with the parameter
trees of Lux's `LayerNorm` and `GroupNorm`. On ordinary arrays
`layernorm` and `groupnorm` call LuxLib and `mean_var` uses
`Statistics`; under Reactant, the `LuximmReactantExt` extension gives
them traced methods whose backward skips the identically zero
∂σ²/∂μ reduction while staying exact at every derivative order.

```@docs
Layers.LuximmLayerNorm
Layers.LuximmGroupNorm
Layers.layernorm
Layers.groupnorm
Layers.mean_var
```

## Helpers

```@docs
se_make_divisible
rel_pos_index
```

## Initializers

```@docs
kaiming_normal_fan_out
normal_init
```
