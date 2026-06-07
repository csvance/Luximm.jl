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
