```@meta
CurrentModule = Luximm
```

# Interop

PyTorch and HuggingFace plumbing: applying a PyTorch `state_dict` to a
Lux `(ps, st)` pair, resolving and caching weights through the
HuggingFace Hub, and loading `.safetensors` blobs.

## State-dict application

```@docs
apply_state_dict
read_parity
```

## Weight-layout transforms

Per-tensor transforms passed in `apply_state_dict` mappings to bridge
PyTorch's stored layout and the Lux-natural layout (see
[Porting Backbones](../porting.md)).

```@docs
axis_reverse
pyperm
as_channel4d
as_token_norm
adapt_input_conv
```

## HuggingFace Hub

```@docs
hf_hub_download
hf_download
hf_hub_cache_dir
```

## SafeTensors

```@docs
load_safetensors_state_dict
```