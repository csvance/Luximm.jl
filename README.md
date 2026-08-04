<p align="center">
    <img width="300px" src="docs/src/assets/logo.svg"/>
</p> 
<div align="center">

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://csvance.github.io/Luximm.jl/dev/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://csvance.github.io/Luximm.jl/stable/)
[![CI](https://img.shields.io/github/checks-status/csvance/Luximm.jl/master?label=CI)](https://github.com/csvance/Luximm.jl/commits/master)

</div>

Julia ports of [`timm`][timm] (PyTorch Image Models, by Ross Wightman) backbones
for [Lux.jl][lux], with pretrained weights loaded directly from HuggingFace Hub
in `.safetensors` format. The name is an homage to the project we port from.

## Status

Most of Luximm was written by AI agents driving the porting workflow
encoded in `.claude/skills/timm-to-lux/`, with human review at each
phase and the parity tests as the correctness backstop. The code is
already being used in real projects, so the registered backbones work
for forward inference with the released weights. That said: **expect
bugs and rough edges**, especially around anything the parity tests do
not exercise (custom training loops, mixed-precision paths, exotic
input shapes). File issues and PRs.

## Available backbones

| Family                      | Variant prefix                       | Weights | Weight License                     | Commercial Use |
|-----------------------------|--------------------------------------|---------|------------------------------------|----------------|
| [ResNet][resnet]            | [`:resnet*`][prefix-resnet]          | 5       | [Apache 2.0][license-apache2]      | ✅              |
| [SE-ResNet][seresnet]       | [`:seresnet*`][prefix-seresnet]      | 1       | [Apache 2.0][license-apache2]      | ✅              |
| [BiT ResNetV2][bit]         | [`:resnetv2_*_bit_*`][prefix-bit]    | 15      | [Apache 2.0][license-apache2]      | ✅              |
| [ConvNeXt][convnextv1]      | [`:convnext_*`][prefix-convnext]     | 19      | [Apache 2.0][license-apache2]      | ✅              |
| [ConvNeXt (DINOv3)][dinov3] | [`:convnext_*`][prefix-convnext]     | 4       | [DINOv3 License][license-dinov3]   | ⚠️             |
| [ConvNeXt V2][convnextv2]   | [`:convnextv2_*`][prefix-convnextv2] | 26      | [CC BY-NC 4.0][license-convnextv2] | ❌              |
| [VGG][vgg]                  | [`:vgg*`][prefix-vgg]                | 8       | [CC BY 4.0][license-ccby4]         | ✅              |
| [ViT][vit]                  | [`:vit_*`][prefix-vit]               | 1       | [Apache 2.0][license-apache2]      | ✅              |
| [CoAtNet][coatnet]          | [`:coatnet_*`][prefix-coatnet]       | 5       | [Apache 2.0][license-apache2]      | ✅              |

## Basic usage

```julia
using Luximm, Lux, Random

# ResNet50 with the trained 1000-class ImageNet head.
# `create_pretrained` is family-agnostic; the symbol selects the family.
# It returns the model and a closure that loads the released weights
# into `(ps, st)` once you've run `Lux.setup`.
model, load = create_pretrained(:resnet50_a1_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)                     # BatchNorm/Dropout in eval mode

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)              # (1000, 1)
top1 = argmax(vec(logits))                # ImageNet class index
```

`create_model(variant; ...)` (without weight loading) is also exported
for from-scratch training. For the full walkthrough, including
feature-extractor mode (`num_classes = 0`), single-channel inputs
(`in_chans = 1`), and the HuggingFace cache layout, see the
[Getting Started][docs-getting-started] docs page.

## Composing with a pretrained backbone

Drop a feature-extractor backbone into your own `@compact` block and
let the loader fill in just the backbone's subtree. The
`prefix = (:backbone,)` tuple matches the slot name in the outer
model, so `load_backbone` writes only into `ps.backbone.*` and
`st.backbone.*`, leaving the head at its random initialization for
downstream training:

```julia
using Luximm, Lux, NNlib, Random

backbone, load_backbone = create_pretrained(:resnet50_a1_in1k;
    num_classes = 0, prefix = (:backbone,))

model = @compact(
    backbone = backbone,
    head     = Dense(2048 => 10),   # custom 10-class head
) do x
    feats  = backbone(x)                                   # (7, 7, 2048, N)
    pooled = NNlib.meanpool(feats, size(feats)[1:2])       # (1, 1, 2048, N)
    head(reshape(pooled, size(pooled, 3), size(pooled, 4)))
end

ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_backbone(ps, st)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)                               # (10, 1)
```

For multi-backbone composition and deeper nesting patterns, see the
[Getting Started][docs-getting-started] docs page.

## Feature pyramids for dense prediction

`features_only = true` mirrors timm's `features_only=True`: the forward
returns a tuple of intermediate feature maps ordered by increasing
reduction, which is what a UNet or FPN decoder consumes. `feature_info`
is the tap table, so the decoder can be sized before it is built.

```julia
using Luximm, Lux, Random

info = feature_info(:resnet18_a1_in1k)
info.reductions                            # (2, 4, 8, 16, 32)
info.channels                              # (64, 64, 128, 256, 512)

model, load = create_pretrained(:resnet18_a1_in1k; features_only = true)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
feats, _ = model(x, ps, st)                # 5-tuple, finest first
size(feats[1]), size(feats[end])           # (112,112,64,1), (7,7,512,1)
```

`out_indices` selects a subset of the taps (1-based and strictly
increasing, so timm's `out_indices=(1,2,3,4)` is Luximm's `(2,3,4,5)`).
A features-only model builds the same parameter tree as the plain
`num_classes = 0` extractor, so the released weights load into it
unchanged.

| Family        | Pyramid | Reductions       |
|---------------|---------|------------------|
| ResNet        | ✅       | 2, 4, 8, 16, 32  |
| SE-ResNet     | ✅       | 2, 4, 8, 16, 32  |
| BiT ResNetV2  | ✅       | 2, 4, 8, 16, 32  |
| ConvNeXt      | ✅       | 4, 8, 16, 32     |
| ConvNeXt V2   | ✅       | 4, 8, 16, 32     |
| VGG           | ❌       | n/a              |
| ViT           | ❌       | n/a              |
| CoAtNet       | ❌       | n/a              |

The unsupported families raise an error explaining why. Two caveats are
worth knowing, both matching timm: the ConvNeXt families have no
reduction-2 tap (their patch stem strides by 4 in one convolution), and
BiT's reduction-32 tap is the raw pre-activation `stage4` output rather
than the `final_norm`-applied map the same model returns at
`num_classes = 0`. See the [Getting Started][docs-getting-started] page
for details.

## License and attribution

Luximm.jl is licensed under the Apache License, Version 2.0 (see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The license matches upstream
`timm`. The Julia code in this repository is original, but layer naming,
hyperparameters, padding ordering, and `state_dict` key layout are
deliberately taken from `timm` so pretrained weights load directly.

## Acknowledgements

Thanks to Ross Wightman for `timm`, to [HuggingFace][huggingface] for
hosting the `.safetensors` weights that Luximm.jl loads at runtime, to
the Julia ML ecosystem maintainers whose work makes a port like this
plausible, and to [Medical Metrics Inc.][medicalmetrics] for allowing
me to work on and open-source the project.

[huggingface]: https://huggingface.co/

[timm]: https://github.com/huggingface/pytorch-image-models

[lux]: https://lux.csail.mit.edu/

[docs]: https://csvance.github.io/Luximm.jl/

[docs-getting-started]: https://csvance.github.io/Luximm.jl/dev/getting_started/

[prefix-resnet]: https://csvance.github.io/Luximm.jl/dev/api/models/#ResNet

[prefix-seresnet]: https://csvance.github.io/Luximm.jl/dev/api/models/#SE-ResNet

[prefix-bit]: https://csvance.github.io/Luximm.jl/dev/api/models/#BiT-ResNetV2

[prefix-convnext]: https://csvance.github.io/Luximm.jl/dev/api/models/#ConvNeXt

[prefix-convnextv2]: https://csvance.github.io/Luximm.jl/dev/api/models/#ConvNeXt-V2

[prefix-vgg]: https://csvance.github.io/Luximm.jl/dev/api/models/#VGG

[prefix-vit]: https://csvance.github.io/Luximm.jl/dev/api/models/#ViT

[prefix-coatnet]: https://csvance.github.io/Luximm.jl/dev/api/models/#CoAtNet

[medicalmetrics]: https://medicalmetrics.com/

[license-apache2]: https://www.apache.org/licenses/LICENSE-2.0

[license-dinov3]: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

[license-convnextv2]: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE

[license-ccby4]: https://creativecommons.org/licenses/by/4.0/

[bit]: https://arxiv.org/abs/1912.11370

[convnextv1]: https://arxiv.org/abs/2201.03545

[dinov3]: https://arxiv.org/abs/2508.10104

[convnextv2]: https://arxiv.org/abs/2301.00808

[resnet]: https://arxiv.org/abs/1512.03385

[seresnet]: https://arxiv.org/abs/1709.01507

[vgg]: https://arxiv.org/abs/1409.1556

[vit]: https://arxiv.org/abs/2010.11929

[coatnet]: https://arxiv.org/abs/2106.04803