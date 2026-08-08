```@meta
CurrentModule = Luximm
```

# Luximm.jl

Julia ports of [`timm`](https://github.com/huggingface/pytorch-image-models)
(PyTorch Image Models, by Ross Wightman) backbones for
[Lux.jl](https://lux.csail.mit.edu/), with pretrained weights loaded
directly from HuggingFace Hub in `.safetensors` format. The name is an
homage to the project we port from.

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

| Family                                                                              | Variant prefix                                  | Weights | Weight License                                                                                             | Commercial Use |
|-------------------------------------------------------------------------------------|-------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------|----------------|
| [ResNet](https://arxiv.org/abs/1512.03385)                                          | [`:resnet*`](@ref RESNET_VARIANTS)              | 5       | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                  | ✅              |
| [BiT ResNetV2](https://arxiv.org/abs/1912.11370)                                    | [`:resnetv2_*_bit_*`](@ref BIT_VARIANTS)        | 15      | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                  | ✅              |
| [ConvNeXt](https://arxiv.org/abs/2201.03545)                                        | [`:convnext_*`](@ref CONVNEXT_VARIANTS)         | 19      | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                  | ✅              |
| [ConvNeXt (DINOv3)](https://arxiv.org/abs/2508.10104)                               | [`:convnext_*`](@ref CONVNEXT_VARIANTS)         | 4       | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)                          | ⚠️             |
| [ConvNeXt V2](https://arxiv.org/abs/2301.00808)                                     | [`:convnextv2_*`](@ref CONVNEXTV2_VARIANTS)     | 26      | [CC BY-NC 4.0](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE)                         | ❌              |
| [VGG](https://arxiv.org/abs/1409.1556)                                              | [`:vgg*`](@ref VGG_VARIANTS)                    | 8       | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)                                                  | ✅              |
| [SE-ResNet](https://arxiv.org/abs/1709.01507)                                       | [`:seresnet*`](@ref SERESNET_VARIANTS)          | 1       | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                  | ✅              |
| [ViT](https://arxiv.org/abs/2010.11929)                                             | [`:vit_*`](@ref VIT_VARIANTS)                   | 4       | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                  | ✅              |
| [CoAtNet](https://arxiv.org/abs/2106.04803)                                         | [`:coatnet_*`](@ref COATNET_VARIANTS)           | 5       | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                  | ✅              |

## At a glance

```julia
using Luximm, Lux, Random

# ResNet50 with the trained 1000-class ImageNet head.
# `create_pretrained` is family-agnostic; the symbol selects the
# family. It returns the model and a closure that loads the released
# weights into `(ps, st)` once you've run `Lux.setup`.
model, load = create_pretrained(:resnet50_a1_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)                 # BatchNorm/Dropout in eval mode

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)          # (1000, 1)
top1 = argmax(vec(logits))            # ImageNet class index
```

See [Getting Started](getting_started.md) for the full walkthrough,
including feature-extractor mode (`num_classes = 0`) and single-channel
inputs.

## Where to go next

- [Getting Started](getting_started.md): end-to-end prediction
  example, switching families, grayscale inputs, HuggingFace cache.
- [Porting Backbones](porting.md): contributor guide for adding a new
  `timm` backbone, with the parity-driven workflow.
- [Testing](testing.md): how the parity test suite is structured, the
  env-var filters, and how to dump a fixture for a new variant.
- [API Reference](api/index.md): every exported function and type.
