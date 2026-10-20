# Channels-on-axis-3 LayerNorm for WHCN tensors.
#
# Matches timm's `LayerNorm2d` (timm/layers/norm.py), which permutes NCHW to
# NHWC, applies LayerNorm over the trailing channel dim, and permutes back.
# In Lux's (W, H, C, N) layout, channels already sit on axis 3, so we
# normalize over `dims = 3` directly without any permutation.
#
# Per-channel learnable affine. Variance is population (`corrected = false`),
# which is what `LuxLib.layernorm` uses internally, matching PyTorch's
# `F.layer_norm`.

"""
    layernorm2d(C; eps=1f-6) -> LuximmLayerNorm

LayerNorm with per-channel affine parameters for a 4D tensor in Lux's
`(W, H, C, N)` layout. Normalizes over the channel axis at each spatial
location and batch element, then applies a learnable scale and bias.
Numerically equivalent to timm's `LayerNorm2d`.

Implemented as `LuximmLayerNorm((1, 1, C), 3; epsilon = eps)`, which has the
parameter tree of `Lux.LayerNorm((1, 1, C); dims = 3)`: the affine leaves
`:scale` and `:bias` have shape `(1, 1, C, 1)`.
PyTorch state-dict entries `<prefix>.weight` and `<prefix>.bias` are stored
as `(C,)` and must be reshaped to `(1, 1, C, 1)` when loading (see
`Luximm.Interop.as_channel4d`).
"""
function layernorm2d(C::Int; eps::Float32 = 1.0f-6)
    return LuximmLayerNorm((1, 1, C), 3; epsilon = eps)
end
