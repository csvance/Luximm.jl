# Global Response Normalization (GRN), ConvNeXtV2's drop-in for LayerScale.
#
# Matches timm/layers/grn.py exactly:
#   g = ||x||_2  over the spatial dims (W, H), keepdim
#   n = g / (mean(g, channel_dim) + eps)
#   out = x + bias + scale * (x * n)
#
# The L2 norm is *unstabilized* (no eps under the sqrt); eps applies only at
# the mean-division step. Gamma (`scale`) and beta (`bias`) are zero-initialized
# so the layer is identity at the start of training.

"""
    grn_layer(C; eps=1f-6) -> @compact block

Global Response Normalization for `(W, H, C, N)` tensors. Computes the L2
norm of each channel's spatial map, normalizes that by the channel-mean of
those norms, and applies a per-channel affine `x + bias + scale * (x * n)`
with both parameters initialized to zero.

Named `grn_layer` to keep the `@compact` field name `:grn` available for
use in containing blocks, so PyTorch keys like `mlp.grn.weight` map directly
to `(..., :grn, :scale)`.

PyTorch state-dict keys `<prefix>.weight` and `<prefix>.bias` map to the
`:scale` and `:bias` leaves of this layer with the `identity` transform.
"""
function grn_layer(C::Int; eps::Float32 = 1.0f-6)
    return @compact(scale = zeros32(C), bias = zeros32(C), eps = eps) do x
        g = sqrt.(sum(x .^ 2; dims = (1, 2)))         # (1, 1, C, N)
        n = g ./ (mean(g; dims = 3) .+ eps)            # (1, 1, C, N)
        s = reshape(scale, 1, 1, :, 1)
        b = reshape(bias, 1, 1, :, 1)
        @return x .+ b .+ s .* (x .* n)
    end
end
