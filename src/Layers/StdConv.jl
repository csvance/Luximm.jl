# Weight-standardized 2D convolution.
#
# Standardizes the kernel at forward time: per output channel, subtract mean
# and divide by sqrt(var + eps), where the statistics are taken over
# `(kW, kH, in)`. Matches timm's `StdConv2d` exactly (population variance via
# PyTorch's BN-with-no-running-stats trick).
#
# Drops into `NNlib.conv` directly with `flipkernel = true` so the result is
# a cross-correlation (matching PyTorch), since `Lux.Conv(...; cross_correlation = true)`
# is incompatible with manually standardizing the kernel before the op.

"""
    std_conv(kW, kH, in, out; stride=1, pad=0, eps=1f-8) -> @compact block

Cross-correlation convolution whose kernel is standardized at forward time.
Use this in place of `Conv` when porting a `timm` model that wraps its
convolutions in `StdConv2d` (BiT-ResNet, NFNet, etc.).
"""
function std_conv(
    kW::Int,
    kH::Int,
    in_ch::Int,
    out_ch::Int;
    stride::Int = 1,
    pad::Int = 0,
    eps::Float32 = 1.0f-8,
    init_weight = glorot_uniform,
)
    @compact(
        conv = Conv(
            (kW, kH),
            in_ch => out_ch;
            stride = stride,
            pad = pad,
            use_bias = false,
            cross_correlation = true,
            init_weight = init_weight,
        ),
        stride = stride,
        pad = pad,
        eps = eps,
        kW = kW,
        kH = kH,
        in_ch = in_ch,
        out_ch = out_ch,
    ) do x
        w = conv.ps.weight  # (kW, kH, in, out); reach through StatefulLuxLayer
        w_flat = reshape(w, (:, out_ch))                          # (kW*kH*in, out)
        # LuxLib's `mean_var`, so weight standardization gets whatever its backends specialize
        # (under Reactant, a backward without the identically zero ∂σ²/∂μ term).
        μ, σ² = Lux.LuxLib.Impl.mean_var(w_flat; dims = 1, corrected = false)   # (1, out) each
        ŵ_flat = (w_flat .- μ) ./ sqrt.(σ² .+ eps)
        ŵ = reshape(ŵ_flat, (kW, kH, in_ch, out_ch))
        cdims = NNlib.DenseConvDims(
            size(x),
            size(ŵ);
            stride = stride,
            padding = pad,
            flipkernel = true,
        )
        @return NNlib.conv(x, ŵ, cdims)
    end
end
