# Squeeze-and-Excitation block, Lux port of timm's `SEModule`
# (timm/layers/squeeze_excite.py).
#
# Matches timm exactly:
#   x_se = x.mean((H, W), keepdim=True)   # global average pool
#   x_se = fc1(x_se)                       # 1x1 conv C -> rd
#   x_se = act(x_se)                       # ReLU
#   x_se = fc2(x_se)                       # 1x1 conv rd -> C
#   out  = x * gate(x_se)                  # sigmoid channel gate
#
# timm's SEModule uses 1x1 *convolutions* (not Linear) for fc1/fc2, so the
# PyTorch keys `se.fc1.weight` / `se.fc2.weight` are 4-D `(out, in, 1, 1)` and
# map to Lux Conv weights with the `identity` transform. The optional BatchNorm
# between fc1 and act is `Identity` in every checkpoint ported here, so it is
# omitted. Reused by SE-ResNet (and later CoAtNet's MBConv stages).

"""
    se_make_divisible(v, divisor=8; min_value=divisor) -> Int

timm's `make_divisible`: round `v` to the nearest multiple of `divisor`, never
dropping below `min_value` and never losing more than 10% of `v`. Used to size
the SE bottleneck channel count.
"""
function se_make_divisible(v::Real, divisor::Int = 8; min_value::Int = divisor)
    new_v = max(min_value, (floor(Int, v + divisor / 2) ÷ divisor) * divisor)
    new_v < 0.9 * v && (new_v += divisor)
    return new_v
end

"""
    se_block(C; rd_ratio=1/16, rd_divisor=8, rd_channels=nothing, act=NNlib.relu) -> @compact block

Squeeze-and-excitation channel-attention block for `(W, H, C, N)` tensors.
Global-average-pools each channel to a scalar, runs a two-layer 1x1-conv
bottleneck (`fc1` C→rd → `act` → `fc2` rd→C), and rescales the input by the
per-channel sigmoid gate. The bottleneck width `rd` is `rd_channels` when
given (e.g. CoAtNet's `int(attn_ratio * mid_chs)`), otherwise
`se_make_divisible(C * rd_ratio, rd_divisor)`, matching timm's `SEModule`. `act`
is the bottleneck activation (ReLU by default; CoAtNet's later recipes use
SiLU).

PyTorch keys `<prefix>.fc1.weight/bias` and `<prefix>.fc2.weight/bias` map to
the `:fc1` / `:fc2` Conv leaves with the `identity` transform.
"""
function se_block(
    C::Int;
    rd_ratio::Real = 1 // 16,
    rd_divisor::Int = 8,
    rd_channels::Union{Nothing,Int} = nothing,
    act = NNlib.relu,
)
    rd = rd_channels === nothing ? se_make_divisible(C * rd_ratio, rd_divisor) : rd_channels
    @compact(
        fc1 = Conv(
            (1, 1),
            C => rd;
            use_bias = true,
            cross_correlation = true,
            init_bias = zeros32,
        ),
        fc2 = Conv(
            (1, 1),
            rd => C;
            use_bias = true,
            cross_correlation = true,
            init_bias = zeros32,
        ),
    ) do x
        s = mean(x; dims = (1, 2))     # (1, 1, C, N) global average pool
        s = act.(fc1(s))
        s = fc2(s)
        @return x .* NNlib.sigmoid.(s)
    end
end
