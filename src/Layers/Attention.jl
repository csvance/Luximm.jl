# Multi-head self-attention, Lux port of timm's `Attention`
# (timm/models/vision_transformer.py).
#
# Token layout is `(dim, T, N)`. The fused `qkv = Dense(dim => 3*dim)` matches
# timm's single `attn.qkv` Linear; the split mirrors timm's
# `qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)`.
#
# qkv-split ordering (the #1 numeric risk): timm's `(3, num_heads, head_dim)`
# row-major reshape has `head_dim` fastest, then `num_heads`, then the q/k/v
# selector. After loading, the Lux qkv output axis 1 has the *same* element
# order as PyTorch's `3*dim` axis, so a Julia column-major
# `reshape(_, head_dim, num_heads, 3, T, N)` (head_dim fastest) recovers the
# same q/k/v/head decomposition.
#
# The two score/context matmuls are exact `NNlib.batched_mul` calls after
# folding `(num_heads, batch)` onto the trailing batch axis. Softmax is taken
# over the **key** axis. All ops are GPU/AD-safe (no scalar indexing, no
# mutation).

"""
    mhsa(dim; num_heads, qkv_bias=true) -> @compact block

Multi-head self-attention over a `(dim, T, N)` token tensor. Splits a fused
`qkv` projection into `num_heads` heads of width `dim ÷ num_heads`, computes
scaled dot-product attention (scale `1/sqrt(head_dim)`, softmax over the key
axis), merges heads, and applies the output `proj`. Numerically matches timm's
`Attention` with `qk_norm=False`.

PyTorch keys map as: `attn.qkv.weight` → `(:qkv, :weight)` (`axis_reverse`),
`attn.qkv.bias` → `(:qkv, :bias)` (`identity`), `attn.proj.weight` →
`(:proj, :weight)` (`axis_reverse`), `attn.proj.bias` → `(:proj, :bias)`
(`identity`).
"""
function mhsa(dim::Int; num_heads::Int, qkv_bias::Bool = true)
    head_dim = dim ÷ num_heads
    scale = Float32(1 / sqrt(head_dim))
    @compact(
        qkv = Dense(dim => 3dim; use_bias = qkv_bias, init_bias = zeros32),
        proj = Dense(dim => dim; init_bias = zeros32),
    ) do x
        D, T, N = size(x)
        y = qkv(x)                                     # (3D, T, N)
        y = reshape(y, head_dim, num_heads, 3, T, N)
        # Each slice: (head_dim, num_heads, T, N) -> (head_dim, T, num_heads*N).
        q = reshape(permutedims(y[:, :, 1, :, :], (1, 3, 2, 4)), head_dim, T, num_heads * N)
        k = reshape(permutedims(y[:, :, 2, :, :], (1, 3, 2, 4)), head_dim, T, num_heads * N)
        v = reshape(permutedims(y[:, :, 3, :, :], (1, 3, 2, 4)), head_dim, T, num_heads * N)
        # scores[i, j, b] = sum_d q[d, i, b] * k[d, j, b]
        scores = NNlib.batched_mul(NNlib.batched_transpose(q), k) .* scale  # (T, T, HN)
        attn = NNlib.softmax(scores; dims = 2)                              # over key axis
        # ctx[d, i, b] = sum_j v[d, j, b] * attn[i, j, b]
        ctx = NNlib.batched_mul(v, NNlib.batched_transpose(attn))           # (head_dim, T, HN)
        ctx = reshape(ctx, head_dim, T, num_heads, N)
        ctx = permutedims(ctx, (1, 3, 2, 4))                               # (head_dim, num_heads, T, N)
        ctx = reshape(ctx, dim, T, N)
        @return proj(ctx)
    end
end
