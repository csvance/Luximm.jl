# 2D multi-head self-attention with relative-position bias, Lux port of timm's
# `Attention2d` + `RelPosBias` (timm/models/maxxvit.py, timm/layers/
# pos_embed_rel.py). Used by CoAtNet's transformer stages.
#
# Operates on a WHCN feature map `(W, H, dim, N)`: the spatial grid is the token
# sequence (no class token), attention is global over all `L = W*H` tokens, and
# a learned per-head bias indexed by relative position is added to the scores
# before softmax.
#
# qkv-split ordering: timm uses `head_first=True`, i.e.
# `qkv.view(B, num_heads, dim_head*3, L).chunk(3, dim=2)`, so the channel axis
# `3*dim` decomposes (row-major) as `(num_heads, [q|k|v], dim_head)` with
# `dim_head` fastest. A Julia column-major `reshape(_, dim_head, 3, num_heads)`
# (dim_head fastest, then the q/k/v selector, then heads) recovers the same
# decomposition. This differs from the plain-ViT `mhsa` ordering.
#
# Token order: timm flattens `(H, W)` row-major (width fastest); Lux's WHCN
# `(W, H)` column-major reshape is also width-fastest, so the orders match (and
# match the relative-position index grid built the same way).

"""
    rel_pos_index(Wh, Ww) -> Matrix{Int}

Relative-position index grid for an `Wh x Ww` window, matching timm's
`gen_relative_position_index`. Returns a `(L, L)` matrix (`L = Wh*Ww`, tokens
in width-fastest order) of **1-based** row indices into a
`(2*Wh-1)*(2*Ww-1)`-row bias table. Entry `[i, j]` is the bias-table row for
the relative offset between query token `i` and key token `j`.
"""
function rel_pos_index(Wh::Int, Ww::Int)
    L = Wh * Ww
    # token t (1-based) ↔ (h, w) with width fastest: t-1 = h*Ww + w.
    hs = [div(t - 1, Ww) for t = 1:L]
    ws = [mod(t - 1, Ww) for t = 1:L]
    idx = Matrix{Int}(undef, L, L)
    span = 2 * Ww - 1
    @inbounds for j = 1:L, i = 1:L
        dh = hs[i] - hs[j] + (Wh - 1)
        dw = ws[i] - ws[j] + (Ww - 1)
        idx[i, j] = dh * span + dw + 1   # +1 for 1-based table indexing
    end
    return idx
end

"""
    rel_pos_attention(dim, dim_out; dim_head=32, window) -> @compact block

2D relative-position-bias multi-head self-attention over a `(dim, ...)` WHCN
feature map. `qkv` is a fused 1x1 conv `dim => 3*dim`; `proj` is a 1x1 conv
`dim => dim_out`. Heads number `dim ÷ dim_head`. A learned bias from
`rel_pos_bias_table` (gathered by a precomputed relative-position index for the
`window = (Wh, Ww)` grid) is added to the attention scores before the
key-axis softmax. Numerically matches timm's `Attention2d(head_first=True,
expand_first=False)` with `RelPosBias`.

PyTorch keys: `qkv.weight/bias` → `(:qkv, :weight/:bias)`
(`identity`/`identity`, 1x1 conv), `proj.weight/bias` → `(:proj, :weight/:bias)`,
`rel_pos.relative_position_bias_table` → `(:rel_pos_bias_table,)`
(`axis_reverse`, to `(num_rel, num_heads)`).
"""
function rel_pos_attention(dim::Int, dim_out::Int; dim_head::Int = 32, window::Tuple{Int,Int})
    num_heads = dim ÷ dim_head
    scale = Float32(1 / sqrt(dim_head))
    Wh, Ww = window
    L = Wh * Ww
    num_rel = (2 * Wh - 1) * (2 * Ww - 1)
    # Closure constant (not a trainable param): the gather index. Captured by
    # the do-block, mirroring how `num_heads`/`scale` are captured.
    ridx = vec(rel_pos_index(Wh, Ww))   # (L*L,), column-major (i fastest)
    @compact(
        qkv = Conv(
            (1, 1),
            dim => 3dim;
            use_bias = true,
            cross_correlation = true,
            init_bias = zeros32,
        ),
        proj = Conv(
            (1, 1),
            dim => dim_out;
            use_bias = true,
            cross_correlation = true,
            init_bias = zeros32,
        ),
        rel_pos_bias_table = zeros32(num_rel, num_heads),
    ) do x
        W, H, _, N = size(x)
        @assert W * H == L "rel_pos_attention: input grid $(W)x$(H) does not match " *
            "window $(Wh)x$(Ww)"
        y = qkv(x)                                  # (W, H, 3dim, N)
        y = reshape(y, L, dim_head, 3, num_heads, N)
        # Each slice (L, dim_head, num_heads, N) -> (dim_head, L, num_heads*N).
        q = reshape(permutedims(y[:, :, 1, :, :], (2, 1, 3, 4)), dim_head, L, num_heads * N)
        k = reshape(permutedims(y[:, :, 2, :, :], (2, 1, 3, 4)), dim_head, L, num_heads * N)
        v = reshape(permutedims(y[:, :, 3, :, :], (2, 1, 3, 4)), dim_head, L, num_heads * N)
        q = q .* scale
        scores = NNlib.batched_mul(NNlib.batched_transpose(q), k)   # (L, L, num_heads*N)
        # bias[i, j, h] = table[ridx[i, j], h]; broadcast over batch.
        bias = reshape(rel_pos_bias_table[ridx, :], L, L, num_heads)  # (L, L, num_heads)
        scores = reshape(scores, L, L, num_heads, N) .+ reshape(bias, L, L, num_heads, 1)
        attn = NNlib.softmax(scores; dims = 2)                       # over key axis
        attn = reshape(attn, L, L, num_heads * N)
        ctx = NNlib.batched_mul(v, NNlib.batched_transpose(attn))    # (dim_head, L, num_heads*N)
        ctx = reshape(ctx, dim_head, L, num_heads, N)
        ctx = permutedims(ctx, (1, 3, 2, 4))                        # (dim_head, num_heads, L, N)
        ctx = reshape(ctx, dim, L, N)                                # channel = head*dim_head + d
        ctx = permutedims(ctx, (2, 1, 3))                           # (L, dim, N)
        ctx = reshape(ctx, W, H, dim, N)                             # WHCN
        @return proj(ctx)
    end
end
