# Patch embedding, Lux port of timm's `PatchEmbed` (timm/layers/patch_embed.py).
#
# The only WHCN→sequence bridge in the ViT stack: a stride-`patch` conv tiles
# the image into non-overlapping patches, then the spatial grid is flattened to
# a token axis. Tokens are laid out `(embed_dim, num_tokens, batch)` = `(C, T,
# N)` so that downstream `Dense`/attention contract axis 1 with no permute.
#
# Token-order trap: timm does `conv -> flatten(2) -> transpose(1, 2)`, i.e. the
# spatial grid `(H, W)` is flattened row-major with **width fastest**, giving
# token `t = h*W + w`. Lux's conv output is `(W, H, C, N)`, so a column-major
# `reshape` of the `(W, H)` axes (after moving channels to axis 1) also runs
# **width fastest** — `t = w + h*W` — which is the same order. Confirmed against
# the pos-embed fixture at parity time.

"""
    patch_embed(in_chans, embed_dim; patch=16) -> @compact block

Split a `(W, H, in_chans, N)` image into `patch x patch` patches via a
stride-`patch` conv and return a token tensor `(embed_dim, num_tokens, N)`
where `num_tokens = (W/patch) * (H/patch)` and tokens run width-fastest,
matching timm's `PatchEmbed`.

PyTorch keys `<prefix>.proj.weight` / `<prefix>.proj.bias` map to the `:proj`
Conv leaves (`identity`, or `adapt_input_conv` for the weight when
`in_chans != 3`).
"""
function patch_embed(in_chans::Int, embed_dim::Int; patch::Int = 16)
    @compact(
        proj = Conv(
            (patch, patch),
            in_chans => embed_dim;
            stride = patch,
            pad = 0,
            use_bias = true,
            cross_correlation = true,
            init_bias = zeros32,
        ),
    ) do x
        y = proj(x)                       # (Wp, Hp, D, N)
        Wp, Hp, D, N = size(y)
        y = permutedims(y, (3, 1, 2, 4))  # (D, Wp, Hp, N)
        @return reshape(y, D, Wp * Hp, N) # (D, T, N), width fastest
    end
end
