# Pre-norm transformer encoder block, Lux port of timm's ViT `Block`
# (timm/models/vision_transformer.py).
#
#   x = x + attn(norm1(x))
#   x = x + mlp(norm2(x))   with mlp = fc2 ∘ gelu_erf ∘ fc1
#
# The LayerNorm here normalizes over the **channel axis 1** of a `(dim, T, N)`
# token tensor (a different layer than `layernorm2d`, which normalizes axis 3
# of a WHCN map). timm ViT uses `eps = 1e-6`, which differs from the `1e-5`
# used by the conv families.

"""
    vit_layernorm(dim; eps=1f-6) -> Lux.LayerNorm

LayerNorm over the channel axis (axis 1) of a `(dim, T, N)` token tensor, with
per-channel affine. Matches timm's ViT `nn.LayerNorm(dim, eps=1e-6)`. The
affine `:scale` / `:bias` leaves have shape `(dim, 1, 1)`, so the PyTorch
`(dim,)` parameter is reshaped with `Luximm.Interop.as_token_norm` when loading.
"""
vit_layernorm(dim::Int; eps::Float32 = 1.0f-6) =
    Lux.LayerNorm((dim, 1); dims = 1, epsilon = eps)

"""
    vit_block(dim; num_heads, mlp_ratio=4, eps=1f-6) -> @compact block

A single pre-norm ViT encoder block over a `(dim, T, N)` token tensor:
LayerNorm → [`mhsa`](@ref) → residual, then LayerNorm → MLP (`fc1` → exact
GELU → `fc2`) → residual.

PyTorch keys map as: `norm1/norm2.{weight,bias}` → `(:norm1/:norm2,
:scale/:bias)` (`as_token_norm`), `attn.*` via [`mhsa`](@ref), `mlp.fc1/fc2.*`
→ `(:fc1/:fc2, :weight/:bias)` (`axis_reverse`/`identity`).
"""
function vit_block(dim::Int; num_heads::Int, mlp_ratio::Int = 4, eps::Float32 = 1.0f-6)
    hidden = mlp_ratio * dim
    @compact(
        norm1 = vit_layernorm(dim; eps = eps),
        attn = mhsa(dim; num_heads = num_heads),
        norm2 = vit_layernorm(dim; eps = eps),
        fc1 = Dense(dim => hidden; init_bias = zeros32),
        fc2 = Dense(hidden => dim; init_bias = zeros32),
    ) do x
        x = x .+ attn(norm1(x))
        h = fc2(NNlib.gelu_erf.(fc1(norm2(x))))
        @return x .+ h
    end
end
