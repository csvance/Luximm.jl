# HDF5 parity-fixture loader and weight-mapping helper.
#
# Vendored from MMILux.jl/src/Parity.jl. Kept byte-identical in function
# signatures and docstrings so that the `timm-to-lux` skill snippets
# at recipes/julia/skills/timm-to-lux/SKILL.md apply to both packages
# without modification.
#
# Fixtures are written by the Python sidecars in test/parity/ and have layout:
#
#   /input              dataset
#   /output             dataset OR group with one dataset per named output
#   /state_dict/<key>   dataset per PyTorch state_dict entry
#
# All tensors are stored in their **native PyTorch row-major byte order**.
# HDF5.jl reads them back as Julia arrays whose axes are **reversed** relative
# to the PyTorch logical order, but pointing at the same bytes. So:
#
#   PyTorch (N, C, H, W) tensor → Julia (W, H, C, N)  (matches Lux WHCN)
#   PyTorch (out, in, kH, kW) conv weight → Julia (kW, kH, in, out)
#                                            (matches Lux Conv layout)
#   PyTorch (C,) bias / norm scale → Julia (C,)       (matches Lux)
#
# For tensors where the user-side Julia layout is **not** simply the reverse
# of the PyTorch layout, the caller passes a per-tensor permutation via the
# mapping table.

"""
    read_parity(path) -> NamedTuple

Read a parity HDF5 fixture and return `(input, state_dict, output)`. Tensors
are returned as `Float32` arrays in their HDF5-natural Julia layout (PyTorch
axes reversed). Per-tensor permutations are the caller's responsibility.

`output` is a single `Array{Float32}` if the fixture wrote `/output` as a
dataset, or a `Dict{String, Array{Float32}}` if it wrote `/output/<name>`.
"""
function read_parity(path::AbstractString)
    HDF5.h5open(path, "r") do f
        input = Float32.(read(f["input"]))
        state_dict = _read_state_dict(f["state_dict"])
        output = _read_output(f["output"])
        return (; input, state_dict, output)
    end
end

function _read_state_dict(g)
    out = Dict{String,Array{Float32}}()
    for k in keys(g)
        out[k] = Float32.(read(g[k]))
    end
    return out
end

function _read_output(node)
    if node isa HDF5.Dataset
        return Float32.(read(node))
    elseif node isa HDF5.Group
        out = Dict{String,Array{Float32}}()
        for k in keys(node)
            out[k] = Float32.(read(node[k]))
        end
        return out
    else
        error("Unexpected /output node type: $(typeof(node))")
    end
end

"""
    apply_state_dict(ps, state_dict, mapping) -> ps

Rebuild a Lux parameter NamedTuple by replacing leaves according to `mapping`,
an iterable of `(pytorch_key, lux_path, transform)` triples where:

- `pytorch_key`: dotted name as written by Python's `model.state_dict().keys()`.
- `lux_path`: tuple of Symbols naming the leaf in `ps`, e.g.
  `(:stage1, :layer_1, :norm1, :gn, :scale)`.
- `transform`: function `Array{Float32} -> Array{Float32}` applied to the raw
  HDF5-read array. Common transforms are `identity` (HDF5-natural matches Lux)
  and `axis_reverse` (full axis reversal, used when the Julia tensor layout
  matches PyTorch's logical axis order rather than its reversed storage order).

The original `ps` is not mutated; the caller must bind the return value.
"""
function apply_state_dict(ps, state_dict::Dict{String,<:AbstractArray}, mapping)
    out = ps
    for (pykey, lux_path, transform) in mapping
        haskey(state_dict, pykey) || error("missing PyTorch state_dict key: $pykey")
        leaf = transform(state_dict[pykey])
        out = _set_leaf(out, lux_path, leaf)
    end
    return out
end

function _set_leaf(nt::NamedTuple, path::NTuple{N,Symbol}, leaf) where {N}
    head = path[1]
    haskey(nt, head) || error("leaf path missing key: $head (have: $(propertynames(nt)))")
    if N == 1
        return merge(nt, (; head => leaf))
    else
        sub = _set_leaf(getfield(nt, head), Base.tail(path), leaf)
        return merge(nt, (; head => sub))
    end
end

_set_leaf(nt::NamedTuple, path::Tuple, leaf) = _set_leaf(nt, Tuple(Symbol.(path)), leaf)
_set_leaf(nt::NamedTuple, path::AbstractVector, leaf) =
    _set_leaf(nt, Tuple(Symbol.(path)), leaf)

"""
    axis_reverse(a) -> Array

Permutes all axes in reverse order: `(d1, d2, ..., dN)` -> `(dN, ..., d2, d1)`.
Use this for tensors whose Julia layout was *designed* in PyTorch axis order.
The HDF5 read gives back the reversed layout; this permute restores the
original axis order.
"""
function axis_reverse(a::AbstractArray)
    return Float32.(permutedims(a, ntuple(i -> ndims(a) + 1 - i, ndims(a))))
end

"""
    pyperm(perm) -> Function

Build a transform that applies a specific permutation to the HDF5-read array.
Useful when the Julia axis order is neither the HDF5-natural reverse nor a
full reverse.
"""
pyperm(perm) = a -> Float32.(permutedims(a, perm))

"""
    as_channel4d(a) -> Array

Reshape a `(C,)` PyTorch norm parameter into `(1, 1, C, 1)`, the shape used
by `Lux.LayerNorm((1, 1, C); dims = 3)` (and so by `Luximm.Layers.layernorm2d`)
for its `:scale` / `:bias` leaves on WHCN 4D inputs.
"""
as_channel4d(a::AbstractArray) = reshape(Float32.(a), 1, 1, :, 1)

"""
    as_token_norm(a) -> Array

Reshape a `(C,)` PyTorch norm parameter into `(C, 1, 1)`, the shape used by a
channel-axis LayerNorm over a `(C, T, N)` token tensor (see
`Luximm.Layers.vit_layernorm`) for its `:scale` / `:bias` leaves.
"""
as_token_norm(a::AbstractArray) = reshape(Float32.(a), :, 1, 1)

"""
    adapt_input_conv(in_chans) -> transform

Build a state-dict transform that adapts a stem conv weight to the requested
input channel count, mirroring `timm.models._helpers.adapt_input_conv`.

The transform takes an HDF5-natural Julia conv weight in Lux's
`(kW, kH, I, O)` layout (returned by `read_parity` or
`load_safetensors_state_dict`) and returns a `(kW, kH, in_chans, O)`
weight following the timm recipe:

- `in_chans == I`: no-op (identity copy as `Float32`).
- `in_chans == 1`, `I == 3`: sum across the input-channel axis (the
  canonical RGB-to-grayscale collapse).
- `in_chans == 1`, `I > 3`, `I % 3 == 0`: reshape to `(kW, kH, 3, I÷3, O)`
  and sum the size-3 axis, leaving `(kW, kH, I÷3, O)`. This branch matches
  timm's special case for space-to-depth stems.
- `in_chans != I`, `I == 3`: tile the weight across the input-channel axis
  to cover `in_chans`, truncate, and rescale by `3 / in_chans` so the per-
  output-element response on a uniform input is preserved.
- Any other shape combination raises; timm itself does not support it.

Plug into `apply_state_dict` mappings in place of `identity` on the stem
weight entry when `in_chans != 3`.
"""
function adapt_input_conv(in_chans::Int)
    in_chans >= 1 || error("adapt_input_conv: in_chans must be ≥ 1, got $in_chans")
    return function (w::AbstractArray)
        ndims(w) == 4 ||
            error("adapt_input_conv expects a 4-D conv weight; " * "got ndims=$(ndims(w))")
        w32 = Float32.(w)
        kW, kH, I, O = size(w32)
        if in_chans == I
            return w32
        elseif in_chans == 1
            if I > 3
                I % 3 == 0 || error(
                    "adapt_input_conv: cannot collapse $I-channel " *
                    "stem weight to 1 channel: I % 3 must be 0",
                )
                grouped = reshape(w32, kW, kH, 3, I ÷ 3, O)
                return dropdims(sum(grouped; dims = 3); dims = 3)
            else
                return sum(w32; dims = 3)
            end
        else
            I == 3 || error(
                "adapt_input_conv: only supports adapting from I=3 to " *
                "in_chans=$in_chans; got I=$I",
            )
            n_repeat = cld(in_chans, 3)
            tiled = repeat(w32, 1, 1, n_repeat, 1)
            cropped = tiled[:, :, 1:in_chans, :]
            scale = Float32(3 / in_chans)
            return scale .* cropped
        end
    end
end
