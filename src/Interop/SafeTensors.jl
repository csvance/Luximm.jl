# SafeTensors -> Julia state dict adapter.
#
# Resolves the axis-order mismatch between SafeTensors and the HDF5 parity
# fixtures: SafeTensors returns arrays in PyTorch logical axis order (NCHW
# for activations, (out, in, kH, kW) for conv weights); `read_parity` returns
# the reversed (Lux-natural) layout. With `reverse_axes = true` (the default),
# the dict this function returns matches what `read_parity` produces, so the
# same `<model>_mapping` table works for both fixture-driven parity tests and
# production HuggingFace loading.

"""
    load_safetensors_state_dict(path; reverse_axes=true) -> Dict{String, Array{Float32}}

Read a `.safetensors` file from disk into a dict of `Float32` arrays.

When `reverse_axes = true` (default), every tensor's axes are reversed so the
resulting layout matches `read_parity`'s HDF5-natural Julia layout (PyTorch
axes reversed). Set `reverse_axes = false` to keep PyTorch's logical axis
order if a caller wants that layout explicitly.
"""
function load_safetensors_state_dict(path::AbstractString; reverse_axes::Bool = true)
    raw = SafeTensors.load_safetensors(path)
    out = Dict{String, Array{Float32}}()
    for (k, v) in raw
        a = Float32.(v)
        if !(a isa AbstractArray)
            a = fill(Float32(a))
        elseif reverse_axes && ndims(a) > 1
            a = permutedims(a, ntuple(i -> ndims(a) + 1 - i, ndims(a)))
        end
        out[k] = a
    end
    return out
end
