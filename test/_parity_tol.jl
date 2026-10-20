# Shared parity tolerance and check for all family test files.
# Include with: isdefined(@__MODULE__, :PARITY_RTOL) || include("_parity_tol.jl")
#
# Every parity comparison (features, logits, pyramid taps) uses one metric, the
# relative L2 error ‖y - ref‖ / ‖ref‖, against one bar. The suite runs on CPU in
# FP32, where summation order, BLAS threading and CPU kernels move the error by a
# few 1e-6; porting bugs (a wrong weight mapping, a transposed reshape, a missing
# layer, the wrong activation) land at 1e-2 and above. The bar sits between the
# two with room for the machine-to-machine spread. Differences small enough to
# pass, such as a slightly different norm epsilon, do not affect transfer learning.

using LinearAlgebra: norm, dot

const PARITY_RTOL = 1.0f-3

rel_l2(y, ref) = norm(y .- ref) / max(norm(ref), floatmin(Float32))

# Set JIMM_PARITY_REPORT to a file path to append one CSV row per check.
const _PARITY_REPORT = get(ENV, "JIMM_PARITY_REPORT", "")
const _PARITY_REPORT_HEADER = "variant,check,rel_l2,cosine,max_abs,pass"

function _report_parity(variant, check, rel, cosine, max_abs, pass)
    isempty(_PARITY_REPORT) && return nothing
    new = !isfile(_PARITY_REPORT) || filesize(_PARITY_REPORT) == 0
    open(_PARITY_REPORT, "a") do io
        new && println(io, _PARITY_REPORT_HEADER)
        println(io, join((variant, check, rel, cosine, max_abs, pass), ','))
    end
    return nothing
end

"""
    check_parity(variant, check, y, ref)

Test that `y` has the shape of `ref` and is within `PARITY_RTOL` of it in
relative L2, log the metrics, and append them to the report when one is set.
"""
function check_parity(variant, check, y, ref)
    @test size(y) == size(ref)
    size(y) == size(ref) || return nothing
    rel = rel_l2(y, ref)
    cosine = dot(vec(y), vec(ref)) / max(norm(y) * norm(ref), floatmin(Float32))
    max_abs = maximum(abs.(y .- ref))
    pass = rel < PARITY_RTOL
    @info "$(variant) $(check): rel-L2 = $rel, cosine = $cosine, max-abs-diff = $max_abs"
    _report_parity(variant, check, rel, cosine, max_abs, pass)
    @test rel < PARITY_RTOL
    return nothing
end
