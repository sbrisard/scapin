module Scapin

export Hooke, in_size, out_size, greenop_apply!

const SQRT2 = sqrt(2)
const SQRT1_2 = 1 / sqrt(2)

struct Hooke{T, DIM}
    μ::T
    ν::T
end

function in_size(::Hooke{T, 2}) where {T}
    return 3
end

function out_size(::Hooke{T, 2}) where {T}
    return 3
end

function greenop_apply!(hooke::Hooke{T, 2}, k::AbstractArray{T,1},
                        τ::AbstractArray{T,1}, out::AbstractArray{T,1}) where {T}
    k² = k[1]*k[1]+k[2]*k[2]
    τk₁ = τ[1]*k[1]+SQRT1_2*τ[3]*k[2]
    τk₂ = τ[2]*k[2]+SQRT1_2*τ[3]*k[1]
    nτn = (k[1]*τk₁+k[2]*τk₂) / k²
    const1 = nτn / (1-hooke.ν)
    const2 = 1 / (2*hooke.μ*k²)
    out[1] = const2*(k[1]*(2*τk₁-const1*k[1]))
    out[2] = const2*(k[2]*(2*τk₂-const1*k[2]))
    const3 = SQRT2*const2
    out[3] = const3*(k[1]*τk₂+k[2]*τk₁-const1*k[1]*k[2])
    return out
end

end
