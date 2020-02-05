module Scapin
using StaticArrays
import Base: size, Matrix

export Hooke, block_apply!, block_matrix

struct Hooke{T, DIM}
    μ::T
    ν::T
end

size(::Hooke{T, 2}) where T = (3, 3)

function block_apply!(out, hooke::Hooke{T, 2}, k, τ) where {T}
    τk₁ = τ[1]*k[1]+τ[3]*k[2] / sqrt(2)
    τk₂ = τ[2]*k[2]+τ[3]*k[1] / sqrt(2)
    nτn = (k[1]*τk₁+k[2]*τk₂) / sum(abs2, k)
    const1 = nτn / (1-hooke.ν)
    const2 = 1 / (2*hooke.μ*sum(abs2, k))
    out[1] = const2*(k[1]*(2*τk₁-const1*k[1]))
    out[2] = const2*(k[2]*(2*τk₂-const1*k[2]))
    const3 = sqrt(2)*const2
    out[3] = const3*(k[1]*τk₂+k[2]*τk₁-const1*k[1]*k[2])
    return out
end

function block_matrix(op::Hooke{T, DIM}, k::SVector{DIM, T}) where {T, DIM}
    nrows, ncols = size(op)
    mat = zeros(T, nrows, ncols)
    τ = zeros(T, ncols)
    for i in 1:ncols
        τ[i] = one(T)
        block_apply!(view(mat, :, i), op, k, τ)
        τ[i] = zero(T)
    end
    mat
end

end
