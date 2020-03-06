module Scapin

using AbstractFFTs
using StaticArrays
import Base: size

export Hooke, block_apply!, block_matrix, TruncatedGreenOperator

struct Hooke{T,DIM}
    μ::T
    ν::T
end

size(::Hooke{T,2}) where {T} = (3, 3)
size(::Hooke{T,3}) where {T} = (6, 6)

function block_apply!(out, hooke::Hooke{T,2}, k, τ) where {T}
    k² = sum(abs2, k)
    τk₁ = τ[1] * k[1] + τ[3] * k[2] / sqrt(2 * one(T))
    τk₂ = τ[2] * k[2] + τ[3] * k[1] / sqrt(2 * one(T))
    nτn = (k[1] * τk₁ + k[2] * τk₂) / k²
    const1 = nτn / (1 - hooke.ν)
    const2 = 1 / (2 * hooke.μ * k²)
    out[1] = const2 * (k[1] * (2 * τk₁ - const1 * k[1]))
    out[2] = const2 * (k[2] * (2 * τk₂ - const1 * k[2]))
    const3 = sqrt(2 * one(T)) * const2
    out[3] = const3 * (k[1] * τk₂ + k[2] * τk₁ - const1 * k[1] * k[2])
    return out
end

function block_apply!(out, hooke::Hooke{T,3}, k, τ) where {T}
    k² = sum(abs2, k)
    τk₁ = τ[1] * k[1] + (τ[6] * k[2] + τ[5] * k[3]) / sqrt(2 * one(T))
    τk₂ = τ[2] * k[2] + (τ[6] * k[1] + τ[4] * k[3]) / sqrt(2 * one(T))
    τk₃ = τ[3] * k[3] + (τ[5] * k[1] + τ[4] * k[2]) / sqrt(2 * one(T))
    nτn = (k[1] * τk₁ + k[2] * τk₂ + k[3] * τk₃) / k²
    const1 = nτn / (1 - hooke.ν)
    const2 = 1 / (2 * hooke.μ * k²)
    out[1] = const2 * (k[1] * (2 * τk₁ - const1 * k[1]))
    out[2] = const2 * (k[2] * (2 * τk₂ - const1 * k[2]))
    out[3] = const2 * (k[3] * (2 * τk₃ - const1 * k[3]))
    const3 = sqrt(2 * one(T)) * const2
    out[4] = const3 * (k[2] * τk₃ + k[3] * τk₂ - const1 * k[2] * k[3])
    out[5] = const3 * (k[3] * τk₁ + k[1] * τk₃ - const1 * k[3] * k[1])
    out[6] = const3 * (k[1] * τk₂ + k[2] * τk₁ - const1 * k[1] * k[2])
    return out
end

function block_matrix(op::Hooke{T,DIM}, k::SVector{DIM,T}) where {T,DIM}
    nrows, ncols = size(op)
    mat = zeros(T, nrows, ncols)
    τ = zeros(T, ncols)
    for i = 1:ncols
        τ[i] = one(T)
        block_apply!(view(mat, :, i), op, k, τ)
        τ[i] = zero(T)
    end
    mat
end

struct TruncatedGreenOperator{T, DIM}
    Γ::Hooke{T, DIM}
    N::SVector{DIM, Int}
    L::SVector{DIM, T}
    # TODO: I would like k to be an AbstractArray
    k::AbstractArray{Frequencies{T}, 1}
    function TruncatedGreenOperator{T, DIM}(Γ::Hooke{T, DIM},
                                            N::SVector{DIM, Int},
                                            L::SVector{DIM, T}) where {T, DIM}
        k = [fftfreq(N[i], 2π*N[i]/L[i]) for i in 1:DIM]
        new(Γ, N, L, k)
    end
end

# function apply!(out, Γ_h::TruncatedGreenOperator{T, DIM}, i, τ) where {T, DIM}
#     block_apply!(out, Γ_h.Γ, Γ_h.k[i], τ)
#     return τ
# end

end # module
