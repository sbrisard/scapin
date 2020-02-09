using StaticArrays
import Base: size

export Hooke, block_apply!, block_matrix

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
    grid_size::SVector{DIM, Int}
    grid_dim::SVector{DIM, T}
    k::AbstractArray{AbstractArray{T, 1}, 1}
    function TruncatedGreenOperator(Γ::Hooke{T, DIM},
                                    grid_size::SVector{DIM, Int},
                                    grid_dim::SVector{DIM, T}) where {T, DIM}
        k = [fftfreq(grid_size[i], 2π*grid_size[i]/grid_dim[i]) for i in 1:DIM]
        new(Γ, grid_size, grid_dim, k)
    end
end

function apply!(out, Γ_h::TruncatedGreenOperator{T, DIM}, i, τ)
    block_apply!(out, Γ_h.Γ, Γ_h.k[i], τ)
    return τ
end

# ==================== TESTS ====================

using LinearAlgebra
using StaticArrays
using Test

function block_matrix_ref(hooke::Hooke{T,DIM}, k::SVector{DIM,T}) where {T,DIM}
    if DIM == 2
        sym = 3
        ij2i = [1, 2, 1]
        ij2j = [1, 2, 2]
    elseif DIM == 3
        sym = 6
        ij2i = [1, 2, 3, 2, 3, 1]
        ij2j = [1, 2, 3, 3, 1, 2]
    else
        throw(ArgumentError("DIM must be 2 or 3 (was $DIM)"))
    end

    mat = zeros(T, sym, sym)
    n = k / norm(k)
    for ij = 1:sym
        i = ij2i[ij]
        j = ij2j[ij]
        w_ij = ij <= DIM ? one(T) : sqrt(2 * one(T))
        for kl = 1:sym
            k = ij2i[kl]
            l = ij2j[kl]
            w_kl = kl <= DIM ? one(T) : sqrt(2 * one(T))
            δik_nj_nl = i == k ? n[j] * n[l] : zero(T)
            δil_nj_nk = i == l ? n[j] * n[k] : zero(T)
            δjk_ni_nl = j == k ? n[i] * n[l] : zero(T)
            δjl_ni_nk = j == l ? n[i] * n[k] : zero(T)
            aux1 = (δik_nj_nl + δil_nj_nk + δjk_ni_nl + δjl_ni_nk) / 4
            aux2 = n[i] * n[j] * n[k] * n[l] / (2 * (1 - hooke.ν))
            mat[ij, kl] = w_ij * w_kl * (aux1 - aux2) / hooke.μ
        end
    end
    mat
end

@testset "Green operator for 2D linear elasticity" begin
    hooke = Hooke{Float64,2}(5.6, 0.3)
    for k_norm ∈ [0.12, 2.3, 14.5]
        for θ ∈ LinRange(0.0, 2 * π, 21)[1:end-1]
            k = @SVector [k_norm * cos(θ), k_norm * sin(θ)]
            act = block_matrix(hooke, k)
            exp = block_matrix_ref(hooke, k)

            @test all(isapprox.(act, exp, atol = 1e-15))
        end
    end
end

@testset "Green operator for 3D linear elasticity" begin
    hooke = Hooke{Float64,3}(5.6, 0.3)
    for k_norm ∈ [0.12, 2.3, 14.5]
        for φ ∈ LinRange(0.0, 2 * π, 21)[1:end-1]
            for θ ∈ LinRange(0.0, π, 11)
                k = k_norm * (@SVector [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)])
                act = block_matrix(hooke, k)
                exp = block_matrix_ref(hooke, k)

                @test all(isapprox.(act, exp, atol = 1e-15))
            end
        end
    end
end
