using LinearAlgebra
using Scapin
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

function ms94_frequencies(N, L)
    # TODO Ensure that π and L are of the same type
    # TODO Can we remove type parameters?
    [2π*(2n < N ? n : n-N)/L for n=1:N]
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

@testset "Discrete Green operator [MS94], Hooke 2D" begin
    T = Float64
    DIM = 2
    Γ = Hooke{T, DIM}(1.2, 0.3)
    N = @SVector [5, 6]
    L = @SVector [3.4, 5.6]
    Γ_h = TruncatedGreenOperator{T, DIM}(Γ, N, L)
    @test Γ_h.N == N
    @test Γ_h.L == L
    for i = 1:DIM
        @test Γ_h.k[i] == ms94_frequencies(N[i], L[i])
    end
end

@testset "Discrete Green operator [MS94], Hooke 3D" begin
    T = Float64
    DIM = 3
    Γ = Hooke{T, DIM}(1.2, 0.3)
    N = @SVector [5, 6, 7]
    L = @SVector [3.4, 5.6, 7.8]
    Γ_h = TruncatedGreenOperator{T, DIM}(Γ, N, L)
    @test Γ_h.N == N
    @test Γ_h.L == L
    for i = 1:DIM
        @test Γ_h.k[i] == ms94_frequencies(N[i], L[i])
    end
end
