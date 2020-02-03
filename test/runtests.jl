using LinearAlgebra
using Test
using Scapin
using StaticArrays

function greenop_matrix_ref(ν, n::SVector{DIM, T}) where {T, DIM}
    out = zeros(T, 3, 3)
    sym = 3
    ij2i = [1, 2, 1]
    ij2j = [1, 2, 2]

    for ij = 1:sym
        i = ij2i[ij]
        j = ij2j[ij]
        w_ij = ij <= DIM ? one(T) : sqrt(2one(T))
        for kl = 1:sym
            k = ij2i[kl]
            l = ij2j[kl]
            w_kl = kl <= DIM ? one(T) : sqrt(2one(T))
            δ_ik = i == k ? 1 : 0
            δ_il = i == l ? 1 : 0
            δ_jk = j == k ? 1 : 0
            δ_jl = j == l ? 1 : 0
            out[ij, kl] = w_ij*w_kl*(0.25*(δ_ik*n[j]*n[l]+δ_il*n[j]*n[k]+
                                           δ_jk*n[i]*n[l]+δ_jl*n[i]*n[k])-
                                     n[i]*n[j]*n[k]*n[l]/(2*(1-ν)))
        end
    end
    out
end

@testset "Green operator for 2D linear elasticity" begin
    hooke = Hooke(1.0, 0.3)
    for θ ∈ LinRange(0., 2*π, 21)[1:end-1]
        n = @SVector [cos(θ), sin(θ)]  # Dimension and floating point type inferred here!
        Γ_act = Matrix(kblock(hooke, n))
        Γ_exp = greenop_matrix_ref(hooke.ν, n)

        @test Γ_act ≈ Γ_exp atol=1e-15
    end
end
