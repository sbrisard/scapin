using LinearAlgebra
using Test
using Scapin
using StaticArrays

function block_matrix_ref(hooke::Hooke{T, DIM}, k::SVector{DIM, T}) where {T, DIM}
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
    n = k/norm(k)
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
            mat[ij, kl] = w_ij*w_kl*(0.25*(δ_ik*n[j]*n[l]+δ_il*n[j]*n[k]+
                                           δ_jk*n[i]*n[l]+δ_jl*n[i]*n[k])-
                                     n[i]*n[j]*n[k]*n[l]/(2*(1-hooke.ν)))
            mat[ij, kl] *= hooke.μ
        end
    end
    mat
end

@testset "Green operator for 2D linear elasticity" begin
    hooke = Hooke{Float64, 2}(1.0, 0.3)
    for θ ∈ LinRange(0., 2*π, 21)[1:end-1]
        n = @SVector [cos(θ), sin(θ)]
        Γ_act = block_matrix(hooke, n)
        Γ_exp = block_matrix_ref(hooke, n)

        @test all(isapprox.(Γ_act, Γ_exp, atol=1e-15))
    end
end
