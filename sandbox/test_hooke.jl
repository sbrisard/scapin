using LinearAlgebra
using Test

include("Scapin.jl")

using .Scapin

function greenop_matrix!(hooke::Hooke{T, DIM}, k::AbstractArray{T, 1},
                         out::AbstractArray{T,2}) where {T, DIM}
    zero_ = zero(T)
    one_ = one(T)
    n = in_size(hooke)
    τ = zeros(T, n)
    for i = 1:n
        τ[i] = one_
        greenop_apply!(hooke, k, τ, view(out, :, i))
        τ[i] = zero_
    end
    return out
end

function greenop_matrix_ref!(ν, n::AbstractArray{T,1}, out::AbstractArray{T,2}) where{T}
    one_ = one(T)
    sqrt2 = sqrt(2*one_)
    dim = 2
    sym = 3
    ij2i = [1, 2, 1]
    ij2j = [1, 2, 2]

    for ij = 1:sym
        i = ij2i[ij]
        j = ij2j[ij]
        w_ij = ij <= dim ? one_ : sqrt2
        for kl = 1:sym
            k = ij2i[kl]
            l = ij2j[kl]
            w_kl = kl <= dim ? one_ : sqrt2
            δ_ik = i == k ? 1 : 0
            δ_il = i == l ? 1 : 0
            δ_jk = j == k ? 1 : 0
            δ_jl = j == l ? 1 : 0
            out[ij, kl] = w_ij*w_kl*(0.25*(δ_ik*n[j]*n[l]+δ_il*n[j]*n[k]+
                                           δ_jk*n[i]*n[l]+δ_jl*n[i]*n[k])-
                                     n[i]*n[j]*n[k]*n[l]/(2*(1-ν)))
        end
    end
    return out
end

@testset "Green operator for 2D linear elasticity" begin
    T = Float64
    DIM = 2
    hooke = Hooke{T, DIM}(1.0, 0.3)
    Γ_act = zeros(T, in_size(hooke), out_size(hooke))
    Γ_exp = zeros(T, in_size(hooke), out_size(hooke))
    n = zeros(T, DIM)
    for θ ∈ LinRange(0., 2*π, 21)[1:end-1]
        n[1] = cos(θ)
        n[2] = sin(θ)
        greenop_matrix!(hooke, n, Γ_act)
        greenop_matrix_ref!(hooke.ν, n, Γ_exp)

        @test all(isapprox.(Γ_act, Γ_exp, atol=1e-15))
    end
end
