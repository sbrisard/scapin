module Scapin
using StaticArrays
import Base: size, Matrix

export Hooke, HookeBlock, kblock, mul!

struct Hooke
    μ
    ν
end

struct HookeBlock{T, DIM}
    hooke::Hooke
    k::SVector{DIM, T}
end

kblock(hooke::Hooke, k::SVector{DIM, T}) where {DIM, T} = HookeBlock{T, DIM}(hooke, k)

size(::HookeBlock{T, 2}) where T = (3, 3)

# Notice this is general and will work also for other dimensions without any extra code
function Matrix(block::HookeBlock{T}) where T
    n, m = size(block)
    mat = zeros(T, n, m)
    τ = zeros(T, m)
    for i in 1:m
        τ[i] = one(T)
        mul!(view(mat, :, i), block, τ)
        τ[i] = zero(T)
    end
    mat
end

function mul!(out, block::HookeBlock{T, 2}, τ) where {T}
    hooke = block.hooke
    k = block.k

    # I think this is too much micro-optimisation for the stage of your code.
    # Why do you not first write ti down more readable and then only optimise the
    # parts the Julia compiler has trouble with?
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

end
