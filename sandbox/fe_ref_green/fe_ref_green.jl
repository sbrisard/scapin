using StaticArrays

struct CartesianGrid{DIM}
    L::SVector{DIM,Float64}
    N::SVector{DIM,Int}
    function CartesianGrid{DIM}(L::SVector{DIM,Float64}, N::SVector{DIM,Int}) where {DIM}
        if DIM < 2 || DIM > 3
            throw(DomainError(DIM))
        end
        new(L, N)
    end
end

function modal_strain_displacement(
    grid::CartesianGrid{DIM},
    k::SVector{DIM,Int},
) where {DIM}
    h⁻¹ = grid.N ./ grid.L
    α = π * k ./ grid.N
    c = cos.(α)
    s = sin.(α)
    prefactor = 2 * im * exp(im * sum(α))
    if DIM == 2
        return prefactor * [h⁻¹[1] * s[1] * c[2], h⁻¹[2] * c[1] * s[2]]
    elseif DIM == 3
        return prefactor * [
            h⁻¹[1] * s[1] * c[2] * c[3],
            h⁻¹[2] * c[1] * s[2] * c[3],
            h⁻¹[3] * c[1] * c[2] * s[3],
        ]
    else
        # This should never occur
        throw(DomainError(DIM))
    end
end

function modal_stiffness(grid::CartesianGrid{DIM}, k::SVector{DIM,Int}) where {DIM}
    # {φ, χ, ψ}[i] = {φ, χ, ψ}(zᵢ) in the notation of [Bri16]
    h⁻¹ = grid.N ./ grid.L
    β = 2π * k ./ grid.N
    φ = 2 * (1 .- cos.(β))
    χ = (2 .+ cos.(β)) / 3
    ψ = sin.(β)

    if DIM == 2
        return [
            [h⁻¹[1] * h⁻¹[1] * φ[1] * χ[2], h⁻¹[1] * h⁻¹[2] * ψ[1] * ψ[2]],
            [h⁻¹[1] * h⁻¹[2] * ψ[1] * ψ[2], h⁻¹[2]^2 * χ[1] * φ[2]],
        ]
    elseif DIM == 3
        return [
            [
                h⁻¹[1] * h⁻¹[1] * φ[1] * χ[2] * χ[3],
                h⁻¹[1] * h⁻¹[2] * ψ[1] * ψ[2] * χ[3],
                h⁻¹[1] * h⁻¹[3] * ψ[1] * χ[2] * ψ[3],
            ],
            [
                h⁻¹[2] * h⁻¹[1] * ψ[1] * ψ[2] * χ[3],
                h⁻¹[2] * h⁻¹[2] * χ[1] * φ[2] * χ[3],
                h⁻¹[2] * h⁻¹[3] * χ[1] * ψ[2] * ψ[3],
            ],
            [
                h⁻¹[3] * h⁻¹[1] * ψ[1] * χ[2] * ψ[3],
                h⁻¹[3] * h⁻¹[2] * χ[1] * ψ[2] * ψ[3],
                h⁻¹[3] * h⁻¹[3] * χ[1] * χ[2] * φ[3],
            ],
        ]
    else
        # This should never occur
        throw(DomainError(DIM))
    end
end

DIM = 2
L2 = @SVector [0.5, 1.0]
N2 = SVector{DIM,Int}(32, 64)
grid2 = CartesianGrid{DIM}(L2, N2)
k2 = @SVector [0.0, 0.0]
B2 = modal_strain_displacement(grid2, k2)
K2 = modal_stiffness(grid2, k2)

DIM = 3
L3 = @SVector [0.5, 1.0, 2.0]
N3 = SVector{DIM,Int}(32, 64, 128)
grid3 = CartesianGrid{DIM}(L3, N3)
k3 = @SVector [0.0, 0.0]
B3 = modal_strain_displacement(grid3, k3)
K3 = modal_stiffness(grid3, k3)

# Should throw an exception
# DIM = 4
# L4 = @SVector [0.5, 1., 2., 4.]
# N4 = SVector{DIM, Int}(32, 64, 128, 256)
# grid4 = CartesianGrid{DIM}(L4, N4)
