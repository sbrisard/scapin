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
    h⁻¹ = @SVector zeros(DIM)
    α = @SVector zeros(DIM)
    c = @SVector zeros(DIM)
    s = @SVector zeros(DIM)

    for i = 1:DIM
        h⁻¹[i] = grid.N[i] / grid.L[i]
        α[i] = π * k[i] / grid.N[i]
        c[i] = cos(α[i])
        s[i] = sin(α[i])
    end

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
    h⁻¹ = @SVector zeros(DIM)
    φ = @SVector zeros(DIM)
    χ = @SVector zeros(DIM)
    ψ = @SVector zeros(DIM)

    for i = 1:DIM
        h⁻¹[i] = grid.N[i] / grid.L[i]
        β = 2π * k[i] / grid.N[i]
        φ[i] = 2 * h⁻¹ * (1 - cos(β))
        χ[i] = h⁻¹ * (2 + cos(β)) / 3
        ψ[i] = h⁻¹ * sin(β)
    end

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

L2 = @SVector [0.5, 1.]
N2 = @SVector [32, 64]
grid2 = CartesianGrid{2}(L2, N2)
k2 = @SVector [0, 0]
B2 = modal_strain_displacement(grid2, k2)
K2 = modal_stiffness(grid2, k2)

L3 = @SVector [0.5, 1., 2.]
N3 = @SVector [32, 64, 128]
grid3 = CartesianGrid{3}(L3, N3)
B3 = modal_strain_displacement(grid3, k3)
K3 = modal_stiffness(grid3, k3)

L4 = @SVector [0.5, 1., 2., 4.]
N4 = @SVector [32, 64, 128, 256]
grid4 = CartesianGrid{4}(L4, N4)
