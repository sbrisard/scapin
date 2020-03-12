using StaticArrays

struct CartesianGrid{DIM}
    L::SVector{DIM,Float64}
    N::SVector{DIM,Int}
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
        throw(DomainError(DIM))
    end
end
