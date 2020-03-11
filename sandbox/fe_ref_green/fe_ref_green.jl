using StaticArrays

struct CartesianGrid{DIM}
    L::SVector{DIM,Float64}
    N::SVector{DIM,Int}
end

function modal_strain_displacement(grid::CartesianGrid{2}, k::SVector{2,Int})
    α₁ = π * k[1] / grid.N[1]
    α₂ = π * k[2] / grid.N[2]
    2 *
    im *
    exp(im * (α₁ + α₂)) *
    [N[1] / L[1] * sin(α₁) * cos(α₂), N[2] / L[2] * cos(α₁) * sin(α₂)]
end

function modal_strain_displacement(grid::CartesianGrid{3}, k::SVector{3,Int})
    α₁ = π * k[1] / grid.N[1]
    α₂ = π * k[2] / grid.N[2]
    α₃ = π * k[3] / grid.N[3]
    2 *
    im *
    exp(im * (α₁ + α₂ + α₃)) *
    [
        N[1] / L[1] * sin(α₁) * cos(α₂) * cos(α₃),
        N[2] / L[2] * cos(α₁) * sin(α₂) * cos(α₃),
        N[3] / L[3] * cos(α₁) * cos(α₂) * sin(α₃),
    ]
end

function modal_stiffness(grid::CartesianGrid{2}, k::SVector{2,Int})
    h₁ = L[1] / N[1]
    β₁ = 2π * k[1] / grid.N[1]
    φ₁ = 2 * (1 - cos(β₁))
    χ₁ = (2 + cos(β₁)) / 3
    ψ₁ = sin(β₁)

    h₂ = L[2] / N[2]
    β₂ = 2π * k[2] / grid.N[2]
    φ₂ = 2 * (1 - cos(β₂))
    χ₂ = (2 + cos(β₂)) / 3
    ψ₂ = sin(β₂)

    H = [
        [φ₁ * χ₂ / h₁^2 ψ₁ * ψ₂ / h₁ / h₂]
        [ψ₁ * ψ₂ / h₁ / h₂ χ₁ * φ₂ / h₂^2]
    ]
end
