import numpy as np
import pytest

from pyscapin.hooke import HookeFloat64_2D, HookeFloat64_3D

SQRT2 = np.sqrt(2.0)

_ij2i = {2: np.array([0, 1, 0]), 3: np.array([0, 1, 2, 1, 2, 0])}

_ij2j = {2: np.array([0, 1, 1]), 3: np.array([0, 1, 2, 2, 0, 1])}


def directions_2D(num_theta):
    out = np.empty((num_theta, 2), dtype=np.float64)
    theta = np.linspace(0.0, np.pi, num=num_theta, endpoint=False)
    out[:, 0] = np.cos(theta)
    out[:, 1] = np.sin(theta)
    return out


def directions_3D(num_theta, num_phi):
    theta = np.linspace(0.0, np.pi, num=num_theta)
    cos_theta, sin_theta = np.cos(theta)[:, None], np.sin(theta)[:, None]
    phi = np.linspace(0.0, 2 * np.pi, num=num_phi, endpoint=False)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    out = np.empty((num_theta, num_phi, 3), dtype=np.float64)
    out[..., 0] = sin_theta * cos_phi
    out[..., 1] = sin_theta * sin_phi
    out[..., 2] = cos_theta
    return out.reshape((num_theta * num_phi, -1))


def actual_hooke_matrix(hooke, n):
    out = np.empty((hooke.osize, hooke.isize), dtype=np.float64)
    tau = np.zeros((hooke.isize,), dtype=np.float64)
    eps = np.zeros((hooke.osize,), dtype=np.float64)
    for j in range(hooke.isize):
        tau[j] = 1.0
        hooke.apply(n, tau, eps)
        out[:, j] = eps
        tau[j] = 0.0
    return out


def expected_hooke_matrix(n, nu, dim):
    sym = (dim * (dim + 1)) // 2
    out = np.empty((sym, sym), dtype=np.float64)

    ij2i = _ij2i[dim]
    ij2j = _ij2j[dim]

    for ij in range(sym):
        i = ij2i[ij]
        j = ij2j[ij]
        w_ij = 1.0 if ij < dim else SQRT2
        for kl in range(sym):
            k = ij2i[kl]
            l = ij2j[kl]
            w_kl = 1.0 if kl < dim else SQRT2
            delta_ik = 1 if i == k else 0
            delta_il = 1 if i == l else 0
            delta_jk = 1 if j == k else 0
            delta_jl = 1 if j == l else 0
            out[ij, kl] = (
                w_ij
                * w_kl
                * (
                    0.25
                    * (
                        delta_ik * n[j] * n[l]
                        + delta_il * n[j] * n[k]
                        + delta_jk * n[i] * n[l]
                        + delta_jl * n[i] * n[k]
                    )
                    - 0.5 * n[i] * n[j] * n[k] * n[l] / (1.0 - nu)
                )
            )
    return out


@pytest.mark.parametrize("Hooke", [HookeFloat64_2D, HookeFloat64_3D])
def test_hooke_apply(Hooke, rtol=1e-15, atol=1e-15):
    mu, nu = 1.0, 0.3
    k_norm = np.array([1.2, 3.4, 5.6])
    hooke = Hooke(mu, nu)
    n = directions_2D(20) if hooke.dim == 2 else directions_3D(10, 20)
    for i in range(n.shape[0]):
        exp = expected_hooke_matrix(n[i], nu, hooke.dim)
        for j in range(k_norm.shape[0]):
            k = k_norm[j] * n[i]
            act = actual_hooke_matrix(hooke, k)
            np.testing.assert_allclose(act, exp, rtol=rtol, atol=atol)
