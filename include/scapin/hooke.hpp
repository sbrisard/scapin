#pragma once

#include <cmath>
#include <numbers>
#include <ostream>

#include "core.hpp"

namespace scapin {
template <typename T, int DIM>
requires spatial_dimension<DIM> class Hooke {
 public:
  static constexpr int dim = DIM;
  static constexpr int isize = (DIM * (DIM + 1)) / 2;
  static constexpr int osize = (DIM * (DIM + 1)) / 2;
  using Scalar = typename T;
  using Real = remove_complex_t<T>;

  Real const mu;
  Real const nu;
  Real const atol;
  Hooke(Real const mu, Real const nu,
        Real const atol = 10 * std::numeric_limits<Real>::epsilon())
      : mu(mu), nu(nu), atol(atol) {}

  std::string repr() const {
    std::ostringstream stream;
    stream << "Hooke<" << typeid(T).name() << "," << DIM << ">{mu=" << mu << ","
           << "nu=" << nu << "}";
    return stream.str();
  }

  void apply(Real kx, Real ky, T const* tau, T* out) const {
    static_assert(DIM == 2, "should only be called when DIM == 2");
    auto k2 = kx * kx + ky * ky;
    if (k2 <= atol) {
      for (int i = 0; i < osize; i++) out[i] = 0;
    } else {
      auto const sqrt2 = std::numbers::sqrt2_v<Real>;
      auto tau_k_x = tau[0] * kx + tau[2] * ky / sqrt2;
      auto tau_k_y = tau[1] * ky + tau[2] * kx / sqrt2;
      auto n_tau_n = (kx * tau_k_x + ky * tau_k_y) / k2;
      auto const1 = n_tau_n / (1. - nu);
      auto const2 = 1. / (2. * mu * k2);
      out[0] = const2 * (kx * (2. * tau_k_x - const1 * kx));
      out[1] = const2 * (ky * (2. * tau_k_y - const1 * ky));
      auto const3 = sqrt2 * const2;
      out[2] = const3 * (kx * tau_k_y + ky * tau_k_x - const1 * kx * ky);
    }
  }

  void apply(Real kx, Real ky, Real kz, T const* tau, T* out) const {
    static_assert(DIM == 3, "should only be called when DIM == 3");
    auto k2 = kx * kx + ky * ky + kz * kz;
    if (k2 <= atol) {
      for (int i = 0; i < osize; i++) out[i] = 0;
    } else {
      auto const sqrt2 = std::numbers::sqrt2_v<Real>;
      auto tau_k_x = tau[0] * kx + (tau[5] * ky + tau[4] * kz) / sqrt2;
      auto tau_k_y = tau[1] * ky + (tau[5] * kx + tau[3] * kz) / sqrt2;
      auto tau_k_z = tau[2] * kz + (tau[4] * kx + tau[3] * ky) / sqrt2;
      auto n_tau_n = (kx * tau_k_x + ky * tau_k_y + kz * tau_k_z) / k2;
      auto const1 = n_tau_n / (1. - nu);
      auto const2 = 1. / (2. * mu * k2);
      out[0] = const2 * (kx * (2. * tau_k_x - const1 * kx));
      out[1] = const2 * (ky * (2. * tau_k_y - const1 * ky));
      out[2] = const2 * (kz * (2. * tau_k_z - const1 * kz));
      auto const const3 = sqrt2 * const2;
      out[3] = const3 * (ky * tau_k_z + kz * tau_k_y - const1 * ky * kz);
      out[4] = const3 * (kz * tau_k_x + kx * tau_k_z - const1 * kz * kx);
      out[5] = const3 * (kx * tau_k_y + ky * tau_k_x - const1 * kx * ky);
    }
  }

  void apply(Real const* k, T const* tau, T* out) const {
    if constexpr (DIM == 2) apply(k[0], k[1], tau, out);
    if constexpr (DIM == 3) apply(k[0], k[1], k[2], tau, out);
  }

  void stress_strain(T const* sig, T* eps) const {
    if constexpr (DIM == 2) {
      auto lambda = 0.;
      auto tr_eps = eps[0] + eps[1];
    }
  }
};

template <typename T, int DIM>
std::ostream& operator<<(std::ostream& os, const Hooke<T, DIM>& hooke) {
  return os << hooke.repr();
}
}  // namespace scapin
