#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <ostream>

#include "core.hpp"

template <typename T, size_t DIM>
class Hooke {
 public:
  static constexpr size_t dim = DIM;
  static constexpr size_t isize = (DIM * (DIM + 1)) / 2;
  static constexpr size_t osize = (DIM * (DIM + 1)) / 2;
  static constexpr T zero{0};  // This is used to retrieve the data type

  const double mu;
  const double nu;
  Hooke(const double mu, const double nu) : mu(mu), nu(nu) {}

  void apply(double kx, double ky, const T* tau, T* out) const;
  void apply(double kx, double ky, double kz, const T* tau, T* out) const;
  void apply(const double* k, const T* tau, T* out) const;
};

template <typename T, size_t DIM>
void Hooke<T, DIM>::apply(double kx, double ky, const T* tau, T* out) const {
  static_assert(DIM == 2, "should only be called when DIM == 2");
  auto k2 = kx * kx + ky * ky;
  auto tau_k_x = tau[0] * kx + M_SQRT1_2 * tau[2] * ky;
  auto tau_k_y = tau[1] * ky + M_SQRT1_2 * tau[2] * kx;
  auto n_tau_n = (kx * tau_k_x + ky * tau_k_y) / k2;
  auto const1 = n_tau_n / (1. - nu);
  auto const2 = 1. / (2. * mu * k2);
  out[0] = const2 * (kx * (2. * tau_k_x - const1 * kx));
  out[1] = const2 * (ky * (2. * tau_k_y - const1 * ky));
  auto const3 = M_SQRT2 * const2;
  out[2] = const3 * (kx * tau_k_y + ky * tau_k_x - const1 * kx * ky);
}

template <typename T, size_t DIM>
void Hooke<T, DIM>::apply(double kx, double ky, double kz, const T* tau,
                          T* out) const {
  static_assert(DIM == 3, "should only be called when DIM == 3");
  auto k2 = kx * kx + ky * ky + kz * kz;
  auto tau_k_x = tau[0] * kx + M_SQRT1_2 * (tau[5] * ky + tau[4] * kz);
  auto tau_k_y = tau[1] * ky + M_SQRT1_2 * (tau[5] * kx + tau[3] * kz);
  auto tau_k_z = tau[2] * kz + M_SQRT1_2 * (tau[4] * kx + tau[3] * ky);
  auto n_tau_n = (kx * tau_k_x + ky * tau_k_y + kz * tau_k_z) / k2;
  auto const1 = n_tau_n / (1. - nu);
  auto const2 = 1. / (2. * mu * k2);
  out[0] = const2 * (kx * (2. * tau_k_x - const1 * kx));
  out[1] = const2 * (ky * (2. * tau_k_y - const1 * ky));
  out[2] = const2 * (kz * (2. * tau_k_z - const1 * kz));
  auto const const3 = M_SQRT2 * const2;
  out[3] = const3 * (ky * tau_k_z + kz * tau_k_y - const1 * ky * kz);
  out[4] = const3 * (kz * tau_k_x + kx * tau_k_z - const1 * kz * kx);
  out[5] = const3 * (kx * tau_k_y + ky * tau_k_x - const1 * kx * ky);
}

template <typename T, size_t DIM>
void Hooke<T, DIM>::apply(const double* k, const T* tau, T* out) const {
  if constexpr (DIM == 2) apply(k[0], k[1], tau, out);
  if constexpr (DIM == 3) apply(k[0], k[1], k[2], tau, out);
}

template <typename T, size_t DIM>
std::ostream& operator<<(std::ostream& os, const Hooke<T, DIM>& hooke) {
  return os << "Hooke<" << typeid(T).name() << "," << DIM << ">(mu=" << hooke.mu
            << ","
            << "nu=" << hooke.nu << ")";
}
