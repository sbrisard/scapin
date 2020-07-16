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

  const double mu;
  const double nu;
  Hooke(const double mu, const double nu) : mu(mu), nu(nu) {}

  void apply(double kx, double ky, const T* tau, T* out);
  void apply(double kx, double ky, double kz, const T* tau, T* out);
  void apply(const double* k, const T* tau, T* out);
};

template <typename T, size_t DIM>
void Hooke<T, DIM>::apply(double kx, double ky, const T* tau, T* out) {
  static_assert(DIM == 2, "should only be called when DIM == 2");
  const double k2 = kx * kx + ky * ky;
  const double tau_k_x = tau[0] * kx + M_SQRT1_2 * tau[2] * ky;
  const double tau_k_y = tau[1] * ky + M_SQRT1_2 * tau[2] * kx;
  const double n_tau_n = (kx * tau_k_x + ky * tau_k_y) / k2;
  const double const1 = n_tau_n / (1. - nu);
  const double const2 = 1. / (2. * mu * k2);
  out[0] = const2 * (kx * (2. * tau_k_x - const1 * kx));
  out[1] = const2 * (ky * (2. * tau_k_y - const1 * ky));
  const double const3 = M_SQRT2 * const2;
  out[2] = const3 * (kx * tau_k_y + ky * tau_k_x - const1 * kx * ky);
}

template <typename T, size_t DIM>
void Hooke<T, DIM>::apply(double kx, double ky, double kz, const T* tau,
                          T* out) {
  static_assert(DIM == 3, "should only be called when DIM == 3");
  const double k2 = kx * kx + ky * ky + kz * kz;
  const double tau_k_x = tau[0] * kx + M_SQRT1_2 * (tau[5] * ky + tau[4] * kz);
  const double tau_k_y = tau[1] * ky + M_SQRT1_2 * (tau[5] * kx + tau[3] * kz);
  const double tau_k_z = tau[2] * kz + M_SQRT1_2 * (tau[4] * kx + tau[3] * ky);
  const double n_tau_n = (kx * tau_k_x + ky * tau_k_y + kz * tau_k_z) / k2;
  const double const1 = n_tau_n / (1. - nu);
  const double const2 = 1. / (2. * mu * k2);
  out[0] = const2 * (kx * (2. * tau_k_x - const1 * kx));
  out[1] = const2 * (ky * (2. * tau_k_y - const1 * ky));
  out[2] = const2 * (kz * (2. * tau_k_z - const1 * kz));
  double const const3 = M_SQRT2 * const2;
  out[3] = const3 * (ky * tau_k_z + kz * tau_k_y - const1 * ky * kz);
  out[4] = const3 * (kz * tau_k_x + kx * tau_k_z - const1 * kz * kx);
  out[5] = const3 * (kx * tau_k_y + ky * tau_k_x - const1 * kx * ky);
}

template <typename T, size_t DIM>
void Hooke<T, DIM>::apply(const double* k, const T* tau, T* out) {
  if constexpr (DIM == 2) apply(k[0], k[1], tau, out);
  if constexpr (DIM == 3) apply(k[0], k[1], k[2], tau, out);
}

template <typename T, size_t DIM>
std::ostream& operator<<(std::ostream& os, const Hooke<T, DIM>& hooke) {
  return os << "Hooke<" << typeid(T).name() << "," << DIM << ">(mu=" << hooke.mu
            << ","
            << "nu=" << hooke.nu << ")";
}
