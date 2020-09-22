#pragma once

#include <array>
#include <ostream>

#include "core.hpp"
#include "hooke.hpp"

#include "fft_helper.hpp"

template <typename T>
T *create_copy(size_t n, const T *src) {
  auto dest = new T[n];
  std::copy(src, src + n, dest);
  return dest;
}

template <typename T>
class MoulinecSuquet94 {
 public:
  const T gamma;
  const std::array<size_t, T::dim> N;
  const double *L;

  // TODO : L should default to (1., 1., ...)
  MoulinecSuquet94(T gamma, const std::array<size_t, T::dim> N, const double *L)
      : gamma(gamma), N(N), L(create_copy(T::dim, L)), k(create_k(N, L)) {}

  void apply(const size_t *n, const typename T::Scalar *tau,
             typename T::Scalar *out) const {
    if constexpr (T::dim == 2) gamma.apply(k[0][n[0]], k[1][n[1]], tau, out);
    if constexpr (T::dim == 3)
      gamma.apply(k[0][n[0]], k[1][n[1]], k[2][n[2]], tau, out);
  }

  ~MoulinecSuquet94() { delete[] L; }

 private:
  const std::array<double *, T::dim> k;

  static std::array<double *, T::dim> create_k(
      const std::array<size_t, T::dim> N, const double *L) {
    std::array<double *, T::dim> k;
    for (size_t i = 0; i < T::dim; i++) {
      k[i] = new double[N[i]];
      fft_helper::fftwavnum(N[i], L[i], k[i]);
    }
    return k;
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const MoulinecSuquet94<T> &gamma_h) {
  os << "MoulinecSuquet1994(gamma=" << gamma_h.gamma << ", N=[";
  for (auto N_ : gamma_h.N) os << N_ << ",";
  os << "],L=[";
  for (size_t i = 0; i < T::dim; i++) os << gamma_h.L[i] << ",";
  return os << "])";
}