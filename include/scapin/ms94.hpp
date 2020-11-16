#pragma once

#include <array>
#include <ostream>

#include "core.hpp"
#include "hooke.hpp"

#include "fft_helper.hpp"

namespace scapin {
template <typename T, size_t N>
std::string repr(std::array<T, N> a) {
  std::ostringstream stream;
  stream << "[";
  for (auto a_ : a) stream << a_ << ",";
  stream << "]";
  return stream.str();
}

template <typename T, size_t N>
std::array<T, N> to_std_array(const T *data) {
  std::array<T, N> array{};
  for (size_t i = 0; i < N; i++) {
    array[i] = data[i];
  }
  return array;
}

template <typename T>
class MoulinecSuquet94 {
 public:
  const T &gamma;
  const std::array<int, T::dim> N;
  const std::array<double, T::dim> L;

  // TODO : L should default to (1., 1., ...)
  MoulinecSuquet94(const T &gamma, const int *N, const double *L)
      : gamma(gamma),
        N(to_std_array<int, T::dim>(N)),
        L(to_std_array<double, T::dim>(L)),
        k(create_k(this->N, this->L)) {}

  std::string repr() const {
    std::ostringstream stream;
    stream << "MoulinecSuquet1994(gamma=" << gamma << ", N=" << ::repr(N)
           << ",L=" << ::repr(L);
    return stream.str();
  }

  void apply(const int *n, const typename T::Scalar *tau,
             typename T::Scalar *out) const {
    if constexpr (T::dim == 2) gamma.apply(k[0][n[0]], k[1][n[1]], tau, out);
    if constexpr (T::dim == 3)
      gamma.apply(k[0][n[0]], k[1][n[1]], k[2][n[2]], tau, out);
  }

 private:
  const std::array<double *, T::dim> k;

  static std::array<double *, T::dim> create_k(
      const std::array<int, T::dim> N, const std::array<double, T::dim> L) {
    std::array<double *, T::dim> k;
    for (int i = 0; i < T::dim; i++) {
      k[i] = new double[N[i]];
      fft_helper::fftwavnum(N[i], L[i], k[i]);
    }
    return k;
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const MoulinecSuquet94<T> &gamma_h) {
  return os << gamma_h.repr();
}
}  // namespace scapin