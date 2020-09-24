#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <complex>
#include <iostream>

#include <fftw3.h>
#include <numeric>
#include <vector>

#include "scapin/ms94.hpp"
#include "scapin/scapin.hpp"

using complex128 = std::complex<double>;
using Scalar = std::complex<double>;

template <typename T, size_t N>
std::array<T, N> array_strides(const std::array<T, N> size) {
  std::array<T, N> strides;
  strides[N - 1] = 1;
  for (size_t k = 0; k < N - 1; ++k) {
    size_t kk = N - 2 - k;
    strides[kk] = strides[kk + 1] * size[kk + 1];
  }
  return strides;
}

template <typename T, size_t N>
T array_num_cells(const std::array<T, N> size) {
  constexpr T one{1};
  constexpr auto mul = std::multiplies<T>();
  return std::accumulate(std::begin(size), std::end(size), one, mul);
}

template <typename T>
std::string repr_array_2D(const std::array<size_t, 2> size,
                          const std::array<size_t, 2> stride, const T *array) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i0 = 0; i0 < size[0]; ++i0) {
    stream << "[";
    size_t i = stride[0] * i0;
    for (size_t i1 = 0; i1 < size[1]; ++i1, i += stride[1]) {
      stream << array[i] << ",";
    }
    stream << "]";
    if (i0 + 1 < size[0]) stream << std::endl;
  }
  stream << "]";
  return stream.str();
}

template <typename T>
void nninterp(const std::array<size_t, 2> isize,
              const std::array<size_t, 2> istride, const T *in,
              std::array<size_t, 2> osize, std::array<size_t, 2> ostride,
              T *out) {
  /*
   * i0, i1: indices in output array
   * i: offset in output array
   * j0, j1: indices in input array
   * j: offset in input array
   */
  std::array<size_t, 2> r;
  for (size_t k = 0; k < 2; ++k) {
    r[k] = osize[k] / isize[k];
  }
  for (size_t i0 = 0; i0 < osize[0]; ++i0) {
    size_t j0 = i0 / r[0];
    size_t i = ostride[0] * i0;
    for (size_t i1 = 0; i1 < osize[1]; ++i1, i += ostride[1]) {
      size_t j1 = i1 / r[1];
      size_t j = istride[0] * j0 + istride[1] * j1;
      out[i] = in[j];
    }
  }
}

class FFTWComplexBuffer {
 public:
  const size_t size;
  fftw_complex *const c_data;
  complex128 *const cpp_data;

  FFTWComplexBuffer(size_t n)
      : size(n),
        c_data((fftw_complex *)fftw_malloc(n * sizeof(fftw_complex))),
        cpp_data(reinterpret_cast<complex128 *const>(c_data)) {}

  ~FFTWComplexBuffer() { fftw_free(c_data); }
  FFTWComplexBuffer(const FFTWComplexBuffer &) = delete;
  FFTWComplexBuffer &operator=(const FFTWComplexBuffer &) = delete;
  FFTWComplexBuffer(const FFTWComplexBuffer &&) = delete;
  FFTWComplexBuffer &operator=(const FFTWComplexBuffer &&) = delete;
};

template <typename GREENC>
void run(GREENC gamma, std::array<size_t, GREENC::dim> Nc,
         std::array<size_t, GREENC::dim> Nf, Scalar *eta_f) {
  std::array<double, gamma.dim> L;
  L.fill(1.);

  /*
   * tau_in: value of the prescribed polarization inside the patch
   * tau_out: value of the prescribed polarization outside the patch
   */
  std::array<double, gamma.dim> patch_ratio;
  patch_ratio.fill(0.125);
  std::array<size_t, gamma.dim> patch_size;
  for (size_t k = 0; k < gamma.dim; ++k) {
    patch_size[k] = size_t(std::round(patch_ratio[k] * Nc[k]));
  }

  std::array<double, gamma.isize> tau_in, tau_out;
  tau_in.fill(0.);
  tau_in[gamma.isize - 1] = 1.;
  tau_out.fill(0.);

  FFTWComplexBuffer tau{array_num_cells(Nc) * gamma.isize};
  // TODO: This is not dimension independent
  for (size_t i0 = 0, i = 0; i0 < Nc[0]; ++i0) {
    for (size_t i1 = 0; i1 < Nc[1]; ++i1, i += gamma.isize) {
      bool in = (i0 < patch_size[0]) && (i1 < patch_size[1]);
      for (size_t k = 0; k < gamma.isize; ++k) {
        tau.cpp_data[i + k] = in ? tau_in[k] : tau_out[k];
      }
    }
  }

  // Set tau to the DFT of the polarization
  std::array<int, gamma.dim> n;

  // TODO: this ugly conversion from size_t to int is needed
  for (size_t k = 0; k < gamma.dim; ++k) {
    n[k] = int(Nc[k]);
  }
  auto p = fftw_plan_many_dft(gamma.dim, n.data(), gamma.isize, tau.c_data,
                              nullptr, gamma.isize, 1, tau.c_data, nullptr,
                              gamma.isize, 1, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Set eta to the DFT of gamma_h(tau)
  MoulinecSuquet94<decltype(gamma)> gamma_h{gamma, Nc, L};
  FFTWComplexBuffer eta{array_num_cells(Nc) * gamma.osize};
  // TODO: this is not dimension independent
  for (size_t i0 = 0, i = 0, j = 0; i0 < Nc[0]; ++i0) {
    for (size_t i1 = 0; i1 < Nc[1]; ++i1, i += gamma.isize, j += gamma.osize) {
      size_t n[gamma.dim] = {i0, i1};
      gamma_h.apply(n, tau.cpp_data + i, eta.cpp_data + j);
    }
  }

  // Handle the null-frequency case
  for (size_t k = 0; k < gamma.osize; k++) {
    eta.cpp_data[k] = 0.;
  }

  // Set eta to gamma_h(tau)
  p = fftw_plan_many_dft(gamma.dim, n.data(), gamma.osize, eta.c_data, nullptr,
                         gamma.osize, 1, eta.c_data, nullptr, gamma.osize, 1,
                         FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Normalize inverse DFT
  const double normalization = 1. / array_num_cells(Nc);
  for (size_t k = 0; k < eta.size; ++k) {
    eta.cpp_data[k] *= normalization;
  }

  // Refine eta
  std::array<size_t, gamma.dim> istrides, ostrides;
  istrides[gamma.dim - 1] = gamma.osize;
  ostrides[gamma.dim - 1] = gamma.osize;
  for (int k = gamma.dim - 2; k >= 0; k--) {
    istrides[k] = istrides[k + 1] * Nc[k + 1];
    ostrides[k] = ostrides[k + 1] * Nf[k + 1];
  }
  for (size_t k = 0; k < gamma.osize; k++) {
    nninterp(Nc, istrides, eta.cpp_data + k, Nf, ostrides, eta_f + k);
  }
}

int main() {
  const size_t dim = 2;
  Hooke<Scalar, dim> gamma{1.0, 0.3};
  const size_t num_refinements = 6;
  std::array<size_t, num_refinements> N;
  N[0] = 8;
  for (size_t k = 1; k < num_refinements; ++k) {
    N[k] = 2 * N[k - 1];
  }
  std::cout << repr(N) << std::endl;

  std::array<size_t, dim> Nc, Nf;
  Nf.fill(N[num_refinements - 1]);
  size_t num_cells = array_num_cells(Nf);

  Scalar *results[num_refinements];
  for (size_t r = 0; r < num_refinements; r++) {
    Nc.fill(N[r]);
    auto eta = new Scalar[num_cells * gamma.osize];
    run(gamma, Nc, Nf, eta);
    results[r] = eta;
  }

  auto eta_ref = results[num_refinements - 1];
  for (size_t r = 0; r < num_refinements; ++r) {
    auto eta = results[r];
    double norm = 0.;
    for (size_t i = 0; i < num_cells * gamma.osize; ++i) {
      norm += std::norm(eta[i] - eta_ref[i]);
    }
    std::cout << N[r] << ", " << norm << std::endl;
  }
}
