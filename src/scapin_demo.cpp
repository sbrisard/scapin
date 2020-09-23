#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <complex>
#include <iostream>

#include <fftw3.h>
#include <numeric>

#include "scapin/ms94.hpp"
#include "scapin/scapin.hpp"

using complex128 = std::complex<double>;

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

int main() {
  using Scalar = std::complex<double>;
  Hooke<Scalar, 2> gamma{1.0, 0.3};

  const std::array<double, gamma.dim> L = {1., 1.};

  /* Coarse grid */
  const std::array<size_t, gamma.dim + 1> size_c{16, 16, gamma.isize};
  const auto stride_c = array_strides(size_c);
  const size_t ncells_c = array_num_cells(size_c);
  std::cout << repr(stride_c) << std::endl;

  /* Size of coarse grid. */
  const decltype(size_c) size_f{256, 256, gamma.isize};
  const auto stride_f = array_strides(size_f);
  const size_t ncells_f = array_num_cells(size_f);
  std::cout << repr(stride_f) << std::endl;

  /*
   * tau_in: value of the prescribed polarization inside the patch
   * tau_out: value of the prescribed polarization outside the patch
   */
  const std::array<double, gamma.dim> patch_ratio{0.125, 0.125};
  std::array<size_t, gamma.dim> patch_size;
  for (size_t k = 0; k < gamma.dim; ++k) {
    patch_size[k] = size_t(std::round(patch_ratio[k] * size_c[k]));
  }

  std::array<double, gamma.isize> tau_in, tau_out;
  tau_in.fill(0.);
  tau_in[gamma.isize - 1] = 1.;
  tau_out.fill(0.);

  FFTWComplexBuffer tau{ncells_c};
  // TODO: This is not dimension independent
  for (size_t i0 = 0, i = 0; i0 < size_c[0]; ++i0) {
    for (size_t i1 = 0; i1 < size_c[1]; ++i1, i += gamma.isize) {
      bool in = (i0 < patch_size[0]) && (i1 < patch_size[1]);
      for (size_t k = 0; k < gamma.isize; ++k) {
        tau.cpp_data[i + k] = in ? tau_in[k] : tau_out[k];
      }
    }
  }

  // Set tau to the DFT of the polarization
  std::array<int, gamma.dim> n;
  for (size_t k = 0; k < gamma.dim; ++k) {
    n[k] = int(size_c[k]);
  }
  auto p = fftw_plan_many_dft(gamma.dim, n.data(), gamma.isize, tau.c_data,
                              nullptr, gamma.isize, 1, tau.c_data, nullptr,
                              gamma.isize, 1, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Set eta to the DFT of gamma_h(tau)
  // TODO: this is not dimension independent
  MoulinecSuquet94<decltype(gamma)> gamma_h{
      gamma, std::array<size_t, gamma.dim>{size_c[0], size_c[1]}, L};
  std::cout << gamma_h << std::endl;
  FFTWComplexBuffer eta{tau.size};
  for (size_t i0 = 0, i = 0; i0 < size_c[0]; ++i0) {
    for (size_t i1 = 0; i1 < size_c[1]; ++i1, i += gamma.isize) {
      gamma_h.apply(size_c.data(), tau.cpp_data + i, eta.cpp_data + i);
    }
  }

  // Set eta to gamma_h(tau)
  p = fftw_plan_many_dft(gamma.dim, n.data(), gamma.isize, eta.c_data, nullptr,
                         gamma.isize, 1, eta.c_data, nullptr, gamma.isize, 1,
                         FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Refine eta
  auto eta_f = new double[ncells_f * gamma.osize];
  //  nninterp(size_c, stride_c, eta, size_f, stride_f, eta_f);

  delete[eta_f];
}
