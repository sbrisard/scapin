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

//<template typename T, size_t DIM>
// void nninterp(const std::array<size_t, DIM) isize, const std::array<size_t,
// DIM> istride, const T* in, const std::array<size_t, DIM> factor, const
// std::array<size_t, DIM> ostride, T* out) {
//  for (size_t i0 = 0; i0 < isize[0]; i0++) {
//    offset = istride[0] * i0;
//    for (size_t i1 = 0; i1 < isize[1]; i1++, offset += istride[1]) {
//    }
//  }
//}

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

/**
 * Compute in Fourier space the nearest-neighbour interpolation kernel.
 *
 * Let `xc[0], ..., xc[nc-1]` be data points defined on the coarse grid (size:
 * `nc`). The nearest-neighbour interpolation `xf` is defined on the fine grid
 * (size: `nf = r * nc`) as follows: `xf[i] = xc[i / r]`.
 *
 * In Fourier space, this formula reads: `yf[i] = kernel[i] * yc[i % nc]`,
 * where `yc` and `yf` are the DFT of `xc` and `xf`, respectively.
 *
 * If `kernel == nullptr`, it is allocated. Otherwise, it must be a preallocated
 * `std::complex<double>[nf]` array.
 *
 * @param nc the size of the coarse grid
 * @param r the refinement factor
 * @param kernel the coefficients of the kernel
 * @returns kernel
 */
complex128 *fft_nninterp(size_t nc, size_t r, complex128 *kernel) {
  const size_t nf = r * nc;
  if (kernel == nullptr) {
    kernel = new complex128[nf];
  }
  kernel[0] = r;
  for (size_t i = 1; i < nf; i++) {
    const double xc = i * M_PI / (double)nc;
    const double xf = i * M_PI / (double)nf;
    const complex128 z{cos((r - 1) * xf), -sin((r - 1) * xf)};
    kernel[i] = sin(xc) / sin(xf) * z;
  }
  return kernel;
}

int main() {
  using Scalar = std::complex<double>;
  Hooke<Scalar, 2> gamma{1.0, 0.3};
  std::cout << gamma << std::endl;
  using MultiIndex = std::array<size_t, gamma.dim>;

  const double L[] = {1., 1.};
  const size_t grid_size[] = {16, 16};

  const size_t num_cells =
      std::accumulate(std::begin(grid_size), std::end(grid_size),
                      std::size_t{1}, std::multiplies<size_t>());

  const double patch_ratio[] = {0.125, 0.125};

  /*
   * tau_in: value of the prescribed polarization inside the patch
   * tau_out: value of the prescribed polarization outside the patch
   */
  std::array<double, gamma.isize> tau_in, tau_out;
  tau_in.fill(0.);
  tau_in[gamma.isize - 1] = 1.;
  tau_out.fill(0.);

  FFTWComplexBuffer tau{num_cells * gamma.isize};

  MultiIndex patch_size;
  for (size_t k = 0; k < gamma.dim; k++) {
    patch_size[k] = size_t(std::round(patch_ratio[k] * grid_size[k]));
  }
  // TODO: This is not dimension independent
  for (size_t i0 = 0, offset = 0; i0 < grid_size[0]; i0++) {
    for (size_t i1 = 0; i1 < grid_size[1]; i1++, offset += gamma.isize) {
      bool in = (i0 < patch_size[0]) && (i1 < patch_size[1]);
      for (size_t k = 0; k < gamma.isize; k++) {
        tau.cpp_data[offset + k] = in ? tau_in[k] : tau_out[k];
      }
    }
  }

  std::array<int, gamma.dim> n;
  for (size_t k = 0; k < gamma.dim; k++) {
    n[k] = int(grid_size[k]);
  }
  FFTWComplexBuffer tau_hat_exp{tau.size};
  auto p = fftw_plan_many_dft(
      gamma.dim, n.data(), gamma.isize, tau.c_data, nullptr, gamma.isize, 1,
      tau_hat_exp.c_data, nullptr, gamma.isize, 1, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // TODO This is a bit ugly
  auto tau_hat = new Scalar[num_cells * gamma.isize];
  for (size_t i0 = 0, offset = 0; i0 < grid_size[0]; i0++) {
    auto s0 = i0 == 0 ? patch_size[0]
                      : sin(M_PI * patch_size[0] * i0 / grid_size[0]) /
                            sin(M_PI * i0 / grid_size[0]);
    for (size_t i1 = 0; i1 < grid_size[1]; i1++, offset += gamma.isize) {
      auto s1 = i1 == 0 ? patch_size[1]
                        : sin(M_PI * patch_size[1] * i1 / grid_size[1]) /
                              sin(M_PI * i1 / grid_size[1]);
      auto phase = M_PI * ((patch_size[0] - 1.) / grid_size[0] * i0 +
                           (patch_size[1] - 1.) / grid_size[1] * i1);
      auto s = Scalar{cos(phase), -sin(phase)};
      for (size_t k = 0; k < gamma.isize; k++) {
        tau_hat[offset + k] = s0 * s1 * s * tau_in[k];
      }
    }
  }

  double rtol = 1e-15, atol = 1e-15;
  for (size_t offset = 0; offset < num_cells * gamma.isize; offset++) {
    auto exp = tau_hat_exp.cpp_data[offset];
    auto act = tau_hat[offset];
    if (abs(act - exp) > rtol * abs(exp) + atol) {
      std::cout << "exp[" << offset << "] = " << exp << "   ";
      std::cout << "act[" << offset << "] = " << act << std::endl;
    }
  }

  MoulinecSuquet94<decltype(gamma)> gamma_h{gamma, grid_size, L};
  std::cout << gamma_h << std::endl;
  //  auto eta_hat = new Scalar[num_cells * gamma.isize];
  //  FFTWComplexBuffer eta{tau.size};
  //  for (size_t i0 = 0, offset = 0; i0 < grid_size[0]; i0++) {
  //    for (size_t i1 = 0; i1 < grid_size[1]; i1++, offset += gamma.isize) {
  //      gamma_h.apply(grid_size, tau_hat + offset, eta_hat + offset);
  //      for (size_t k = 0; k < gamma.isize; k++) {
  //        eta.cpp_data[offset + k] = eta_hat[offset + k];
  //      }
  //    }
  //  }
  //
  //  auto p1 = fftw_plan_many_dft(gamma.dim, n.data(), gamma.isize, eta.c_data,
  //                               nullptr, gamma.isize, 1, eta.c_data, nullptr,
  //                               gamma.isize, 1, FFTW_BACKWARD,
  //                               FFTW_ESTIMATE);
  //  fftw_execute(p1);
  //  fftw_destroy_plan(p1);
  //
  //  delete[] tau_hat;

  //  std::array<double, dim> L = {1.0, 1.0};
  //  MultiIndex Nc = {16, 16};
  //
  //  size_t num_refinements = 6;
  //  size_t r_max = 1 << num_refinements;
  //  MultiIndex Nf = {r_max * Nc[0], r_max * Nc[1]};
  //
  //  FFTWComplexBuffer tau_c{Nc[0] * Nc[1] * gamma.isize};
  //  tau_c.cpp_data[0] = 1.;
  //  for (size_t i = 0; i < tau_c.size; i++) tau_c.cpp_data[i] = 0.;
  //
  //  for (size_t r = 1; r <= r_max; r *= 2) {
  //    MultiIndex N = {r * Nc[0], r * Nc[1]};
  //
  //    std::cout << gamma_h << std::endl;
  //
  //    int n[] = {static_cast<int>(N[0]), static_cast<int>(N[1])};
  //    FFTWComplexBuffer tau{N[0] * N[1] * gamma.isize};
  //    FFTWComplexBuffer eps{N[0] * N[1] * gamma.osize};
  //
  //    for (MultiIndex i{0}, ic{0}; i[0] < N[0]; i[0]++) {
  //      ic[0] = i[0] * Nc[0] / N[0];
  //      for (i[1] = 0; i[1] < N[1]; i[1]++) {
  //        ic[1] = i[1] * Nc[1] / N[1];
  //        for (size_t j = 0; j < gamma.isize; j++) {
  //          auto adr = (i[0] * N[1] + i[1]) * gamma.isize + j;
  //          auto adr_c = (ic[0] * Nc[1] + ic[1]) * gamma.isize + j;
  //          tau.cpp_data[adr] = tau_c.cpp_data[adr_c];
  //        }
  //      }
  //    }
  //
  //
  //    auto tau_i = tau.cpp_data;
  //    auto eps_i = eps.cpp_data;
  //    for (MultiIndex i{0}; i[0] < N[0]; i[0]++) {
  //      for (i[1] = 0; i[1] < N[1]; i[1]++) {
  //        gamma_h.apply(i.data(), tau_i, eps_i);
  //        tau_i += gamma.isize;
  //        eps_i += gamma.osize;
  //      }
  //    }
  //
  //    auto p2 = fftw_plan_many_dft(dim, n, gamma.osize, eps.c_data, nullptr,
  //                                 gamma.osize, 1, eps.c_data, nullptr,
  //                                 gamma.osize, 1, FFTW_BACKWARD,
  //                                 FFTW_ESTIMATE);
  //    fftw_execute(p2);
  //    fftw_destroy_plan(p2);
  //  }
}
