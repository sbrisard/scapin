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

#include "blitz/array.h"

using complex128 = std::complex<double>;

double squared_modulus(complex128 x) { return std::norm(x); }
BZ_DECLARE_FUNCTION(squared_modulus)

template <typename T, int RANK>
void nninterp(const blitz::Array<T, RANK> &in, blitz::Array<T, RANK> &out) {
  /*
   * i0, i1: indices in output array
   * j0, j1: indices in input array
   */
  auto ishape = in.shape();
  auto oshape = out.shape();
  //  blitz::TinyVector<int, RANK> r;
  //  for (int i = 0; i < RANK; i++) r[i] = oshape[i] / ishape[i];
  auto r = out.shape() / in.shape();
  // TODO: this loop is not dimension independent
  for (int i0 = 0; i0 < oshape[0]; ++i0) {
    int j0 = i0 / r[0];
    for (int i1 = 0; i1 < oshape[1]; ++i1) {
      int j1 = i1 / r[1];
      out(i0, i1) = in(j0, j1);
    }
  }
}

template <size_t RANK>
blitz::Array<complex128, RANK> create_array(
    blitz::TinyVector<int, RANK> shape) {
  auto size = blitz::product(shape);
  auto c_data = fftw_malloc(size * sizeof(fftw_complex));
  auto cpp_data = reinterpret_cast<complex128 *>(c_data);
  return blitz::Array<complex128, RANK>(cpp_data, shape,
                                        blitz::neverDeleteData);
  // TODO: Huge memory leak!
}

template <typename GREENC>
void run(GREENC gamma, blitz::TinyVector<int, GREENC::dim> Nc,
         blitz::TinyVector<int, GREENC::dim> Nf,
         blitz::Array<complex128, GREENC::dim + 1> &eta_f) {
  blitz::TinyVector<double, GREENC::dim> L;
  L = 1.;

  /*
   * tau_in: value of the prescribed polarization inside the patch
   * tau_out: value of the prescribed polarization outside the patch
   */
  blitz::TinyVector<double, GREENC::dim> patch_ratio;
  patch_ratio = 0.125;
  blitz::TinyVector<int, GREENC::dim> patch_size;
  for (int i = 0; i < GREENC::dim; i++) {
    patch_size[i] = int(std::round(patch_ratio[i] * Nc[i]));
  }

  blitz::TinyVector<double, GREENC::isize> tau_in, tau_out;
  tau_in = 0.;
  tau_in[gamma.isize - 1] = 1.;
  tau_out = 0.;

  blitz::TinyVector<int, GREENC::dim + 1> tau_shape;
  for (int i = 0; i < gamma.dim; i++) tau_shape[i] = Nc[i];
  tau_shape[GREENC::dim] = GREENC::isize;
  auto tau = create_array(tau_shape);

  // TODO: This is not dimension independent
  for (int i0 = 0, i = 0; i0 < Nc[0]; ++i0) {
    for (int i1 = 0; i1 < Nc[1]; ++i1, i += gamma.isize) {
      bool in = (i0 < patch_size[0]) && (i1 < patch_size[1]);
      for (int k = 0; k < gamma.isize; ++k) {
        tau(i0, i1, k) = in ? tau_in[k] : tau_out[k];
      }
    }
  }

  auto tau_data = reinterpret_cast<fftw_complex *>(tau.data());
  auto p = fftw_plan_many_dft(GREENC::dim, Nc.data(), GREENC::isize, tau_data,
                              nullptr, GREENC::isize, 1, tau_data, nullptr,
                              GREENC::isize, 1, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Set eta to the DFT of gamma_h(tau)
  // TODO: remove with issue #4
  std::array<size_t, GREENC::dim> Nc_{Nc[0], Nc[1]};
  MoulinecSuquet94<decltype(gamma)> gamma_h{gamma, Nc_.data(), L.data()};
  auto eta_shape{tau_shape};
  eta_shape[GREENC::dim] = GREENC::osize;
  auto eta = create_array(eta_shape);
  const auto all = blitz::Range::all();
  // TODO: this is not dimension independent
  for (int i0 = 0; i0 < Nc[0]; ++i0) {
    for (int i1 = 0; i1 < Nc[1]; ++i1) {
      size_t n[GREENC::dim] = {i0, i1};
      gamma_h.apply(n, tau(i0, i1, all).data(), eta(i0, i1, all).data());
    }
  }

  eta(0, 0, all) = 0.;

  // Set eta to gamma_h(tau)
  auto eta_data = reinterpret_cast<fftw_complex *>(eta.data());
  p = fftw_plan_many_dft(GREENC::dim, Nc.data(), GREENC::osize, eta_data,
                         nullptr, GREENC::osize, 1, eta_data, nullptr,
                         GREENC::osize, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Normalize inverse DFT
  const double normalization = 1. / blitz::product(Nc);
  eta *= normalization;

  // TODO This is not dimension independent
  for (size_t k = 0; k < gamma.osize; k++) {
    nninterp(eta(all, all, k), eta_f(all, all, k));
  }
}

int main() {
  const size_t dim = 2;
  Hooke<complex128, dim> gamma{1.0, 0.3};
  const size_t num_refinements = 6;
  std::array<int, num_refinements> N;
  N[0] = 8;
  for (size_t k = 1; k < num_refinements; ++k) {
    N[k] = 2 * N[k - 1];
  }
  std::cout << repr(N) << std::endl;

  blitz::TinyVector<int, dim> Nc, Nf;
  Nf = N[num_refinements - 1];
  auto num_cells = blitz::product(Nf);

  blitz::TinyVector<int, dim + 1> eta_shape;
  for (int i = 0; i < dim; i++) eta_shape[i] = Nf[i];
  eta_shape[dim] = gamma.osize;
  blitz::Array<complex128, dim + 1> results[num_refinements];
  for (size_t r = 0; r < num_refinements; r++) {
    Nc = N[r];
    blitz::Array<complex128 , dim+1> eta{eta_shape};
    run(gamma, Nc, Nf, eta);
    results[r].reference(eta);
  }

  auto eta_ref = results[num_refinements - 1];
  for (size_t r = 0; r < num_refinements; ++r) {
    auto eta = results[r];
    auto eps = eta - eta_ref;
    complex128 norm2 = blitz::sum(eps * blitz::conj(eps));
    std::cout << N[r] << ", " << norm2 << std::endl;
  }
}
