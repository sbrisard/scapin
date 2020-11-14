#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <complex>
#include <iostream>

#include <fftw3.h>
#include <numeric>
#include <vector>

#include "bri17/bri17.hpp"
#include "scapin/ms94.hpp"
#include "scapin/scapin.hpp"

#include "blitz/array.h"

using complex128 = std::complex<double>;

double squared_modulus(complex128 x) { return std::norm(x); }
BZ_DECLARE_FUNCTION(squared_modulus)

template <typename T, int RANK>
void nninterp(const blitz::Array<T, RANK> &in, blitz::Array<T, RANK> &out) {
  auto r = out.shape() / in.shape();
  for (auto out_i = out.begin(); out_i != out.end(); out_i++) {
    blitz::TinyVector<int, RANK> j = out_i.position() / r;
    *out_i = in(j);
  }
}

template <typename GREENC>
class ConvergenceTest {
 public:
  static constexpr int num_refinements = 6;
  const GREENC &gamma;

  blitz::TinyVector<int, GREENC::dim> Nf;
  blitz::TinyVector<double, GREENC::dim> L;
  blitz::TinyVector<double, GREENC::dim> patch_ratio;
  blitz::TinyVector<double, GREENC::isize> tau_in, tau_out;

  ConvergenceTest(GREENC &gamma) : gamma(gamma) {
    Nf = 256;
    L = 1.0;
    patch_ratio = 0.125;
    tau_in = 0.;
    tau_in[GREENC::isize - 1] = 1.;
    tau_out = 0.;
  }

  blitz::Array<complex128, GREENC::dim + 1> create_tau_hat(int refinement) {
    blitz::TinyVector<int, GREENC::dim> patch_size;
    blitz::TinyVector<int, GREENC::dim + 1> tau_shape;
    for (int i = 0; i < GREENC::dim; i++) {
      int Nc = Nf[i] >> (num_refinements - 1 - refinement);
      patch_size[i] = int(std::round(patch_ratio[i] * Nc));
      tau_shape[i] = Nc;
    }
    tau_shape[GREENC::dim] = GREENC::isize;

    auto tau_data = (fftw_complex *)fftw_malloc(blitz::product(tau_shape) *
                                                sizeof(fftw_complex));
    auto tau = blitz::Array<complex128, GREENC::dim + 1>(
        reinterpret_cast<complex128 *>(tau_data), tau_shape,
        blitz::neverDeleteData);

    // TODO: This is not dimension independent
    for (int i0 = 0; i0 < tau_shape[0]; ++i0) {
      for (int i1 = 0; i1 < tau_shape[1]; ++i1) {
        bool in = (i0 < patch_size[0]) && (i1 < patch_size[1]);
        for (int k = 0; k < GREENC::isize; ++k) {
          tau(i0, i1, k) = in ? tau_in[k] : tau_out[k];
        }
      }
    }

    auto p = fftw_plan_many_dft(GREENC::dim, tau_shape.data(), GREENC::isize,
                                tau_data, nullptr, GREENC::isize, 1, tau_data,
                                nullptr, GREENC::isize, 1, FFTW_FORWARD,
                                FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // TODO: memory leak, tau is never destroyed. Use unique pointers?
    return tau;
  }

  void run(int refinement, blitz::Array<complex128, GREENC::dim + 1> &eta_f) {
    auto tau = create_tau_hat(refinement);
    auto tau_shape = tau.shape();

    // Set eta to the DFT of gamma_h(tau)
    MoulinecSuquet94<GREENC> gamma_h{gamma, tau_shape.data(), L.data()};
    auto eta_shape{tau_shape};
    eta_shape[GREENC::dim] = GREENC::osize;
    auto eta_data = (fftw_complex *)fftw_malloc(blitz::product(eta_shape) *
                                                sizeof(fftw_complex));
    auto eta = blitz::Array<complex128, GREENC::dim + 1>(
        reinterpret_cast<complex128 *>(eta_data), eta_shape,
        blitz::neverDeleteData);

    const auto all = blitz::Range::all();
    // TODO: this is not dimension independent
    for (int i0 = 0; i0 < tau_shape[0]; ++i0) {
      for (int i1 = 0; i1 < tau_shape[1]; ++i1) {
        int n[GREENC::dim] = {i0, i1};
        gamma_h.apply(n, tau(i0, i1, all).data(), eta(i0, i1, all).data());
      }
    }

    eta(0, 0, all) = 0.;

    // Set eta to gamma_h(tau)
    auto p = fftw_plan_many_dft(GREENC::dim, tau_shape.data(), GREENC::osize,
                                eta_data, nullptr, GREENC::osize, 1, eta_data,
                                nullptr, GREENC::osize, 1, FFTW_BACKWARD,
                                FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // Normalize inverse DFT
    int num_cells = 1;
    for (int i = 0; i < GREENC::dim; i++) {
      num_cells *= tau_shape[i];
    }
    eta /= (double)num_cells;

    // TODO This is not dimension independent
    for (int k = 0; k < gamma.osize; k++) {
      nninterp(eta(all, all, k), eta_f(all, all, k));
    }

    fftw_free(eta_data);
  }
};

template <int DIM>
void compute_reference(double mu, double nu, blitz::TinyVector<int, DIM> Nc,
                       blitz::TinyVector<int, DIM> Nf,
                       blitz::Array<complex128, DIM + 1> &eta_f) {}

int main() {
  const int dim = 2;
  Hooke<complex128, dim> gamma{1.0, 0.3};
  ConvergenceTest<decltype(gamma)> test{gamma};

  blitz::TinyVector<int, dim> Nc;
  blitz::TinyVector<int, dim + 1> eta_shape;
  for (int i = 0; i < dim; i++) eta_shape[i] = test.Nf[i];
  eta_shape[dim] = gamma.osize;
  blitz::Array<complex128, dim + 1> results[test.num_refinements];
  for (int r = 0; r < test.num_refinements; r++) {
    blitz::Array<complex128, dim + 1> eta{eta_shape};
    test.run(r, eta);
    results[r].reference(eta);
  }

  auto eta_ref = results[test.num_refinements - 1];
  for (int r = 0; r < test.num_refinements; ++r) {
    auto eta = results[r];
    auto eps = eta - eta_ref;
    complex128 norm2 = blitz::sum(eps * blitz::conj(eps));
    std::cout << r << ", " << norm2 << std::endl;
  }
}
