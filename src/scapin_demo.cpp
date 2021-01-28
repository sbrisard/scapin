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

#include <xtensor/xarray.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using complex128 = std::complex<double>;

template <typename GREENC>
class ConvergenceTest {
 public:
  using Field = xt::xtensor<complex128, GREENC::dim + 1>;
  static constexpr int num_refinements = 6;
  const GREENC &gamma;

  std::array<int, GREENC::dim> Nf;
  std::array<double, GREENC::dim> L, patch_ratio;
  std::array<complex128, GREENC::isize> tau_in, tau_out;

  explicit ConvergenceTest(GREENC &gamma) : gamma(gamma) {
    static_assert((GREENC::dim == 2) || (GREENC::dim == 3),
                  "unexpected number of spatial dimensions");
    fill(Nf.begin(), Nf.end(), 512);
    fill(L.begin(), L.end(), 1.0);
    fill(patch_ratio.begin(), patch_ratio.end(), 0.125);
    fill(tau_in.begin(), tau_in.end(), 0.0);
    tau_in[GREENC::isize - 1] = 1.;
    fill(tau_out.begin(), tau_out.end(), 0.0);
  }

  Field create_tau_hat(int refinement) {
    std::array<size_t, GREENC::dim> patch_size;
    std::array<size_t, GREENC::dim + 1> tau_shape;
    std::transform(Nf.cbegin(), Nf.cend(), tau_shape.begin(),
                   [this, refinement](int n) {
                     return n >> (this->num_refinements - 1 - refinement);
                   });
    tau_shape[GREENC::dim] = GREENC::isize;
    std::transform(patch_ratio.cbegin(), patch_ratio.cend(), tau_shape.cbegin(),
                   patch_size.begin(), [](double r, size_t n) {
                     return size_t(std::round(r * n));
                   });

    Field tau{tau_shape};
    for (size_t k = 0; k < GREENC::isize; ++k) {
      xt::strided_view(tau, {xt::ellipsis(), k}) = tau_out[k];
      xt::xstrided_slice_vector sv;
      for (size_t i = 0; i < GREENC::dim; ++i) {
        sv.push_back(xt::range(0, patch_size[i]));
      }
      sv.push_back(k);
      xt::strided_view(tau, sv) = tau_in[k];
    }

    // TODO: this should be simplified
    std::array<int, GREENC::dim + 1> tau_shape_int{};
    std::transform(tau_shape.cbegin(), tau_shape.cend(), tau_shape_int.begin(),
                   [](size_t n) -> int { return n; });
    auto tau_data = reinterpret_cast<fftw_complex *>(tau.data());
    auto p = fftw_plan_many_dft(GREENC::dim, tau_shape_int.data(),
                                GREENC::isize, tau_data, nullptr, GREENC::isize,
                                1, tau_data, nullptr, GREENC::isize, 1,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    // End of simplification

    return tau;
  }

  Field run(int refinement) {
    auto tau = create_tau_hat(refinement);
    // This conversion is required by FFTW
    std::array<int, GREENC::dim + 1> tau_shape;
    std::transform(tau.shape().cbegin(), tau.shape().cend(), tau_shape.begin(),
                   [](size_t n) -> int { return n; });

    // Set eta to the DFT of gamma_h(tau)
    scapin::MoulinecSuquet94<GREENC> gamma_h{gamma, tau_shape.data(), L.data()};
    auto eta_shape{tau_shape};
    eta_shape[GREENC::dim] = GREENC::osize;
    auto eta = Field::from_shape(eta_shape);
    auto eta_data = reinterpret_cast<fftw_complex *>(eta.data());

    // TODO -- Remove ugly pointers
    auto tau_ = tau.data();
    auto eta_ = eta.data();
    if constexpr (GREENC::dim == 2) {
      for (int i0 = 0; i0 < tau_shape[0]; ++i0) {
        for (int i1 = 0; i1 < tau_shape[1]; ++i1) {
          std::array<int, GREENC::dim> n{i0, i1};
          gamma_h.apply(n.data(), tau_, eta_);
          // TODO Ugly pointers
          tau_ += GREENC::isize;
          eta_ += GREENC::osize;
        }
      }
    } else if constexpr (GREENC::dim == 3) {
      for (int i0 = 0; i0 < tau_shape[0]; ++i0) {
        for (int i1 = 0; i1 < tau_shape[1]; ++i1) {
          for (int i2 = 0; i2 < tau_shape[2]; ++i2) {
            std::array<int, GREENC::dim> n{i0, i1, i2};
            gamma_h.apply(n.data(), tau_, eta_);
            tau_ += GREENC::isize;
            eta_ += GREENC::osize;
          }
        }
      }
    }

    // TODO -- This should be simplified (xtensor-fftw)
    auto p = fftw_plan_many_dft(GREENC::dim, eta_shape.data(), GREENC::osize,
                                eta_data, nullptr, GREENC::osize, 1, eta_data,
                                nullptr, GREENC::osize, 1, FFTW_BACKWARD,
                                FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // Normalize inverse DFT
    int num_cells = std::reduce(tau_shape.cbegin(), tau_shape.cend() - 1, 1,
                                std::multiplies());
    xt::strided_view(eta, {xt::ellipsis()}) /= (double)num_cells;

    std::array<size_t, GREENC::dim + 1> eta_f_shape;
    std::copy(Nf.cbegin(), Nf.cend(), eta_f_shape.begin());
    eta_f_shape[GREENC::dim] = gamma.osize;
    auto eta_f = Field::from_shape(eta_f_shape);
    // Use int since eta_shape is an int (this is a bit ugly).
    std::array<int, GREENC::dim> ratio{};
    std::transform(eta_f_shape.cbegin(), eta_f_shape.cend() - 1,
                   eta_shape.cbegin(), ratio.begin(), std::divides());
    const int factor = eta_f_shape[0] / eta_shape[0];
    if constexpr (GREENC::dim == 2) {
      for (size_t i0 = 0; i0 < eta_f_shape[0]; ++i0) {
        size_t j0 = i0 / ratio[0];
        for (size_t i1 = 0; i1 < eta_f_shape[1]; ++i1) {
          size_t j1 = i1 / ratio[1];
          for (size_t i2 = 0; i2 < eta_f_shape[2]; ++i2) {
            eta_f(i0, i1, i2) = eta(j0, j1, i2);
          }
        }
      }
    } else if constexpr (GREENC::dim == 3) {
      for (size_t i0 = 0; i0 < eta_f_shape[0]; ++i0) {
        size_t j0 = i0 / ratio[0];
        for (size_t i1 = 0; i1 < eta_f_shape[1]; ++i1) {
          size_t j1 = i1 / ratio[1];
          for (size_t i2 = 0; i2 < eta_f_shape[2]; ++i2) {
            size_t j2 = i2 / ratio[2];
            for (size_t i3 = 0; i3 < eta_f_shape[3]; ++i3) {
              eta_f(i0, i1, i2, i3) = eta(j0, j1, j2, i3);
            }
          }
        }
      }
    }

    return eta_f;
  }
};

int main() {
  const int dim = 2;
  scapin::Hooke<complex128, dim> gamma{1.0, 0.3};
  ConvergenceTest<decltype(gamma)> test{gamma};

  std::vector<xt::xtensor<complex128, dim + 1>> results{};
  for (int r = 0; r < decltype(test)::num_refinements; r++) {
    results.push_back(test.run(r));
  }

  auto eta_ref = *results.rbegin();
  for (auto eta : results) {
    auto eps = eta - eta_ref;
    auto err = xt::norm_l2(eps);
    std::cout << err << std::endl;
  }
  return 0;
}
