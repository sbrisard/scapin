#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include <fftw3.h>

#include "bri17/bri17.hpp"
#include "scapin/ms94.hpp"
#include "scapin/scapin.hpp"

using scalar_t = std::complex<double>;

auto distance(size_t n, scalar_t const *x1, scalar_t const *x2) {
  double output = 0.0;
  auto x1_ = x1;
  auto x2_ = x2;
  for (size_t i = 0; i < n; i++, x1_++, x2_++) {
    output += std::norm(*x2_ - *x1_);
  }
  return sqrt(output);
}

template <typename GREENC>
class ConvergenceTest {
 public:
  static constexpr int num_refinements = 6;
  const GREENC &gamma;

  std::array<int, GREENC::dim> Nf;
  std::array<double, GREENC::dim> L, patch_ratio;
  std::array<scalar_t, GREENC::isize> tau_in, tau_out;

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

  std::array<int, GREENC::dim> get_grid_shape(int refinement) {
    std::array<int, GREENC::dim> shape;
    std::transform(Nf.cbegin(), Nf.cend(), shape.begin(),
                   [this, refinement](int n) {
                     return n >> (this->num_refinements - 1 - refinement);
                   });
    return shape;
  }

  std::array<int, GREENC::dim> get_patch_shape(std::array<int, GREENC::dim> grid_shape) {
    std::array<int, GREENC::dim> patch_shape;
    std::transform(
        patch_ratio.cbegin(), patch_ratio.cend(), grid_shape.cbegin(), patch_shape.begin(),
        [](double r, int n) { return int(std::round(r * n)); });
    return patch_shape;
  }

  scalar_t *create_tau_hat(int refinement) {
    auto grid_shape = get_grid_shape(refinement);
    auto tau_size = std::accumulate(grid_shape.cbegin(), grid_shape.cend(),
                                    GREENC::isize, std::multiplies());
    auto patch_shape = get_patch_shape(grid_shape);
    auto tau = new scalar_t[tau_size];
    if constexpr (GREENC::dim == 2) {
      auto tau_ = tau;
      for (int i0 = 0; i0 < grid_shape[0]; i0++) {
        bool in0 = i0 < patch_shape[0];
        for (int i1 = 0; i1 < grid_shape[1]; i1++, tau_ += GREENC::isize) {
          auto tau_act = in0 && i1 < patch_shape[1] ? tau_in : tau_out;
          for (int k = 0; k < GREENC::isize; k++) {
            tau_[k] = tau_act[k];
          }
        }
      }
    }
    if constexpr (GREENC::dim == 3) {
      static_assert(GREENC::dim != 3, "not implemented");
    }

    // TODO: this should be simplified
    std::array<int, GREENC::dim + 1> tau_shape;
    std::copy(grid_shape.cbegin(), grid_shape.cend(), tau_shape.begin());
    tau_shape[GREENC::dim] = GREENC::isize;
    auto tau_data = reinterpret_cast<fftw_complex *>(tau);
    auto p = fftw_plan_many_dft(GREENC::dim, tau_shape.data(), GREENC::isize,
                                tau_data, nullptr, GREENC::isize, 1, tau_data,
                                nullptr, GREENC::isize, 1, FFTW_FORWARD,
                                FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    // End of simplification

    return tau;
  }

  scalar_t *run(int refinement) {
    auto grid_shape = get_grid_shape(refinement);
    auto grid_size = std::reduce(grid_shape.cbegin(), grid_shape.cend(), 1,
                                 std::multiplies());
    auto tau = create_tau_hat(refinement);

    std::array<int, GREENC::dim + 1> tau_shape;
    std::transform(grid_shape.cbegin(), grid_shape.cend(), tau_shape.begin(),
                   [](size_t n) { return int(n); });
    tau_shape[GREENC::dim] = GREENC::isize;

    // Set eta to the DFT of gamma_h(tau)
    scapin::MoulinecSuquet94<GREENC> gamma_h{gamma, tau_shape.data(), L.data()};
    auto eta_shape{tau_shape};
    eta_shape[GREENC::dim] = GREENC::osize;
    auto eta_size = std::accumulate(eta_shape.cbegin(), eta_shape.cend(), 1,
                                    std::multiplies());
    auto eta = new scalar_t[eta_size];
    auto eta_data = reinterpret_cast<fftw_complex *>(eta);

    auto tau_ = tau;
    auto eta_ = eta;
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
    for (int i = 0; i < eta_size; i++) {
      eta[i] /= (double)grid_size;
    }

    std::array<int, GREENC::dim + 1> eta_f_shape;
    std::copy(Nf.cbegin(), Nf.cend(), eta_f_shape.begin());
    eta_f_shape[GREENC::dim] = gamma.osize;
    auto eta_f_size = std::accumulate(eta_f_shape.cbegin(), eta_f_shape.cend(),
                                      1, std::multiplies());
    auto eta_f = new scalar_t[eta_f_size];
    std::array<int, GREENC::dim> ratio{};
    std::transform(eta_f_shape.cbegin(), eta_f_shape.cend() - 1,
                   eta_shape.cbegin(), ratio.begin(), std::divides());
    if constexpr (GREENC::dim == 2) {
      for (int i0 = 0; i0 < eta_f_shape[0]; ++i0) {
        auto j0 = i0 / ratio[0];
        for (int i1 = 0; i1 < eta_f_shape[1]; ++i1) {
          auto j1 = i1 / ratio[1];
          for (int k = 0; k < GREENC::osize; ++k) {
            int i = (i0 * eta_f_shape[1] + i1) * eta_f_shape[2] + k;
            int j = (j0 * eta_shape[1] + j1) * eta_shape[2] + k;
            eta_f[i] = eta[j];
          }
        }
      }
    } else if constexpr (GREENC::dim == 3) {
      static_assert(GREENC::dim != 3, "check implementation");
      //      for (size_t i0 = 0; i0 < eta_f_shape[0]; ++i0) {
      //        size_t j0 = i0 / ratio[0];
      //        for (size_t i1 = 0; i1 < eta_f_shape[1]; ++i1) {
      //          size_t j1 = i1 / ratio[1];
      //          for (size_t i2 = 0; i2 < eta_f_shape[2]; ++i2) {
      //            size_t j2 = i2 / ratio[2];
      //            for (size_t i3 = 0; i3 < eta_f_shape[3]; ++i3) {
      //              eta_f(i0, i1, i2, i3) = eta(j0, j1, j2, i3);
      //            }
      //          }
      //        }
      //      }
    }

    delete[] tau;
    delete[] eta;
    return eta_f;
  }
};

int main() {
  const int dim = 2;
  scapin::Hooke<scalar_t, dim> gamma{1.0, 0.3};
  ConvergenceTest<decltype(gamma)> test{gamma};

  auto eta_size = std::accumulate(test.Nf.cbegin(), test.Nf.cend(), gamma.osize,
                                  std::multiplies());

  std::vector<scalar_t *> results(test.num_refinements);
  for (int r = 0; r < test.num_refinements; r++) {
    results[r] = test.run(r);
  }

  auto eta_ref = *results.rbegin();
  for (auto eta : results) {
    auto err = distance(eta_size, eta, eta_ref);
    std::cout << err << std::endl;
    delete[] eta;
  }
  return 0;
}
