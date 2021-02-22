#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include <fftw3.h>

#include "bri17/bri17.hpp"
#include "scapin/ms94.hpp"
#include "scapin/scapin.hpp"

using scalar_t = std::complex<double>;

template <typename T>
auto distance(std::vector<T> const &v1, std::vector<T> const &v2) {
  auto squared_difference = [](T x, T y) { return std::norm(y - x); };
  return sqrt(std::transform_reduce(v1.cbegin(), v1.cend(), v2.cbegin(), T{0},
                                    std::plus(), squared_difference));
}

template <typename T>
void create_and_execute_plan(std::span<int> shape, int howmany, std::span<T> in,
                             int sign = FFTW_FORWARD,
                             unsigned flags = FFTW_ESTIMATE);

template <>
void create_and_execute_plan(std::span<int> shape, int howmany,
                             std::span<std::complex<double>> in, int sign,
                             unsigned flags) {
  auto in_ = reinterpret_cast<fftw_complex *>(in.data());
  auto p =
      fftw_plan_many_dft(shape.size(), shape.data(), howmany, in_, nullptr,
                         howmany, 1, in_, nullptr, howmany, 1, sign, flags);
  if (sign == FFTW_BACKWARD) {
    int size =
        std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies());
    double factor = 1. / (double)size;
    std::transform(in.cbegin(), in.cend(), in.begin(),
                   [factor](auto x) { return factor * x; });
  }
  fftw_execute(p);
  fftw_destroy_plan(p);
}

template <typename GREENC>
class ConvergenceTest {
 public:
  using shape_t = std::array<int, GREENC::dim>;
  static constexpr int num_refinements = 6;

  const GREENC &gamma;

  shape_t finest_grid_shape;
  int finest_grid_size;
  std::array<double, GREENC::dim> L, patch_ratio;
  std::array<scalar_t, GREENC::isize> tau_in, tau_out;

  explicit ConvergenceTest(GREENC &gamma) : gamma(gamma) {
    static_assert((GREENC::dim == 2) || (GREENC::dim == 3),
                  "unexpected number of spatial dimensions");
    fill(finest_grid_shape.begin(), finest_grid_shape.end(), 512);
    finest_grid_size =
        std::accumulate(finest_grid_shape.cbegin(), finest_grid_shape.cend(), 1,
                        std::multiplies());

    fill(L.begin(), L.end(), 1.0);
    fill(patch_ratio.begin(), patch_ratio.end(), 0.125);
    fill(tau_in.begin(), tau_in.end(), 0.0);
    tau_in[GREENC::isize - 1] = 1.;
    fill(tau_out.begin(), tau_out.end(), 0.0);
  }

  shape_t get_grid_shape(int refinement) {
    shape_t grid_shape;
    std::transform(finest_grid_shape.cbegin(), finest_grid_shape.cend(),
                   grid_shape.begin(), [this, refinement](int n) {
                     return n >> (this->num_refinements - 1 - refinement);
                   });
    return grid_shape;
  }

  shape_t get_patch_shape(shape_t grid_shape) {
    shape_t patch_shape;
    std::transform(patch_ratio.cbegin(), patch_ratio.cend(),
                   grid_shape.cbegin(), patch_shape.begin(),
                   [](double r, int n) { return int(std::round(r * n)); });
    return patch_shape;
  }

  std::vector<scalar_t> compute_reference();

  std::vector<scalar_t> create_tau_hat(int refinement);

  std::vector<scalar_t> run(int refinement) {
    auto grid_shape = get_grid_shape(refinement);
    auto grid_size = std::reduce(grid_shape.cbegin(), grid_shape.cend(), 1,
                                 std::multiplies());
    auto tau = create_tau_hat(refinement);
    scapin::MoulinecSuquet94<GREENC> gamma_h{gamma, grid_shape.data(),
                                             L.data()};
    std::vector<scalar_t> eta(grid_size * GREENC::osize);
    gamma_h.apply(tau.data(), eta.data());
    create_and_execute_plan<scalar_t>(grid_shape, GREENC::osize, eta,
                                      FFTW_BACKWARD);

    std::vector<scalar_t> eta_f(finest_grid_size * GREENC::osize);
    std::array<int, GREENC::dim> ratio{};
    std::transform(finest_grid_shape.cbegin(), finest_grid_shape.cend(),
                   grid_shape.cbegin(), ratio.begin(), std::divides());
    if constexpr (GREENC::dim == 2) {
      for (int i0 = 0; i0 < finest_grid_shape[0]; ++i0) {
        auto j0 = i0 / ratio[0];
        for (int i1 = 0; i1 < finest_grid_shape[1]; ++i1) {
          auto j1 = i1 / ratio[1];
          auto i = i0 * finest_grid_shape[1] + i1;
          auto j = j0 * grid_shape[1] + j1;
          std::span eta_f_{eta_f.data() + i * GREENC::osize, GREENC::osize};
          std::span eta_{eta.data() + j * GREENC::osize, GREENC::osize};
          std::copy(eta_.cbegin(), eta_.cend(), eta_f_.begin());
        }
      }
    } else if constexpr (GREENC::dim == 3) {
      for (int i0 = 0; i0 < finest_grid_shape[0]; ++i0) {
        auto j0 = i0 / ratio[0];
        for (int i1 = 0; i1 < finest_grid_shape[1]; ++i1) {
          auto j1 = i1 / ratio[1];
          for (int i2 = 0; i2 < finest_grid_shape[2]; ++i2) {
            auto j2 = i2 / ratio[2];
            int i =
                (i0 * finest_grid_shape[1] + i1) * finest_grid_shape[2] + i2;
            int j = (j0 * grid_shape[1] + j1) * grid_shape[2] + j2;
            std::span eta_f_{eta_f.data() + i * GREENC::osize, GREENC::osize};
            std::span eta_{eta.data() + j * GREENC::osize, GREENC::osize};
            std::copy(eta_.cbegin(), eta_.cend(), eta_f_.begin());
          }
        }
      }
    }
    return eta_f;
  }
};

template <typename GREENC>
std::vector<scalar_t> ConvergenceTest<GREENC>::compute_reference() {
  // TODO Use templated scalar types
  auto refinement = num_refinements - 1;
  auto shape = finest_grid_shape;
  auto size = finest_grid_size;
  bri17::CartesianGrid<double, GREENC::dim> grid{shape, L};
  bri17::Hooke<double, GREENC::dim> hooke{gamma.mu, gamma.nu, grid};
  auto tau = create_tau_hat(refinement);
  std::vector<scalar_t> eta(size * GREENC::osize);

  if constexpr (GREENC::dim == 2) {
    auto tau_ = tau.data();
    auto eta_ = eta.data();
    std::array<int, GREENC::dim> k{};
    for (k[0] = 0; k[0] < shape[0]; k[0]++) {
      for (k[1] = 0; k[1] < shape[1]; k[1]++) {
        hooke.modal_eigenstress_to_opposite_strain(k.data(), tau_, eta_);
        tau_ += GREENC::isize;
        eta_ += GREENC::osize;
      }
    }
  } else if constexpr (GREENC::dim == 3) {
    auto tau_ = tau.data();
    auto eta_ = eta.data();
    std::array<int, GREENC::dim> k{};  // Frequency multi-index
    for (k[0] = 0; k[0] < shape[0]; k[0]++) {
      for (k[1] = 0; k[1] < shape[1]; k[1]++) {
        for (k[2] = 0; k[2] < shape[2]; k[2]++) {
          hooke.modal_eigenstress_to_opposite_strain(k.data(), tau_, eta_);
          tau_ += GREENC::isize;
          eta_ += GREENC::osize;
        }
      }
    }
  }
  create_and_execute_plan<scalar_t>(shape, GREENC::osize, eta, FFTW_BACKWARD);
  return eta;
}

template <typename GREENC>
std::vector<scalar_t> ConvergenceTest<GREENC>::create_tau_hat(int refinement) {
  auto grid_shape = get_grid_shape(refinement);
  auto grid_size = std::accumulate(grid_shape.cbegin(), grid_shape.cend(), 1,
                                   std::multiplies());
  auto patch_shape = get_patch_shape(grid_shape);
  std::vector<scalar_t> tau(grid_size * GREENC::isize);
  for (int i = 0, offset = 0; i < grid_size; i++, offset += GREENC::isize) {
    std::copy(tau_out.cbegin(), tau_out.cend(), tau.data() + offset);
  }
  if constexpr (GREENC::dim == 2) {
    for (int i0 = 0; i0 < patch_shape[0]; i0++) {
      auto tau_ = tau.data() + (i0 * grid_shape[1] * GREENC::isize);
      for (int i1 = 0; i1 < patch_shape[1]; i1++, tau_ += GREENC::isize) {
        std::copy(tau_in.cbegin(), tau_in.cend(), tau_);
      }
    }
  }
  if constexpr (GREENC::dim == 3) {
    for (int i0 = 0; i0 < patch_shape[0]; i0++) {
      for (int i1 = 0; i1 < grid_shape[1]; i1++) {
        auto offset = (i0 * grid_shape[1] + i1) * grid_shape[2] * GREENC::isize;
        auto tau_ = tau.data() + offset;
        for (int i2 = 0; i2 < grid_shape[2]; i2++, tau_ += GREENC::isize) {
          std::copy(tau_in.cbegin(), tau_in.cend(), tau_);
        }
      }
    }
  }
  create_and_execute_plan<scalar_t>(grid_shape, GREENC::isize, tau);
  return tau;
}

int main() {
  const int dim = 2;
  scapin::Hooke<scalar_t, dim> gamma{1.0, 0.3};
  ConvergenceTest<decltype(gamma)> test{gamma};

  std::vector<std::vector<scalar_t>> results(test.num_refinements);
  for (int r = 0; r < test.num_refinements; r++) {
    results[r] = test.run(r);
  }

  // auto eta_ref = *results.rbegin();
  auto eta_ref = test.compute_reference();
  for (auto const &eta : results) {
    auto err = distance(eta, eta_ref);
    std::cout << err << std::endl;
  }
  return 0;
}
