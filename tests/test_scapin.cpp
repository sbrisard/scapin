#define _USE_MATH_DEFINES

#include <array>
#include <cmath>

#include "catch2/catch.hpp"
#include "scapin/hooke.hpp"

template <size_t DIM>
void test_data() {
  const double mu = 10.;
  const double nu = 0.3;

  Hooke<DIM> hooke{mu, nu};
  REQUIRE(hooke.isize == (DIM * (DIM + 1)) / 2);
  REQUIRE(hooke.osize == hooke.osize);
  REQUIRE(hooke.mu == mu);
  REQUIRE(hooke.nu == nu);
}

template <size_t DIM>
void green_operator_matrix(double *n, double nu, double *out) {
  constexpr size_t sym = (DIM * (DIM + 1)) / 2;
  constexpr size_t const ij2i_2d[] = {0, 1, 0};
  constexpr size_t const ij2j_2d[] = {0, 1, 1};
  constexpr size_t const ij2i_3d[] = {0, 1, 2, 1, 2, 0};
  constexpr size_t const ij2j_3d[] = {0, 1, 2, 2, 0, 1};

  auto const &ij2i = DIM == 2 ? ij2i_2d : ij2i_3d;
  auto const &ij2j = DIM == 2 ? ij2j_2d : ij2j_3d;

  double *gamma_ijkl = out;

  for (size_t ij = 0; ij < sym; ij++) {
    size_t const i = ij2i[ij];
    size_t const j = ij2j[ij];
    double const w_ij = ij < DIM ? 1. : M_SQRT2;
    for (size_t kl = 0; kl < sym; kl++, gamma_ijkl++) {
      size_t const k = ij2i[kl];
      size_t const l = ij2j[kl];
      double const w_kl = kl < DIM ? 1. : M_SQRT2;
      double const delta_ik = i == k ? 1. : 0.;
      double const delta_il = i == l ? 1. : 0.;
      double const delta_jk = j == k ? 1. : 0.;
      double const delta_jl = j == l ? 1. : 0.;
      gamma_ijkl[0] =
          w_ij * w_kl *
          (0.25 * (delta_ik * n[j] * n[l] + delta_il * n[j] * n[k] +
                   delta_jk * n[i] * n[l] + delta_jl * n[i] * n[k]) -
           0.5 * n[i] * n[j] * n[k] * n[l] / (1. - nu));
    }
  }
}

template <size_t DIM>
class DirectionGenerator {
 private:
  size_t const num_theta;
  size_t const num_phi;
  double const delta_theta;
  double const delta_phi;

 public:
  DirectionGenerator(size_t num_theta = 1, size_t num_phi = 1)
      : num_theta(num_theta),
        num_phi(num_phi),
        delta_theta(M_PI / (num_theta - 1.)),
        delta_phi(2 * M_PI / (double)num_phi) {
    if ((DIM == 2) && (num_phi != 1)) {
      throw std::invalid_argument("num_phi must be equal to 1 when DIM = 2");
    }
  }

  std::array<double, DIM> as_array(size_t i) const {
    if constexpr (DIM == 2) {
      double const theta = delta_theta * i;
      return {cos(theta), sin(theta)};
    }
    if constexpr (DIM == 3) {
      double const theta = delta_theta * (i / num_phi);
      double const phi = delta_phi * (i % num_phi);
      double const sin_theta = sin(theta);
      return {sin_theta * cos(phi), sin_theta * sin(phi), cos(theta)};
    }
  }

  class DirectionIterator {
   private:
    size_t current;
    const DirectionGenerator &generator;

   public:
    DirectionIterator(const DirectionGenerator &generator, size_t current)
        : generator(generator), current(current){};

    std::array<double, DIM> operator*() { return generator.as_array(current); };
    bool operator!=(const DirectionIterator &j) const {
      return current != j.current;
    }
    void operator++() { ++current; }
  };

  DirectionIterator begin() const { return DirectionIterator(*this, 0); }
  DirectionIterator end() const {
    return DirectionIterator(*this, num_theta * num_phi);
  }
};

template <size_t DIM>
void test_grop_hooke_apply() {
  constexpr size_t const sym = (DIM * (DIM + 1)) / 2;
  double const mu = 1.0;
  double const nu = 0.3;

  std::array<double, 3> k_norm = {1.2, 3.4, 5.6};

  Hooke<DIM> gamma{mu, nu};
  double exp[sym * sym];
  double act[sym * sym];
  double tau[sym];
  double eps[sym];

  size_t const num_theta = DIM == 2 ? 20 : 10;
  size_t const num_phi = DIM == 2 ? 1 : 20;
  for (auto n : DirectionGenerator<DIM>(num_theta, num_phi)) {
    green_operator_matrix<DIM>(n.data(), nu, exp);
    for (size_t i = 0; i < k_norm.size(); i++) {
      double k_vec[DIM];
      for (size_t j = 0; j < DIM; j++) {
        k_vec[j] = k_norm[i] * n[j];
      }
      for (size_t col = 0; col < sym; col++) {
        tau[col] = 1.;
        gamma.apply(k_vec, tau, eps);
        for (size_t row = 0; row < sym; row++) {
          act[col + sym * row] = eps[row];
        }
        tau[col] = 0.;
      }
      for (size_t ijkl = 0; ijkl < sym * sym; ijkl++) {
        REQUIRE(act[ijkl] == Approx(exp[ijkl]).epsilon(1e-12).margin(1e-12));
      }
    }
  }
}

TEST_CASE("Continuous Green operator") {
  SECTION("Hooke model") {
    SECTION("Data") {
      SECTION("2D") { test_data<2>(); }
      SECTION("3D") { test_data<3>(); }
    }
    SECTION("Apply") {
      SECTION("2D") { test_grop_hooke_apply<2>(); }
      SECTION("3D") { test_grop_hooke_apply<3>(); }
    }
  }
}
