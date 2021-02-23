#include <array>
#include <numbers>
#include <stdexcept>

#include "catch2/catch.hpp"
#include "scapin/hooke.hpp"

template <int DIM>
inline constexpr int sym = (DIM * (DIM + 1)) / 2;

template <int DIM>
std::pair<int, int> unravel_index(int ij);

template <>
std::pair<int, int> unravel_index<2>(int ij) {
  switch (ij) {
    case 0:
      return std::make_pair(0, 0);
    case 1:
      return std::make_pair(1, 1);
    case 2:
      return std::make_pair(0, 1);
    default:
      throw std::invalid_argument("unexpected value");
  }
}

template <>
std::pair<int, int> unravel_index<3>(int ij) {
  switch (ij) {
    case 0:
      return std::make_pair(0, 0);
    case 1:
      return std::make_pair(1, 1);
    case 2:
      return std::make_pair(2, 2);
    case 3:
      return std::make_pair(1, 2);
    case 4:
      return std::make_pair(2, 0);
    case 5:
      return std::make_pair(0, 1);
    default:
      throw std::invalid_argument("unexpected argument");
  }
}

template <int DIM>
void green_operator_matrix(std::array<double, DIM> const &n, double nu,
                           std::array<double, sym<DIM> * sym<DIM>> &out) {
  int ijkl = 0;
  for (int ij = 0; ij < sym<DIM>; ij++) {
    auto const [i, j] = unravel_index<DIM>(ij);
    double const w_ij = ij < DIM ? 1. : std::numbers::sqrt2;
    for (int kl = 0; kl < sym<DIM>; kl++, ijkl++) {
      auto const [k, l] = unravel_index<DIM>(kl);
      double const w_kl = kl < DIM ? 1. : std::numbers::sqrt2;
      double const delta_ik = i == k ? 1. : 0.;
      double const delta_il = i == l ? 1. : 0.;
      double const delta_jk = j == k ? 1. : 0.;
      double const delta_jl = j == l ? 1. : 0.;
      out[ijkl] = w_ij * w_kl *
                  (0.25 * (delta_ik * n[j] * n[l] + delta_il * n[j] * n[k] +
                           delta_jk * n[i] * n[l] + delta_jl * n[i] * n[k]) -
                   0.5 * n[i] * n[j] * n[k] * n[l] / (1. - nu));
    }
  }
}

std::vector<std::array<double, 2>> gen_directions(int num_theta) {
  double const delta_theta = std::numbers::pi / (num_theta - 1.);
  std::vector<std::array<double, 2>> directions;
  for (int i = 0; i < num_theta; i++) {
    double theta = i * delta_theta;
    directions.push_back({cos(theta), sin(theta)});
  }
  return directions;
}

std::vector<std::array<double, 3>> gen_directions(int num_theta, int num_phi) {
  double const delta_theta = std::numbers::pi / (num_theta - 1.);
  double const delta_phi = 2 * std::numbers::pi / double(num_phi);
  std::vector<std::array<double, 3>> directions{};
  for (int i = 0; i < num_theta; i++) {
    double const theta = i * delta_theta;
    double const cos_theta = cos(theta);
    double const sin_theta = sin(theta);
    for (int j = 0; j < num_phi; j++) {
      double const phi = j * delta_phi;
      directions.push_back(
          {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta});
    }
  }
  return directions;
}

template <int DIM>
void test_hooke_apply() {
  double const mu = 1.0;
  double const nu = 0.3;

  std::vector<double> norms{1.2, 3.4, 5.6};

  scapin::Hooke<double, DIM> gamma{mu, nu};
  std::array<double, sym<DIM> * sym<DIM>> exp{};
  std::array<double, sym<DIM> * sym<DIM>> act{};
  std::array<double, sym<DIM>> tau{};
  std::array<double, sym<DIM>> eps{};

  std::vector<std::array<double, DIM>> directions;
  if constexpr (DIM == 2) {
    directions = gen_directions(20);
  } else if constexpr (DIM == 3) {
    directions = gen_directions(10, 20);
  }
  for (auto n : directions) {
    green_operator_matrix<DIM>(n, nu, exp);
    for (auto const norm : norms) {
      std::array<double, DIM> k{};
      std::transform(n.cbegin(), n.cend(), k.begin(),
                     [norm](auto n_) { return norm * n_; });
      for (int col = 0; col < sym<DIM>; col++) {
        tau[col] = 1.;
        gamma.apply(k.data(), tau.data(), eps.data());
        for (int row = 0; row < sym<DIM>; row++) {
          act[col + sym<DIM> * row] = eps[row];
        }
        tau[col] = 0.;
      }
      for (int ijkl = 0; ijkl < sym<DIM> * sym<DIM>; ijkl++) {
        REQUIRE(act[ijkl] == Approx(exp[ijkl]).epsilon(1e-12).margin(1e-12));
      }
    }
  }
}

TEST_CASE("Continuous Green operator") {
  SECTION("Hooke model") {
    SECTION("Apply") {
      SECTION("2D") { test_hooke_apply<2>(); }
      SECTION("3D") { test_hooke_apply<3>(); }
    }
  }
}
