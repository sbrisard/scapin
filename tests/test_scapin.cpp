#include <numbers>
#include <stdexcept>

#include "catch2/catch.hpp"

#include <Eigen/Dense>

#include "scapin/hooke.hpp"

template <int DIM>
inline constexpr int sym = (DIM * (DIM + 1)) / 2;

template <int DIM>
using Vector = Eigen::Matrix<double, DIM, 1>;

template <int DIM>
using Tensor2 = Eigen::Matrix<double, sym<DIM>, 1>;

template <int DIM>
using Tensor4 = Eigen::Matrix<double, sym<DIM>, sym<DIM>>;

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
Tensor4<DIM> green_operator_matrix(Vector<DIM> const &n, double nu) {
  Tensor4<DIM> out;
  for (int ij = 0; ij < sym<DIM>; ij++) {
    auto const [i, j] = unravel_index<DIM>(ij);
    double const w_ij = ij < DIM ? 1. : std::numbers::sqrt2;
    for (int kl = 0; kl < sym<DIM>; kl++) {
      auto const [k, l] = unravel_index<DIM>(kl);
      double const w_kl = kl < DIM ? 1. : std::numbers::sqrt2;
      double const delta_ik = i == k ? 1. : 0.;
      double const delta_il = i == l ? 1. : 0.;
      double const delta_jk = j == k ? 1. : 0.;
      double const delta_jl = j == l ? 1. : 0.;
      out(ij, kl) = w_ij * w_kl *
                    (0.25 * (delta_ik * n(j) * n(l) + delta_il * n(j) * n(k) +
                             delta_jk * n(i) * n(l) + delta_jl * n(i) * n(k)) -
                     0.5 * n(i) * n(j) * n(k) * n(l) / (1. - nu));
    }
  }
  return out;
}

template <int DIM>
double bulk_modulus(double mu, double nu);

template <>
double bulk_modulus<2>(double mu, double nu) {
  return mu / (1. - 2. * nu);
}

template <>
double bulk_modulus<3>(double mu, double nu) {
  return 2. * mu * (1. + nu) / 3. / (1. - 2. * nu);
}

template <int DIM>
std::pair<Tensor4<DIM>, Tensor4<DIM>> isotropic_projectors() {
  Tensor2<DIM> I2 = Tensor2<DIM>::Zero();
  for (int i = 0; i < DIM; i++) I2(i) = 1.;
  Tensor4<DIM> I = Tensor4<DIM>::Identity();
  Tensor4<DIM> J = I2 * I2.transpose() / DIM;
  Tensor4<DIM> K = I - J;
  return std::make_pair(J, K);
}

template <int DIM>
Tensor4<DIM> stiffness_matrix(double mu, double nu) {
  auto const [J, K] = isotropic_projectors<DIM>();
  auto kappa = bulk_modulus<DIM>(mu, nu);
  Tensor4<DIM> C = DIM * kappa * J + 2 * mu * K;
  return C;
}

template <int DIM>
Tensor4<DIM> compliance_matrix(double mu, double nu) {
  auto const [J, K] = isotropic_projectors<DIM>();
  auto kappa = bulk_modulus<DIM>(mu, nu);
  Tensor4<DIM> S = J / (DIM * kappa) + K / (2 * mu);
  return S;
}

std::vector<Vector<2>> gen_directions(int num_theta) {
  double const delta_theta = std::numbers::pi / (num_theta - 1.);
  std::vector<Vector<2>> directions;
  for (int i = 0; i < num_theta; i++) {
    double theta = i * delta_theta;
    directions.push_back(Vector<2>{cos(theta), sin(theta)});
  }
  return directions;
}

std::vector<Vector<3>> gen_directions(int num_theta, int num_phi) {
  double const delta_theta = std::numbers::pi / (num_theta - 1.);
  double const delta_phi = 2 * std::numbers::pi / double(num_phi);
  std::vector<Vector<3>> directions{};
  for (int i = 0; i < num_theta; i++) {
    double const theta = i * delta_theta;
    double const cos_theta = cos(theta);
    double const sin_theta = sin(theta);
    for (int j = 0; j < num_phi; j++) {
      double const phi = j * delta_phi;
      directions.push_back(
          Vector<3>{sin_theta * cos(phi), sin_theta * sin(phi), cos_theta});
    }
  }
  return directions;
}

template <int DIM>
void test_hooke_apply() {
  std::vector<double> norms{1.2, 3.4, 5.6};
  scapin::Hooke<double, DIM> gamma{1.0, 0.3};
  std::vector<Vector<DIM>> directions;
  if constexpr (DIM == 2) {
    directions = gen_directions(20);
  } else if constexpr (DIM == 3) {
    directions = gen_directions(10, 20);
  }
  for (auto n : directions) {
    auto exp = green_operator_matrix<DIM>(n, gamma.nu);
    Tensor4<DIM> act;
    for (auto const norm : norms) {
      Vector<DIM> k = norm * n;
      for (int i = 0; i < sym<DIM>; i++) {
        Tensor2<DIM> tau = Tensor2<DIM>::Zero();
        tau(i) = 1.;
        Tensor2<DIM> eps;
        gamma.apply(k.data(), tau.data(), eps.data());
        act.col(i) = eps;
      }
      for (int i = 0; i < sym<DIM>; ++i) {
        for (int j = 0; j < sym<DIM>; ++j) {
          REQUIRE(act(i, j) == Approx(exp(i, j)).epsilon(1e-12).margin(1e-12));
        }
      }
    }
  }
}

template <int DIM>
void test_apply_stiffness() {
  scapin::Hooke<double, DIM> gamma{1.2, 0.3};
  Tensor4<DIM> C_exp = stiffness_matrix<DIM>(gamma.mu, gamma.nu);
  Tensor4<DIM> C_act;
  Tensor2<DIM> eps = Tensor2<DIM>::Zero();
  Tensor2<DIM> sig = Tensor2<DIM>::Zero();
  for (int i = 0; i < sym<DIM>; i++) {
    eps(i) = 1.0;
    gamma.apply_stiffness(eps.data(), sig.data());
    C_act.col(i) = sig;
    eps(i) = 0.0;
  }

  for (int i = 0; i < sym<DIM>; i++) {
    for (int j = 0; j < sym<DIM>; j++) {
      REQUIRE(C_act(i, j) == Approx(C_exp(i, j)).epsilon(1e-15).margin(1e-15));
    }
  }
}

template <int DIM>
void test_apply_compliance() {
  scapin::Hooke<double, DIM> gamma{1.2, 0.3};
  Tensor4<DIM> S_exp = compliance_matrix<DIM>(gamma.mu, gamma.nu);
  Tensor4<DIM> S_act;
  Tensor2<DIM> eps = Tensor2<DIM>::Zero();
  Tensor2<DIM> sig = Tensor2<DIM>::Zero();
  for (int i = 0; i < sym<DIM>; i++) {
    sig(i) = 1.0;
    gamma.apply_compliance(sig.data(), eps.data());
    S_act.col(i) = eps;
    sig(i) = 0.0;
  }

  for (int i = 0; i < sym<DIM>; i++) {
    for (int j = 0; j < sym<DIM>; j++) {
      REQUIRE(S_act(i, j) == Approx(S_exp(i, j)).epsilon(1e-15).margin(1e-15));
    }
  }
}

TEST_CASE("Continuous Green operator") {
  SECTION("Hooke model") {
    SECTION("Apply") {
      SECTION("2D") { test_hooke_apply<2>(); }
      SECTION("3D") { test_hooke_apply<3>(); }
    }
    SECTION("apply_stiffness") {
      SECTION("2D") { test_apply_stiffness<2>(); }
      SECTION("3D") { test_apply_stiffness<3>(); }
    }
    SECTION("apply_compliance") {
      SECTION("2D") { test_apply_compliance<2>(); }
      SECTION("3D") { test_apply_compliance<3>(); }
    }
  }
}
