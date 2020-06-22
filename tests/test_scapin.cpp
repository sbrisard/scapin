#include <cmath>

#include "catch2/catch.hpp"
#include <scapin/scapin.hpp>

void test_grop_hooke_data(size_t dim) {
  const size_t isize = (dim * (dim + 1)) / 2;
  const size_t osize = isize;
  const double mu = 10.;
  const double nu = 0.3;

  ScapinGreenOperator *op = scapin_grop_hooke_new(dim, mu, nu);
  REQUIRE(op->type->dim == dim);
  REQUIRE(op->type->isize == isize);
  REQUIRE(op->type->osize == osize);
  REQUIRE(scapin_grop_hooke_mu(op) == mu);
  REQUIRE(scapin_grop_hooke_nu(op) == nu);
}

void grop_hooke_matrix(size_t dim, double *n, double nu, double *out) {
  size_t const sym = (dim * (dim + 1)) / 2;
  size_t const ij2i_2d[] = {0, 1, 0};
  size_t const ij2i_3d[] = {0, 1, 2, 1, 2, 0};
  size_t const ij2j_2d[] = {0, 1, 1};
  size_t const ij2j_3d[] = {0, 1, 2, 2, 0, 1};
  size_t const *ij2i = (dim == 2) ? ij2i_2d : ij2i_3d;
  size_t const *ij2j = (dim == 2) ? ij2j_2d : ij2j_3d;

  double *gamma_ijkl = out;

  for (size_t ij = 0; ij < sym; ij++) {
    size_t const i = ij2i[ij];
    size_t const j = ij2j[ij];
    double const w_ij = ij < dim ? 1. : M_SQRT2;
    for (size_t kl = 0; kl < sym; kl++, gamma_ijkl++) {
      size_t const k = ij2i[kl];
      size_t const l = ij2j[kl];
      double const w_kl = kl < dim ? 1. : M_SQRT2;
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

void test_grop_hooke_2d_apply() {
  size_t const dim = 2;
  size_t const sym = 3;
  size_t const num_theta = 20;
  double const mu = 1.0;
  double const nu = 0.3;

  const size_t num_k_norms = 3;
  double const k_norm[] = {1.2, 3.4, 5.6};

  double const delta_theta = 2 * M_PI / (double)num_theta;
  double const theta_max = (num_theta - 0.5) * delta_theta;

  ScapinGreenOperator *gamma = scapin_grop_hooke_new(dim, mu, nu);
  double exp[sym * sym];
  double act[sym * sym];
  double tau[sym];
  double eps[sym];
  double n[dim];
  for (double theta = 0; theta < theta_max; theta += delta_theta) {
    n[0] = sin(theta);
    n[1] = cos(theta);
    grop_hooke_matrix(dim, n, nu, exp);
    for (size_t i = 0; i < num_k_norms; i++) {
      double const k_vec[] = {k_norm[i] * n[0], k_norm[i] * n[1],
                              k_norm[i] * n[2]};
      for (size_t col = 0; col < sym; col++) {
        tau[col] = 1.;
        gamma->type->apply(gamma, k_vec, tau, eps);
        for (size_t row = 0; row < sym; row++) {
          act[col + sym * row] = eps[row];
        }
        tau[col] = 0.;
      }
      for (size_t ijkl = 0; ijkl < sym * sym; ijkl++) {
        double const err = fabs(act[ijkl] - exp[ijkl]);
        REQUIRE(act[ijkl] == Approx(exp[ijkl]).epsilon(1e-12).margin(1e-12));
      }
    }
  }
  scapin_grop_free(gamma);
}

void test_grop_hooke_3d_apply() {
  size_t const dim = 3;
  size_t const sym = 6;
  size_t const num_theta = 10;
  size_t const num_phi = 20;
  double const mu = 1.0;
  double const nu = 0.3;

  const size_t num_k_norms = 3;
  double const k_norm[] = {1.2, 3.4, 5.6};

  double const delta_theta = M_PI / (num_theta - 1.);
  double const theta_max = (num_theta - 0.5) * delta_theta;
  double const delta_phi = 2 * M_PI / (double)num_phi;
  double const phi_max = (num_phi - 0.5) * delta_phi;

  ScapinGreenOperator *gamma = scapin_grop_hooke_new(dim, mu, nu);
  double exp[sym * sym];
  double act[sym * sym];
  double tau[sym];
  double eps[sym];
  double n[dim];
  for (double theta = 0; theta < theta_max; theta += delta_theta) {
    double const sin_theta = sin(theta);
    double const cos_theta = cos(theta);
    for (double phi = 0; phi < phi_max; phi += delta_phi) {
      n[0] = sin_theta * cos(phi);
      n[1] = sin_theta * sin(phi);
      n[2] = cos_theta;
      grop_hooke_matrix(dim, n, nu, exp);
      for (size_t i = 0; i < num_k_norms; i++) {
        double const k_vec[] = {k_norm[i] * n[0], k_norm[i] * n[1],
                                k_norm[i] * n[2]};
        for (size_t col = 0; col < sym; col++) {
          tau[col] = 1.;
          gamma->type->apply(gamma, k_vec, tau, eps);
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
  scapin_grop_free(gamma);
}

TEST_CASE("Continuous Green operator") {
  SECTION("Hooke model") {
    SECTION("Data") {
      SECTION("2D") {
        test_grop_hooke_data(2);
      }
      SECTION("3D") {
        test_grop_hooke_data(3);
      }
    }
    SECTION("Apply") {
      SECTION("2D") { test_grop_hooke_2d_apply(); }
      SECTION("3D") { test_grop_hooke_3d_apply(); }
    }
  }
}
