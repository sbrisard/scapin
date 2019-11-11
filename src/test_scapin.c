#include <math.h>

#include <glib.h>

#include "scapin.h"

#include <stdio.h>

void test_grop_hooke_data(gconstpointer data) {
  const size_t ndims = *((size_t *)data);
  const size_t isize = (ndims * (ndims + 1)) / 2;
  const size_t osize = isize;
  const double mu = 10.;
  const double nu = 0.3;

  ScapinGreenOperator *op = scapin_grop_hooke_new(ndims, mu, nu);
  g_assert_cmpint(op->type->ndims, ==, ndims);
  g_assert_cmpint(op->type->isize, ==, isize);
  g_assert_cmpint(op->type->osize, ==, osize);
  g_assert_cmpfloat(scapin_grop_hooke_mu(op), ==, mu);
  g_assert_cmpfloat(scapin_grop_hooke_nu(op), ==, nu);
}

size_t const ij2i[] = {0, 1, 2, 1, 2, 0};
size_t const ij2j[] = {0, 1, 2, 2, 0, 1};

void grop_hooke_3d_matrix(double *n, double nu, double *out) {
  size_t const dim = 3;
  size_t const sym = 6;
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

void test_grop_hooke_3d_apply() {
  size_t const dim = 3;
  size_t const sym = 6;
  size_t const num_theta = 10;
  size_t const num_phi = 20;
  double const mu = 1.0;
  double const nu = 0.3;

  ScapinGreenOperator *gamma = scapin_grop_hooke_new(dim, mu, nu);
  double *exp = malloc(sym * sym * sizeof(double));
  double *act = malloc(sym * sym * sizeof(double));
  double *tau = malloc(sym * sizeof(double));
  double *eps = malloc(sym * sizeof(double));
  double *n = malloc(dim * sizeof(double));

  for (size_t i = 0; i < num_theta; i++) {
    double const theta = M_PI * i / (num_theta - 1.);
    for (size_t j = 0; j < num_phi; j++) {
      double const phi = 2. * M_PI * j / (double)num_phi;
      n[0] = sin(theta) * cos(phi);
      n[1] = sin(theta) * sin(phi);
      n[2] = cos(theta);
      grop_hooke_3d_matrix(n, nu, exp);
      for (size_t k = 0; k < sym; k++) {
        tau[k] = 1.;
        gamma->type->apply(gamma, n, tau, eps);
        for (size_t l = 0; l < sym; l++) {
          act[k + sym * l] = eps[l];
        }
        tau[k] = 0.;
      }
      for (size_t ijkl = 0; ijkl < sym * sym; ijkl++) {
        double const err = fabs(act[ijkl] - exp[ijkl]);
        g_assert_cmpfloat(err, <=, 1e-12 * fabs(exp[ijkl]) + 1e-12);
      }
    }
  }

  free(n);
  free(eps);
  free(tau);
  free(act);
  free(exp);
  scapin_grop_free(gamma);
}

void test_grop_hooke_setup_tests() {
  size_t *data1 = malloc(sizeof(size_t));
  data1[0] = 2;
  g_test_add_data_func_full("/Hooke2D/data", data1, test_grop_hooke_data, free);

  size_t *data2 = malloc(sizeof(size_t));
  data2[0] = 3;
  g_test_add_data_func_full("/Hooke3D/data", data2, test_grop_hooke_data, free);

  g_test_add_func("/Hooke3D/apply", test_grop_hooke_3d_apply);
}

int main(int argc, char **argv) {
  g_test_init(&argc, &argv, NULL);
  test_grop_hooke_setup_tests();
  return g_test_run();
}
