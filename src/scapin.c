#include <scapin.h>

typedef sruct ScapinGreenOperator_ ScapinGreenOperator;
typedef void scapin_green_operator_apply_t(ScapinGreenOperator *, double *,
                                           double *);

typedef struct ScapinGreenOperator_ {
  size_t ndims;
  size_t isize;
  size_t osize;
  scapin_green_operator_apply_t *apply;
}

void green_apply(double *tau, double *k, double mu, double nu, double *out) {
  const size_t dim = 3;
  const size_t sym = 6;
  const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
  double *tau_dot_k = g_new(double, dim);
  tau_dot_k[0] = tau[0] * k[0] + M_SQRT1_2 * (tau[5] * k[1] + tau[4] * k[2]);
  tau_dot_k[1] = tau[1] * k[1] + M_SQRT1_2 * (tau[5] * k[0] + tau[3] * k[2]);
  tau_dot_k[2] = tau[2] * k[2] + M_SQRT1_2 * (tau[4] * k[0] + tau[3] * k[1]);
  double k_dot_tau_dot_k =
      k[0] * tau_dot_k[0] + k[1] * tau_dot_k[1] + k[2] * tau_dot_k[2];
  const double const1 = 1. / mu;
  out[0] = const1 * (k[0] * tau_dot_k[0]);
  out[1] = const1 * (k[1] * tau_dot_k[1]);
  out[2] = const1 * (k[2] * tau_dot_k[2]);
  out[3] = const1 * (M_SQRT1_2 * (k[1] * tau_dot_k[2] + tau_dot_k[1] * k[2]));
  out[4] = const1 * (M_SQRT1_2 * (k[2] * tau_dot_k[0] + tau_dot_k[2] * k[0]));
  out[5] = const1 * (M_SQRT1_2 * (k[0] * tau_dot_k[1] + tau_dot_k[0] * k[1]));

  g_free(tau_dot_k);
}
