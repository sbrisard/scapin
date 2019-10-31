#include <scapin.h>

ScapinGreenOperator *scapin_grop_new() {
  return malloc(sizeof(scapin_grop_new));
}

void scapin_grop_free(ScapinGreenOperator *grop) {
  if (grop->type->dispose) {
    grop->type->dispose(grop->data);
  } else {
    free(grop->data);
  }
  free(grop);
}

#define SCAPIN_GROP_HOOKE_DATA(op) ((GrOpHookeData *)((op)->data))

typedef struct GrOpHookeData_ {
  double mu;
  double nu;
} GrOpHookeData;

ScapinGreenOperatorType const Hooke2D = {.name = "Hooke2D",
                                         .ndims = 2,
                                         .isize = 3,
                                         .osize = 3,
                                         .apply = NULL,
                                         .dispose = NULL};

ScapinGreenOperatorType const Hooke3D = {.name = "Hooke3D",
                                         .ndims = 3,
                                         .isize = 6,
                                         .osize = 6,
                                         .apply = NULL,
                                         .dispose = NULL};

ScapinGreenOperator *scapin_grop_hooke_new(size_t ndims, double mu, double nu) {
  ScapinGreenOperator *op = scapin_grop_new();
  switch (ndims) {
    case 2:
      op->type = &Hooke2D;
      break;
    case 3:
      op->type = &Hooke3D;
      break;
    default:
      /* TODO Return error message. */
      return NULL;
  }
  GrOpHookeData *data = malloc(sizeof(GrOpHookeData));
  data->mu = mu;
  data->nu = nu;
  op->data = data;
  return op;
}

double scapin_grop_hooke_mu(ScapinGreenOperator *op) {
  return SCAPIN_GROP_HOOKE_DATA(op)->mu;
}

double scapin_grop_hooke_nu(ScapinGreenOperator *op) {
  return SCAPIN_GROP_HOOKE_DATA(op)->nu;
}

/* void green_apply(double *tau, double *k, double mu, double nu, double *out) {
 */
/*   const size_t dim = 3; */
/*   const size_t sym = 6; */
/*   const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2]; */
/*   double *tau_dot_k = g_new(double, dim); */
/*   tau_dot_k[0] = tau[0] * k[0] + M_SQRT1_2 * (tau[5] * k[1] + tau[4] * k[2]);
 */
/*   tau_dot_k[1] = tau[1] * k[1] + M_SQRT1_2 * (tau[5] * k[0] + tau[3] * k[2]);
 */
/*   tau_dot_k[2] = tau[2] * k[2] + M_SQRT1_2 * (tau[4] * k[0] + tau[3] * k[1]);
 */
/*   double k_dot_tau_dot_k = */
/*       k[0] * tau_dot_k[0] + k[1] * tau_dot_k[1] + k[2] * tau_dot_k[2]; */
/*   const double const1 = 1. / mu; */
/*   out[0] = const1 * (k[0] * tau_dot_k[0]); */
/*   out[1] = const1 * (k[1] * tau_dot_k[1]); */
/*   out[2] = const1 * (k[2] * tau_dot_k[2]); */
/*   out[3] = const1 * (M_SQRT1_2 * (k[1] * tau_dot_k[2] + tau_dot_k[1] *
 * k[2])); */
/*   out[4] = const1 * (M_SQRT1_2 * (k[2] * tau_dot_k[0] + tau_dot_k[2] *
 * k[0])); */
/*   out[5] = const1 * (M_SQRT1_2 * (k[0] * tau_dot_k[1] + tau_dot_k[0] *
 * k[1])); */

/*   g_free(tau_dot_k); */
/* } */
