#include "math.h"

#include "scapin.h"

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

void scapin_grop_hooke_3d_apply(ScapinGreenOperator const *, double const *,
                                double const *, double *);

ScapinGreenOperatorType const Hooke3D = {.name = "Hooke3D",
                                         .ndims = 3,
                                         .isize = 6,
                                         .osize = 6,
                                         .apply = scapin_grop_hooke_3d_apply,
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

void scapin_grop_hooke_3d_apply(ScapinGreenOperator const *op, double const *k,
                                double const *tau, double *out) {
  double const mu = SCAPIN_GROP_HOOKE_DATA(op)->mu;
  double const nu = SCAPIN_GROP_HOOKE_DATA(op)->nu;
  double const k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
  double tau_k[] = {
      tau[0] * k[0] + M_SQRT1_2 * (tau[5] * k[1] + tau[4] * k[2]),
      tau[1] * k[1] + M_SQRT1_2 * (tau[5] * k[0] + tau[3] * k[2]),
      tau[2] * k[2] + M_SQRT1_2 * (tau[4] * k[0] + tau[3] * k[1])};
  double const n_tau_n =
      (k[0] * tau_k[0] + k[1] * tau_k[1] + k[2] * tau_k[2]) / k2;
  double const const1 = n_tau_n / (1. - nu);
  double const const2 = 1. / (2. * mu * k2);
  out[0] = const2 * (k[0] * (2. * tau_k[0] - const1 * k[0]));
  out[1] = const2 * (k[1] * (2. * tau_k[1] - const1 * k[1]));
  out[2] = const2 * (k[2] * (2. * tau_k[2] - const1 * k[2]));
  double const const3 = M_SQRT2 * const2;
  out[3] = const3 * (k[1] * tau_k[2] + k[2] * tau_k[1] - const1 * k[1] * k[2]);
  out[4] = const3 * (k[2] * tau_k[0] + k[0] * tau_k[2] - const1 * k[2] * k[0]);
  out[5] = const3 * (k[0] * tau_k[1] + k[1] * tau_k[0] - const1 * k[0] * k[1]);
}
