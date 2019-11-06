#include <mandel.h>

mandel3d_tens2_dot_vec(double const *a, double *x, double *y) {
  y[0] = a[0] * x[0] + M_SQRT1_2 * (a[5] * x[1] + a[4] * x[2]);
  y[1] = a[1] * x[1] + M_SQRT1_2 * (a[5] * x[0] + a[3] * x[2]);
  y[2] = a[2] * x[2] + M_SQRT1_2 * (a[4] * x[0] + a[3] * x[1]);
}
