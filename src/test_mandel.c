#include <glib.h>

#include <mandel.h>

void test_mandel3d_tens2_dot_vec(gconstpointer data) {
  size_t const dim = 3;
  size_t const sym = 6;
  double const *a = data;
  double const *x = data+sym;
  double act[dim];
}
