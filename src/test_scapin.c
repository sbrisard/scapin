#include <glib.h>

#include <scapin.h>

void test_grop_hooke_data(gconstpointer data) {
  const size_t ndims = *((size_t*)data);
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

void test_grop_hooke_setup_tests() {
  size_t *data1 = malloc(sizeof(size_t));
  data1[0] = 2;
  g_test_add_data_func("/Hooke2D/data", data1, free);

  size_t *data2 = malloc(sizeof(size_t));
  data2[0] = 3;
  g_test_add_data_func("/Hooke3D/data", data2, free);
}

int main(int argc, char **argv) {
  g_test_init(&argc, &argv, NULL);
  test_grop_hooke_setup_tests();
  return g_test_run();
}
